package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yashp5/inference-serving-infra/internal/types"
)

const (
	defaultOutPath      = "bench/results/runs.csv"
	defaultStatsOutPath = "bench/results/stats.csv"
)

func main() {
	targetUrlPtr := flag.String("url", "http://localhost:8080/v1/completions", "target url")
	concurrencyPtr := flag.Int("concurrency", 3, "number of concurrent workers sending requests; ignored when -sweep is set")
	totalRequestsPtr := flag.Int("totalRequest", 1, "total requests to be made, per concurrency level; ignored when -requestsPerWorker is set")
	reqPerWorkerPtr := flag.Int("requestsPerWorker", 0, "requests each concurrent worker sends; total per level = this x concurrency. Overrides -totalRequest")
	sweepPtr := flag.String("sweep", "", "comma-separated concurrency levels to run in sequence, e.g. 1,10,50,100,500")
	outPtr := flag.String("out", defaultOutPath, "csv file to append one summary row per level to")
	statsOutPtr := flag.String("statsOut", defaultStatsOutPath, "csv file to append stats to")
	statsIntervalPtr := flag.Duration("statsInterval", 100*time.Millisecond, "stats interval duration")
	statsUrlPtr := flag.String("statsUrl", "", "stats endpoint; defaults to /stats on the -url host")
	flag.Parse()

	levels, err := parseSweep(*sweepPtr, *concurrencyPtr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid -sweep: %v\n", err)
		os.Exit(1)
	}

	if *reqPerWorkerPtr < 0 {
		fmt.Fprintf(os.Stderr, "-requestsPerWorker must not be negative, got %d\n", *reqPerWorkerPtr)
		os.Exit(1)
	}
	if *reqPerWorkerPtr == 0 && *totalRequestsPtr <= 0 {
		fmt.Fprintf(os.Stderr, "-totalRequest must be greater than 0, got %d\n", *totalRequestsPtr)
		os.Exit(1)
	}

	statsUrl := strings.TrimSpace(*statsUrlPtr)
	if statsUrl == "" {
		if statsUrl, err = deriveStatsUrl(*targetUrlPtr); err != nil {
			fmt.Fprintf(os.Stderr, "cannot derive -statsUrl from -url: %v\n", err)
			os.Exit(1)
		}
	}

	for _, level := range levels {
		// a fixed -totalRequest cannot hold total >= 10*concurrency across a
		// sweep: whatever value suits level 500 drowns level 1 in requests, and
		// whatever suits level 1 leaves level 500 measuring mostly ramp-up.
		// sizing per worker instead pins the ramp/drain artifact at
		// 0.5/requestsPerWorker regardless of level -- 10 gives 5%, 20 gives 2.5%
		total := *totalRequestsPtr
		if *reqPerWorkerPtr > 0 {
			total = *reqPerWorkerPtr * level
		}

		b := NewBenchmark(*targetUrlPtr, total, level)

		var poller *statsPoller
		if *statsOutPtr != "" {
			poller = NewStatsPoller(statsUrl, *statsIntervalPtr, level)
			poller.Start()
		}

		report := b.Run()

		// stop the poller before printing, so the samples cover exactly the run
		if poller != nil {
			samples := poller.Stop()
			records := make([][]string, 0, len(samples))
			for i := range samples {
				records = append(records, buildRecord(&samples[i]))
			}
			// buffered until here on purpose: writing every tick would put file
			// i/o inside the window being measured
			if err := writeToCSV(*statsOutPtr, csvHeader(statsSample{}), records...); err != nil {
				fmt.Fprintf(os.Stderr, "error writing stats csv: %v\n", err)
			}
		}

		printReport(report)
		if err := writeToCSV(*outPtr, csvHeader(Report{}), buildRecord(report)); err != nil {
			fmt.Fprintf(os.Stderr, "error writing runs csv: %v\n", err)
		}
	}
}

func printReport(report *Report) {
	fmt.Printf("concurrency: %d, total: %d, wall_ms: %d, util: %.1f%%, goodput_rps: %.2f, p50_ms: %d, p95_ms: %d, p99_ms: %d, ok: %d, rejected(429): %d (%.1f%%), timeout(408/504): %d, 5xx: %d, other: %d, errors: %d\n",
		report.concurreny, report.total, report.wallClockMs, report.utilizationPct,
		report.throughputRps,
		report.p50Ms, report.p95Ms, report.p99Ms,
		report.ok, report.rejected, report.rejectRatePct,
		report.timedout, report.serverErr, report.otherErr, len(report.errors))

	// a sample, not the whole slice: one level at concurrency 500 can produce
	// hundreds of identical errors, which buries every row printed after it
	if len(report.errors) > 0 {
		fmt.Printf("  first error: %v\n", report.errors[0])
	}
}

func parseSweep(sweep string, fallback int) ([]int, error) {
	if strings.TrimSpace(sweep) == "" {
		return []int{fallback}, nil
	}
	levels := []int{}
	for field := range strings.SplitSeq(sweep, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		n, err := strconv.Atoi(field)
		if err != nil {
			return nil, fmt.Errorf("%q is not a number", field)
		}
		if n <= 0 {
			return nil, fmt.Errorf("concurrency must be greater than 0, got %d", n)
		}
		levels = append(levels, n)
	}
	// catches a value that is non-empty but has no levels in it, e.g. ",,,"
	if len(levels) == 0 {
		return nil, fmt.Errorf("no levels found")
	}
	return levels, nil
}

// statsSample is one row of the stats csv. It carries the level and the offset
// into the run because one stats file accumulates every level of a sweep --
// without those two columns the samples can't be told apart or lined up.
type statsSample struct {
	Concurrency int   `csv:"concurrency"`
	ElapsedMs   int64 `csv:"elapsed_ms"`
	QueueDepth  int   `csv:"queue_depth"`
	InFlight    int64 `csv:"in_flight"`
}

type statsPoller struct {
	url         string
	interval    time.Duration
	concurrency int

	samples []statsSample
	stop    chan struct{}
	done    chan struct{}
}

func NewStatsPoller(url string, interval time.Duration, concurrency int) *statsPoller {
	return &statsPoller{
		url:         url,
		interval:    interval,
		concurrency: concurrency,
		stop:        make(chan struct{}),
		done:        make(chan struct{}),
	}
}

// Start polls until Stop is called. samples is written only by this goroutine
// and read only after Stop has seen done close, which orders the two -- so the
// slice needs no lock.
func (s *statsPoller) Start() {
	go func() {
		defer close(s.done)

		// one sequential request per tick, so a single connection is enough:
		// sizing this pool to the load level would reserve hundreds it never
		// uses. the timeout stays well under the interval, or a stalled server
		// turns the poller into a source of gaps instead of a measurement
		c := &http.Client{Timeout: 2 * time.Second}

		ticker := time.NewTicker(s.interval)
		defer ticker.Stop()

		start := time.Now()
		for {
			select {
			case <-s.stop:
				return
			case <-ticker.C:
				sample, err := s.sample(c, time.Since(start))
				if err != nil {
					continue // a missed tick is a gap in the series, not a failed run
				}
				s.samples = append(s.samples, sample)
			}
		}
	}()
}

func (s *statsPoller) sample(c *http.Client, elapsed time.Duration) (statsSample, error) {
	resp, err := c.Get(s.url)
	if err != nil {
		return statsSample{}, err
	}
	body, readErr := io.ReadAll(resp.Body)
	resp.Body.Close() // unconditional: an early return here would leak the conn
	if readErr != nil {
		return statsSample{}, readErr
	}
	if resp.StatusCode != http.StatusOK {
		return statsSample{}, fmt.Errorf("stats endpoint returned %d", resp.StatusCode)
	}

	// decode into the server's own type rather than a local copy: a field rename
	// on the server side then breaks the build here instead of silently
	// unmarshalling into zeros
	var st types.StatsResponse
	if err := json.Unmarshal(body, &st); err != nil {
		return statsSample{}, err
	}

	return statsSample{
		Concurrency: s.concurrency,
		ElapsedMs:   elapsed.Milliseconds(),
		QueueDepth:  st.QueueDepth,
		InFlight:    st.InFlight,
	}, nil
}

// Stop ends the polling goroutine and returns everything it collected.
func (s *statsPoller) Stop() []statsSample {
	close(s.stop)
	<-s.done
	return s.samples
}

// deriveStatsUrl points at /stats on the same host as -url so the two flags
// can't drift to different ports.
func deriveStatsUrl(target string) (string, error) {
	u, err := url.Parse(target)
	if err != nil {
		return "", err
	}
	if u.Host == "" {
		return "", fmt.Errorf("%q has no host", target)
	}
	u.Path = "/stats"
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

type Report struct {
	concurreny int `csv:"concurrency"`
	total      int `csv:"total"`
	// wall clock is what every derived rate divides by, so it belongs in the
	// csv rather than being back-computed from ok/goodput_rps
	wallClockMs int64 `csv:"wall_clock_ms"`
	// sum of per-request latencies, not elapsed time -- the old total_time_ms
	// name read like wall clock and is what made that easy to confuse
	sumLatencyMs int `csv:"sum_latency_ms"`
	// sumLatencyMs / (wallClockMs * concurrency): the share of the run during
	// which the offered load was actually at the requested level. counts only
	// 2xx, so it understates once a level starts shedding
	utilizationPct float64 `csv:"utilization_pct"`
	throughputRps  float64 `csv:"goodput_rps"`
	p50Ms          int     `csv:"p50_ms"`
	p95Ms          int     `csv:"p95_ms"`
	p99Ms          int     `csv:"p99_ms"`
	ok             int     `csv:"ok"`
	rejected       int     `csv:"rejected"`
	timedout       int     `csv:"timedout"`
	serverErr      int     `csv:"server_err"`
	otherErr       int     `csv:"other_err"`
	rejectRatePct  float64 `csv:"reject_rate_pct"`
	errors         []error `csv:"-"`
}

type Benchmark struct {
	url           string
	totalRequests int
	concurreny    int
	requestsMade  atomic.Int64

	ok        atomic.Int32 // 2xx
	rejected  atomic.Int32 // 429
	timedOut  atomic.Int32 // 504
	serverErr atomic.Int32 // 5xx (excluding 504)
	otherErr  atomic.Int32 // everything else (4xx, 3xx, ...)

	latencies []time.Duration
	errors    []error
	ch        chan time.Duration
	errch     chan error
}

func NewBenchmark(url string, totalRequests int, concurrency int) *Benchmark {
	return &Benchmark{
		url:           url,
		totalRequests: totalRequests,
		concurreny:    concurrency,
		requestsMade:  atomic.Int64{},
		latencies:     make([]time.Duration, 0, totalRequests),
		errors:        make([]error, 0, totalRequests),
		ch:            make(chan time.Duration, totalRequests),
		errch:         make(chan error, totalRequests),
	}
}

func (b *Benchmark) Run() *Report {
	c := http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			MaxIdleConnsPerHost: b.concurreny,
			MaxIdleConns:        b.concurreny,
		},
	}

	wallClockStart := time.Now()
	var wg sync.WaitGroup
	for range b.concurreny {
		wg.Add(1)
		go func(c http.Client, totalRequests int) {
			defer wg.Done()
			for b.requestsMade.Add(1) <= int64(totalRequests) {
				payload := &types.CompletionsRequest{
					Prompt:      "Once upon a time",
					MaxTokens:   20,
					Temperature: 0.7,
				}
				body, err := json.Marshal(payload)
				if err != nil {
					b.errch <- err
					continue
				}

				req, err := http.NewRequest("POST", b.url, bytes.NewBuffer(body))
				if err != nil {
					b.errch <- err
					continue
				}
				req.Header.Set("Content-Type", "application/json")

				now := time.Now()
				resp, err := c.Do(req)
				if err != nil {
					b.errch <- err
					continue
				}

				// Drain fully so keep-alive can reuse the connection, then close
				// Without this every request leaks a conn and the pool churns
				// which iteself inflates latency
				_, copyErr := io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				latency := time.Since(now) // measured after drain: full response time

				if copyErr != nil {
					b.errch <- fmt.Errorf("read body (status %d): %w", resp.StatusCode, copyErr)
					continue
				}

				switch {
				case resp.StatusCode >= 200 && resp.StatusCode < 300:
					b.ok.Add(1)
					b.ch <- latency
				case resp.StatusCode == http.StatusTooManyRequests: // 429
					b.rejected.Add(1)
				case resp.StatusCode == http.StatusGatewayTimeout || // 504
					resp.StatusCode == http.StatusRequestTimeout: // 408
					b.timedOut.Add(1)
				case resp.StatusCode >= 500:
					b.serverErr.Add(1)
				default:
					b.otherErr.Add(1)
					b.errch <- fmt.Errorf("unexpected status %d", resp.StatusCode)
				}
			}
		}(c, b.totalRequests)
	}

	wg.Wait()

	close(b.ch)
	close(b.errch)
	for v := range b.ch {
		b.latencies = append(b.latencies, v)
	}
	for e := range b.errch {
		b.errors = append(b.errors, e)
	}

	return b.BuildReport(time.Since(wallClockStart))
}

func (b *Benchmark) BuildReport(wallClock time.Duration) *Report {
	sumLatencyMs := 0
	for _, l := range b.latencies {
		sumLatencyMs += int(l.Milliseconds())
	}
	slices.Sort(b.latencies)

	ok := int(b.ok.Load())
	rejected := int(b.rejected.Load())
	timedout := int(b.timedOut.Load())
	serverErr := int(b.serverErr.Load())
	otherErr := int(b.otherErr.Load())

	completed := ok + rejected + timedout + serverErr + otherErr
	var rejectRate float64
	if completed > 0 {
		rejectRate = 100 * float64(rejected) / float64(completed)
	}

	report := &Report{
		concurreny:   b.concurreny,
		total:        b.totalRequests,
		wallClockMs:  wallClock.Milliseconds(),
		sumLatencyMs: sumLatencyMs,

		ok:            ok,
		rejected:      rejected,
		timedout:      timedout,
		serverErr:     serverErr,
		otherErr:      otherErr,
		rejectRatePct: rejectRate,
		errors:        b.errors,
	}

	if n := len(b.latencies); n > 0 {
		report.p50Ms = int(b.latencies[int(math.Ceil(float64(n)*0.5))-1].Milliseconds())
		report.p95Ms = int(b.latencies[int(math.Ceil(float64(n)*0.95))-1].Milliseconds())
		report.p99Ms = int(b.latencies[int(math.Ceil(float64(n)*0.99))-1].Milliseconds())
	}

	if secs := wallClock.Seconds(); secs > 0 {
		report.throughputRps = float64(ok) / secs
	}

	// busy-worker-milliseconds available over the run. a level that reads well
	// under 100% spent a visible slice of its wall clock ramping up or draining,
	// which depresses goodput_rps independently of anything the server did
	if capacityMs := float64(wallClock.Milliseconds()) * float64(b.concurreny); capacityMs > 0 {
		report.utilizationPct = 100 * float64(sumLatencyMs) / capacityMs
	}

	return report
}

// csvHeader and buildRecord both walk the csv tags of whatever struct they are
// handed, so header and row can't drift apart -- and both row types (Report and
// statsSample) share one implementation. Either takes a struct or a pointer.
func csvHeader(v any) []string {
	header := []string{}
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	for i := 0; i < t.NumField(); i++ {
		name := t.Field(i).Tag.Get("csv")
		if name == "" || name == "-" {
			continue
		}
		header = append(header, name)
	}
	return header
}

func buildRecord(r any) []string {
	record := []string{}
	v := reflect.ValueOf(r)
	if v.Kind() == reflect.Pointer {
		v = v.Elem()
	}
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		if name := t.Field(i).Tag.Get("csv"); name == "" || name == "-" {
			continue
		}
		value := v.Field(i)

		var str string
		switch value.Kind() {
		case reflect.String:
			str = value.String()
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			str = strconv.Itoa(int(value.Int()))
		case reflect.Float32, reflect.Float64:
			str = fmt.Sprintf("%f", value.Float())
		case reflect.Bool:
			str = fmt.Sprintf("%t", value.Bool())
		default:
			// Interface() panics on unexported fields, and every field of Report
			// is unexported -- so anything not handled above must not reach it
			if !value.CanInterface() {
				str = fmt.Sprintf("<unsupported %s>", value.Kind())
				break
			}
			str = fmt.Sprint(value.Interface())
		}
		record = append(record, str)
	}
	return record
}

func writeToCSV(filename string, header []string, records ...[]string) error {
	// MkdirAll, not Mkdir: it is idempotent, so levels 2..n of a sweep don't
	// fail with EEXIST once level 1 has created the directory
	if dir := filepath.Dir(filename); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// size rather than existence: O_CREATE means the file is always there by
	// now, so an empty one is the only kind that still needs a header
	info, err := file.Stat()
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if info.Size() == 0 {
		if err := writer.Write(header); err != nil {
			return err
		}
	}
	for _, record := range records {
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	// Write only buffers; errors surface on flush
	writer.Flush()
	return writer.Error()
}
