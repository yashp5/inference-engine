package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yashp5/inference-engine/internal/types"
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
	maxTokensDistPtr := flag.String("maxTokensDist", "20:1", "max_tokens mix as tokens:weight pairs, e.g. 16:0.8,256:0.2")
	seedPtr := flag.Uint64("seed", 1, "seed for the max_tokens sequence; same seed, same workload")
	labelPtr := flag.String("label", "", "tag written to every csv row, e.g. static / continuous")
	reqOutPtr := flag.String("reqOut", "bench/results/requests.csv", "per-request csv")
	timeoutPtr := flag.Duration("timeout", 60*time.Second, "per-request client timeout")
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

	mixEntry, err := parseMix(*maxTokensDistPtr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot parse mix entry: %v\n", err)
		os.Exit(1)
	}
	for _, level := range levels {
		total := *totalRequestsPtr
		if *reqPerWorkerPtr > 0 {
			total = *reqPerWorkerPtr * level
		}

		workload := buildWorkload(mixEntry, total, *seedPtr)

		b := NewBenchmark(*targetUrlPtr, *labelPtr, *timeoutPtr, total, level, workload)

		var poller *statsPoller
		if *statsOutPtr != "" {
			poller = NewStatsPoller(statsUrl, *labelPtr, *statsIntervalPtr, level)
			poller.Start()
		}

		report, reqRecords := b.Run()

		if poller != nil {
			samples := poller.Stop()
			records := make([][]string, 0, len(samples))
			for i := range samples {
				records = append(records, buildRecord(&samples[i]))
			}
			if err := writeToCSV(*statsOutPtr, csvHeader(statsSample{}), records...); err != nil {
				fmt.Fprintf(os.Stderr, "error writing stats csv: %v\n", err)
			}
		}

		fmt.Printf("label: %s, concurrency: %d, total: %d, wall_ms: %d, util: %.1f%%, goodput_rps: %.2f, output_tok_s: %.1f, p50_ms: %d, p95_ms: %d, p99_ms: %d, ttft_p50_ms: %d, ttft_p99_ms: %d, ok: %d, rejected(429): %d (%.1f%%), timeout(408/504): %d, 5xx: %d, other: %d, errors: %d\n",
			report.label, report.concurreny, report.total, report.wallClockMs, report.utilizationPct,
			report.throughputRps, report.outputTokS,
			report.p50Ms, report.p95Ms, report.p99Ms, report.ttftP50Ms, report.ttftP99Ms,
			report.ok, report.rejected, report.rejectRatePct,
			report.timedout, report.serverErr, report.otherErr, len(report.errors))

		if len(report.errors) > 0 {
			fmt.Printf("  first error: %v\n", report.errors[0])
		}

		if err := writeToCSV(*outPtr, csvHeader(Report{}), buildRecord(report)); err != nil {
			fmt.Fprintf(os.Stderr, "error writing runs csv: %v\n", err)
		}

		rows := make([][]string, 0, len(reqRecords))
		for i := range reqRecords {
			rows = append(rows, buildRecord(&reqRecords[i]))
		}
		if err := writeToCSV(*reqOutPtr, csvHeader(reqRecord{}), rows...); err != nil {
			fmt.Fprintf(os.Stderr, "error writing req records csv: %v\n", err)
		}
	}
}

// ---------- Benchmark ----------

type Benchmark struct {
	url           string
	label         string
	reqTimeout    time.Duration
	totalRequests int
	workload      []int
	concurreny    int
	requestsMade  atomic.Int64
	start         time.Time // set before any worker runs; StartMs offsets from it

	ok        atomic.Int32 // 2xx
	rejected  atomic.Int32 // 429
	timedOut  atomic.Int32 // 504
	serverErr atomic.Int32 // 5xx (excluding 504)
	otherErr  atomic.Int32 // everything else (4xx, 3xx, ...)

	latencies  []time.Duration
	reqRecords []reqRecord
	errors     []error
	ch         chan time.Duration
	reqCh      chan reqRecord
	errch      chan error
}

type reqRecord struct {
	Label       string `csv:"label"`
	Concurrency int    `csv:"concurrency"`
	Idx         int    `csv:"idx"`
	MaxTokens   int    `csv:"max_tokens"`
	Status      int    `csv:"status"` // 0: no response (client timeout, conn error)
	StartMs     int64  `csv:"start_ms"`
	LatencyMs   int64  `csv:"latency_ms"`
	QueueMs     int    `csv:"queue_ms"`
	TTFTMs      int    `csv:"ttft_ms"` // -1: 200 whose body didn't decode
	Tokens      int    `csv:"tokens"`
}

type Report struct {
	label      string `csv:"label"`
	concurreny int    `csv:"concurrency"`
	total      int    `csv:"total"`
	// wall clock is what every derived rate divides by, so it belongs in the
	// csv rather than being back-computed from ok/goodput_rps
	wallClockMs int64 `csv:"wall_clock_ms"`
	// sum of per-request latencies, not elapsed time
	sumLatencyMs int `csv:"sum_latency_ms"`
	// sumLatencyMs / (wallClockMs * concurrency): the share of the run during
	// which the offered load was actually at the requested level. counts only
	// 2xx, so it understates once a level starts shedding
	utilizationPct float64 `csv:"utilization_pct"`
	throughputRps  float64 `csv:"goodput_rps"`
	outputTokS     float64 `csv:"output_tok_s"` // sum of tokens / wall clock
	p50Ms          int     `csv:"p50_ms"`
	p95Ms          int     `csv:"p95_ms"`
	p99Ms          int     `csv:"p99_ms"`
	ttftP50Ms      int     `csv:"ttft_p50_ms"`
	ttftP99Ms      int     `csv:"ttft_p99_ms"`
	ok             int     `csv:"ok"`
	rejected       int     `csv:"rejected"`
	timedout       int     `csv:"timedout"`
	serverErr      int     `csv:"server_err"`
	otherErr       int     `csv:"other_err"`
	rejectRatePct  float64 `csv:"reject_rate_pct"`
	errors         []error `csv:"-"`
}

func NewBenchmark(url string, label string, reqTimeout time.Duration, totalRequests int, concurrency int, workload []int) *Benchmark {
	return &Benchmark{
		url:           url,
		label:         label,
		reqTimeout:    reqTimeout,
		totalRequests: totalRequests,
		concurreny:    concurrency,
		workload:      workload,
		requestsMade:  atomic.Int64{},
		latencies:     make([]time.Duration, 0, totalRequests),
		reqRecords:    make([]reqRecord, 0, totalRequests),
		errors:        make([]error, 0, totalRequests),
		ch:            make(chan time.Duration, totalRequests),
		reqCh:         make(chan reqRecord, totalRequests),
		errch:         make(chan error, totalRequests),
	}
}

func (b *Benchmark) Run() (*Report, []reqRecord) {
	c := http.Client{
		// a 256-token request queued behind a static batch can take well over
		// the old hardcoded 60s, which turned slow successes into client errors
		Timeout: b.reqTimeout,
		Transport: &http.Transport{
			MaxIdleConnsPerHost: b.concurreny,
			MaxIdleConns:        b.concurreny,
		},
	}

	wallClockStart := time.Now()
	b.start = wallClockStart
	var wg sync.WaitGroup
	for range b.concurreny {
		wg.Add(1)
		go func(c http.Client, totalRequests int) {
			defer wg.Done()
			for {
				n := b.requestsMade.Add(1)
				if n > int64(b.totalRequests) {
					return
				}
				idx := int(n - 1)
				payload := &types.CompletionsRequest{
					Prompt:      "Once upon a time",
					MaxTokens:   b.workload[idx],
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

				sent := time.Now()
				rec := reqRecord{
					Label:       b.label,
					Concurrency: b.concurreny,
					Idx:         idx,
					MaxTokens:   b.workload[idx],
					StartMs:     sent.Sub(b.start).Milliseconds(),
				}

				resp, err := c.Do(req)
				if err != nil {
					// still a row: a client timeout is exactly the slow request
					// the per-request csv exists to show
					rec.LatencyMs = time.Since(sent).Milliseconds()
					b.reqCh <- rec
					b.errch <- err
					continue
				}

				body, readErr := io.ReadAll(resp.Body)
				resp.Body.Close()
				latency := time.Since(sent)
				rec.Status = resp.StatusCode
				rec.LatencyMs = latency.Milliseconds()

				if readErr != nil {
					b.reqCh <- rec
					b.errch <- fmt.Errorf("read body (status %d): %w", resp.StatusCode, readErr)
					continue
				}

				if resp.StatusCode == http.StatusOK {
					var cr types.CompletionsReponse
					if err := json.Unmarshal(body, &cr); err != nil {
						// otherwise it lands as ttft 0 / tokens 0 and drags the percentiles down
						rec.TTFTMs = -1 // excluded from the ttft percentiles
						b.errch <- fmt.Errorf("decode 200 body: %w", err)
					} else {
						rec.QueueMs = cr.QueueTimeMs
						rec.TTFTMs = cr.TTFTMs
						rec.Tokens = cr.TokensGenerated
					}
				}
				b.reqCh <- rec

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
	close(b.reqCh)
	close(b.errch)
	for v := range b.ch {
		b.latencies = append(b.latencies, v)
	}
	for r := range b.reqCh {
		b.reqRecords = append(b.reqRecords, r)
	}
	for e := range b.errch {
		b.errors = append(b.errors, e)
	}

	return b.BuildReport(time.Since(wallClockStart)), b.reqRecords
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
		label:        b.label,
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

	// ttft and tokens only exist on parsed 200s; everything else carries zeros
	var ttfts []int
	var tokens int
	for _, r := range b.reqRecords {
		if r.Status == http.StatusOK && r.TTFTMs >= 0 {
			ttfts = append(ttfts, r.TTFTMs)
			tokens += r.Tokens
		}
	}
	if n := len(ttfts); n > 0 {
		slices.Sort(ttfts)
		report.ttftP50Ms = ttfts[int(math.Ceil(float64(n)*0.5))-1]
		report.ttftP99Ms = ttfts[int(math.Ceil(float64(n)*0.99))-1]
	}

	if secs := wallClock.Seconds(); secs > 0 {
		report.throughputRps = float64(ok) / secs
		report.outputTokS = float64(tokens) / secs
	}

	// busy-worker-milliseconds available over the run. a level that reads well
	// under 100% spent a visible slice of its wall clock ramping up or draining,
	// which depresses goodput_rps independently of anything the server did
	if capacityMs := float64(wallClock.Milliseconds()) * float64(b.concurreny); capacityMs > 0 {
		report.utilizationPct = 100 * float64(sumLatencyMs) / capacityMs
	}

	return report
}

// ---------- Stats Sampler ----------

type statsSample struct {
	Label       string `csv:"label"`
	Concurrency int    `csv:"concurrency"`
	ElapsedMs   int64  `csv:"elapsed_ms"`
	QueueDepth  int    `csv:"queue_depth"`
	InFlight    int64  `csv:"in_flight"`
	ActiveSlots int    `csv:"active_slots"` // -1 when the server runs no engine
	Waiting     int    `csv:"engine_waiting"`
	Step        int64  `csv:"step"`
}

type statsPoller struct {
	url         string
	label       string
	interval    time.Duration
	concurrency int

	samples []statsSample
	stop    chan struct{}
	done    chan struct{}
}

func NewStatsPoller(url string, label string, interval time.Duration, concurrency int) *statsPoller {
	return &statsPoller{
		url:         url,
		label:       label,
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
	resp.Body.Close()
	if readErr != nil {
		return statsSample{}, readErr
	}
	if resp.StatusCode != http.StatusOK {
		return statsSample{}, fmt.Errorf("stats endpoint returned %d", resp.StatusCode)
	}

	var st types.StatsResponse
	if err := json.Unmarshal(body, &st); err != nil {
		return statsSample{}, err
	}

	out := statsSample{
		Label:       s.label,
		Concurrency: s.concurrency,
		ElapsedMs:   elapsed.Milliseconds(),
		QueueDepth:  st.QueueDepth,
		InFlight:    st.InFlight,
		ActiveSlots: -1,
		Waiting:     -1,
	}
	if st.Engine != nil {
		out.ActiveSlots = st.Engine.ActiveSlots
		out.Waiting = st.Engine.Waiting
		out.Step = st.Engine.Step
	}
	return out, nil
}

// Stop ends the polling goroutine and returns everything it collected. It
// waits on done: a tick mid-sample would otherwise still be appending while
// the caller reads the slice.
func (s *statsPoller) Stop() []statsSample {
	close(s.stop)
	<-s.done
	return s.samples
}
