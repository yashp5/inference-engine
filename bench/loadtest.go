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
	"os"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yashp5/inference-serving-infra/internal/types"
)

const (
	csvFilePath = "/Users/yash/build/inference-engine/bench/runs.csv"
)

func main() {
	targetUrlPtr := flag.String("url", "http://localhost:8080/infer", "target url")
	concurrencyPtr := flag.Int("concurrency", 3, "number of concurrent workers sending requests")
	totalRequestsPtr := flag.Int("totalRequest", 1, "total requests to be made")
	flag.Parse()

	b := NewBenchmark(*targetUrlPtr, *totalRequestsPtr, *concurrencyPtr)
	report := b.Run()

	fmt.Printf("concurrency: %d, total: %d, goodput_rps: %.2f, p50_ms: %d, p95_ms: %d, p99_ms: %d, ok: %d, rejected(429): %d (%.1f%%), timeout(504): %d, 5xx: %d, other: %d, errors: %v\n",
		report.concurreny, report.total, report.throughputRps,
		report.p50Ms, report.p95Ms, report.p99Ms,
		report.ok, report.rejected, report.rejectRatePct,
		report.timedout, report.serverErr, report.otherErr, report.errors)

	record := buildRecord(report)
	err := writeToCSV(csvFilePath, record)
	if err != nil {
		fmt.Printf("error writing to csv: %v", err)
	}
}

type Report struct {
	concurreny    int
	total         int
	totalTime     int
	throughputRps float64
	p50Ms         int
	p95Ms         int
	p99Ms         int
	ok            int
	rejected      int
	timedout      int
	serverErr     int
	otherErr      int
	rejectRatePct float64
	errors        []error
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
				case resp.StatusCode == http.StatusGatewayTimeout: // 504
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
	totalTime := 0
	for _, l := range b.latencies {
		totalTime += int(l.Milliseconds())
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
		concurreny:    b.concurreny,
		total:         b.totalRequests,
		totalTime:     totalTime,
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

	return report
}

func buildRecord(r *Report) []string {
	record := []string{}
	v := reflect.ValueOf(r).Elem()
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.Name == "errors" {
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
			str = fmt.Sprint(value.Interface())
		}
		record = append(record, str)
	}
	return record
}

func writeToCSV(filename string, record []string) error {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	return writer.Write(record)
}
