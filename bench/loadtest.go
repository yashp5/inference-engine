package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const (
	workerCount = 5
	csvFilePath = "/Users/yash/build/inference-engine/bench/runs.csv"
)

func main() {
	targetUrlPtr := flag.String("url", "http://localhost:8080/infer", "target url")
	concurrencyPtr := flag.Int("concurrency", 3, "number of concurrent workers sending requests")
	totalRequestsPtr := flag.Int("totalRequest", 1, "total requests to be made")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	b := NewBenchmark(*targetUrlPtr, *totalRequestsPtr, *concurrencyPtr)
	report, errs := b.Run(ctx)
	if len(errs) != 0 {
		fmt.Printf("%v\n", errs)
		return
	}

	fmt.Printf("concurrency: %d, total: %d, throughput_rps: %d, p50_ms: %d, p95_ms: %d, p99_ms: %d, errors: %v\n", report.concurreny, report.total, report.throughputRps, report.p50Ms, report.p95Ms, report.p99Ms, report.errors)

	record := buildRecord(report)
	err := writeToCSV(csvFilePath, record)
	if err != nil {
		fmt.Printf("error writing to csv: %v", err)
	}
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

type Benchmark struct {
	url           string
	totalRequests int
	concurreny    int
	requestsMade  atomic.Int64
	latencies     []int
	errors        []error
	ch            chan int
	errch         chan error
}

type Report struct {
	concurreny    int
	total         int
	totalTime     int
	throughputRps int
	p50Ms         int
	p95Ms         int
	p99Ms         int
	errors        []error
}

func NewBenchmark(url string, totalRequests int, concurrency int) *Benchmark {
	return &Benchmark{
		url:           url,
		totalRequests: totalRequests,
		concurreny:    concurrency,
		requestsMade:  atomic.Int64{},
		latencies:     make([]int, 0, totalRequests),
		errors:        make([]error, 0, totalRequests),
		ch:            make(chan int, totalRequests),
		errch:         make(chan error, totalRequests),
	}
}

type CompletionsRequest struct {
	RequestId   string  `json:"request_id"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

type CompletionsReponse struct {
	RequestId       string `json:"request_id"`
	GeneratedText   string `json:"generated_text"`
	TokensGenerated int    `json:"tokens_generated"`
	InferenceTimeMs int    `json:"inference_time_ms"`
}

type ErrorResponse struct {
	RequestId string `json:"request_id,omitempty"`
	Error     string `json:"error"`
}

func (b *Benchmark) Run(ctx context.Context) (*Report, []error) {
	c := http.Client{
		Timeout: 10 * time.Second,
	}

	wallclockstart := time.Now()
	var wg sync.WaitGroup
	for range b.concurreny {
		wg.Add(1)
		go func(c http.Client, totalRequests int) {
			defer wg.Done()
			for b.requestsMade.Add(1) <= int64(totalRequests) {
				payload := &CompletionsRequest{
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
				req.Header.Set("Authorization", "Bearer token")

				now := time.Now()
				_, err = c.Do(req)
				latency := time.Since(now)
				if err != nil {
					b.errch <- err
					continue
				}
				b.ch <- int(latency)
			}
		}(c, b.totalRequests)
	}

	wg.Wait()
	wallclockSeconds := time.Since(wallclockstart)

	close(b.ch)
	close(b.errch)
	for v := range b.ch {
		b.latencies = append(b.latencies, v)
	}
	for e := range b.errch {
		b.errors = append(b.errors, e)
	}

	return b.BuildReport(int(wallclockSeconds)), nil
}

func (b *Benchmark) BuildReport(wallClockSeconds int) *Report {
	totalTime := 0
	for _, l := range b.latencies {
		totalTime += l
	}
	slices.Sort(b.latencies)
	n := len(b.latencies)
	p50Ms := b.latencies[int((float64(n) * 0.5))]
	p95Ms := b.latencies[int((float64(n) * 0.95))]
	p99Ms := b.latencies[int(float64(n)*0.99)]
	report := &Report{
		concurreny:    b.concurreny,
		total:         b.totalRequests,
		totalTime:     totalTime,
		throughputRps: b.totalRequests / wallClockSeconds,
		p50Ms:         p50Ms,
		p95Ms:         p95Ms,
		p99Ms:         p99Ms,
		errors:        b.errors,
	}
	return report
}
