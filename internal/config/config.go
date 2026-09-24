package config

import (
	"flag"
	"time"
)

type Config struct {
	HTTPAddr               string        // 127.0.0.1:8080
	WorkerAddr             string        // 127.0.0.1:50051
	MaxQueueDepth          int           // step 4: 429 past this
	QueueTimeout           time.Duration // time-to-admission bound: 504 past this
	RequestTimeout         time.Duration // whole-request bound, generation included
	AgingInterval          time.Duration
	ContinuousBatching     bool // use the Engine bidi stream
	EngineSlots            int  // must match the worker's --engine-slots
	MaxBatchSize           int  // =1 disables batching
	MaxBatchWait           time.Duration
	MaxInflight            int // semaphore width
	WorkerCount            int
	RateLimit              bool
	RateLimitN             int
	RateLimitWindow        time.Duration
	RateLimitBucketTTL     time.Duration
	RateLimitSweepInterval time.Duration
}

func Load() *Config {
	httpAddr := flag.String("httpAddr", "127.0.0.1:8080", "go server http address")
	workerAddr := flag.String("workerAddr", "127.0.0.1:50051", "python server grpc address")
	maxQueueDepth := flag.Int("maxQueueDepth", 1000, "max request queue depth")
	queueTimeout := flag.Duration("queueTimeout", 5*time.Second, "max time a request may wait for admission")
	requestTimeout := flag.Duration("requestTimeout", 60*time.Second, "overall request deadline, queue plus generation")
	agingInterval := flag.Duration("agingInterval", 5*time.Second, "queue request aging interval")
	continuousBatching := flag.Bool("continuousBatching", false, "use the bidirectional Engine stream instead of per-request GenerateStream")
	engineSlots := flag.Int("engineSlots", 8, "engine slot count; must match the worker's --engine-slots")
	maxBatchSize := flag.Int("maxBatchSize", 8, "max batch size")
	maxBatchWait := flag.Duration("maxBatchWait", 10*time.Millisecond, "max batch wait")
	maxInFlight := flag.Int("maxInFlight", 10, "max in flight requests")
	workerCount := flag.Int("workerCount", 4, "scheduler worker count")
	rateLimit := flag.Bool("rateLimit", false, "rate limiter enablement")
	rateLimitN := flag.Int("rateLimitN", 100, "rate limiter request count")
	rateLimitWindow := flag.Duration("rateLimitWindow", time.Second, "rate limiter window")
	rateLimitBucketTTL := flag.Duration("rateLimitBucketTTL", 5*time.Minute, "rate limiter bucket TTL")
	rateLimitSweepInterval := flag.Duration("rateLimitSweepInterval", 1*time.Minute, "rate limiter sweep interval")
	flag.Parse()

	return &Config{
		HTTPAddr:               *httpAddr,
		WorkerAddr:             *workerAddr,
		MaxQueueDepth:          *maxQueueDepth,
		QueueTimeout:           *queueTimeout,
		RequestTimeout:         *requestTimeout,
		AgingInterval:          *agingInterval,
		ContinuousBatching:     *continuousBatching,
		EngineSlots:            *engineSlots,
		MaxBatchSize:           *maxBatchSize,
		MaxBatchWait:           *maxBatchWait,
		MaxInflight:            *maxInFlight,
		WorkerCount:            *workerCount,
		RateLimit:              *rateLimit,
		RateLimitN:             *rateLimitN,
		RateLimitWindow:        *rateLimitWindow,
		RateLimitBucketTTL:     *rateLimitBucketTTL,
		RateLimitSweepInterval: *rateLimitSweepInterval,
	}
}
