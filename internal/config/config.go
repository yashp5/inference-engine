package config

import (
	"flag"
	"time"
)

type Config struct {
	HTTPAddr               string        // 127.0.0.1:8080
	WorkerAddr             string        // 127.0.0.1:50051
	MaxQueueDepth          int           // step 4: 429 past this
	QueueTimeout           time.Duration // step 4: 504 past this
	AgingInterval          time.Duration
	MaxBatchSize           int // =1 disables batching
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
	queueTimeout := flag.Duration("queueTimeout", 5*time.Second, "queue request timeout")
	agingInterval := flag.Duration("agingInterval", 5*time.Second, "queue request aging interval")
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
		AgingInterval:          *agingInterval,
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
