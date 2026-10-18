package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/yashp5/inference-engine/internal/api"
	"github.com/yashp5/inference-engine/internal/config"
	"github.com/yashp5/inference-engine/internal/pipeline"
	"github.com/yashp5/inference-engine/internal/worker"
)

func main() {
	cfg := config.Load()

	ctx := context.Background()

	inferClient, conn, err := worker.New(cfg.WorkerAddr)
	if err != nil {
		panic(err)
	}
	conn.Connect()
	defer conn.Close()

	// N=0 is NewRateLimiter's off switch: it returns the no-op limiter
	rateLimitN := 0
	if cfg.RateLimit {
		rateLimitN = cfg.RateLimitN
	}
	r := api.NewRateLimiter(ctx, "TOKEN_BUCKET", rateLimitN, cfg.RateLimitWindow, cfg.RateLimitBucketTTL, cfg.RateLimitSweepInterval)
	log.Printf("rate limiter: enabled=%t n=%d window=%s", cfg.RateLimit, cfg.RateLimitN, cfg.RateLimitWindow)

	pipelineCfg := config.PipelineConfig{
		ContinuousBatching: cfg.ContinuousBatching,
		EngineSlots:        cfg.EngineSlots,
		MaxBatchSize:       cfg.MaxBatchSize,
		MaxBatchWait:       cfg.MaxBatchWait,
		WorkerCount:        cfg.WorkerCount,
		MaxInflight:        cfg.MaxInflight,
		MaxQueueDepth:      cfg.MaxQueueDepth,
		QueueTimeout:       cfg.QueueTimeout,
	}
	p := pipeline.New(ctx, inferClient, pipelineCfg)

	h := api.NewHandler(inferClient, conn, r, p, cfg.QueueTimeout, cfg.RequestTimeout, cfg.AgingInterval)
	mux := api.NewMux(h)

	srv := &http.Server{Addr: cfg.HTTPAddr, Handler: mux}
	go srv.ListenAndServe()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	srv.Shutdown(ctx)
	p.Close()
}
