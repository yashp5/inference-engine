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
	"github.com/yashp5/inference-engine/internal/batcher"
	"github.com/yashp5/inference-engine/internal/config"
	"github.com/yashp5/inference-engine/internal/dispatcher"
	"github.com/yashp5/inference-engine/internal/queue"
	"github.com/yashp5/inference-engine/internal/scheduler"
	"github.com/yashp5/inference-engine/internal/types"
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
	pq := queue.NewPriorityQueue(cfg.MaxQueueDepth, cfg.QueueTimeout)

	reqCh := make(chan *types.InferRequest)
	dispatcher.NewDispatcher(pq, reqCh).Start(ctx)

	// nil in static mode: only the engine stream produces step stats
	var engineStats func() *types.EngineStats
	if cfg.ContinuousBatching {
		// The dispatcher feeds the engine directly. Batching now happens inside
		// the worker's step loop, so a Batcher here would be pure added latency.
		engine := scheduler.NewEngine(inferClient, reqCh, cfg.EngineSlots)
		engine.Start(ctx)
		engineStats = engine.Stats
		log.Printf("scheduler: continuous batching, engineSlots=%d", cfg.EngineSlots)
	} else {
		batchedReqCh := make(chan types.Batch)
		batcher.NewBatcher(reqCh, batchedReqCh, cfg.MaxBatchSize, cfg.MaxBatchWait).Start(ctx)
		scheduler.NewScheduler(inferClient, batchedReqCh, cfg.WorkerCount, cfg.MaxInflight).Start(ctx)
		log.Printf("scheduler: static batching, maxBatchSize=%d workerCount=%d maxInFlight=%d",
			cfg.MaxBatchSize, cfg.WorkerCount, cfg.MaxInflight)
	}

	h := api.NewHandler(inferClient, conn, r, pq, cfg.QueueTimeout, cfg.RequestTimeout, cfg.AgingInterval, engineStats)
	mux := api.NewMux(h)

	srv := &http.Server{Addr: cfg.HTTPAddr, Handler: mux}
	go srv.ListenAndServe()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	srv.Shutdown(ctx)
}
