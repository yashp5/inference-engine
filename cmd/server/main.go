package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/yashp5/inference-serving-infra/internal/api"
	"github.com/yashp5/inference-serving-infra/internal/batcher"
	"github.com/yashp5/inference-serving-infra/internal/config"
	"github.com/yashp5/inference-serving-infra/internal/dispatcher"
	"github.com/yashp5/inference-serving-infra/internal/queue"
	"github.com/yashp5/inference-serving-infra/internal/scheduler"
	"github.com/yashp5/inference-serving-infra/internal/types"
	"github.com/yashp5/inference-serving-infra/internal/worker"
)

func main() {
	cfg := config.Load()

	ctx := context.Background()

	inferClient, conn, err := worker.New(cfg.WorkerAddr)
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	r := api.NewRateLimiter(ctx, "TOKEN_BUCKET", cfg.RateLimitN, cfg.RateLimitWindow, cfg.RateLimitBucketTTL, cfg.RateLimitSweepInterval)
	pq := queue.NewPriorityQueue(cfg.MaxQueueDepth, cfg.QueueTimeout)

	reqCh := make(chan *types.InferRequest)
	dispatcher.NewDispatcher(pq, reqCh).Start(ctx)

	batchedReqCh := make(chan types.Batch)
	batcher.NewBatcher(reqCh, batchedReqCh, cfg.MaxBatchSize, cfg.MaxBatchWait).Start(ctx)

	scheduler.NewScheduler(inferClient, batchedReqCh, cfg.WorkerCount).Start(ctx)

	h := api.NewHandler(inferClient, conn, r, pq)
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
