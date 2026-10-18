package pipeline

import (
	"context"
	"log"

	inferencepb "github.com/yashp5/inference-engine/gen"
	"github.com/yashp5/inference-engine/internal/batcher"
	"github.com/yashp5/inference-engine/internal/config"
	"github.com/yashp5/inference-engine/internal/dispatcher"
	"github.com/yashp5/inference-engine/internal/queue"
	"github.com/yashp5/inference-engine/internal/scheduler"
	"github.com/yashp5/inference-engine/internal/types"
)

type Pipeline struct {
	Pq          *queue.PriorityQueue
	EngineStats func() *types.EngineStats
	Cancel      context.CancelFunc
}

func New(ctx context.Context, inferClient inferencepb.InferenceClient, cfg config.PipelineConfig) *Pipeline {
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

	_, cancel := context.WithCancel(ctx)
	return &Pipeline{
		Pq:          pq,
		EngineStats: engineStats,
		Cancel:      cancel,
	}
}

func (p *Pipeline) Submit(req *types.InferRequest) error {
	return p.Pq.Push(req)
}

func (p *Pipeline) Close() {
	p.Cancel()
}
