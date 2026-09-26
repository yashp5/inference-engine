// Package pipeline is one model's request path: priority queue → dispatcher →
// (batcher → scheduler | engine), bound to a single worker connection.
//
// Today cmd/server/main.go builds exactly one of these inline. Phase 5 needs
// one per loaded model, built on load and torn down on unload, so that
// construction moves here. See docs/phase5-multi-model.md, step 1.
package pipeline

import (
	"context"
	"errors"
	"time"

	inferencepb "github.com/yashp5/inference-engine/gen"
	"github.com/yashp5/inference-engine/internal/types"
)

var ErrNotImplemented = errors.New("pipeline: not implemented")

// Config is the per-model slice of what config.Config holds globally today.
type Config struct {
	ContinuousBatching bool
	EngineSlots        int // continuous mode; Go now starts the worker, so it passes the same value to both sides

	MaxBatchSize int // static mode
	MaxBatchWait time.Duration
	WorkerCount  int
	MaxInflight  int

	MaxQueueDepth int
	QueueTimeout  time.Duration
}

// Pipeline owns one model's queue and the goroutines that drain it.
type Pipeline struct {
	// TODO(phase5): *queue.PriorityQueue, the engine's Stats func (nil in static
	// mode), and the cancel func for the context every goroutine here runs under.
}

// New builds and starts the chain main.go builds today, under a child of ctx.
//
// TODO(phase5): move the wiring out of main.go. Everything started here must
// stop when Close cancels ctx; scheduler.Engine already exits its reconnect
// loop on ctx.Done(), which is what stops it retrying a dead worker forever.
func New(ctx context.Context, client inferencepb.InferenceClient, cfg Config) (*Pipeline, error) {
	return nil, ErrNotImplemented
}

// Submit enqueues a request. Returns queue.ErrQueueFull when the queue is at
// MaxQueueDepth, which the handler maps to 429 as it does today.
func (p *Pipeline) Submit(req *types.InferRequest) error {
	return ErrNotImplemented
}

// Stats reports this model's queue depth and, in continuous mode, the latest
// engine step stats.
func (p *Pipeline) Stats() types.StatsResponse {
	return types.StatsResponse{}
}

// Close cancels the pipeline's context. Requests still in flight are answered
// with errors by the dispatcher and engine as their channels shut down.
func (p *Pipeline) Close() {}
