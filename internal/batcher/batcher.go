package batcher

import (
	"context"
	"time"

	"github.com/yashp5/inference-engine/internal/types"
)

type Batcher struct {
	batchedReqCh chan<- types.Batch
	reqCh        <-chan *types.InferRequest
	batch        []*types.InferRequest
	maxBatchSize int
	maxBatchWait time.Duration
}

func NewBatcher(reqCh <-chan *types.InferRequest, batchedReqCh chan<- types.Batch, maxBatchSize int, maxBatchWait time.Duration) *Batcher {
	return &Batcher{
		reqCh:        reqCh,
		batchedReqCh: batchedReqCh,
		batch:        make([]*types.InferRequest, 0, maxBatchSize),
		maxBatchSize: maxBatchSize,
		maxBatchWait: maxBatchWait,
	}
}

func (b *Batcher) Start(ctx context.Context) {
	var timer *time.Timer
	var timerch <-chan time.Time
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case req := <-b.reqCh:
				b.batch = append(b.batch, req)
				if len(b.batch) == 1 {
					timer = time.NewTimer(b.maxBatchWait)
					timerch = timer.C
				}
				if len(b.batch) >= b.maxBatchSize {
					timer.Stop()
					timerch = nil
					b.Flush()
				}
			case <-timerch:
				b.Flush()
				timer.Stop()
				timerch = nil
			}
		}
	}()
}

func (b *Batcher) Flush() {
	if len(b.batch) == 0 {
		return
	}
	out := make([]*types.InferRequest, len(b.batch))
	copy(out, b.batch)
	b.batchedReqCh <- types.Batch(out) // blocking call
	b.batch = b.batch[:0]
}
