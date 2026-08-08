package batcher

import (
	"context"
	"time"

	"github.com/yashp5/inference-serving-infra/internal/types"
)

const (
	maxBatchSize       = 8
	maxBatchWaitTimeMs = 100
)

type Batcher struct {
	batchedReqCh chan<- types.Batch
	reqCh        <-chan *types.InferRequest
	batch        []*types.InferRequest
}

func NewBatcher(reqCh <-chan *types.InferRequest, batchedReqCh chan<- types.Batch) *Batcher {
	return &Batcher{
		reqCh:        reqCh,
		batchedReqCh: batchedReqCh,
		batch:        make([]*types.InferRequest, 0, maxBatchSize),
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
					timer = time.NewTimer(maxBatchWaitTimeMs * time.Millisecond)
					timerch = timer.C
				}
				if len(b.batch) >= maxBatchSize {
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
	b.batchedReqCh <- types.Batch(out)
	b.batch = b.batch[:0]
}
