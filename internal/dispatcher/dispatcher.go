package dispatcher

import (
	"context"
	"errors"
	"time"

	"github.com/yashp5/inference-serving-infra/internal/queue"
	"github.com/yashp5/inference-serving-infra/internal/types"
)

type Dispatcher struct {
	pq    *queue.PriorityQueue
	reqCh chan<- *types.InferRequest
}

func NewDispatcher(pq *queue.PriorityQueue, reqCh chan<- *types.InferRequest) *Dispatcher {
	return &Dispatcher{
		pq:    pq,
		reqCh: reqCh,
	}
}

func (d *Dispatcher) Start(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-d.pq.Signal():
				for {
					req, err := d.pq.Pop()
					if errors.Is(err, queue.ErrQueueEmpty) {
						break
					}
					if req.Ctx.Err() != nil || time.Since(req.EnqueuedAt) > d.pq.QueueTimeout() {
						continue
					}
					req.DispatchedAt = time.Now()
					d.reqCh <- req
				}
			}
		}
	}()
}
