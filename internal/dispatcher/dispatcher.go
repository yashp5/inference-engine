package dispatcher

import (
	"context"
	"errors"
	"time"

	"github.com/yashp5/inference-engine/internal/queue"
	"github.com/yashp5/inference-engine/internal/types"
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
					if req.Ctx.Err() != nil {
						// Caller is already gone; nobody is reading RespCh.
						continue
					}
					if time.Since(req.EnqueuedAt) > d.pq.QueueTimeout() {
						// Answer it. Dropping silently leaves the handler blocked
						// until the much longer overall deadline fires.
						req.RespCh <- &types.InferResponse{Error: types.ErrQueueTimeout}
						continue
					}
					req.DispatchedAt = time.Now()
					d.reqCh <- req
				}
			}
		}
	}()
}
