package dispatcher

import (
	"context"

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
				reqs, err := d.pq.PopAll() // fix ? need to use pop as after pop, a new req addition will result in different dispatch order
				if err != nil {
					continue
				}
				for _, req := range reqs {
					d.reqCh <- req // not buffered? will get blocked if the batcher cannot send the batch to the scheduler
				}
			}
		}
	}()
}
