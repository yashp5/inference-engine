package queue

import (
	"container/heap"
	"fmt"
	"sync"

	"github.com/yashp5/inference-serving-infra/internal/types"
)

type queue []*types.InferRequest

func (q queue) Len() int           { return len(q) }
func (q queue) Less(i, j int) bool { return q[i].Priority > q[j].Priority }
func (q queue) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }
func (q *queue) Push(x any)        { *q = append(*q, x.(*types.InferRequest)) }
func (q *queue) Pop() any {
	n := len(*q)
	x := (*q)[n-1]
	*q = (*q)[:n-1]
	return x
}

type PriorityQueue struct {
	mu     sync.Mutex
	buf    queue
	signal chan struct{}
}

func NewPriorityQueue() *PriorityQueue {
	return &PriorityQueue{
		buf:    queue{},
		signal: make(chan struct{}, 1),
	}
}

func (p *PriorityQueue) Push(r *types.InferRequest) {
	p.mu.Lock()
	defer p.mu.Unlock()
	heap.Push(&p.buf, r)
	select {
	case p.signal <- struct{}{}:
	default:
	}
}

func (p *PriorityQueue) Pop() (*types.InferRequest, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.buf) == 0 {
		return nil, fmt.Errorf("priority queue is empty")
	}
	return heap.Pop(&p.buf).(*types.InferRequest), nil
}

func (p *PriorityQueue) PopAll() ([]*types.InferRequest, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.buf) == 0 {
		return nil, fmt.Errorf("priority queue is empty")
	}
	reqs := make([]*types.InferRequest, 0, len(p.buf))
	for len(p.buf) > 0 {
		reqs = append(reqs, heap.Pop(&p.buf).(*types.InferRequest))
	}
	return reqs, nil
}

func (p *PriorityQueue) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.buf)
}

func (p *PriorityQueue) Signal() <-chan struct{} {
	return p.signal
}
