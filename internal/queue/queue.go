package queue

import (
	"container/heap"
	"errors"
	"sync"
	"time"

	"github.com/yashp5/inference-engine/internal/types"
)

var (
	ErrQueueEmpty = errors.New("queue is empty")
	ErrQueueFull  = errors.New("queue is full")
)

// an aged request may climb at most this far, which lets a starved LOW draw
// level with a fresh HIGH but never outrank one
const maxAgingBoost = int(types.PRIORITY_HIGH)

type queue []*types.InferRequest

// score is priority plus one point per AgingInterval spent waiting. an
// AgingInterval of 0 disables aging rather than dividing by zero -- Less runs
// on the dispatcher goroutine, where a panic takes the whole process down.
func score(r *types.InferRequest) int {
	if r.AgingInterval <= 0 {
		return int(r.Priority)
	}
	boost := int(time.Since(r.EnqueuedAt) / r.AgingInterval)
	if boost > maxAgingBoost {
		boost = maxAgingBoost
	}
	return int(r.Priority) + boost
}

func (q queue) Len() int           { return len(q) }
func (q queue) Less(i, j int) bool { return score(q[i]) > score(q[j]) }
func (q queue) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }
func (q *queue) Push(x any)        { *q = append(*q, x.(*types.InferRequest)) }
func (q *queue) Pop() any {
	n := len(*q)
	x := (*q)[n-1]
	*q = (*q)[:n-1]
	return x
}

type PriorityQueue struct {
	mu            sync.Mutex
	buf           queue
	maxQueueDepth int
	queueTimeout  time.Duration
	signal        chan struct{}
}

func NewPriorityQueue(maxQueueDepth int, queueTimeout time.Duration) *PriorityQueue {
	return &PriorityQueue{
		buf:           queue{},
		maxQueueDepth: maxQueueDepth,
		queueTimeout:  queueTimeout,
		signal:        make(chan struct{}, 1),
	}
}

func (p *PriorityQueue) Push(r *types.InferRequest) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.buf) >= p.maxQueueDepth {
		return ErrQueueFull
	}
	heap.Push(&p.buf, r)
	select {
	case p.signal <- struct{}{}:
	default:
	}
	return nil
}

func (p *PriorityQueue) Pop() (*types.InferRequest, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.buf) == 0 {
		return nil, ErrQueueEmpty
	}
	return heap.Pop(&p.buf).(*types.InferRequest), nil
}

func (p *PriorityQueue) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.buf)
}

func (p *PriorityQueue) Signal() <-chan struct{} {
	return p.signal
}

func (p *PriorityQueue) QueueTimeout() time.Duration {
	return p.queueTimeout
}
