package scheduler

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	inferencepb "github.com/yashp5/inference-engine/gen"
	"github.com/yashp5/inference-engine/internal/types"
)

const (
	workerChBufferSize = 3
)

type Scheduler struct {
	inferClient  inferencepb.InferenceClient
	batchedReqCh <-chan types.Batch
	workerChs    []chan types.Batch
	sem          chan struct{}
	next         int
}

func NewScheduler(inferClient inferencepb.InferenceClient, batchedReqCh <-chan types.Batch, workerCount int, maxInFlight int) *Scheduler {
	workerChs := make([]chan types.Batch, 0, workerCount)
	for range workerCount {
		batchch := make(chan types.Batch, workerChBufferSize)
		workerChs = append(workerChs, batchch)
	}

	return &Scheduler{
		inferClient:  inferClient,
		batchedReqCh: batchedReqCh,
		workerChs:    workerChs,
		sem:          make(chan struct{}, maxInFlight),
		next:         0,
	}
}

func (s *Scheduler) Start(ctx context.Context) {
	for _, wch := range s.workerChs {
		worker := NewWorker(s.inferClient, wch, s.sem)
		worker.Start(ctx)
	}

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case b := <-s.batchedReqCh:
				s.Schedule(b)
			}
		}
	}()
}

func (s *Scheduler) Schedule(b types.Batch) {
	s.workerChs[s.next%len(s.workerChs)] <- types.Batch(b)
	s.next++
}

type Worker struct {
	InferClient inferencepb.InferenceClient
	Batchch     chan types.Batch
	sem         chan struct{}
}

func NewWorker(inferClient inferencepb.InferenceClient, batchch chan types.Batch, sem chan struct{}) *Worker {
	return &Worker{
		InferClient: inferClient,
		Batchch:     batchch,
		sem:         sem,
	}
}

func (w *Worker) Start(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case b := <-w.Batchch:
				w.Process(b) // batch needs to handled by the engine
				// engine.admitch <- req
			}
		}
	}()
}

func (w *Worker) acquireSem() {
	w.sem <- struct{}{}
}

func (w *Worker) releaseSem() {
	<-w.sem
}

func (w *Worker) Process(b types.Batch) {
	wg := sync.WaitGroup{}

	for _, req := range b {
		wg.Add(1)
		go func(req *types.InferRequest) {
			defer wg.Done()

			b := strings.Builder{}
			tokensGenerated := 0

			w.acquireSem()
			defer w.releaseSem()

			// after the semaphore, so queueing behind a full worker pool is not
			// reported to the client as model time
			inferenceStart := time.Now()

			stream, err := w.InferClient.GenerateStream(req.Ctx, &inferencepb.GenerateRequest{
				RequestId:   req.Body.RequestId,
				Prompt:      req.Body.Prompt,
				MaxTokens:   int32(req.Body.MaxTokens),
				Temperature: float32(req.Body.Temperature),
			})
			if err != nil {
				req.RespCh <- &types.InferResponse{Body: nil, Error: err}
				return
			}

			for {
				resp, err := stream.Recv()
				if err != nil {
					if err == io.EOF {
						break
					}
					req.RespCh <- &types.InferResponse{Body: nil, Error: err}
					return
				}
				b.WriteString(resp.Token)
				tokensGenerated = int(resp.TokensGenerated)
				if resp.Finished {
					break
				}
			}

			inferenceTime := time.Since(inferenceStart)
			respBody := &types.CompletionsReponse{
				RequestId:       req.Body.RequestId,
				GeneratedText:   b.String(),
				TokensGenerated: tokensGenerated,
				InferenceTimeMs: int(inferenceTime.Milliseconds()),
				QueueTimeMs:     int(inferenceStart.Sub(req.EnqueuedAt).Milliseconds()),
			}
			req.RespCh <- &types.InferResponse{Body: respBody, Error: nil}
		}(req)
	}

	wg.Wait()
}

type Engine struct {
	client   inferencepb.InferenceClient
	admitCh  chan *types.InferRequest
	cancelCh chan string
	sem      chan struct{} // width = n_slots
	mu       sync.Mutex
	inflight map[string]*inflight // request_id -> accumulator
	stats    atomic.Pointer[types.EngineStats]
}

type inflight struct {
	req       *types.InferRequest
	text      strings.Builder
	tokens    int
	startedAt time.Time
	done      chan struct{} // closed by take(); stops the watcher
}

func NewEngine(client inferencepb.InferenceClient, admitCh chan *types.InferRequest, nSlots int) *Engine {
	return &Engine{
		client:  client,
		admitCh: admitCh,
		// Buffered: watch() must never park here, or a request that finishes at
		// the same instant its context expires leaks a goroutine.
		cancelCh: make(chan string, nSlots),
		sem:      make(chan struct{}, nSlots),
		mu:       sync.Mutex{},
		inflight: make(map[string]*inflight),
	}
}

func (e *Engine) Start(ctx context.Context) {
	go func() {
		const minBackoff, maxBackoff = 100 * time.Millisecond, 5 * time.Second
		backoff := minBackoff
		for {
			if ctx.Err() != nil {
				return
			}
			began := time.Now()
			err := e.session(ctx)
			if ctx.Err() != nil {
				return
			}
			// A stream that survied a while is not a flapping worker
			if time.Since(began) > 30*time.Second {
				backoff = minBackoff
			}
			log.Printf("engine stream ended: %v; reconnecting in %s", err, backoff)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			if backoff *= 2; backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}()
}

// Stats returns the latest StepStats from the worker, or nil when no session is
// live. Safe to call from any goroutine; the snapshot must not be mutated.
func (e *Engine) Stats() *types.EngineStats {
	return e.stats.Load()
}

func (e *Engine) lookup(id string) (*inflight, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	f, ok := e.inflight[id]
	return f, ok
}

func (e *Engine) take(id string) (*inflight, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	f, ok := e.inflight[id]
	if ok {
		delete(e.inflight, id)
		// take is the only remover, so this closes exactly once. It stops the
		// cancel watcher for a request that has already been answered.
		close(f.done)
	}
	return f, ok
}

// respond hands the caller its answer and returns the slot to the pool. Only
// valid for an inflight obtained from take, which guarantees exactly once.
func (e *Engine) respond(f *inflight, resp *types.InferResponse) {
	<-e.sem
	f.req.RespCh <- resp
}

func (e *Engine) failAll(err error) {
	e.mu.Lock()
	pending := make([]*inflight, 0, len(e.inflight))
	for id, f := range e.inflight {
		pending = append(pending, f)
		delete(e.inflight, id)
		close(f.done)
	}
	e.mu.Unlock()
	for _, f := range pending {
		<-e.sem
		f.req.RespCh <- &types.InferResponse{Error: fmt.Errorf("engine stream failed: %w", err)}
	}
}

func (e *Engine) watch(f *inflight) {
	go func() {
		select {
		case <-f.done:
		case <-f.req.Ctx.Done():
			select {
			case e.cancelCh <- f.req.Body.RequestId:
			case <-f.done: // finished while we were trying to send
			}
		}
	}()
}

// sendLoop owns stream.Send. grpc.ClientStream.SendMsg is NOT safe for
// concurrent use, so this must be the only writer for the stream's lifetime.
func (e *Engine) sendLoop(ctx context.Context, stream inferencepb.Inference_EngineClient) {
	for {
		// Cancels first: freeing a slot is worth more than filling one, and a
		// plain select gives them equal odds.
		select {
		case id := <-e.cancelCh:
			if !e.sendCancel(stream, id) {
				return
			}
			continue
		default:
		}

		select {
		case <-ctx.Done():
			return
		case id := <-e.cancelCh:
			if !e.sendCancel(stream, id) {
				return
			}
		case req := <-e.admitCh:
			if !e.sendAdmit(ctx, stream, req) {
				return
			}
		}
	}
}

// sendCancel reports whether the stream is still usable.
func (e *Engine) sendCancel(stream inferencepb.Inference_EngineClient, id string) bool {
	err := stream.Send(&inferencepb.EngineRequest{
		Payload: &inferencepb.EngineRequest_Cancel{
			Cancel: &inferencepb.Cancel{RequestId: id},
		},
	})
	if err != nil {
		// A failed Send means the stream is gone; the receiver will see it too.
		log.Printf("engine: cancel send failed request_id=%s: %v", id, err)
		return false
	}
	return true
}

// sendAdmit reports whether the stream is still usable.
func (e *Engine) sendAdmit(ctx context.Context, stream inferencepb.Inference_EngineClient, req *types.InferRequest) bool {
	// The dispatcher checks this at pop time, but there is a window between
	// there and here. Don't burn a slot on a request nobody is waiting for.
	if req.Ctx.Err() != nil {
		return true
	}

	// A bare `e.sem <- struct{}{}` would park outside any select, so a dying
	// stream could not unblock it and this goroutine would outlive its session,
	// then race the next session's sender for admitCh.
	select {
	case e.sem <- struct{}{}:
	case <-ctx.Done():
		req.RespCh <- &types.InferResponse{Error: errors.New("engine stream closed")}
		return false
	}

	f := &inflight{req: req, startedAt: time.Now(), done: make(chan struct{})}
	e.mu.Lock()
	e.inflight[req.Body.RequestId] = f
	e.mu.Unlock()
	e.watch(f)

	err := stream.Send(&inferencepb.EngineRequest{
		Payload: &inferencepb.EngineRequest_Admit{
			Admit: &inferencepb.Admit{
				RequestId:   req.Body.RequestId,
				Prompt:      req.Body.Prompt,
				MaxTokens:   int32(req.Body.MaxTokens),
				Temperature: float32(req.Body.Temperature),
			},
		},
	})
	if err != nil {
		// It never reached the worker, so no event will ever come back for it.
		// Routed through take so the cancel watcher is stopped as well.
		if f, ok := e.take(req.Body.RequestId); ok {
			e.respond(f, &types.InferResponse{Error: err})
		}
		return false
	}
	return true
}

func (e *Engine) session(ctx context.Context) error {
	streamCtx, cancel := context.WithCancel(ctx)
	stream, err := e.client.Engine(streamCtx)
	if err != nil {
		cancel()
		return err
	}

	// Deferred calls run LIFO, so the declaration order here is deliberate:
	// cancel() runs first and unblocks sendLoop, then wg.Wait() collects it.
	// Reversing these two deadlocks on shutdown.
	var wg sync.WaitGroup
	defer wg.Wait()
	defer cancel()
	defer e.stats.Store(nil)

	wg.Add(1)
	go func() {
		defer wg.Done()
		e.sendLoop(streamCtx, stream)
	}()

	// receiver
	for {
		ev, err := stream.Recv()
		if err != nil {
			// Nobody else will answer these, and their sem tokens must come back
			// or the semaphore narrows permanently across reconnects.
			e.failAll(err)
			return err
		}
		switch p := ev.Payload.(type) {
		case *inferencepb.EngineEvent_Admitted:
			log.Printf("engine: admitted request_id=%s slot=%d",
				p.Admitted.RequestId, p.Admitted.SlotId)
		case *inferencepb.EngineEvent_Token:
			if f, ok := e.lookup(p.Token.RequestId); ok {
				f.text.WriteString(p.Token.Text)
				f.tokens = int(p.Token.Index) + 1
			}
		case *inferencepb.EngineEvent_Finished:
			if f, ok := e.take(p.Finished.RequestId); ok {
				// A mismatch means Token events went missing on the wire.
				if f.tokens != int(p.Finished.TokensGenerated) {
					log.Printf("engine: token count mismatch request_id=%s received=%d reported=%d",
						p.Finished.RequestId, f.tokens, p.Finished.TokensGenerated)
				}
				switch p.Finished.Reason {
				case inferencepb.FinishReason_FINISH_REASON_EOS, inferencepb.FinishReason_FINISH_REASON_LENGTH:
				default:
					// Cancelled, error (KV eviction, failed decode) or unset: the
					// text is truncated, so it must not go out as a 200.
					e.respond(f, &types.InferResponse{Error: fmt.Errorf("generation ended: %s after %d tokens",
						p.Finished.Reason, p.Finished.TokensGenerated)})
					continue
				}
				e.respond(f, &types.InferResponse{Body: &types.CompletionsReponse{
					RequestId:       p.Finished.RequestId,
					GeneratedText:   f.text.String(),
					TokensGenerated: int(p.Finished.TokensGenerated),
					InferenceTimeMs: int(time.Since(f.startedAt).Milliseconds()),
					QueueTimeMs:     int(f.startedAt.Sub(f.req.EnqueuedAt).Milliseconds()),
				}})
			}
		case *inferencepb.EngineEvent_Rejected:
			if f, ok := e.take(p.Rejected.RequestId); ok {
				e.respond(f, &types.InferResponse{Error: errors.New(p.Rejected.Reason)})
			}
		case *inferencepb.EngineEvent_Stats:
			st := p.Stats
			e.stats.Store(&types.EngineStats{
				Step:          st.Step,
				ActiveSlots:   int(st.ActiveSlots),
				FreeSlots:     int(st.FreeSlots),
				Waiting:       int(st.Waiting),
				BatchTokens:   int(st.BatchTokens),
				PrefillTokens: int(st.PrefillTokens),
				StepTimeUs:    int(st.StepTimeUs),
				KVUsed:        int(st.KvUsed),
				ObservedAt:    time.Now(),
			})
		}
	}
}
