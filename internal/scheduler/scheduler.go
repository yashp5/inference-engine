package scheduler

import (
	"context"
	"io"
	"strings"
	"sync"
	"time"

	inferencepb "github.com/yashp5/inference-serving-infra/gen"
	"github.com/yashp5/inference-serving-infra/internal/types"
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

// TODO: continous batching
// next batch is processed only once the current batchs all req are processed.
// right now, per batch each request is processed as soon as all the tokens are streamed
// still we do cannot process another req in the empty slot of the Batchch
// if out of 8 req, 2 have already completed, these slots remain empty until the complete batch is processed
// we need some kind of bidirectional stream that allows us to submit new requests while the worker keeps streaming tokens
// worker can either can admit a new request or reject it
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
	sem      chan struct{} // width = n_slots
	mu       sync.Mutex
	inflight map[string]*inflight // request_id -> accumulator
}

type inflight struct {
	req       *types.InferRequest
	text      strings.Builder
	tokens    int
	startedAt time.Time
}

func NewEngine(client inferencepb.InferenceClient, admitCh chan *types.InferRequest, nSlots int) *Engine {
	return &Engine{
		client:   client,
		admitCh:  admitCh,
		sem:      make(chan struct{}, nSlots),
		mu:       sync.Mutex{},
		inflight: make(map[string]*inflight),
	}
}

func (e *Engine) start(ctx context.Context) error {
	stream, err := e.client.Engine(ctx)
	if err != nil {
		return err
	}

	// sender: one goroutine owns send. grpc.ClientStream.SendMsg is NOT
	// safe for concurrent use. This must be the only writer.
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case req := <-e.admitCh:
				e.sem <- struct{}{}
				e.mu.Lock()
				e.inflight[req.Body.RequestId] = &inflight{req: req, startedAt: time.Now()}
				e.mu.Unlock()
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
					// the request never reached the worker, so no event will ever
					// come back for it: unwind the slot and answer the caller here.
					e.mu.Lock()
					delete(e.inflight, req.Body.RequestId)
					e.mu.Unlock()
					<-e.sem
					req.RespCh <- &types.InferResponse{Body: nil, Error: err}
					return
				}
			}
		}
	}()

	// receiver
	for {
		ev, err := stream.Recv()
		if err != nil {
			return err
		}
		switch p := ev.Payload.(type) {
		case *inferencepb.EngineEvent_Token:
			e.mu.Lock()
			e.inflight[p.Token.RequestId].text.WriteString(p.Token.Text)
			e.mu.Unlock()
		case *inferencepb.EngineEvent_Finished:
			e.mu.Lock()
			b := &types.CompletionsReponse{
				RequestId:       p.Finished.RequestId,
				GeneratedText:   e.inflight[p.Finished.RequestId].text.String(),
				TokensGenerated: int(p.Finished.TokensGenerated),
			}
			<-e.sem
			e.inflight[p.Finished.RequestId].req.RespCh <- &types.InferResponse{Body: b, Error: err}
			delete(e.inflight, p.Finished.RequestId)
			e.mu.Unlock()
		case *inferencepb.EngineEvent_Rejected:
			e.mu.Lock()
			req := e.inflight[p.Rejected.RequestId]
			delete(e.inflight, p.Rejected.RequestId)
			e.mu.Unlock()
			<-e.sem
			req.req.RespCh <- &types.InferResponse{Body: nil, Error: err}
		case *inferencepb.EngineEvent_Stats:
			// metrics
		}
	}
}
