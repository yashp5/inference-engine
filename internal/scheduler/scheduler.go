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
				w.Process(b)
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

	// TODO: continous batching
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
