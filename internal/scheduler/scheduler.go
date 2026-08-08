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

type Worker struct {
	InferClient inferencepb.InferenceClient
	Batchch     chan types.Batch
}

func NewWorker(inferClient inferencepb.InferenceClient, batchch chan types.Batch) *Worker {
	return &Worker{
		InferClient: inferClient,
		Batchch:     batchch,
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

func (w *Worker) Process(b types.Batch) {
	wg := sync.WaitGroup{}

	// slot based scheduler
	// at every decode step, the worker reports which slots finished
	// shceduler immediately swaps in waiting requests to fill freed slots
	// the batch stays as full as possible at all times

	for _, req := range b {
		wg.Add(1)
		go func(req *types.InferRequest) {
			defer wg.Done()

			inferenceStart := time.Now()

			b := strings.Builder{}
			tokensGenerated := 0

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
				tokensGenerated += int(resp.TokensGenerated)
				if resp.Finished {
					break
				}
			}

			inferenceTime := time.Since(inferenceStart)
			respBody := &types.CompletionsReponse{
				RequestId:       req.Body.RequestId,
				GeneratedText:   b.String(),
				TokensGenerated: tokensGenerated,
				InferenceTimeMs: int(inferenceTime),
			}
			req.RespCh <- &types.InferResponse{Body: respBody, Error: nil}
		}(req)
	}

	wg.Wait()
}
