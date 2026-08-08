package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	inferencepb "github.com/yashp5/inference-serving-infra/gen"
	"github.com/yashp5/inference-serving-infra/internal/batcher"
	"github.com/yashp5/inference-serving-infra/internal/dispatcher"
	"github.com/yashp5/inference-serving-infra/internal/queue"
	"github.com/yashp5/inference-serving-infra/internal/scheduler"
	"github.com/yashp5/inference-serving-infra/internal/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
)

const (
	rateLimiterRequests = 10
	rateLimiterWindowMs = 100
	workerCount         = 3
)

type Handler struct {
	inferClient   inferencepb.InferenceClient
	conn          *grpc.ClientConn
	priorityQueue *queue.PriorityQueue
	rateLimiter   RateLimiter
}

func NewHandler(ctx context.Context, inferClient inferencepb.InferenceClient, conn *grpc.ClientConn) *Handler {
	pq := queue.NewPriorityQueue()

	reqCh := make(chan *types.InferRequest)
	dispatcher := dispatcher.NewDispatcher(pq, reqCh)
	dispatcher.Start(ctx)

	wchs := make([]chan types.Batch, 0, workerCount)
	for range workerCount {
		batchch := make(chan types.Batch, 10)
		worker := scheduler.NewWorker(inferClient, batchch)
		worker.Start(ctx)
		wchs = append(wchs, worker.Batchch)
	}
	batcher := batcher.NewBatcher(wchs, reqCh)
	batcher.Start(ctx)

	return &Handler{
		inferClient:   inferClient,
		conn:          conn,
		rateLimiter:   NewTokenBucketRateLimiter(rateLimiterRequests, rateLimiterWindowMs),
		priorityQueue: pq,
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	state := h.conn.GetState()
	if state == connectivity.Shutdown || state == connectivity.TransientFailure {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "worker_unavailable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handler) Infer(w http.ResponseWriter, r *http.Request) {
	if !h.rateLimiter.allow(r.RemoteAddr) {
		writeJSON(w, http.StatusTooManyRequests, "rate limited")
		return
	}
	id, _ := uuid.NewV7()
	requestId := id.String()

	reqBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, types.ErrorResponse{RequestId: requestId, Error: "failed to read request body"})
		return
	}
	defer r.Body.Close()

	reqBody := &types.CompletionsRequest{}
	if err := json.Unmarshal(reqBytes, reqBody); err != nil {
		writeJSON(w, http.StatusBadRequest, types.ErrorResponse{RequestId: requestId, Error: "failed to unmarshal req body"})
		return
	}

	id, _ = uuid.NewV7()
	reqBody.RequestId = id.String()

	if errMsg := reqBody.Validate(); errMsg != "" {
		writeJSON(w, http.StatusBadRequest, types.ErrorResponse{RequestId: requestId, Error: errMsg})
		return
	}

	req := &types.InferRequest{
		Body:     reqBody,
		Priority: types.PRIORITY_MEDIUM,
		Ctx:      r.Context(),
		RespCh:   make(chan *types.InferResponse),
	}
	h.priorityQueue.Push(req)

	select {
	case <-time.After(5 * time.Second):
		writeJSON(w, http.StatusRequestTimeout, types.ErrorResponse{RequestId: requestId, Error: "request timeout"})
	case resp := <-req.RespCh:
		if resp.Error != nil {
			writeJSON(w, http.StatusInternalServerError, types.ErrorResponse{RequestId: requestId, Error: resp.Error.Error()})
			return
		}
		writeJSON(w, http.StatusOK, resp.Body)
	}
}
