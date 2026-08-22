package api

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	inferencepb "github.com/yashp5/inference-serving-infra/gen"
	"github.com/yashp5/inference-serving-infra/internal/queue"
	"github.com/yashp5/inference-serving-infra/internal/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
)

type Handler struct {
	inferClient   inferencepb.InferenceClient
	conn          *grpc.ClientConn
	priorityQueue *queue.PriorityQueue
	rateLimiter   RateLimiter
	inflight      atomic.Int64
}

func NewHandler(inferClient inferencepb.InferenceClient, conn *grpc.ClientConn, r RateLimiter, pq *queue.PriorityQueue) *Handler {
	return &Handler{
		inferClient:   inferClient,
		conn:          conn,
		rateLimiter:   r,
		priorityQueue: pq,
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (h *Handler) Healthz(w http.ResponseWriter, r *http.Request) {
	state := h.conn.GetState()
	if state == connectivity.Idle {
		h.conn.Connect()
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "worker_not_ready"})
		return
	}
	if !(state == connectivity.Ready) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "worker_unavailable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func clientKey(r *http.Request) string {
	if k := r.Header.Get("X-API-Key"); k != "" {
		return k
	}
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return host
	}
	return r.RemoteAddr
}

func (h *Handler) Stats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, types.StatsResponse{
		QueueDepth: h.priorityQueue.Len(),
		InFlight:   h.inflight.Load(),
	})
}

func (h *Handler) Completions(w http.ResponseWriter, r *http.Request) {
	receivedAt := time.Now()
	if !h.rateLimiter.allow(clientKey(r)) {
		writeJSON(w, http.StatusTooManyRequests, "rate limited")
		return
	}

	h.inflight.Add(1)
	defer h.inflight.Add(-1)

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

	reqBody.RequestId = id.String()

	if errMsg := reqBody.Validate(); errMsg != "" {
		writeJSON(w, http.StatusBadRequest, types.ErrorResponse{RequestId: requestId, Error: errMsg})
		return
	}

	req := &types.InferRequest{
		Body:       reqBody,
		Priority:   types.PRIORITY_MEDIUM,
		Ctx:        r.Context(),
		RespCh:     make(chan *types.InferResponse, 1),
		ReceivedAt: receivedAt,
	}
	req.EnqueuedAt = time.Now()
	h.priorityQueue.Push(req)

	select {
	case <-time.After(5 * time.Second):
		writeJSON(w, http.StatusRequestTimeout, types.ErrorResponse{RequestId: requestId, Error: "request timeout"})
	case resp := <-req.RespCh:
		if resp.Error != nil {
			writeJSON(w, http.StatusInternalServerError, types.ErrorResponse{RequestId: requestId, Error: resp.Error.Error()})
			return
		}
		resp.Body.TotalTimeMs = int(time.Since(req.ReceivedAt).Milliseconds())
		writeJSON(w, http.StatusOK, resp.Body)
	}
}
