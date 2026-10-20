package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/yashp5/inference-engine/internal/queue"
	"github.com/yashp5/inference-engine/internal/registry"
	"github.com/yashp5/inference-engine/internal/types"
	"google.golang.org/grpc/connectivity"
)

type Handler struct {
	reg         *registry.Registry
	rateLimiter RateLimiter
	inflight    atomic.Int64
	// queueTimeout bounds time-to-admission and is enforced by the dispatcher.
	// requestTimeout bounds the whole request, generation included, and is the
	// deadline on req.Ctx. Keeping them separate matters: req.Ctx.Done() is what
	// triggers a Cancel to the worker, so putting the (short) queue timeout there
	// would abort healthy generations mid-stream.
	queueTimeout   time.Duration
	requestTimeout time.Duration
	agingInterval  time.Duration
}

func NewHandler(r RateLimiter, reg *registry.Registry, queueTimeout time.Duration, requestTimeout time.Duration, agingInterval time.Duration) *Handler {
	return &Handler{
		reg:            reg,
		rateLimiter:    r,
		queueTimeout:   queueTimeout,
		requestTimeout: requestTimeout,
		agingInterval:  agingInterval,
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// Healthz tracks the default model only: once models load on demand, a
// non-default model being down is normal and shouldn't fail the health check.
func (h *Handler) Healthz(w http.ResponseWriter, r *http.Request) {
	state := h.reg.DefaultConnState()
	if state == connectivity.Idle {
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
	models := h.reg.Stats()
	resp := types.StatsResponse{
		InFlight: h.inflight.Load(),
		// top-level engine keeps the pre-multi-model shape for existing callers
		Engine: models[h.reg.Resolve("")].Engine,
		Models: models,
	}
	for _, m := range models {
		resp.QueueDepth += m.QueueDepth
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) Completions(w http.ResponseWriter, r *http.Request) {
	receivedAt := time.Now()

	id, _ := uuid.NewV7()
	requestId := id.String()

	if !h.rateLimiter.allow(clientKey(r)) {
		writeJSON(w, http.StatusTooManyRequests, types.ErrorResponse{RequestId: requestId, Error: "rate limited"})
		return
	}

	h.inflight.Add(1)
	defer h.inflight.Add(-1)
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

	ctx, cancel := context.WithTimeout(r.Context(), h.requestTimeout)
	defer cancel()

	lease, err := h.reg.Acquire(ctx, reqBody.Model)
	if err != nil {
		if errors.Is(err, registry.ErrUnknownModel) {
			writeJSON(w, http.StatusNotFound, types.ErrorResponse{RequestId: requestId, Error: fmt.Sprintf("unknown model %q", reqBody.Model)})
		} else {
			writeJSON(w, http.StatusServiceUnavailable, types.ErrorResponse{RequestId: requestId, Error: err.Error()})
		}
		return
	}
	defer lease.Release()
	// carry the resolved name, so queue/scheduler logs never see ""
	reqBody.Model = lease.Model

	var priority types.Priority
	switch reqBody.Priority {
	case "low":
		priority = types.PRIORITY_LOW
	case "medium":
		priority = types.PRIORITY_MEDIUM
	case "high":
		priority = types.PRIORITY_HIGH
	default:
		// Validate rejects anything else; medium is the documented default
		priority = types.PRIORITY_MEDIUM
	}

	req := &types.InferRequest{
		Body:          reqBody,
		Priority:      priority,
		AgingInterval: h.agingInterval,
		Ctx:           ctx,
		RespCh:        make(chan *types.InferResponse, 1),
		ReceivedAt:    receivedAt,
	}
	req.EnqueuedAt = time.Now()
	err = lease.Pipeline.Submit(req)
	if errors.Is(err, queue.ErrQueueFull) {
		w.Header().Set("Retry-After", "1")
		writeJSON(w, http.StatusTooManyRequests, types.ErrorResponse{RequestId: requestId, Error: "queue full"})
		return
	}

	select {
	case <-ctx.Done():
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			writeJSON(w, http.StatusGatewayTimeout, types.ErrorResponse{RequestId: requestId, Error: "request timeout"})
		}
	case resp := <-req.RespCh:
		if resp.Error != nil {
			status := http.StatusInternalServerError
			if errors.Is(resp.Error, types.ErrQueueTimeout) {
				status = http.StatusGatewayTimeout
			}
			writeJSON(w, status, types.ErrorResponse{RequestId: requestId, Error: resp.Error.Error()})
			return
		}
		resp.Body.Model = lease.Model
		resp.Body.TotalTimeMs = int(time.Since(req.ReceivedAt).Milliseconds())
		writeJSON(w, http.StatusOK, resp.Body)
	}
}
