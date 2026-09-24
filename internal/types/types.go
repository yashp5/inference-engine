package types

import (
	"context"
	"errors"
	"fmt"
	"time"
)

const maxTokensUpperBound = 4096

// ErrQueueTimeout means the request aged out of the priority queue before a
// worker could pick it up. Distinct from the overall request deadline so the
// API can answer 504 immediately instead of letting the caller block.
var ErrQueueTimeout = errors.New("queue timeout")

type CompletionsRequest struct {
	RequestId   string  `json:"request_id"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	Priority    string  `json:"priority"`
}

func (r *CompletionsRequest) Validate() string {
	if r.Prompt == "" {
		return "prompt must not be empty"
	}
	if r.MaxTokens <= 0 {
		return "max_tokens must be greater than 0"
	}
	if r.MaxTokens > maxTokensUpperBound {
		return fmt.Sprintf("max_tokens must not exceed %d", maxTokensUpperBound)
	}
	if r.Temperature < 0.0 || r.Temperature > 2.0 {
		return "temperature must be between 0.0 and 2.0"
	}
	switch r.Priority {
	case "":
		r.Priority = "medium"
	case "low", "medium", "high":
	default:
		return `priority must be one of "low", "medium", "high"`
	}
	return ""
}

type CompletionsReponse struct {
	RequestId       string `json:"request_id"`
	GeneratedText   string `json:"generated_text"`
	TokensGenerated int    `json:"tokens_generated"`
	InferenceTimeMs int    `json:"inference_time_ms"`
	QueueTimeMs     int    `json:"queue_time_ms"`
	TotalTimeMs     int    `json:"total_time_ms"`
}

type ErrorResponse struct {
	RequestId string `json:"request_id,omitempty"`
	Error     string `json:"error"`
}

type Priority int

const (
	PRIORITY_LOW Priority = iota
	PRIORITY_MEDIUM
	PRIORITY_HIGH
)

// StatsResponse is what GET /stats returns. InFlight counts requests accepted
// past the rate limiter and not yet answered, so it includes whatever is still
// sitting in the queue: InFlight-QueueDepth is roughly what the workers are on.
type StatsResponse struct {
	QueueDepth int          `json:"queue_depth"`
	InFlight   int64        `json:"in_flight"`
	Engine     *EngineStats `json:"engine,omitempty"` // continuous batching only
}

// EngineStats is the worker's most recent StepStats. Nil until the first step
// of an engine session, and reset to nil when the session ends so a dead
// stream doesn't keep reporting its last healthy numbers.
type EngineStats struct {
	Step          int64     `json:"step"`
	ActiveSlots   int       `json:"active_slots"`
	FreeSlots     int       `json:"free_slots"`
	Waiting       int       `json:"waiting"`
	BatchTokens   int       `json:"batch_tokens"`
	PrefillTokens int       `json:"prefill_tokens"`
	StepTimeUs    int       `json:"step_time_us"`
	KVUsed        int       `json:"kv_used"`
	ObservedAt    time.Time `json:"observed_at"`
}

type InferResponse struct {
	Body  *CompletionsReponse
	Error error
}

type InferRequest struct {
	Body          *CompletionsRequest
	ReceivedAt    time.Time     `json:"received_at"`
	EnqueuedAt    time.Time     `json:"enqueued_at"`
	DispatchedAt  time.Time     `json:"dispatched_at"`
	AgingInterval time.Duration `json:"aging_interval"`
	Priority      Priority
	RespCh        chan *InferResponse
	Ctx           context.Context
}

type Batch []*InferRequest
