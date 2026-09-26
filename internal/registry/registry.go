// Package registry decides which models are loaded: it owns the catalog, each
// model's lifecycle, the memory budget and LRU eviction.
//
// The rules that keep it correct are in docs/phase5-multi-model.md, sections
// 4-6 and the concurrency checklist. The short version: hold mu only to read or
// change state, never while starting, waiting on or stopping a process.
package registry

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/yashp5/inference-engine/internal/launcher"
	"github.com/yashp5/inference-engine/internal/pipeline"
)

var (
	ErrNotImplemented = errors.New("registry: not implemented")
	// ErrUnknownModel: not in the catalog. 404.
	ErrUnknownModel = errors.New("unknown model")
	// ErrInsufficientMemory: evicting every idle model still doesn't free
	// enough, or the model is bigger than the whole budget. 503.
	ErrInsufficientMemory = errors.New("insufficient memory")
	// ErrDraining: the model is being unloaded. 503.
	ErrDraining = errors.New("model draining")
)

type State int

const (
	StateUnloaded State = iota
	StateLoading
	StateReady
	StateDraining // no new leases; waiting for refs to hit 0, then the process to exit
	StateFailed   // crash loop; stays down until an explicit Load
)

type model struct {
	spec  ModelSpec
	state State
	refs  int // live leases; a model with refs > 0 is never evicted

	// ready is closed when a load finishes, success or not. Concurrent
	// Acquires for a Loading model wait on it rather than starting their own
	// worker, then re-check state and loadErr.
	ready   chan struct{}
	loadErr error

	lastLoad time.Duration
	proc     launcher.Process
	pipe     *pipeline.Pipeline
}

type Registry struct {
	mu       sync.Mutex
	models   map[string]*model
	lru      *lru
	budget   *budget
	launcher launcher.Launcher

	loadTimeout  time.Duration
	drainTimeout time.Duration
}

func New(cat *Catalog, l launcher.Launcher, memoryLimitMB int, loadTimeout, drainTimeout time.Duration) *Registry {
	// TODO(phase5): one Unloaded model per catalog entry.
	return &Registry{
		models:       make(map[string]*model),
		lru:          newLRU(),
		budget:       newBudget(memoryLimitMB),
		launcher:     l,
		loadTimeout:  loadTimeout,
		drainTimeout: drainTimeout,
	}
}

// Lease pins a model for the life of one request: while any lease is held the
// model can't be evicted, and an unload waits for it.
type Lease struct {
	Pipeline *pipeline.Pipeline
	// LoadWait is how long this request waited for a cold load; 0 when warm.
	LoadWait time.Duration
	// TODO(phase5): back-pointer to the model, and a guard so Release is
	// idempotent.
}

// Release drops the lease. Wakes an unload waiting for the model to go idle.
func (l *Lease) Release() {}

// Acquire returns a lease on a Ready model, loading it first if needed and
// evicting idle models to make room. Blocks until ready, ctx's deadline, or
// loadTimeout.
//
// TODO(phase5):
//   - Ready: refs++, lru.touch, return
//   - Loading: wait on ready or ctx.Done(), then loop
//   - Unloaded: reserve memory (evicting if needed) and move to Loading, all in
//     one critical section; then start the worker and build the pipeline
//     outside the lock
//   - Draining/Failed: ErrDraining / the stored error
func (r *Registry) Acquire(ctx context.Context, name string) (*Lease, error) {
	return nil, ErrNotImplemented
}

// Load is Acquire without keeping the lease, for POST /v1/models/load and
// preloading. Idempotent.
func (r *Registry) Load(ctx context.Context, name string) (ModelInfo, error) {
	return ModelInfo{}, ErrNotImplemented
}

// Unload drains the model: no new leases, wait up to drainTimeout for existing
// ones, then close the pipeline, stop the process and release the memory once
// it has exited.
func (r *Registry) Unload(ctx context.Context, name string) error {
	return ErrNotImplemented
}

// Preload starts background loads for every catalog entry with Preload set.
func (r *Registry) Preload(ctx context.Context) {}

// Shutdown unloads everything. main must call it before returning, or every
// restart leaves orphaned worker processes holding their memory.
func (r *Registry) Shutdown(ctx context.Context) error {
	return ErrNotImplemented
}

// ModelInfo is one row of GET /v1/models.
type ModelInfo struct {
	Name             string    `json:"name"`
	State            string    `json:"state"`
	Refs             int       `json:"refs"`
	MemoryReservedMB int       `json:"memory_reserved_mb"`
	MemoryObservedMB int       `json:"memory_observed_mb,omitempty"` // worker RSS; informational only
	LastUsed         time.Time `json:"last_used,omitzero"`
	LastLoadMs       int       `json:"last_load_ms,omitempty"`
	Pid              int       `json:"pid,omitempty"`
}

func (r *Registry) List() []ModelInfo {
	return nil
}
