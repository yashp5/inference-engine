package registry

import (
	"context"
	"errors"

	"github.com/yashp5/inference-engine/internal/config"
	"github.com/yashp5/inference-engine/internal/pipeline"
	"github.com/yashp5/inference-engine/internal/types"
	"github.com/yashp5/inference-engine/internal/worker"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
)

var (
	ErrUnknownModel = errors.New("unknown model")
)

type entry struct {
	spec     ModelSpec
	conn     *grpc.ClientConn
	pipeline *pipeline.Pipeline
}

// Registry maps model names to their worker connection and pipeline. In this
// step the set is fixed at startup, so the map is read-only and needs no lock.
type Registry struct {
	models       map[string]*entry
	defaultModel string
}

func New(ctx context.Context, catalog []ModelSpec, cfg *config.Config) (*Registry, error) {
	r := &Registry{
		models:       make(map[string]*entry),
		defaultModel: cfg.DefaultModel,
	}
	for _, model := range catalog {
		inferClient, conn, err := worker.New(model.WorkerAddr)
		if err != nil {
			// don't leak the conns and pipelines already built
			r.Close()
			return nil, err
		}
		conn.Connect()
		p := pipeline.New(ctx, inferClient, model.PipelineConfig(cfg))
		r.models[model.Name] = &entry{model, conn, p}
	}
	if _, ok := r.models[r.defaultModel]; !ok {
		r.Close()
		return nil, errors.New("default model is not in the catalog")
	}
	return r, nil
}

func (r *Registry) Close() {
	for _, e := range r.models {
		e.pipeline.Close()
		e.conn.Close()
	}
}

// Resolve maps an empty model name to the default, so requests without a
// "model" field keep working.
func (r *Registry) Resolve(name string) string {
	if name == "" {
		return r.defaultModel
	}
	return name
}

type Lease struct {
	Model    string
	Pipeline *pipeline.Pipeline
}

// Acquire is a plain lookup for now. Step 4 adds refcounts and on-demand loads
// behind the same signature, which is why it takes a ctx it doesn't use yet.
func (r *Registry) Acquire(ctx context.Context, name string) (*Lease, error) {
	name = r.Resolve(name)
	e, ok := r.models[name]
	if !ok {
		return nil, ErrUnknownModel
	}
	return &Lease{
		Model:    name,
		Pipeline: e.pipeline,
	}, nil
}

func (l *Lease) Release() {
}

// DefaultConnState is the default model's worker connection state. Idle
// conns are kicked so the next poll can see them come up.
func (r *Registry) DefaultConnState() connectivity.State {
	conn := r.models[r.defaultModel].conn
	state := conn.GetState()
	if state == connectivity.Idle {
		conn.Connect()
	}
	return state
}

func (r *Registry) Stats() map[string]types.ModelStats {
	stats := make(map[string]types.ModelStats, len(r.models))
	for name, e := range r.models {
		stats[name] = types.ModelStats{
			QueueDepth: e.pipeline.QueueDepth(),
			Engine:     e.pipeline.EngineStats(),
		}
	}
	return stats
}
