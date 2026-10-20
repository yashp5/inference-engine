package registry

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/yashp5/inference-engine/internal/config"
)

type ModelSpec struct {
	Name               string `json:"name"`
	WorkerAddr         string `json:"worker_addr"`
	ContinuousBatching bool   `json:"continuous_batching"`
	EngineSlots        int    `json:"engine_slots"`    // must match that worker's --engine-slots until the launcher owns both
	MaxBatchSize       int    `json:"max_batch_size"`  // 0 = global -maxBatchSize
	MaxQueueDepth      int    `json:"max_queue_depth"` // 0 = global -maxQueueDepth
}

// PipelineConfig layers the spec over the global flags. Mode and slots always
// come from the spec; zero-valued overrides fall back to the global value.
func (s ModelSpec) PipelineConfig(global *config.Config) config.PipelineConfig {
	p := config.PipelineConfig{
		ContinuousBatching: s.ContinuousBatching,
		EngineSlots:        s.EngineSlots,
		MaxBatchSize:       s.MaxBatchSize,
		MaxBatchWait:       global.MaxBatchWait,
		WorkerCount:        global.WorkerCount,
		MaxInflight:        global.MaxInflight,
		MaxQueueDepth:      s.MaxQueueDepth,
		QueueTimeout:       global.QueueTimeout,
	}
	if p.MaxBatchSize == 0 {
		p.MaxBatchSize = global.MaxBatchSize
	}
	// the queue rejects at len >= depth, so a zero here would 429 every request
	if p.MaxQueueDepth == 0 {
		p.MaxQueueDepth = global.MaxQueueDepth
	}
	return p
}

// LegacyCatalog is the no -models mode: a catalog of one, built from the
// single-worker flags, so the old invocation keeps working unchanged.
func LegacyCatalog(cfg *config.Config) []ModelSpec {
	return []ModelSpec{{
		Name:               cfg.DefaultModel,
		WorkerAddr:         cfg.WorkerAddr,
		ContinuousBatching: cfg.ContinuousBatching,
		EngineSlots:        cfg.EngineSlots,
	}}
}

func LoadCatalog(path string, defaultModel string) ([]ModelSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var catalog []ModelSpec
	if err := json.Unmarshal(data, &catalog); err != nil {
		return nil, err
	}
	if err := validate(catalog, defaultModel); err != nil {
		return nil, err
	}
	return catalog, nil
}

func validate(catalog []ModelSpec, defaultModel string) error {
	if len(catalog) == 0 {
		return errors.New("catalog has no models")
	}
	names := make(map[string]struct{})
	for _, model := range catalog {
		if model.Name == "" {
			return errors.New("model name is empty")
		}
		if _, ok := names[model.Name]; ok {
			return fmt.Errorf("model name %q is not unique", model.Name)
		}
		names[model.Name] = struct{}{}
		if model.WorkerAddr == "" {
			return fmt.Errorf("model %q: worker_addr is empty", model.Name)
		}
		if model.ContinuousBatching && model.EngineSlots <= 0 {
			return fmt.Errorf("model %q: engine_slots must be > 0 in continuous batching mode", model.Name)
		}
		if model.MaxBatchSize < 0 || model.MaxQueueDepth < 0 {
			return fmt.Errorf("model %q: max_batch_size and max_queue_depth must not be negative", model.Name)
		}
	}
	if _, ok := names[defaultModel]; !ok {
		return fmt.Errorf("default model %q is not in the catalog", defaultModel)
	}
	return nil
}
