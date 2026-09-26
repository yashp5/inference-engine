package registry

// ModelSpec is one catalog entry: everything needed to start a worker for the
// model and to budget for it before it's loaded. See models.example.json.
type ModelSpec struct {
	Name string `json:"name"`
	Path string `json:"path"` // GGUF file
	// MemoryMB is what admission reserves against -memoryLimitMB before the
	// worker starts. Declared rather than measured: RSS is a poor admission
	// signal because llama.cpp mmaps the weights.
	MemoryMB           int  `json:"memory_mb"`
	ContinuousBatching bool `json:"continuous_batching"`
	EngineSlots        int  `json:"engine_slots"`
	PerSeqCtx          int  `json:"per_seq_ctx"`
	// Preload loads the model at startup, in the background.
	Preload bool `json:"preload"`
}

type Catalog struct {
	Models []ModelSpec `json:"models"`
}

// LoadCatalog reads the -models file.
//
// TODO(phase5): decode JSON, then fail fast on a config bug: duplicate names,
// missing file, MemoryMB <= 0, EngineSlots/PerSeqCtx <= 0 in continuous mode.
func LoadCatalog(path string) (*Catalog, error) {
	return nil, ErrNotImplemented
}
