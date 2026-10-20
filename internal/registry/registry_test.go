package registry

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/yashp5/inference-engine/internal/config"
)

func writeCatalog(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "models.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCatalog(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr bool
	}{
		{"valid", `[{"name":"default","worker_addr":"127.0.0.1:50051","continuous_batching":true,"engine_slots":8},
			{"name":"b","worker_addr":"127.0.0.1:50052"}]`, false},
		{"empty catalog", `[]`, true},
		{"empty name", `[{"name":"","worker_addr":"x"}]`, true},
		{"duplicate name", `[{"name":"default","worker_addr":"x"},{"name":"default","worker_addr":"y"}]`, true},
		{"empty worker addr", `[{"name":"default"}]`, true},
		{"continuous without slots", `[{"name":"default","worker_addr":"x","continuous_batching":true}]`, true},
		{"negative queue depth", `[{"name":"default","worker_addr":"x","max_queue_depth":-1}]`, true},
		{"default not in catalog", `[{"name":"a","worker_addr":"x"}]`, true},
		{"malformed json", `[{`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadCatalog(writeCatalog(t, tt.body), "default")
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %t", err, tt.wantErr)
			}
		})
	}
}

func TestLoadCatalogMissingFile(t *testing.T) {
	if _, err := LoadCatalog(filepath.Join(t.TempDir(), "nope.json"), "default"); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}

func TestPipelineConfigOverrides(t *testing.T) {
	global := &config.Config{MaxBatchSize: 8, MaxQueueDepth: 1000}

	p := ModelSpec{}.PipelineConfig(global)
	if p.MaxBatchSize != 8 || p.MaxQueueDepth != 1000 {
		t.Fatalf("zero overrides should fall back to globals, got %+v", p)
	}

	p = ModelSpec{MaxBatchSize: 2, MaxQueueDepth: 5, ContinuousBatching: true, EngineSlots: 4}.PipelineConfig(global)
	if p.MaxBatchSize != 2 || p.MaxQueueDepth != 5 || !p.ContinuousBatching || p.EngineSlots != 4 {
		t.Fatalf("spec values should win, got %+v", p)
	}
}

// grpc.NewClient doesn't dial, so a registry over unreachable addresses is
// enough to exercise routing.
func TestAcquire(t *testing.T) {
	ctx := t.Context()
	cfg := &config.Config{DefaultModel: "a", MaxBatchSize: 1, MaxQueueDepth: 10, WorkerCount: 1, MaxInflight: 1}
	reg, err := New(ctx, []ModelSpec{
		{Name: "a", WorkerAddr: "127.0.0.1:1"},
		{Name: "b", WorkerAddr: "127.0.0.1:2"},
	}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()

	lease, err := reg.Acquire(ctx, "")
	if err != nil || lease.Model != "a" {
		t.Fatalf(`Acquire("") = %+v, %v; want the default model "a"`, lease, err)
	}
	lease, err = reg.Acquire(ctx, "b")
	if err != nil || lease.Model != "b" || lease.Pipeline == nil {
		t.Fatalf(`Acquire("b") = %+v, %v`, lease, err)
	}
	if _, err := reg.Acquire(ctx, "nope"); !errors.Is(err, ErrUnknownModel) {
		t.Fatalf(`Acquire("nope") err = %v, want ErrUnknownModel`, err)
	}
}

func TestNewRejectsMissingDefault(t *testing.T) {
	cfg := &config.Config{DefaultModel: "missing", MaxBatchSize: 1, MaxQueueDepth: 10, WorkerCount: 1, MaxInflight: 1}
	if _, err := New(context.Background(), []ModelSpec{{Name: "a", WorkerAddr: "127.0.0.1:1"}}, cfg); err == nil {
		t.Fatal("expected an error when the default model isn't in the catalog")
	}
}
