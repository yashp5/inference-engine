// Package launcher starts and stops one Python worker process per model.
//
// It hides exec, process groups, ports and readiness behind an interface so the
// registry can be tested with a fake that starts nothing. See
// docs/phase5-multi-model.md, "Launcher: worker process lifecycle".
package launcher

import (
	"context"
	"errors"
)

var ErrNotImplemented = errors.New("launcher: not implemented")

// Spec is what a worker needs to serve one model.
type Spec struct {
	Name               string
	ModelPath          string
	ContinuousBatching bool
	EngineSlots        int
	PerSeqCtx          int
}

// Process is a running worker.
type Process interface {
	// Addr is the gRPC target to dial, e.g. "127.0.0.1:50123" or "unix:///tmp/x.sock".
	Addr() string
	Pid() int
	// Exited is closed when the process exits for any reason. The registry uses
	// it to detect crashes and to know when a model's memory is actually free.
	Exited() <-chan struct{}
	// Stop sends SIGTERM, then SIGKILLs the whole process group at ctx's deadline.
	Stop(ctx context.Context) error
}

// Launcher starts workers. Start returns only once the worker is serving.
type Launcher interface {
	Start(ctx context.Context, spec Spec) (Process, error)
}

// ProcessLauncher runs cmd/worker/worker.py as a child process.
type ProcessLauncher struct {
	Python string // e.g. cmd/worker/venv/bin/python
	Script string // e.g. cmd/worker/worker.py
}

// Start execs the worker and waits until it's ready.
//
// TODO(phase5):
//   - pick a listen address (free TCP port, as test/e2e's freePort does; a unix
//     socket later) and pass it with --model-path and the mode/slot flags
//   - Setpgid, so Stop can kill the real interpreter and not just a launcher
//     shim (see test/e2e startWorker for why that matters on macOS)
//   - a goroutine on cmd.Wait() that closes Exited()
//   - readiness: the worker loads its model before server.start(), so dialing
//     and waiting for connectivity.Ready works; give up at ctx's deadline or if
//     Exited() fires first, and kill the process on the way out
func (l *ProcessLauncher) Start(ctx context.Context, spec Spec) (Process, error) {
	return nil, ErrNotImplemented
}
