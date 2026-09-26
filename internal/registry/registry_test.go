package registry

import (
	"context"
	"testing"

	"github.com/yashp5/inference-engine/internal/launcher"
)

// fakeLauncher starts nothing, so registry logic can be tested without Python
// or a model file, fast and under -race.
//
// TODO(phase5): count Start calls per model (the dedup tests need it), make
// Start block until the test releases it (to hold a model in Loading), and let
// a test fail a Start or crash a process on demand.
type fakeLauncher struct{}

var _ launcher.Launcher = (*fakeLauncher)(nil)

func (f *fakeLauncher) Start(ctx context.Context, spec launcher.Spec) (launcher.Process, error) {
	return nil, launcher.ErrNotImplemented
}

// Each test is a to-do: the skip message is the behavior to pin down.

func TestAcquireLoadsOnDemand(t *testing.T) {
	t.Skip("TODO(phase5): Acquire on an Unloaded model starts one worker and returns a lease on a Ready model")
}

func TestConcurrentAcquireLoadsOnce(t *testing.T) {
	t.Skip("TODO(phase5): 50 goroutines Acquire the same cold model; the launcher sees exactly one Start")
}

func TestAcquireHonorsContextWhileLoading(t *testing.T) {
	t.Skip("TODO(phase5): with Start blocked, Acquire returns ctx.Err() at the deadline and the load carries on")
}

func TestEvictsLeastRecentlyUsed(t *testing.T) {
	t.Skip("TODO(phase5): budget fits two; use A, use B, load C; A is evicted, B stays")
}

func TestNeverEvictsModelWithLeases(t *testing.T) {
	t.Skip("TODO(phase5): hold a lease on the LRU model; loading another evicts the next candidate or fails")
}

func TestInsufficientMemoryWhenNothingEvictable(t *testing.T) {
	t.Skip("TODO(phase5): every loaded model leased; loading one more returns ErrInsufficientMemory without waiting")
}

func TestModelLargerThanBudget(t *testing.T) {
	t.Skip("TODO(phase5): MemoryMB > limit returns ErrInsufficientMemory and evicts nothing")
}

func TestNewLoadWaitsForEvictedProcessToExit(t *testing.T) {
	t.Skip("TODO(phase5): the victim's memory is released only when Exited() fires; the new Start happens after that")
}

func TestFailedLoadReleasesReservationAndWakesWaiters(t *testing.T) {
	t.Skip("TODO(phase5): Start fails; every waiter gets the error and the budget is back where it was")
}

func TestCrashMarksDownAndReloadsOnNextAcquire(t *testing.T) {
	t.Skip("TODO(phase5): close Exited() on a Ready model; its memory is released and the next Acquire starts a new worker")
}

func TestUnloadDrainsBeforeStopping(t *testing.T) {
	t.Skip("TODO(phase5): Unload with a live lease: new Acquires get ErrDraining; Stop happens only after Release")
}

func TestShutdownStopsEveryWorker(t *testing.T) {
	t.Skip("TODO(phase5): after Shutdown, every started process has had Stop called")
}
