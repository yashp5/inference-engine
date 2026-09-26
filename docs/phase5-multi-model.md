# Phase 5 — Multi-Model Management: Design and Build Plan

Working doc for Phase 5 of the [README](../README.md#phase-5--multi-model-management-and-memory).
It describes what has to change in the current code, the pieces to build, the traps,
and an order to build them in. The Go scaffolding on this branch matches it: types,
interfaces and signatures with `TODO(phase5)` markers, and no behavior change yet.

**Done means:** the server reads a catalog of models, routes each request by its
`model` field, starts and stops one worker process per model on demand, never
exceeds a memory budget, evicts the least-recently-used idle model to make room,
and reports load times.

---

## Today vs. after

Today, everything is built once, for one worker you start by hand:

```
HTTP ─► handler ─► queue ─► dispatcher ─► batcher/scheduler or engine ─► gRPC conn ─► worker.py
                   (one)                        (one)                     (one)       (started by you)
```

After Phase 5, the handler asks a registry for the model and gets back that model's
own pipeline. The Go server starts and stops the workers itself:

```
                            ┌──────────► pipeline[tinyllama] ─► worker.py (pid 4121)
HTTP ─► handler ─► registry ┤
          (model field)     └──────────► pipeline[qwen-0.5b] ─► worker.py (pid 4188)

  registry = catalog + state machine + memory budget + LRU + launcher
```

The existing queue, dispatcher, batcher, scheduler and engine code barely changes.
Phase 5 is mostly about **owning lifecycles**: which workers exist, when, and what
they cost.

---

## The pieces

### 1. Model catalog

The set of models the server *may* serve, loaded from a JSON file (`-models`); see
[`models.example.json`](../models.example.json). Each entry is a `registry.ModelSpec`:
name, GGUF path, memory reservation, and worker parameters (mode, engine slots,
per-sequence context, preload).

- Validate at startup: unique names, file exists, `memory_mb > 0`, slots > 0 in
  continuous mode. Fail fast, since a bad catalog is a config bug.
- Per-model engine settings replace the global `-engineSlots`/`-continuousBatching`.
  Because Go now starts the worker and passes both sides the same value, the current
  "must match the worker's `--engine-slots`" footgun goes away.
- **Testing trick:** list the *same* GGUF file under two names. That gives you two
  models, with independent processes and evictions, from one download.

### 2. Per-model pipeline (refactor first)

`cmd/server/main.go` currently wires queue → dispatcher → (batcher → scheduler |
engine) inline for a single worker. Move that into `internal/pipeline` so it can be
built once per loaded model and torn down on unload (`pipeline.New`, `Submit`,
`Stats`, `Close`).

**Why one pipeline per model instead of one shared queue:** the dispatcher's
`d.reqCh <- req` is a blocking send. With a shared queue, a request for a model whose
engine is full blocks the dispatcher, and every other model's requests wait behind it
(head-of-line blocking). Separate pipelines isolate models from each other.

This step has no behavior change, and `make test-e2e` should stay green. Do it first.

### 3. Launcher: worker process lifecycle

`internal/launcher` starts `cmd/worker/worker.py` as a child process for one model and
hides the OS details behind an interface, so the registry can be tested with a fake.

- **Start:** exec the worker with `--model-path`, the mode and slot flags, and a
  listen address. Put it in its own process group (`Setpgid`, as `test/e2e` already
  does) so that stopping it kills the real interpreter, not a launcher shim.
- **Address:** start with a free TCP port (the `freePort` pattern in `test/e2e`). A
  Unix domain socket per model (`unix:///tmp/…/model.sock`) is a clean upgrade that
  avoids port races; both gRPC Python and grpc-go support it.
- **Readiness:** `Start` returns only once the worker is serving. The worker calls
  `server.start()` *after* loading the model, so "gRPC connection reaches `Ready`" is
  a usable first readiness signal and needs no proto change. The gRPC health-checking
  protocol is the more robust upgrade.
- **Stop:** SIGTERM, wait for exit until a deadline, then SIGKILL the process group.
- **Exit watch:** a goroutine on `cmd.Wait()` closes an `Exited()` channel. The
  registry uses it for crash detection and to learn when memory is actually free.

### 4. Registry: state machine and leases

The core of the phase, in `internal/registry`. Each model moves through these states:

```
             Acquire/Load                 worker serving
 Unloaded ─────────────────► Loading ───────────────────► Ready
    ▲                           │                           │
    │                           │ start failed / timeout    │ Unload, evicted,
    │                           ▼                           │ or worker crashed
    │                        Failed                         ▼
    └──────────────────────────────────────────────────  Draining
                    process exited, memory released
```

- **Leases, not raw pointers.** `Acquire(ctx, name)` returns a `Lease` holding the
  model's pipeline and increments the model's reference count. The handler calls
  `defer lease.Release()` around the whole request. A model with `refs > 0` is never
  evicted, and unloading it waits for its leases to be released.
- **Load deduplication.** Ten requests hitting a cold model must start one worker,
  not ten. The first request moves the model to `Loading` and creates a `ready`
  channel; later requests wait on that channel (or `ctx.Done()`) and then re-check
  the state. This is a hand-rolled singleflight that lives in the state machine, with
  no new dependency.
- **Lock discipline.** Hold the registry mutex only to *read or transition state*.
  Never hold it while spawning a process, waiting for readiness, or draining, which
  take seconds. Everything slow happens outside the lock, driven by states and
  channels.
- **LRU touch** happens in `Acquire`.

### 5. Memory budget

`-memoryLimitMB` caps the sum of reservations for models in `Loading`, `Ready` and
`Draining`.

- **Reserve before spawning, under the lock.** If two loads check "does it fit?" and
  then both spawn, both can pass the check even though only one fits (a
  check-then-act race). The reservation is the admission decision.
- **Release only when the process has exited**, not when eviction *starts*. An
  evicted model's memory is still in use while it drains, so the load that triggered
  the eviction must wait on the victim's `Exited()` before spawning.
- **Where the number comes from.** Declare `memory_mb` per model in the catalog.
  Measuring the process after load (RSS) is useful for `/v1/models`, but it's a poor
  admission signal: llama.cpp memory-maps the weights, so RSS rises as pages are
  touched, and GPU/Metal allocations may not show up in it at all. A reasonable
  estimate:

  `GGUF file size + KV cache + compute buffers + Python process overhead`

  KV cache ≈ `2 (K and V) × n_layer × n_ctx × n_embd_kv × 2 bytes (f16)`. For
  TinyLlama 1.1B (22 layers, 4 KV heads × 64 dims, so `n_embd_kv = 256`) in continuous
  mode with 8 slots × 512 context (`n_ctx = 4096`), that's about 88 MiB. On top of
  roughly 640 MiB of Q4_K_M weights, the whole worker lands somewhere around 0.8–1 GB.
  Measure it once and put it in the catalog.

### 6. LRU eviction

When a load doesn't fit, evict models in least-recently-used order, choosing only
among models that are **`Ready` with `refs == 0`**, until the new reservation fits.

- If evicting every eligible model still isn't enough (everything is busy, or the
  model is bigger than the whole budget), fail fast with `ErrInsufficientMemory`
  and return 503. Don't wait on in-flight requests.
- Choose the victims and mark them `Draining` in the same critical section that makes
  the reservation, so another `Acquire` can't lease a model that's being evicted.
- Implementation: `container/list` plus a map from name to list element. `Touch` moves
  an element to the front, and eviction walks from the back.
- **Thrash** is the known failure mode: alternating requests between two models when
  only one fits makes every request a cold start. The benchmark should show it.

### 7. Request routing and cold start

- Add `model` to `CompletionsRequest` (the OpenAI-style field name). It's optional
  and falls back to `-defaultModel`, so the curl example in the README keeps working.
  Unknown model → 404.
- Handler flow:
  `rateLimit → validate → lease := registry.Acquire(ctx, model) → lease.Pipeline.Submit(req) → wait → lease.Release()`.
- **Timeouts:** a cold request waits for the load, bounded by `-loadTimeout` and by
  the overall `-requestTimeout`. Enqueue *after* the load completes, so load time
  doesn't eat into `-queueTimeout` (5s by default, which is shorter than many loads).
- Response gains `model` and `load_wait_ms` (0 when warm), so cold starts are visible
  per request, separately from `queue_time_ms`.

### 8. Admin API

| Endpoint | Behavior |
| :--- | :--- |
| `GET /v1/models` | Every catalog model: state, refs, memory reserved/observed, last used, last load time, pid |
| `POST /v1/models/load` `{"model": "…"}` | Load and block until `Ready` (bounded by `-loadTimeout`). Idempotent: already `Ready` → 200 |
| `DELETE /v1/models/{name}` | Drain (stop admitting, wait for leases up to `-drainTimeout`, then cancel), stop the process, release memory |

Go 1.22+ `ServeMux` patterns handle `{name}` directly. Stub handlers are in
`internal/api/models.go`, not yet routed.

### 9. Worker changes (Python)

- **Load only what the mode needs, eagerly.** Today `InferenceServicer.__init__`
  always builds a `Llama` for the static path, and the continuous-batching `Engine`
  is built lazily on the first stream. In continuous mode that means two model
  objects over the same weights, and the first request pays the engine's
  initialization cost after the worker already looked ready. Add a mode flag
  (`--mode static|continuous`) and construct only that path, before
  `server.start()`, so that serving implies ready.
- **Parent-death watchdog.** If the Go server crashes, its children keep running and
  keep their memory. Linux has `Pdeathsig`; macOS doesn't. A portable fix: pass
  `--parent-pid`, and have a daemon thread exit the process when
  `os.getppid() != parent_pid`.
- **Listen address.** `--port` binds `[::]:port` today. Add a `--listen` flag if you
  move to Unix sockets.

### 10. Observability

- Per model: load time (cold-start latency), load count, eviction count, crash count,
  current state and refs.
- `/stats` becomes per model. Each pipeline has its own queue depth and engine stats.
- `/healthz` currently reflects the single worker connection. It should report that
  the *server* is up, with per-model readiness in `/v1/models`. This leads into
  Phase 6's `/readyz` ("at least one model is ready").

### 11. Preloading

Load the `preload: true` models at startup, in the background, so the port opens
immediately and `/v1/models` shows them as `Loading`. Decide whether preloaded models
are **pinned** (never evicted); see the open questions.

### 12. Crashes and shutdown

- **Worker crash:** when `Exited()` fires while the model is `Ready`, move it to
  `Failed` or `Unloaded`, cancel its pipeline's context, and release its reservation.
  Cancelling the context matters: the `scheduler.Engine` otherwise retries its stream
  forever against a dead process. In-flight requests already get errors through the
  engine's `failAll`. The next `Acquire` reloads the model. Add crash-loop protection
  (for example, 3 failures in 1 minute → stay `Failed` until an explicit load).
- **Server shutdown:** `registry.Shutdown` drains and stops every worker before `main`
  returns. Otherwise every restart leaves orphaned Python processes holding gigabytes.

---

## Concurrency hazards checklist

- [ ] Concurrent `Acquire` on a cold model starts exactly one worker
- [ ] Reservations are made under the lock before spawning, so two loads can't both "fit"
- [ ] A model with active leases is never chosen for eviction
- [ ] Eviction and `Acquire` can't interleave: victims leave `Ready` in the same critical section
- [ ] A new load waits for the evicted process to exit before spawning
- [ ] No process spawn, readiness wait, or drain happens while holding the registry mutex
- [ ] Unloading cancels the pipeline context, so the engine's reconnect loop stops
- [ ] `Lease.Release` is safe to call exactly once, and wakes anyone waiting for the drain
- [ ] `go test -race` on the registry with many goroutines acquiring, loading and unloading

## Status codes

| Situation | Status |
| :--- | :--- |
| Unknown model | 404 |
| Doesn't fit even after evicting every idle model | 503 + `Retry-After` |
| Load failed (worker crashed or exited during startup) | 503 |
| Load exceeded `-loadTimeout` | 504 |
| Model draining (explicit unload in progress) | 503 + `Retry-After` |

## New flags

| Flag | Purpose |
| :--- | :--- |
| `-models` | Catalog file. **Unset means legacy mode:** today's single external worker at `-workerAddr`, unchanged, so the existing e2e test keeps passing while you build |
| `-memoryLimitMB` | Budget for all model reservations |
| `-defaultModel` | Model used when a request omits `model` |
| `-loadTimeout` | Max wait for a worker to become ready |
| `-drainTimeout` | Max wait for leases before an unload cancels them |
| `-workerPython`, `-workerScript` | How to launch the worker (default `cmd/worker/venv/bin/python`, `cmd/worker/worker.py`) |

---

## Testing plan

**Unit tests with a fake launcher (no Python, no model, fast, run under `-race`).**
Skeletons are in `internal/registry/registry_test.go`, where each is a `t.Skip` with
its intent, so `go test ./internal/registry -v` doubles as a to-do list:

- Loads on demand; concurrent `Acquire`s load once
- Evicts the least-recently-used idle model; never evicts a model with leases
- Rejects when nothing evictable frees enough; rejects a model bigger than the budget
- Failed load releases its reservation and wakes the waiters
- A crash marks the model down and reloads it on the next `Acquire`
- Unload drains before stopping
- `Acquire` honors the context deadline while waiting for a load

**E2E** (extend `test/e2e`): two catalog names for the same GGUF, and a budget that
fits exactly one. Request A, then B: A is evicted and `/v1/models` shows it. Kill B's
pid: the next request reloads it. Stop the server: no worker processes left behind.

**Benchmarks:** warm vs. cold latency, load time per model, and the thrash scenario
(alternating models with a budget that fits one).

---

## Suggested build order

Each step leaves the tree building and `make test-e2e` passing.

1. **Extract `internal/pipeline`** from `main.go`, with no behavior change
2. **Catalog + `model` field + registry over pre-started workers.** Skip process
   management and point each model at a worker you started by hand. This proves the
   routing and the per-model pipelines
3. **Launcher**, plus the worker's mode flag, eager load and parent watchdog
4. **Registry state machine:** leases, on-demand load with deduplication, unload and
   drain, and the admin endpoints
5. **Memory budget and LRU eviction**
6. **Preload, cold-start metrics, per-model `/stats`, shutdown cleanup**
7. **E2E and benchmarks**, then update the README's Phase 5 to "What's built"

## Open questions (your call)

1. **Cold request behavior:** block while loading (friendlier) or return 503 +
   `Retry-After` immediately (simpler, and more honest under thrash)? The plan above
   assumes blocking.
2. **Are preloaded models pinned** (never evicted)?
3. **`DELETE /v1/models/{name}`:** block until the process exits, or return 202 and
   let clients poll `/v1/models`?
4. **Memory figure:** declared in the catalog only, or estimated from the file size
   and the KV formula as a default?
5. **Admin endpoint protection:** anyone who can reach the server can unload models.
   Acceptable for now?
