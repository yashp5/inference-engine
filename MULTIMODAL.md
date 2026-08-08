# Multimodal Inference Serving — Part II

A continuation of [README.md](README.md). Phases 1–8 build a text LLM serving system:
queuing, dynamic batching, continuous batching, multi-model management, observability,
horizontal scaling, and KV cache reuse. This document is the roadmap for what comes
after: evolving the same infrastructure to serve **voice, audio, and video** models.

The thesis of Part I was that LLM serving is about keeping the accelerator busy under a
latency constraint. That thesis survives contact with other modalities — but almost
every abstraction built around it (`request`, `batch`, `slot`, `cache`) has to
generalize. The dominant shift:

> **Text is request/response. Audio is sessions. Video is jobs.**

## The Big Picture

| | Text LLM | Voice (ASR/TTS) | Video understanding | Video generation |
|---|---|---|---|---|
| Request shape | request → response | bidirectional stream | request → response | async job |
| Ingress | HTTP POST | WebSocket / gRPC bidi | HTTP + object ref | submit → poll/webhook |
| Payload size | KBs | ~1 MB/min of audio | 100s of MB | 100s of MB out |
| Lifetime | ms–seconds | seconds–minutes | seconds | minutes |
| "Iteration" | decode step | audio frame (~80ms) | encoder pass + decode | denoising step |
| Hard constraint | throughput/latency knob | **RTF < 1** (real time) | prefill latency | GPU-seconds budget |
| State stickiness | per-request | **per-session** (can't re-route) | per-request | per-job (checkpointable) |
| Cost currency | tokens | audio-minutes | frames encoded | GPU-seconds |

The unifying insight from Phase 4 carries over: **continuous batching is a special case
of iteration-level scheduling.** Every modality has its own "iteration" — a decode step,
an audio frame, a denoising step — and the scheduler's job is always the same: at each
iteration boundary, fill every free slot on the accelerator.

## Target Architecture

```
                        ┌────────────────────────────────────────────────────┐
                        │                 Go Control Plane                    │
                        │                                                    │
  HTTP POST ──────────► │  Sync API ────► Queue ─► Batcher ─┐                │
                        │                                   │                │
  WebSocket ──────────► │  Session Mgr ─► Frame Scheduler ──┼─► Dispatch     │
                        │                                   │      │         │
  POST /v1/jobs ──────► │  Job Queue ───► Step Scheduler ───┘      │         │
                        │                                          │         │
                        │        Pipeline Orchestrator (DAGs)      │         │
                        └──────────────────────────────────────────┼─────────┘
                                                                   │
                     ┌──────────────┬──────────────┬───────────────┤
                     ▼              ▼              ▼               ▼
              ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
              │ Preprocess │ │ ASR Worker │ │ LLM Worker │ │ TTS Worker │
              │ (CPU tier) │ │ (frame     │ │ (token     │ │ (frame     │
              │ decode/VAD │ │  batching) │ │  batching) │ │  batching) │
              └────────────┘ └────────────┘ └────────────┘ └────────────┘
                     │
                     ▼
              ┌────────────┐        ┌─────────────────────────┐
              │ Object     │◄──────►│ Diffusion Worker        │
              │ Store      │        │ (step batching, jobs)   │
              └────────────┘        └─────────────────────────┘
```

The Go/Python split from Phase 1 holds: Go owns sessions, jobs, scheduling, and routing;
Python workers own model execution. What changes is that the data plane becomes
**heterogeneous** (different worker types with different hardware profiles) and
**composable** (pipelines of workers, not a single hop).

---

## Phase 9 — Media Ingestion and the Out-of-Band Data Plane

A prompt is a few KB and rides inside the request. A 30-second audio clip is ~1 MB; a
video is hundreds of MB. The first structural change is that **payloads move out of the
request path**.

**What to build:**
- Upload API: `POST /v1/files` accepts multipart uploads, streams to disk (later: S3/GCS),
  returns a `file_id`. Inference requests reference `file_id`, never raw bytes
- Chunked gRPC streaming between Go and workers for media transfer — never a 200 MB
  `bytes` field in a single message
- A **preprocessing worker tier** (CPU-bound, separate process pool): audio resampling
  to 16kHz mono, video demux + frame sampling + JPEG decode. Scales independently of
  GPU workers
- Byte-aware backpressure: the Phase 2 semaphore capped request *count*; now cap
  **bytes in flight**. A weighted semaphore (`golang.org/x/sync/semaphore`) where a
  request's weight is its payload size
- Content validation and limits: max duration, max resolution, container/codec
  allowlist — reject at the front door, not after decode

**Benchmarking:**
- Upload throughput vs. concurrent inference load (do big uploads starve inference?)
- Preprocessing tier saturation: at what request rate does CPU decode become the
  bottleneck before the GPU does?

**Key tradeoff:** Inline payloads are simple and low-latency for small media; references
add a round trip but keep the inference path lean. Production systems support both with
a size cutoff.

**Concepts:** Object storage patterns, weighted semaphores, CPU/GPU tier separation,
streaming uploads

---

## Phase 10 — Session Infrastructure

Everything so far is request/response. Voice requires a new primitive: the
**long-lived bidirectional session**.

**What to build:**
- WebSocket endpoint (`/v1/sessions`) accepting binary audio frames upstream and
  emitting events downstream (partial transcripts, audio chunks, control messages)
- A `Session` struct owning: session ID, connection, inbound/outbound buffers, per-stage
  state, deadline clock — with a goroutine-per-session reader/writer pair
- Session lifecycle: open → active → draining → closed, with idle timeouts and
  max-duration limits
- Bidirectional streaming gRPC to workers (extends the Phase 4 stream): multiplex many
  sessions over one worker connection, demux by session ID
- **Jitter buffering**: client audio arrives bursty; smooth it into fixed-size frames
  before the scheduler sees it
- Session-level admission control: `max_concurrent_sessions` replaces the request
  semaphore as the front-line limiter — a session holds resources for minutes, not ms
- Graceful drain on shutdown: stop accepting new sessions, let active ones finish,
  emit a `reconnect` hint. Killing a server no longer fails a retryable request — it
  **drops a live call**

**Concepts:** WebSocket lifecycle, goroutine-per-connection patterns, stream
multiplexing, jitter buffers, graceful drain

---

## Phase 11 — Streaming ASR with Frame-Level Continuous Batching

The payoff phase. This reuses nearly everything from Phases 2–4 while forcing the
central multimodal idea: **iteration-level scheduling where the iteration is an audio
frame, not a token.**

**What to build:**
- Python ASR worker wrapping `faster-whisper` (or a streaming-native model), serving
  bidirectional gRPC
- Frame scheduler in Go: maintain N session slots (the Phase 4 slot array, but a slot
  lives for an entire call). Every tick (~80ms), collect the newest frame from each
  active session and dispatch one **batched frame step** across all sessions
- Voice activity detection (VAD) and endpointing: detect end-of-utterance to emit
  final transcripts; sessions in silence release compute without releasing their slot
  state
- Partial (interim) transcripts streamed as the model updates its hypothesis; finals
  on endpoint
- **RTF (real-time factor) tracking** per session and per batch: `processing_time /
  audio_duration`. RTF ≥ 1 means the user experiences ever-growing lag

**Benchmarking:**
- Max concurrent sessions at RTF < 1 on your hardware — this number *is* your capacity
- Batch size vs. per-frame latency curve: find the knee
- Word-level latency: time from a word being spoken to it appearing in the transcript

**Key tradeoff:** In Phase 3, larger batches traded latency for throughput — a knob. Here
the frame period is a **hard deadline**: a batch that takes longer than 80ms to process
80ms of audio per stream is not a slower configuration, it is a broken one. Throughput
tuning becomes constraint satisfaction.

**Concepts:** Real-time scheduling, frame-level batching, VAD/endpointing, deadline-aware
batch sizing, RTF

---

## Phase 12 — Streaming TTS

The mirror image of ASR: text in, audio stream out. Structurally similar batching, but
the output side now has real-time obligations.

**What to build:**
- Python TTS worker (e.g., Piper or a small VITS-family model) generating audio in
  chunks, streamed back per-session
- **Time-to-first-audio (TTFA)** as the headline metric: how long from text submission
  to the first audible chunk
- Playback pacing: generate faster than real time, but buffer-manage so a slow client
  doesn't cause unbounded memory growth (bounded outbound buffer + backpressure to the
  generator)
- Sentence-level chunking: synthesize per sentence/clause so long inputs stream
  incrementally instead of blocking on full synthesis
- **Underrun detection**: if generation falls behind playback (RTF > 1 on the output
  side), the user hears a gap. Count and alert on these

**Concepts:** Producer/consumer pacing, bounded buffers, TTFA, audio underruns

---

## Phase 13 — Pipeline Orchestration: the Voice Agent

Compose Phases 11, 12, and the Part I LLM into `ASR → LLM → TTS` — a talking assistant.
This is where the Phase 5 model registry evolves into a **pipeline orchestrator**.

**What to build:**
- Pipeline definitions as DAGs: named stages, each mapping to a worker pool, with typed
  edges (audio → text → text → audio)
- **Streaming between stages**: do not wait for stage N to finish before starting N+1.
  Start LLM prefill on the final transcript the moment the endpoint fires; start TTS on
  the LLM's first complete sentence, not the full reply. This is Phase 3's fan-out/fan-in,
  chained
- End-to-end latency budget: conversational voice needs roughly **300–800ms** from
  end-of-user-speech to first audio out. Instrument each stage's share of the budget
  and surface the breakdown per turn
- Barge-in: if the user starts speaking while TTS is playing, cancel the in-flight LLM
  generation and TTS synthesis via context cancellation (Phase 2's `context.Context`
  propagation, now spanning three workers)
- Per-stage independent scaling: ASR, LLM, and TTS pools have different hardware
  appetites — the orchestrator routes across pools rather than assuming co-location.
  This is the multimodal analog of prefill/decode disaggregation

**Benchmarking:**
- Turn latency distribution (end-of-speech → first audio) and per-stage breakdown
- Barge-in cancellation latency: how much wasted compute per interruption?
- Concurrent conversations sustained within budget

**Concepts:** DAG orchestration, pipelined streaming, latency budgets, cross-worker
cancellation, disaggregated serving

---

## Phase 13.5 — Toward Full-Duplex: Speculative Prefill and Predictive Turn-Taking

The capstone of the voice track. Phase 13's pipeline is *half-duplex* — structurally
turn-based. VAD waits for silence, declares end-of-turn, and only then does the LLM see
any input: ~500–1000ms of dead air by construction, and the agent can never interject.
Real conversation is **full-duplex** — humans respond in ~200ms with no gap because we
predict turn ends and prepare responses *while the other person is still talking*.

The striking thing: closing this gap is mostly a **scheduling problem, not a modeling
problem**. The scheduler starts making bets.

**What to build:**
- **Speculative prefill on partial transcripts**: stream interim ASR hypotheses into
  LLM prefill continuously, before the endpoint fires. Extending a hypothesis extends
  a prefix — this is the Phase 8 prefix cache doing exactly its job. By end-of-turn,
  the response is one decode step away instead of a full prefill away
- **Hypothesis rollback**: when ASR revises an interim word, roll back to the last
  stable prefix and re-prefill the suffix. Reuses the Phase 13 cancellation machinery —
  barge-in and hypothesis revision are the same operation from the scheduler's view
- **Predictive endpointing**: replace "silence = end of turn" with a small
  turn-taking classifier over the transcript (and optionally prosody features) that
  fires *before* the silence threshold. Start with a heuristic (punctuation +
  falling-pitch proxy + partial-hypothesis stability), graduate to a learned model
- **Early-commit policy**: decide when to start *speaking* a speculated response.
  Committing early wins latency but risks talking over the user or answering a
  half-asked question; add a confidence gate and an abort path (stop TTS mid-word if
  the user keeps talking)
- **Speculation accounting**: every discarded prefill is wasted GPU time. Track
  speculation waste rate (wasted tokens / total tokens) and expose the
  aggressiveness/waste tradeoff as a tunable policy

**Benchmarking:**
- Turn latency (end-of-speech → first audio) vs. Phase 13 baseline — target cutting it
  by 2–4x
- Speculation waste rate vs. latency win across policy aggressiveness levels
- False-commit rate: how often does the agent start speaking and have to abort?

**Key tradeoff:** This phase introduces a new resource-allocation regime: spending GPU
cycles on work that might be thrown away, purely to buy latency. The same
bet-and-discard structure appears in speculative decoding and branch prediction — it
is the "keep the accelerator busy" thesis, now with bets.

**The outlook — true full-duplex models:** The radical endpoint collapses the pipeline
entirely. Kyutai's **Moshi** models both audio streams — the user's and its own — as
parallel token streams in a single model, every frame, with no turn structure at all:
it listens while speaking, backchannels, and interrupts naturally at ~200ms latency.
Serving such a model deletes the ASR/LLM/TTS DAG (one model, one worker) but makes
Phases 10–11 *more* central, not less: a full-duplex model is nothing but frame-level
streaming, consuming and producing a frame on every tick — even during silence. The
scheduler's iteration becomes symmetric I/O.

**Concepts:** Speculative execution, turn-taking prediction, rollback/repair,
latency-vs-waste tradeoffs, full-duplex serving

---

## Phase 14 — Vision-Language Understanding

Images and video in, text out. Unlike voice, this stays request/response — but it is
**prefill-dominated**, and it introduces a second scheduler in series with the LLM's.

**What to build:**
- Python VLM worker (e.g., LLaVA-family via llama-cpp-python's multimodal support, or a
  separate vision encoder + projector): image → vision tokens → LLM prefill
- Two-stage scheduling: the vision encoder batches like classic **static batching**
  (fixed-shape image tensors bucket cleanly), then hands token embeddings to the
  Part I continuous batcher for decode
- Video understanding = frame sampling policy (uniform, keyframe, or scene-change
  detection in the preprocessing tier) → batch of images → interleaved prefill
- **Embedding cache** — the Phase 8 idea, generalized: key = hash of image bytes (or
  video segment), value = encoder output. A user asking three questions about the same
  image should run the vision encoder once. Same LRU machinery, new key space
- Shape bucketing in the batcher: images at different resolutions can't share a tensor;
  group by bucket, pad within bucket, track padding waste as a metric

**Benchmarking:**
- Encoder cache hit rate under a chat-with-image workload
- Padding waste vs. bucket granularity
- Prefill time vs. image count per request (video = many images)

**Concepts:** Encoder/decoder disaggregation, shape bucketing, embedding caches,
frame sampling policies

---

## Phase 15 — Async Jobs: Image and Video Generation

Diffusion models generate for minutes, not milliseconds. Synchronous HTTP is the wrong
shape entirely — this phase builds the **job** primitive.

**What to build:**
- Job API: `POST /v1/jobs` → `{job_id}`; `GET /v1/jobs/{id}` for status/progress;
  webhook callback on completion; results written to the object store from Phase 9
- Durable job queue: persist jobs (start with SQLite/bbolt) so a server restart doesn't
  lose the backlog — the first time in this project that queue state outlives the process
- Python diffusion worker (a small SD-family model keeps iteration fast on modest
  hardware) reporting per-step progress over streaming gRPC
- **Step-level continuous batching**: the iteration is now a denoising step. Jobs at the
  same resolution/step-count batch together; new jobs join the batch *between* steps —
  Phase 4's insight, third incarnation. Batches are shape-constrained, so reuse
  Phase 14's bucketing
- Checkpointing: persist intermediate latents every K steps so a crashed worker resumes
  a 500-step video job at step 400, not step 0
- Job scheduling policy: fair-share GPU-seconds across users (extends Phase 2's token
  buckets), plus priority classes with preemption at step boundaries
- Cancellation: `DELETE /v1/jobs/{id}` stops work at the next step boundary and
  releases the slot

**Benchmarking:**
- Backlog drain rate (GPU-seconds of queued work as the autoscaling signal)
- Throughput gain from step batching vs. one-job-at-a-time
- Recovery time and wasted compute after killing a worker mid-job

**Concepts:** Durable queues, checkpointing, step-level batching, preemption,
fair-share scheduling, webhooks

---

## Phase 16 — Multimodal Routing, Affinity, and Autoscaling

Phase 7's load balancer treated affinity as an optimization: prefer the server with the
model warm, fall back anywhere. Streaming sessions break that assumption — **affinity
becomes correctness**.

**What to build:**
- Session-sticky routing: a live session's state (audio buffers, KV cache, VAD state)
  lives on one worker. Consistent hashing moves from model-name granularity to
  **session-ID granularity**, and mid-session re-routing is forbidden
- Reconnect + resume: clients that drop get a session token; the balancer routes the
  reconnect to the original worker if it's alive, or restarts the session cleanly if not
- Modality-aware autoscaling signals, because queue depth no longer means one thing:
  - Text: queue depth + token throughput (Part I)
  - Voice: concurrent sessions × **RTF headroom**
  - Jobs: **GPU-seconds of backlog**
- Drain-aware deploys: rolling restarts wait for session drain (bounded by max session
  duration) instead of killing connections
- Heterogeneous worker pools in the registry: workers advertise capabilities
  (modality, models, hardware class) in their Phase 7 heartbeats; the balancer routes
  on capability, not just load

**Benchmarking:**
- Kill a server: text requests retry invisibly; how many voice sessions drop, and does
  reconnect/resume recover them?
- Deploy latency: how long does a drain-aware rolling restart take at various session
  loads?

**Concepts:** Sticky sessions, state locality, capability-based routing, heterogeneous
autoscaling, drain-aware deploys

---

## Observability Additions

Phase 6's metrics remain, but token throughput stops being the universal currency. New
headline SLIs per modality:

| Modality | SLIs |
|---|---|
| Voice | RTF distribution, time-to-first-audio, underrun count, turn latency (end-of-speech → first audio), session duration, drop rate |
| Vision | encoder cache hit rate, padding waste, prefill latency by image count |
| Jobs | queue wait, job duration, steps/sec, checkpoint recovery rate, backlog GPU-seconds |
| Cost | **GPU-seconds per unit of output** (per audio-minute, per image, per 1K tokens) |

Structured logs gain a `session_id` / `job_id` dimension alongside `request_id`, and
per-turn pipeline traces (which stage spent what share of the latency budget).

## API Sketch

```
POST   /v1/completions          # Part I — text (unchanged)
POST   /v1/files                # Phase 9 — media upload → file_id
GET    /v1/sessions             # Phase 10 — WebSocket upgrade; audio in, events out
POST   /v1/vision/completions   # Phase 14 — prompt + file_id(s) → text
POST   /v1/jobs                 # Phase 15 — generation job → job_id
GET    /v1/jobs/{id}            # Phase 15 — status, progress, result reference
DELETE /v1/jobs/{id}            # Phase 15 — cancel at next step boundary
```

## Suggested Order

**Prerequisite: finish Part I first — especially Phases 3, 4, and 8 with real
benchmarks.** Continuous batching and KV/prefix caching are the crown jewels of this
project; a deep, well-measured Phase 4 is worth more than any phase in this document.
Part II is the reward for finishing, not an alternative to it.

Then the recommended path is the **voice track**:

```
9 (media ingestion) → 10 (sessions) → 11 (streaming ASR) → 13 (voice agent) → 13.5 (full-duplex)
```

- Phases 9–10 are prerequisites for everything else in Part II
- Phase 11 is the payoff: it reuses the Phase 4 slot scheduler almost unchanged and
  forces the real-time constraint — the best proof the Part I work was understood,
  not just implemented
- Phase 13 folds TTS (Phase 12) in as a pipeline stage rather than a standalone
  milestone; the talking agent with barge-in is the most impressive demo per line of
  code in the whole project
- Phase 13.5 is the capstone: an agent that responds with no dead air — and can
  interject before you finish a sentence — via speculative prefill and predictive
  turn-taking

This path covers sessions, real-time scheduling, and pipeline orchestration — the
three ideas Part I structurally couldn't teach — and is not hardware-gated
(`faster-whisper` runs real-time on CPU).

**Optional extensions**, in descending order of marginal value:

- **Phase 14 (vision-language):** worthwhile if vision interests you — the embedding
  cache is a nice generalization of Phase 8 — but much of the rest is tensor-shape
  bookkeeping
- **Phase 16 (routing/affinity):** only interesting once multiple worker types exist;
  do a light version (session stickiness + drain-aware restarts) rather than the
  full build
- **Phase 15 (generation jobs):** lowest marginal value — the durable-job-queue
  pattern is well-trodden ground, and video diffusion is painful on local hardware.
  Skip unless the job primitive itself is the goal

## Reading List

- **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision** (Radford et
  al., 2022) — architecture of the ASR model behind Phase 11; why chunked streaming
  inference over an encoder-decoder is nontrivial.
- **faster-whisper / CTranslate2 docs** — practical batched streaming ASR inference,
  directly usable in the Phase 11 worker.
- **LLaVA: Visual Instruction Tuning** (Liu et al., 2023) — the vision-encoder →
  projector → LLM architecture that Phase 14's two-stage scheduler serves.
- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving**
  (Zhong et al., 2024) — the disaggregation argument that Phase 13 generalizes to
  heterogeneous pipeline stages.
- **Moshi: a speech-text foundation model for real-time dialogue** (Défossez et al.,
  Kyutai, 2024) — the landmark full-duplex model behind Phase 13.5's outlook: both
  audio streams as parallel token streams, no turn structure, ~200ms latency.
- **Voice Activity Projection** (Ekstedt & Skantze) — learned turn-taking prediction,
  the research lineage behind Phase 13.5's predictive endpointing.
- **WebRTC for the Curious** (free online book) — jitter buffers, real-time transport,
  and why the network side of Phase 10 is its own discipline.
- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) — enough diffusion
  background to understand why Phase 15's iteration is a denoising step.
- **SGLang / vLLM multimodal serving docs** — how production engines handle image
  tokens, encoder caches, and mixed text/vision batches today.

## License

MIT
