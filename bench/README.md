# Modern inference engines: benchmarking, profiling and important parameters

This guide explains how to measure and tune inference engines, then maps those concepts to this repository's benchmark tools. For measured results, graphs and observations, see the [main README](../README.md#benchmarks).

An inference engine is built around a constrained objective: serve as much useful work as possible within latency, memory, cost and output-quality requirements. Interactive chat needs a fast first token and steady generation; an offline job can tolerate waiting to achieve higher throughput. A single tokens-per-second number cannot describe both.

## Prefill and decode have different bottlenecks

**Prefill** processes the prompt and builds its key/value (KV) cache. **Decode** repeatedly uses that cache to generate subsequent tokens. Long prefills often put pressure on compute; small-batch decode often puts pressure on memory bandwidth because each step reads model weights and cached state. The bottleneck changes with model, batch size, context length and hardware. Running a large prefill alongside decoding requests can also delay their next tokens; chunked prefill limits how much prompt work enters one iteration. See [vLLM's prefill/decode tuning guidance](https://docs.vllm.ai/en/stable/configuration/optimization/).

This is why benchmarks separate prompt processing, token generation and complete requests. For example, [llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) measures prompt processing, generation, or both, and can vary context depth, batch size, threads and GPU offload. Its measurements exclude tokenization and sampling, so they are not directly comparable to HTTP serving throughput.

## Metrics: what counts as fast?

For a streaming request, let `t0` be client send time, `t1` the first token's arrival, `tN` the last token's arrival and `N` the output token count:

| Metric | Definition | What it tells us |
| :--- | :--- | :--- |
| Time to first token (TTFT) | `t1 - t0` | Initial wait, including queueing, prefill and transport |
| Inter-token latency (ITL) | Time between successive token arrivals | Whether generation is smooth or stalls |
| Time per output token (TPOT) | `(tN - t1) / (N - 1)`, for `N > 1` | Average generation interval after the first token; hides individual stalls |
| End-to-end latency (E2E) | Full response completion minus `t0` | How long the complete answer takes |
| Output throughput | Total output tokens / elapsed benchmark seconds | Aggregate serving capacity, across all requests |
| Request throughput | Completed requests / elapsed seconds | Capacity for a specified request-length distribution |

Report latency distributions (p50, p95, p99), not just averages. A response chunk can contain several tokens, so chunk timing alone does not reveal every token's arrival time. Always state the measurement boundary and token-counting convention. These distinctions follow [NVIDIA's inference metric definitions](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics).

Also report rejected requests, timeouts and errors. **SLO goodput** counts successful requests that meet a stated service-level objective, divided by elapsed time. For example, an illustrative chat SLO might require TTFT ≤ 500 ms and TPOT ≤ 50 ms. Throughput that misses that objective is not usable capacity for that application. [vLLM's serving benchmark](https://docs.vllm.ai/en/stable/cli/bench/serve/#--goodput) supports goodput thresholds for TTFT, TPOT and E2E latency. Cost per successful request or per million output tokens should be compared at the same quality and latency target.

## Parameters engines are designed and tuned around

Keep three things separate: the workload being served, the engine settings being changed, and the metrics being optimized.

| Parameter group | Examples | Why it matters |
| :--- | :--- | :--- |
| Model and numerical format | Parameter count, architecture, weight/KV precision, quantization | Set memory demand and kernel work; quality must be checked when precision changes |
| Request shape | Input/output token distributions, context length, shared prefixes, sampling and EOS behavior | Determine prefill cost, decode duration and cache reuse |
| Offered load | Concurrent clients, requests/second, bursts, cancellations | Determine queueing and whether capacity meets demand |
| Scheduling | Maximum active sequences, tokens per iteration, prefill chunk size, admission policy | Balance batching efficiency against first-token and generation latency |
| Memory | KV-cache budget, context limits, prefix caching, eviction/offload | Determine how much live request state fits and when work must wait or be recomputed |
| Execution and placement | CPU threads, accelerator type/count, kernels, tensor/pipeline parallelism, replicas | Determine compute, bandwidth and communication costs |
| Service policy | Queue bounds, deadlines, priorities, rate limits | Determine fairness and behavior when demand exceeds capacity |

Request count and token budget are different limits: eight requests with long prompts need much more prefill work and KV space than eight short requests. Increasing batch width can improve aggregate throughput while making each iteration slower. Reserving more memory for KV state can admit more sequences, but leaves less headroom for weights and execution buffers. Production tuning exposes these trade-offs through controls such as `max_num_seqs`, `max_num_batched_tokens` and `gpu_memory_utilization`; see [vLLM optimization and tuning](https://docs.vllm.ai/en/stable/configuration/optimization/).

## Benchmarking procedure

1. **Fix the comparison.** Record hardware, model/tokenizer revision, quantization, engine/backend versions, thread and device settings, context limits and cache policy. Change one parameter at a time, or explicitly define a configuration sweep.
2. **Exercise representative shapes.** Include short and long prompts, short and long outputs, mixed lengths and the expected prefix reuse. Report actual token counts as well as requested limits. Test output quality separately from serving speed.
3. **Warm up, then measure.** Separate loading, compilation and cache warmup from steady-state measurements; measure cold starts separately if they matter. Repeat trials, alternate configuration order, and report variation. Retain failures and per-request data, not only successful-request percentiles.
4. **Sweep load.** A closed-loop test holds up to `C` requests outstanding; each client waits before sending another. An open-loop test schedules arrivals at a chosen rate independently of completions. Closed-loop traffic backs off automatically as responses slow, so it does not establish behavior under sustained external arrivals. [vLLM's load controls](https://docs.vllm.ai/en/stable/cli/bench/serve/#--request-rate) distinguish request rate, burstiness and maximum concurrency.
5. **Find the operating point.** Plot throughput and tail latency against load. Identify where throughput flattens, queues grow, or the SLO fails. Choose capacity within the target latency/error budget, and report ramp-up and drain effects for short tests.

## Profiling: explain the measured result

A benchmark measures the outcome; a profile locates the time and resource costs behind it. First correlate request latency with queue depth, active sequences, prefill/decode work and KV usage. Then capture a short trace at a representative load, looking for host scheduling delays, expensive model steps, memory transfers or communication stalls.

For CUDA backends, Nsight Systems can show the CPU/GPU execution timeline; PyTorch Profiler can expose operators, shapes and allocations in PyTorch-based engines. Use a profiler appropriate to the actual backend. Detailed tracing adds overhead, so use separate profiling runs and remeasure improvements without it. See [vLLM's profiling guide](https://docs.vllm.ai/en/stable/contributing/profiling/).

High slot occupancy alone does not prove efficient hardware use. An engine can have every slot occupied while waiting on memory or executing slow steps. Interpret occupancy together with step duration, tokens produced, memory use and hardware counters.

## Benchmarking this repository

The following sections describe our benchmark controls, measurement boundaries and reproduction procedure. Run all commands from the repository root after completing [Getting Started](../README.md#getting-started).

### Test setup

- **Hardware and model:** not recorded with these results. Before using the absolute numbers for external comparisons, record the CPU/GPU and memory, exact GGUF model and quantization, llama.cpp/llama-cpp-python versions, backend/offload configuration and thread counts. The example model in [Getting Started](../README.md#getting-started) does not establish which model produced these runs.
- **Engine:** 8 slots, 512-token context per sequence, served through the continuous-batching `Engine` stream (`-continuousBatching`). The policy is selected with the worker's `--engine-policy`:
  - **`continuous`** refills a freed slot on the next decode step.
  - **`static`** forms a batch only after every slot has drained. It waits up to 20 ms (`--engine-static-fill-wait-ms`) for the batch to fill, then seals it until all of its requests finish.
  - Both admit at most one request per step, so the comparison isolates *when* requests may join.
  - This `static` is an engine policy, not the Go server's static mode from Phase 3 (where requests reach the model one at a time).
- **Workloads:** every request generates exactly `max_tokens` tokens (`--bench-ignore-eos`), so request lengths are controlled. Both workloads have an expected mean of 64 output tokens.
  - **Mixed:** 80% of requests ask for 16 tokens, 20% for 256.
  - **Uniform:** every request asks for 64 tokens.
  - The length sequence is seeded (`-seed 1`), so both policies receive the same generated length list at each concurrency. Arrival timing and generated text are not fixed by that seed. All requests use the prompt `"Once upon a time"` at temperature 0.7.
  - The mixed workload's expected mean is 64 tokens, but its realized mean varies: at concurrency 8, the sample contains 140 short and 20 long requests, averaging 46 tokens. Compare policies within the same workload/concurrency pair; mixed and uniform runs do not have equal total output work.
- **Load:** closed loop. *C* clients share a pool of `20 × C` requests, at *C* = 8, 16 and 32 (160, 320 and 640 requests per run). `-requestsPerWorker 20` sets the total; it does not enforce 20 requests for each client. All 4,480 requests returned 200.
- **Metrics:**
  - **TTFT** is measured from HTTP handler receipt until Go receives the first worker token event (`ttft_ms`). The HTTP response is buffered until generation finishes, so this is internal TTFT, not client-observed streaming TTFT. End-to-end latency is measured by the client through reading the full response body.
  - **Throughput** is output tokens divided by wall-clock time.
  - **Slot utilization** is the mean of `active_slots / 8`, sampled from `GET /stats` every 100 ms.

With 8 slots, concurrency 16 and 32 exceed the active-sequence capacity, so much of their latency is queueing (median queue time ≈ median TTFT). This is contention under closed-loop load, not evidence of an unbounded queue under a fixed external arrival rate. Concurrency 8 is the cleanest head-to-head comparison.

### How the general parameters map to this engine

| Concern | Implementation/control | What these runs exercise |
| :--- | :--- | :--- |
| Active sequences | Worker `--engine-slots`, Go `-engineSlots` | Both set to 8; clients sweep 8, 16, 32 |
| Admission policy | `--engine-policy`; static fill wait `--engine-static-fill-wait-ms` | Refill immediately versus seal and drain; 20 ms static fill window |
| Context/KV capacity | `--engine-per-seq-ctx`; `n_ctx = slots × per_seq_ctx` | 512 tokens per sequence, 4,096 total configured context; prompt plus requested output must fit |
| Tokens in a forward pass | `ENGINE_N_BATCH` / `Engine(n_batch=...)`, default 512 | Shared prefill/decode budget; at most one whole prompt admitted per step; no chunked prefill |
| Length distribution | `-maxTokensDist`, `-seed`; worker `--bench-ignore-eos` | Controlled mixed and uniform outputs; one fixed short prompt |
| Backpressure/deadlines | `-maxQueueDepth`, `-queueTimeout`, `-requestTimeout`; client `-timeout` | Raised timeouts let the high-concurrency runs finish; no rejection-policy sweep |

The Go flags `-maxBatchSize`, `-maxBatchWait`, `-workerCount` and `-maxInFlight` belong to the separate static server path. They do not tune this comparison, because both policies use `-continuousBatching` and the worker's `Engine` RPC.

### Reading the measurements correctly

- `runs.csv` contains output tok/s, successful responses/s (`goodput_rps`), E2E p50/p95/p99 and TTFT p50/p99. Here `goodput_rps` means successful HTTP responses per second; there is no latency-SLO filter. Latency percentiles exclude failures, so read them alongside the status/error counts.
- `requests.csv` records request length, status, client latency, queue time, internal TTFT and generated tokens. In engine mode, queue time runs from Go enqueue until Go receives the worker's `Admitted` event, including waits for capacity and worker admission. Break results down by length to expose fairness trade-offs.
- `runs.csv`'s `utilization_pct` is **client activity**: summed successful-request latency divided by `wall time × concurrency`, expressed as a percentage. The headline **engine slot utilization** instead comes from the mean valid `active_slots / 8` samples in `stats.csv`. Neither is GPU utilization.
- `/stats` also exposes `batch_tokens`, `prefill_tokens`, `kv_used` and `step_time_us`, but the current CSV poller does not save them. `step_time_us` times the `llama_decode` call only; it excludes admission, sampling and event delivery. A 100 ms poll of the latest stats also misses individual steps, so a complete step-latency distribution requires recording each worker stats event.

TTFT is currently populated only by the Go `Engine` client. Against the default static server mode, the response's zero-valued `ttft_ms` is uninstrumented, not an instantaneous first token. Both policies in the recorded comparison use the instrumented engine path.

The measurement code is in [loadtest.go](loadtest.go); [analyze.py](analyze.py) generates the plots and length-specific summaries. The Go report uses nearest-rank percentiles, while the analysis script's per-length summaries use pandas' interpolated quantiles, so small numerical differences are expected.

### Reproducing

Use the same model, backend and machine for each policy. Warm up each configuration before measured runs, and keep warmup output separate. The commands below reproduce the workload and configuration; the missing original environment metadata prevents an exact reproduction of the historical timings.

```bash
# 1. Worker: pick the policy. --bench-ignore-eos makes every request generate exactly max_tokens.
cmd/worker/venv/bin/python cmd/worker/worker.py --model-path "$MODEL_PATH" \
  --engine-slots 8 --engine-policy continuous --bench-ignore-eos

# 2. Server: continuous-batching mode. Raise the timeouts, or requests queued at
#    concurrency 16 and 32 hit the 5 s default queue timeout and return 504.
go run ./cmd/server -continuousBatching -engineSlots 8 -queueTimeout 300s -requestTimeout 300s

# 3. Load: the Makefile defaults are the mixed workload (16:0.8,256:0.2) at concurrency 8, 16 and 32,
#    20 × concurrency total requests at each level.
make bench LABEL=continuous
make bench LABEL=continuous-uniform BENCH_MIX=64:1

# 4. Restart the worker with --engine-policy static, then:
make bench LABEL=static
make bench LABEL=static-uniform BENCH_MIX=64:1

# 5. Summary tables and graphs (needs pandas and matplotlib from requirements.txt)
cmd/worker/venv/bin/python bench/analyze.py
```

The load tester appends to the CSVs. Archive the existing CSVs before a fresh set of runs, or select separate files with `-out`, `-reqOut` and `-statsOut`. `analyze.py` reads `bench/results/`, pairs runs by these four labels and assumes 8 slots; adjust those settings when changing the experiment. Reusing labels for repeated trials without separating the files will merge their samples in the analysis.

The load tester also works on its own, against either server mode:

```bash
# One level: 50 concurrent clients, 1000 requests
go run ./bench -concurrency 50 -totalRequest 1000

# Sweep concurrency levels with a length mix, 20 × concurrency requests per level
go run ./bench -sweep 1,10,50,100,500 -requestsPerWorker 20 -maxTokensDist 16:0.8,256:0.2 -label mytest
```

For each concurrency level it reports p50/p95/p99 latency, TTFT, goodput, and counts of 429s, timeouts and 5xx errors. It appends one summary row to `bench/results/runs.csv` and one row per request to `bench/results/requests.csv`. It also polls `GET /stats` every `-statsInterval` (default 100 ms) into `bench/results/stats.csv`, which records queue depth and engine occupancy over time.

### Extending the measurements

The next useful experiments are repeated trials with an environment manifest; prompt-length and slot-count sweeps; fixed-rate/bursty arrivals with rejection and deadline accounting; and per-step traces for the uniform-workload gap. Add streaming token timestamps to measure user-visible TTFT and ITL. These would turn the current admission-policy comparison into a broader capacity and latency characterization.
