import codecs
from collections import deque
import ctypes
from dataclasses import dataclass, field
import logging
import os
from queue import Queue
import queue
import threading
from time import perf_counter
from typing import Any, Deque, Iterator, List, Optional, Tuple
import llama_cpp
from llama_cpp._internals import LlamaModel, LlamaContext
import inference_pb2

log = logging.getLogger("engine")

MODEL_PATH: str = os.getenv(
    "MODEL_PATH", "/Users/yash/build/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
N_SLOTS: int = int(os.getenv("ENGINE_N_SLOTS", "8"))
PER_SEQ_CTX: int = int(os.getenv("ENGINE_PER_SEQ_CTX", "512"))
N_BATCH: int = int(os.getenv("ENGINE_N_BATCH", "512"))

REASON_EOS = inference_pb2.FINISH_REASON_EOS
REASON_LENGTH = inference_pb2.FINISH_REASON_LENGTH
REASON_CANCELLED = inference_pb2.FINISH_REASON_CANCELLED
REASON_ERROR = inference_pb2.FINISH_REASON_ERROR

_TOP_K = 40
_TOP_P = 0.95

_PIECE_BUF = 128

_backend_lock = threading.Lock()
_backend_ready = False

def _init_backend() -> None:
    global _backend_ready
    with _backend_lock:
        if not _backend_ready:
            llama_cpp.llama_backend_init()
            _backend_ready = True

@dataclass
class Pending:
    """Validated and tokenized, waiting for a free slot."""
    request_id: str
    tokens: List[int]
    max_tokens: int
    temperature: float

@dataclass
class Slot:
    id: int                         # seq_id
    active: bool = False
    request_id: str = ""
    sampler: Any = None
    decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")("replace")
    )
    n_past: int = 0
    max_tokens: int = 0
    generated: int = 0
    last_token: int = -1

"""
1. Admit at most one waiting request into a free slot. Guard it: reject if len(prompt_tokens) + max_tokens > per_seq_ctx, and
    skip admission this step if batch.n_tokens + len(prompt_tokens) > n_batch (a long prefill can't fit alongside the decoding tokens).
    Push prompt tokens with seq_id=slot, pos=0..n-1, logits=1 on the last prompt token only.
2. For each decoding slot, push its previously-sampled token at pos=n_past, logits=1, then n_past += 1.
3. rc = llama_cpp.llama_decode(ctx.ctx, batch). Check the return code — 1 means no KV slot (back off admission, don't crash), <0 is fatal.
4. For each batch index that had logits=1: llama_sampler_sample(slot.sampler, ctx.ctx, idx) → llama_sampler_accept →
    llama_token_to_piece(model.vocab, tok, buf, 64, 0, False) → emit Token.
5. On EOS (llama_vocab_is_eog(model.vocab, tok)) or max_tokens: emit Finished, then llama_memory_seq_rm(mem, slot, -1, -1),
    llama_sampler_free(slot.sampler), mark free.
6. Emit StepStats.
"""

class Engine:
    def __init__(self,
        model_path: str = MODEL_PATH,
        n_slots: int = N_SLOTS,
        per_seq_ctx: int = PER_SEQ_CTX,
        n_batch: int = N_BATCH,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
    ) -> None:
        self.n_slots = n_slots
        self.per_seq_ctx = per_seq_ctx
        self.n_batch = n_batch

        _init_backend()

        self.model = LlamaModel(
            path_model=model_path,
            params=llama_cpp.llama_model_default_params(),
            verbose=False,
        )
        self.vocab = self.model.vocab

        cp = llama_cpp.llama_context_default_params()
        cp.n_ctx = n_slots * per_seq_ctx
        cp.n_seq_max = n_slots
        cp.n_batch = n_batch
        cp.n_ubatch = n_batch
        if n_threads is not None:
            cp.n_threads = n_threads
        if n_threads_batch is not None:
            cp.n_threads_batch = n_threads_batch
        self.ctx = LlamaContext(model=self.model, params=cp, verbose=False)

        mem = llama_cpp.llama_get_memory(self.ctx.ctx)
        if mem is None:
            raise RuntimeError("llama_get_memory returned NULL")
        self.mem = mem
        # 3rd arg is seq_ids PER TOKEN, not the sequence count
        self.batch = llama_cpp.llama_batch_init(n_batch, 0, 1)

        self.slots: List[Slot] = [Slot(id=i) for i in range(n_slots)]
        self.free: Deque[int] = deque(range(n_slots))
        self.waiting: Deque[Pending] = deque()
        self.step = 0
        self._closed = False
        self._piece_buf = (ctypes.c_char * _PIECE_BUF)()

        log.info(
            "engine ready model=%s n_slots=%d per_seq_ctx=%d n_ctx=%d "
            "n_ctx_train=%d n_batch=%d n_vocab=%d",
            self.model.desc(),
            n_slots,
            per_seq_ctx,
            llama_cpp.llama_n_ctx(self.ctx.ctx),
            self.model.n_ctx_train(),
            n_batch,
            self.model.n_vocab(),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _push(self, token: int, seq_id: int, pos: int, logits: bool) -> int:
        i = self.batch.n_tokens
        if i >= self.n_batch:
            raise RuntimeError(f"batch overflow: n_batch={self.n_batch}")

        self.batch.token[i] = token
        self.batch.pos[i] = pos # per-sequence position, drives rope
        self.batch.n_seq_id[i] = 1
        self.batch.seq_id[i][0] = seq_id # the only thing isolating requests
        self.batch.logits[i] = 1 if logits else 0
        self.batch.n_tokens = i+1

        return i

    def _new_sampler(
        self, temperature: float, seed: int = llama_cpp.LLAMA_DEFAULT_SEED
    ) -> llama_cpp.llama_sampler_p:
        """Build a per-request sampler chain.

        Order matters: the truncation samplers run first, temp rescales what
        survives, and dist must be last because it is the one that actually
        picks. The chain takes ownership of everything added to it, so
        llama_sampler_free(chain) frees the members too - never free the parts.
        """
        chain = llama_cpp.llama_sampler_chain_init(
            llama_cpp.llama_sampler_chain_default_params()
        )
        if chain is None:
            raise RuntimeError("llama_sampler_chain_init returned NULL")
        add = llama_cpp.llama_sampler_chain_add
        if temperature <= 0.0:
            add(chain, llama_cpp.llama_sampler_init_greedy())
        else:
            add(chain, llama_cpp.llama_sampler_init_top_k(_TOP_K))
            add(chain, llama_cpp.llama_sampler_init_top_p(_TOP_P, 1))
            add(chain, llama_cpp.llama_sampler_init_temp(temperature))
            add(chain, llama_cpp.llama_sampler_init_dist(seed))
        return chain

    def _occupy(self, p: Pending) -> Slot:
        slot = self.slots[self.free.popleft()]
        slot.active = True
        slot.request_id = p.request_id
        slot.max_tokens = p.max_tokens
        slot.generated = 0
        slot.n_past = 0
        slot.last_token = -1
        slot.sampler = self._new_sampler(p.temperature)
        slot.decoder.reset()
        return slot

    def _retire(self, slot: Slot) -> None:
        llama_cpp.llama_memory_seq_rm(self.mem, slot.id, -1, -1)
        if slot.sampler is not None:
            llama_cpp.llama_sampler_free(slot.sampler)
        slot.active = False
        slot.request_id = ""
        slot.sampler = None
        slot.decoder.reset()   # drop any partial multi-byte sequence
        slot.n_past = 0
        slot.max_tokens = 0
        slot.generated = 0
        slot.last_token = -1
        self.free.append(slot.id)

    def _active(self) -> List[Slot]:
        return [s for s in self.slots if s.active]

    def _accept(self, admit: inference_pb2.Admit) -> Optional[inference_pb2.EngineEvent]:
        rid = admit.request_id
        if not rid:
            log.warning("dropping Admit with empty request_id")
            return None
        if self._known(rid):
            return self._ev_rejected(rid, "duplicate request_id")
        if not admit.prompt:
            return self._ev_rejected(rid, "prompt is required")
        if admit.max_tokens <= 0:
            return self._ev_rejected(rid, "max_tokens must be > 0")
        if not (0.0 <= admit.temperature <= 2.0):
            return self._ev_rejected(rid, "temperature must be between 0.0 and 2.0")

        tokens = self.tokenize(admit.prompt)

        # Enforcing the budget per sequence is what keeps the unified areana from
        # being oversubscribed, given n_ctx = n_slots * per_seq_ctx
        if len(tokens) + admit.max_tokens > self.per_seq_ctx:
            return self._ev_rejected(
                rid,
                f"prompt {len(tokens)} + max_tokens {admit.max_tokens} exceeds "
                f"per-sequence context {self.per_seq_ctx}"
            )
        # Waiting can never help: a prompt is perfilled in a single decode
        if len(tokens) > self.n_batch:
            return self._ev_rejected(
                rid, f"prompt {len(tokens)} exceeds n_batch {self.n_batch}"
            )

        self.waiting.append(Pending(rid, tokens, admit.max_tokens, admit.temperature))
        return None

    def _cancel(self, rid: str) -> List[inference_pb2.EngineEvent]:
        for p in self.waiting:
            if p.request_id == rid:
                self.waiting.remove(p)
                return [self._ev_rejected(rid, "cancelled before admission")]

        for slot in self._active():
            if slot.request_id == rid:
                ev = self._ev_finished(slot, REASON_CANCELLED)
                self._retire(slot)
                return [ev]

        return [] # already finished, or never seen

    def _known(self, rid: str) -> bool:
        return any(p.request_id == rid for p in self.waiting) or any(
            s.active and s.request_id == rid for s in self.slots
        )

    def _ev_rejected(self, rid: str, reason: str) -> inference_pb2.EngineEvent:
        log.info("rejected request_id=%s reason=%s", rid, reason)
        return inference_pb2.EngineEvent(
            rejected=inference_pb2.Rejected(request_id=rid, reason=reason)
        )

    def _ev_finished(self, slot: Slot, reason: "inference_pb2.FinishReason") -> inference_pb2.EngineEvent:
        return inference_pb2.EngineEvent(
           finished=inference_pb2.Finished(
               request_id=slot.request_id,
               slot_id=slot.id,
               reason=reason,
               tokens_generated=slot.generated
           )
        )

    def _ev_stats(self, step_us, prefill_tokens, batch_tokens):
        return inference_pb2.EngineEvent(
            stats=inference_pb2.StepStats(
                step=self.step,
                active_slots=len(self._active()),
                free_slots=len(self.free),
                waiting=len(self.waiting),
                batch_tokens=batch_tokens,
                prefill_tokens=prefill_tokens,
                step_time_us=step_us,
                kv_used=sum(s.n_past for s in self._active())
            )
        )

    def _drain(self, inbox: "Queue[Optional[inference_pb2.EngineRequest]]", block: bool) -> Tuple[bool, List[inference_pb2.EngineEvent]]:
        events: List[inference_pb2.EngineEvent] = []
        half_closed = False
        blocking = block
        while True:
            try:
                item = inbox.get() if blocking else inbox.get_nowait()
            except queue.Empty:
                break
            blocking = False
            if item is None:         # reader thread saw end of stream
                half_closed = True
                continue
            kind = item.WhichOneof("payload")
            if kind == "admit":
                ev = self._accept(item.admit)
                if ev is not None:
                    events.append(ev)
            elif kind == "cancel":
                events.extend(self._cancel(item.cancel.request_id))
            else:
                log.warning("EngineRequest with no payload")
        return half_closed, events

    def run(self, inbox: "Queue[Optional[inference_pb2.EngineRequest]]") -> Iterator[inference_pb2.EngineEvent]:
        closed = False
        backoff = False

        while True:
            # phase 0: drain the inbox
            active = self._active()
            idle = not active and not self.waiting
            half_closed, events = self._drain(inbox, block=(idle and not closed))
            for ev in events:
                yield ev

            if half_closed and not closed:
                closed = True
                # Graceful drain: active slots finish, but queued requests were
                # never started, so answer them now instead of stranding them.
                # Must be inside this branch, or every step rejects the queue.
                while self.waiting:
                    p = self.waiting.popleft()
                    yield self._ev_rejected(p.request_id, "engine stream closed")

            active = self._active()
            if not active and not self.waiting:
                if closed:
                    return
                continue

            self.step += 1
            self.batch.n_tokens = 0
            planned: List[Tuple[Slot, int]] = []
            sample_at: List[Tuple[Slot, int]] = []
            prefill_tokens = 0
            admitted: Optional[Tuple[Slot, Pending]] = None

            # phase 1: admit at most one
            # One per step on purpose: a long prefill dominates the batch and
            # every decoding slot waits for the slowest row, so batching several
            # admissions is an inter-token latency spike for requests already running
            if not closed and not backoff and self.free and self.waiting:
                head = self.waiting[0]
                # Defer, dont reject: it will fit once the batch shrinks
                if len(active) + len(head.tokens) <= self.n_batch:
                    pending = self.waiting.popleft()
                    new_slot = self._occupy(pending)
                    admitted = (new_slot, pending)
                    last = len(pending.tokens) - 1
                    for pos, tok in enumerate(pending.tokens):
                        idx = self._push(tok, new_slot.id, pos, logits=(pos == last))
                        if pos == last:
                            sample_at.append((new_slot, idx))
                    planned.append((new_slot, len(pending.tokens)))
                    prefill_tokens = len(pending.tokens)

            # phase 2: one decode row per already-active slot
            for slot in active:
                idx = self._push(slot.last_token, slot.id, slot.n_past, logits=True)
                sample_at.append((slot, idx))
                planned.append((slot, 1))

            if self.batch.n_tokens == 0:
                # Backoff held the only admission and nothing is decoding
                # llama_decode on an empty batch is an error, so just retry.
                backoff = False
                continue

            # phase 3: the single forward pass
            t0 = perf_counter()
            rc = llama_cpp.llama_decode(self.ctx.ctx, self.batch)
            step_us = int((perf_counter() - t0) * 1_000_000)
            batch_tokens = self.batch.n_tokens

            if rc < 0:
                for slot in self._active():
                    yield self._ev_finished(slot, REASON_ERROR)
                    self._retire(slot)
                raise RuntimeError(f"llama_decode failed rc={rc}")

            if rc == 1:
                # No KV slot. Nothing computed, so no n_past moves and no
                # Admitted was emitted for the rolled-back request.
                if admitted is not None:
                    rolled_back, pending = admitted
                    self._retire(rolled_back)
                    self.waiting.appendleft(pending)  # keep its FIFO place
                    backoff = True
                    log.warning("step %d: no KV slot, rolled back admission", self.step)
                else:
                    victim = max(self._active(), key=lambda s: s.n_past, default=None)
                    if victim is not None:
                        log.error("step %d: KV exhausted, evicting slot %d", self.step, victim.id)
                        yield self._ev_finished(victim, REASON_ERROR)
                        self._retire(victim)
                yield self._ev_stats(step_us, prefill_tokens, batch_tokens)
                continue

            backoff = False
            for slot, pushed in planned:
                slot.n_past += pushed

            if admitted is not None:
                new_slot, _ = admitted
                yield inference_pb2.EngineEvent(
                    admitted=inference_pb2.Admitted(
                        request_id=new_slot.request_id, slot_id=new_slot.id
                    )
                )

            # phase 4/5: sample, emit, retire
            for slot, idx in sample_at:
                # Raw batch index: llama_get_logits_ith maps it through the
                # output table internally.
                tok = llama_cpp.llama_sampler_sample(slot.sampler, self.ctx.ctx, idx)
                llama_cpp.llama_sampler_accept(slot.sampler, tok)

                if llama_cpp.llama_vocab_is_eog(self.vocab, tok):
                    yield self._ev_finished(slot, REASON_EOS)
                    self._retire(slot)
                    continue

                slot.last_token = tok
                yield inference_pb2.EngineEvent(
                    token=inference_pb2.Token(
                        request_id=slot.request_id,
                        slot_id=slot.id,
                        text=slot.decoder.decode(self.piece(tok)),
                        token_id=tok,
                        index=slot.generated,
                    )
                )
                slot.generated += 1

                if slot.generated >= slot.max_tokens:
                    yield self._ev_finished(slot, REASON_LENGTH)
                    self._retire(slot)

            # phase 6: stats
            yield self._ev_stats(step_us, prefill_tokens, batch_tokens)

    def tokenize(self, prompt: str, add_bos: bool = True, special: bool = False) -> List[int]:
        return self.model.tokenize(prompt.encode("utf-8"), add_bos, special)

    def piece(self, token: int):
        """Raw bytes for one token. NOT guaranteed to be valid UTF-8 on its own"""
        n = llama_cpp.llama_token_to_piece(
            self.vocab, token, self._piece_buf, _PIECE_BUF, 0, False
        )
        if n >= 0:
            return bytes(self._piece_buf[:n])
        buf = (ctypes.c_char * (-n))()
        n = llama_cpp.llama_token_to_piece(self.vocab, token, buf, -n, 0, False)
        return bytes(buf[:n]) if n > 0 else b""

    def reset(self) -> None:
        """Drop all per-request state so the next Engine stream starts clean.

        Called when a stream ends. Without this, a Go reconnect inherits a dirty
        KV cache and the previous stream's leaked sampler chains.
        """
        for slot in self._active():
            self._retire(slot)
        self.waiting.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Before the context dies: _retire calls llama_memory_seq_rm, which
        # needs a live ctx, and frees the sampler chains.
        self.reset()
        llama_cpp.llama_batch_free(self.batch)
        # Context before model. Relying on GC order here segfaults at
        # interpreter shutdown.
        self.ctx.close()
        self.model.close()
