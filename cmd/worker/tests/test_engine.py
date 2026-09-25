import codecs
import warnings
from collections.abc import Callable

import pytest
from engine import Engine

from helpers import admit, cancel, drive, kind
import inference_pb2

MakeEngine = Callable[..., Engine]

PROMPTS = {
    "a": "The capital of France is",
    "b": "def fibonacci(n)",
    "c": "Once upon a time,",
}

def test_single_request_completes(eng: Engine):
    run = drive(eng, [admit("r1", "The capital of France is", 8)])
    toks = run.tokens("r1")
    fin = run.finished("r1")

    print(toks)
    print(fin)

    assert fin is not None and fin.reason in (inference_pb2.FINISH_REASON_EOS, inference_pb2.FINISH_REASON_LENGTH)
    assert [t.index for t in toks] == list(range(len(toks)))
    assert fin.tokens_generated == len(toks)
    if fin.reason == inference_pb2.FINISH_REASON_LENGTH:
        assert len(toks) == 8
    assert len(run.of("admitted")) == 1


def test_batched_matches_solo(make_engine: MakeEngine):
    """The KV-isolation test. Greedy output must not depend on who else is in the batch.
    A divergence in the first few tokens means seq_id/pos crosstalk;
    late drift can be legit float noise from different batch shapes.
    """
    eng = make_engine(n_slots=4)
    solo = {rid: drive(eng, [admit(rid, p, 16)]).tokens_ids(rid) for rid, p in PROMPTS.items()}
    batched = drive(eng, [admit(rid, p, 16) for rid, p in PROMPTS.items()])

    for rid in PROMPTS:
        got = batched.tokens_ids(rid)
        assert got[:4] == solo[rid][:4], f"{rid}: batched {got} vs solo {solo[rid]}"
        if got != solo[rid]:
            warnings.warn(f"{rid} drifted late: batched {got} vs solo {solo[rid]}")

def test_more_requests_than_slots(make_engine: MakeEngine):
    eng = make_engine(n_slots=2)
    ids = [f"r{i}" for i in range(5)]
    run = drive(eng, [admit(rid, "Count to ten:", 6) for rid in ids])

    assert all(run.finished(rid) for rid in ids)
    assert {a.slot_id for a in run.of("admitted")} <= {0,1}
    assert max(s.active_slots for s in run.of("stats")) <= 2
    assert max(s.waiting for s in run.of("stats")) >= 1 # queuing actually happened

def test_one_admission_per_step(make_engine: MakeEngine):
    eng = make_engine(n_slots=4)
    run = drive(eng, [admit(rid, p, 4) for rid, p in PROMPTS.items()])

    admitted_this_step = 0
    for ev in run.events:
        if kind(ev) == "admitted":
            admitted_this_step += 1
        elif kind(ev) == "stats":
            assert admitted_this_step <= 1
            assert (ev.stats.prefill_tokens > 0) == (admitted_this_step == 1)
            admitted_this_step = 0

def test_stats_invariants(make_engine: MakeEngine):
    eng = make_engine(n_slots=2)
    run = drive(eng, [admit(f"r{i}", "Hello there", 5) for i in range(4)])

    for s in run.of("stats"):
        assert s.active_slots + s.free_slots == 2
        assert s.prefill_tokens <= s.batch_tokens
        assert s.batch_tokens - s.prefill_tokens <= 2 # at most one decode row per slot
        assert s.kv_used <= 2 * 256

@pytest.mark.parametrize("msg,reason", [
    (admit("v1", "", 4), "prompt is required"),
    (admit("v2", "hi", 0), "max_tokens must be > 0"),
    (admit("v3", "hi", 4, temperature=2.5), "temperature"),
    (admit("v4", "hi", 300), "exceeds per-sequence context"), # per_seq_ctx=256
])
def test_validation_rejects_without_disturbing_others(eng: Engine, msg, reason):
    run = drive(eng, [admit("good", "The sky is", 4), msg])
    rid = msg.admit.request_id

    assert len(run.rejected(rid)) == 1 and reason in run.rejected(rid)[0].reason
    assert run.finished("good") is not None


def test_prompt_longer_than_n_batch(make_engine: MakeEngine):
    eng = make_engine(per_seq_ctx=256, n_batch=64)
    run = drive(eng, [admit("long", "hello " * 100, 4)])
    assert "exceeds n_batch" in run.rejected("long")[0].reason

def test_duplicate_request_id(eng: Engine):
    run = drive(eng, [admit("d", "hi", 4), admit("d", "hi", 4)])
    assert "duplicate" in run.rejected("d")[0].reason
    assert run.finished("d") is not None

def test_cancel_while_waiting(make_engine: MakeEngine):
    eng = make_engine(n_slots=1)
    run = drive(eng, [admit("a", "Tell me a story:", 16), admit("b", "hi", 4), cancel("b")])

    assert run.rejected("b")[0].reason == "cancelled before admission"
    assert not run.tokens("b")
    assert run.finished("a") is not None

def test_cancel_while_running(eng: Engine):
    def on_event(ev, send):
        if kind(ev) == "token" and ev.token.request_id == "r" and ev.token.index == 2:
            send(cancel("r"))

    run = drive(eng, [admit("r", "Write a long story:", 64)], on_event=on_event)
    fin = run.finished("r")
    # The cancel is drained at the start of the next step, so exactly 3 tokens
    assert fin is not None
    assert fin.reason == inference_pb2.FINISH_REASON_CANCELLED
    assert fin.tokens_generated == 3

def test_cancel_unknown_is_noop(eng: Engine):
    run = drive(eng, [admit("r", "hi", 4), cancel("nope")])
    fin = run.finished("r")
    assert fin is not None
    assert fin.reason in (inference_pb2.FINISH_REASON_LENGTH, inference_pb2.FINISH_REASON_EOS)

def test_half_close_drains_active_rejects_queued(make_engine: MakeEngine):
    eng = make_engine(n_slots=1)
    first_token = lambda ev: kind(ev) == "token" and ev.token.request_id == "a"
    run = drive(eng, [admit(r, "Hello", 8) for r in "abc"], close_after=first_token)

    fin = run.finished("a")
    assert fin is not None
    assert fin.reason in (inference_pb2.FINISH_REASON_EOS, inference_pb2.FINISH_REASON_LENGTH) # finished, not cut off
    for rid in "bc":
        assert run.rejected(rid)[0].reason == "engine stream closed"

def test_pieces_roundtrip_utf8(eng: Engine):
    """piece() returns raw bytes that may split a codepoint; the incremental decoder must put them back together"""
    text = "naïve café — こんにちは 🙂"
    toks = eng.tokenize(text, add_bos=False)
    pieces = [eng.piece(t) for t in toks]

    dec = codecs.getincrementaldecoder("utf-8")("replace")
    out = "".join(dec.decode(p) for p in pieces) + dec.decode(b"", final=True)
    assert out.strip() == text

    def invalid_alone(b):
        try:
            b.decode("utf-8")
            return False
        except UnicodeDecodeError:
            return True
    # Otherwise this test isnt exercising the split-codepoint path at all
    assert any(invalid_alone(p) for p in pieces)
