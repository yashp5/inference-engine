import abc
import queue
import time
from collections.abc import Iterator

import grpc
import pytest

import inference_pb2 as pb
import inference_pb2_grpc as pb_grpc
from helpers import TERMINAL, OnEvent, Run, Send, admit, cancel, kind

class EngineCall(grpc.Call, Iterator[pb.EngineEvent], metaclass=abc.ABCMeta):
    """Annotation only: a streaming RPC result is both a grpc.Call (code, details, cancel)
    and an iterator of responses. The generated stub is untyped, so spell it out here."""

class EngineStream:
    def __init__(self, stub: pb_grpc.InferenceStub, timeout: float = 120) -> None:
        self._q: queue.Queue[pb.EngineRequest | None] = queue.Queue()
        self.call: EngineCall = stub.Engine(iter(self._q.get, None), timeout=timeout)

    def send(self, msg: pb.EngineRequest) -> None:
        self._q.put(msg)

    def close(self) -> None:
        self._q.put(None)

    def until_done(self, n: int, on_event: OnEvent | None = None) -> Run:
        """Read until n terminal events, half-close, then drain to end of stream"""
        run, left = Run(), n
        for ev in self.call:
            run.events.append(ev)
            if on_event:
                on_event(ev, self.send)
            if kind(ev) in TERMINAL:
                left -= 1
                if left == 0:
                    self.close()
        return run

def gen_req(
    rid: str = "u1",
    prompt: str = "The capital of france is",
    max_tokens: int = 8,
    temperature: float = 0.0,
) -> pb.GenerateRequest:
    return pb.GenerateRequest(request_id=rid, prompt=prompt, max_tokens=max_tokens, temperature=temperature)

# --------------- unary / server-streaming --------------------------------

def test_generate(stub: pb_grpc.InferenceStub):
    resp: pb.GenerateResponse = stub.Generate(gen_req(), timeout=60)
    assert resp.request_id == "u1"
    assert resp.generated_text
    assert 0 < resp.tokens_generated <= 8

@pytest.mark.parametrize("req", [
    gen_req(rid=""),
    gen_req(prompt=""),
    gen_req(max_tokens=0),
    gen_req(temperature=2.5),
])
def test_generate_validation(stub: pb_grpc.InferenceStub, req: pb.GenerateRequest):
    with pytest.raises(grpc.RpcError) as e:
        stub.Generate(req, timeout=10)
        assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT

def test_generate_stream(stub: pb_grpc.InferenceStub):
    msgs: list[pb.GenerateStreamResponse] = list(stub.GenerateStream(gen_req(rid="s1"), timeout=60))
    assert msgs[-1].finished
    assert not any(m.finished for m in msgs[:-1])
    assert [m.tokens_generated for m in msgs] == list(range(1, len(msgs) + 1))

# ----------------------- Engine bidi ---------------------------------------

def test_engine_two_requets(stub: pb_grpc.InferenceStub):
    s = EngineStream(stub)
    s.send(admit("e1", "The capital of France is", 8))
    s.send(admit("e2", "2+2=", 3))
    run = s.until_done(2)

    assert run.finished("e1") and run.finished("e2")
    assert s.call.code() == grpc.StatusCode.OK

def test_engine_bad_admit_does_not_kill_stream(stub: pb_grpc.InferenceStub):
    s = EngineStream(stub)
    s.send(admit("bad", "", 4))
    s.send(admit("ok", "Hello", 4))
    run = s.until_done(2)

    assert run.rejected("bad") and run.finished("ok")
    assert s.call.code() == grpc.StatusCode.OK


def test_engine_cancel(stub: pb_grpc.InferenceStub):
    def on_event(ev: pb.EngineEvent, send: Send) -> None:
        if kind(ev) == "token" and ev.token.index == 2:
            send(cancel("c1"))

    s = EngineStream(stub)
    s.send(admit("c1", "Write a long story:", 64))
    run = s.until_done(1, on_event=on_event)
    fin = run.finished("c1")
    assert fin is not None
    assert fin.reason == pb.FINISH_REASON_CANCELLED

def test_second_stream_rejected_then_lock_released(stub: pb_grpc.InferenceStub):
    a = EngineStream(stub)
    a.send(admit("a1", "Write a long story:", 32))
    next(a.call)  # a is deinitely running and holds the lock

    b = EngineStream(stub)
    b.close()
    with pytest.raises(grpc.RpcError) as e:
        list(b.call)
    assert e.value.code() == grpc.StatusCode.FAILED_PRECONDITION

    a.until_done(1)

    c = EngineStream(stub)     # lock must be free again
    c.send(admit("c1", "Hi", 2))
    assert(c.until_done(1).finished("c1"))

def test_client_disconnect_resets_engine(stub: pb_grpc.InferenceStub):
    """client vanishes mid-generation. The next stream must get the lock eventually and start with a clean engine (no leftover active slots)"""
    a = EngineStream(stub)
    a.send(admit("gone", "Write a long story:", 64))
    for ev in a.call:
        if kind(ev) == "token" and ev.token.index == 2:
            break
    a.call.cancel()
    a.close()

    deadline = time.monotonic() + 30
    while True:
        b = EngineStream(stub)
        b.send(admit("next", "Hi", 2))
        try:
            run = b.until_done(1)
            break
        except grpc.RpcError as e:
            if e.code() != grpc.StatusCode.FAILED_PRECONDITION or time.monotonic() > deadline:
                raise
            time.sleep(0.5)

    first_stats = run.of("stats")[0]
    assert first_stats.active_slots == 1, "previous stream's slot leaked"
