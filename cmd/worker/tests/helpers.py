"""
This is shared by both Python test files.
It counts Admits sent against terminal events (Finished or Rejected) and half-closes only when they're equal.
Counting events rather than tracking ids means duplicate-id and cancel cases still add up.
"""

import queue
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, cast, overload

from google.protobuf.message import Message

import inference_pb2 as pb
from engine import Engine

EventKind = Literal["admitted", "token", "finished", "stats", "rejected"]
Send = Callable[[pb.EngineRequest], None]
OnEvent = Callable[[pb.EngineEvent, Send], None]
CloseAfter = Callable[[pb.EngineEvent], bool]

TERMINAL: Final[tuple[EventKind, ...]] = ("finished", "rejected")

def admit(rid: str, prompt: str, max_tokens: int = 8, temperature: float = 0.0) -> pb.EngineRequest:
    return pb.EngineRequest(admit=pb.Admit(
        request_id=rid, prompt=prompt, max_tokens=max_tokens, temperature=temperature
    ))

def cancel(rid: str) -> pb.EngineRequest:
    return pb.EngineRequest(cancel=pb.Cancel(request_id=rid))

def kind(ev: pb.EngineRequest | pb.EngineEvent) -> str | None:
    return cast(str | None, ev.WhichOneof("payload"))

@dataclass
class Run:
    events: list[pb.EngineEvent] = field(default_factory=list)

    @overload
    def of(self, k: Literal["admitted"]) -> list[pb.Admitted]: ...
    @overload
    def of(self, k: Literal["token"]) -> list[pb.Token]: ...
    @overload
    def of(self, k: Literal["finished"]) -> list[pb.Finished]: ...
    @overload
    def of(self, k: Literal["stats"]) -> list[pb.StepStats]: ...
    @overload
    def of(self, k: Literal["rejected"]) -> list[pb.Rejected]: ...
    def of(self, k: EventKind) -> Sequence[Message]:
        return [cast(Message, getattr(e, k)) for e in self.events if kind(e) == k]

    def tokens(self, rid: str) -> list[pb.Token]:
        return [t for t in self.of("token") if t.request_id == rid]

    def text(self, rid: str) -> str:
        return "".join(t.text for t in self.tokens(rid))

    def tokens_ids(self, rid: str) -> list[int]:
        return [t.token_id for t in self.tokens(rid)]

    def finished(self, rid: str) -> pb.Finished | None:
        return next((f for f in self.of("finished") if f.request_id == rid), None)

    def rejected(self, rid: str) -> list[pb.Rejected]:
        return [r for r in self.of("rejected") if r.request_id == rid]

def assert_clean(eng: Engine) -> None:
    """No leaked slots, samplers, or queued works. A double _retire shows up as a duplicate in free"""
    assert sorted(eng.free) == list(range(eng.n_slots))
    assert not eng.waiting
    assert all(not s.active and s.sampler is None for s in eng.slots)

def drive(
    eng: Engine,
    requests: Iterable[pb.EngineRequest],
    on_event: OnEvent | None = None,
    close_after: CloseAfter | None = None,
) -> Run:
    """Run the engine in-process until every Admit got a terminal event.
    on_event(ev, send): inject messages mid-run (e.g. a Cancel after N tokens)
    close_after(ev) -> bool: half-close early, to test the graceful drain
    """

    inbox: queue.Queue[pb.EngineRequest | None] = queue.Queue()
    run = Run()
    pending = 0
    closed = False

    def send(msg: pb.EngineRequest) -> None:
        nonlocal pending
        # the engine drops an empty request_id silently, so it never terminates
        if kind(msg) == "admit" and msg.admit.request_id:
            pending += 1
        inbox.put(msg)

    def close() -> None:
        nonlocal closed
        if not closed:
            closed = True
            inbox.put(None)

    for r in requests:
        send(r)
    if pending == 0:
        close()

    for ev in eng.run(inbox):
        run.events.append(ev)
        if on_event:
            on_event(ev, send)
        if kind(ev) in TERMINAL:
            pending -= 1
        if pending == 0 or (close_after and close_after(ev)):
            close()


    assert_clean(eng)
    return run
