import logging, queue
import inference_pb2
from engine import Engine

logging.basicConfig(level=logging.INFO)


def admit(
    rid: str, prompt: str, max_tokens: int, temperature: float = 0.0
) -> inference_pb2.EngineRequest:
    return inference_pb2.EngineRequest(
        admit=inference_pb2.Admit(
            request_id=rid, prompt=prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
    )


inbox: queue.Queue[inference_pb2.EngineRequest | None] = queue.Queue()
inbox.put(admit("R1", "The capital of France is", 8))
inbox.put(admit("R2", "2+2=", 3))
inbox.put(admit("R3", "Explain gRPC in one sentence:", 20))
inbox.put(None)                       # half-close: drain and exit

out: dict[str, list[str]] = {}
with Engine() as eng:
    for ev in eng.run(inbox):
        kind: str | None = ev.WhichOneof("payload")
        if kind == "token":
            out.setdefault(ev.token.request_id, []).append(ev.token.text)
        elif kind == "admitted":
            print(f"  admitted {ev.admitted.request_id} -> slot {ev.admitted.slot_id}")
        elif kind == "finished":
            f = ev.finished
            print(f"  finished {f.request_id} slot={f.slot_id} "
                  f"reason={f.reason} tokens={f.tokens_generated}")
            print(f"    {''.join(out.get(f.request_id, []))!r}")
        elif kind == "rejected":
            print(f"  rejected {ev.rejected.request_id}: {ev.rejected.reason}")
        elif kind == "stats":
            s = ev.stats
            print(f"step {s.step:>3} active={s.active_slots} free={s.free_slots} "
                  f"wait={s.waiting} batch={s.batch_tokens} "
                  f"prefill={s.prefill_tokens} kv={s.kv_used} "
                  f"{s.step_time_us/1000:.1f}ms")
