"""
- make_engine caches one Engine per config, so the model loads once per shape rather than once per test.
- worker_addr starts the real worker.py on a free port and writes its log to a file.
When a gRPC test fails, the worker's own logs are where you'll find the reason.
"""

import os
import signal
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from typing import cast

import grpc
import pytest

import inference_pb2_grpc as pb_grpc
from engine import MODEL_PATH, Engine

WORKER_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MakeEngine = Callable[..., Engine]

def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if not os.path.exists(MODEL_PATH):
        skip = pytest.mark.skip(reason=f"model not found at {MODEL_PATH}")
        for item in items:
            item.add_marker(skip)

@pytest.fixture(scope="session")
def make_engine() -> Iterator[MakeEngine]:
    cache: dict[tuple[int, int, int], Engine] = {}

    def make(n_slots: int = 2, per_seq_ctx: int = 256, n_batch: int = 256) -> Engine:
        key = (n_slots, per_seq_ctx, n_batch)
        if key not in cache:
            cache[key] = Engine(n_slots=n_slots, per_seq_ctx=per_seq_ctx, n_batch=n_batch)
        eng = cache[key]
        eng.reset()
        return eng

    yield make
    for eng in cache.values():
        eng.close()

@pytest.fixture
def eng(make_engine: MakeEngine) -> Engine:
    return make_engine()

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        # typeshed types getsockname() as Any; for AF_INET it's (host, port)
        _, port = cast(tuple[str, int], s.getsockname())
        return port

@pytest.fixture(scope="session")
def worker_addr(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = _free_port()
    log_path = tmp_path_factory.mktemp("worker") / "worker.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "worker.py", "--port", str(port), "--engine-slots", "2",
            "--engine-per-seq-ctx", "256", "--max-threads", "8"],
        cwd=WORKER_DIR, stdout=log, stderr=subprocess.STDOUT
    )
    addr = f"localhost:{port}"
    ch = grpc.insecure_channel(addr)
    try:
        grpc.channel_ready_future(ch).result(timeout=90)
    except grpc.FutureTimeoutError:
        proc.kill()
        pytest.fail(f"worker never became ready, see {log_path}")
    finally:
        ch.close()

    yield addr

    proc.send_signal(signal.SIGTERM)
    try:
        _ = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        _ = proc.wait()
    log.close()

@pytest.fixture(scope="session")
def stub(worker_addr: str) -> Iterator[pb_grpc.InferenceStub]:
    ch = grpc.insecure_channel(worker_addr)
    yield pb_grpc.InferenceStub(ch)
    ch.close()
