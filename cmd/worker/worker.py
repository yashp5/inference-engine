"""
Keep it minimal — a single Python file that does three things:
    loads the model on startup, implements the gRPC Generate method, and serves on a Unix domain socket (or a TCP port, your call).
For the model, use llama-cpp-python with a small GGUF model if you don't have a GPU.
TinyLlama 1.1B Q4 quantized is around 600MB and runs fine on CPU.
If you want even lighter for iteration speed, GPT-2 via transformers works too but is less representative of real LLM inference.
"""

import argparse
from concurrent import futures
import logging
import os
import queue
import signal
import threading
import time
from types import FrameType
from typing import Iterator, Optional, cast

from grpc_reflection.v1alpha import reflection
from engine import Engine as Eng
import inference_pb2
import inference_pb2_grpc
from llama_cpp import Llama
from llama_cpp.llama_types import (
    CompletionUsage,
    CreateCompletionResponse,
    CreateCompletionStreamResponse,
)

import grpc

# ------ Logging ---------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("worker")

# --- Config (overrride via env vars) ------------------------


PORT: str = os.getenv("WORKER_PORT", "50051")
MAX_WORKERS: int = int(os.getenv("WORKER_MAX_THREADS", "4"))
MODEL_PATH: str = os.getenv("MODEL_PATH", "/Users/yash/build/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")


def _validate_request(
    request: inference_pb2.GenerateRequest, context: grpc.ServicerContext
) -> bool:
    """Validation for the unary and server-streaming RPCs only.

    The Engine RPC deliberately does not use this: one malformed Admit must not
    abort a stream that is multiplexing other live requests, so the engine
    validates inline and answers with a Rejected event instead.
    """
    if not request.request_id:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        return False
    if not request.prompt:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompt is required")
        return False
    if request.max_tokens <= 0:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "max_tokens must be > 0")
        return False
    if not (0.0 <= request.temperature <= 2.0):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "temperature must be between 0.0 and 2.0")
        return False
    return True

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, model_path: str, llm_n_ctx: int, engine_slots: int, engine_per_seq_ctx: int) -> None:
        super().__init__()
        self._model_path = model_path
        self._engine_slots = engine_slots
        self._engine_per_seq_ctx = engine_per_seq_ctx
        self._engine: Optional[Eng] = None
        self._engine_init_lock = threading.Lock() # guards construction
        self._engine_busy = threading.Lock() # held for a streams life
        # Llama is not thread safe and the gRPC pool runs RPCs concurrently, so
        # Generate/GenerateStream serialize on this for the whole generation
        self._llm_lock = threading.Lock()
        self.llm: Llama = Llama(
            model_path=model_path,
            n_ctx=llm_n_ctx,
        )

    def _get_engine(self) -> Eng:
        with self._engine_init_lock:
            if self._engine is None:
                log.info("initializing continous batching engine")
                self._engine = Eng(
                    model_path=self._model_path,
                    n_slots=self._engine_slots,
                    per_seq_ctx=self._engine_per_seq_ctx
                )
            return self._engine

    # Unary RPC - waits for the full completion and returns response
    def Generate(self, request: inference_pb2.GenerateRequest, context: grpc.ServicerContext):
        log.info("Generate called request_id=%s prompt_len=%d max_tokens=%d temperature=%.2f",
            request.request_id,
            len(request.prompt),
            request.max_tokens,
            request.temperature,
        )

        if not _validate_request(request, context):
            return inference_pb2.GenerateResponse()

        with self._llm_lock:
            start_ms: float = time.monotonic()
            output = cast(
                CreateCompletionResponse,
                self.llm.create_completion(prompt=request.prompt, max_tokens=request.max_tokens, temperature=request.temperature, stream=False),
            )
            inference_time_ms: int = int((time.monotonic()-start_ms) * 1000)

        generated_text: str = output["choices"][0]["text"]
        usage: CompletionUsage = output.get("usage") or CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        tokens_generated: int = usage["completion_tokens"]

        return inference_pb2.GenerateResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            inference_time_ms=inference_time_ms
        )

    # Server-Streaming RPC - yields one token at a time
    def GenerateStream(self, request: inference_pb2.GenerateRequest, context: grpc.ServicerContext
        ) -> Iterator[inference_pb2.GenerateStreamResponse]:
        log.info("GenerateStream called request_id=%s prompt_len=%d max_tokens=%d temperature=%.2f",
            request.request_id,
            len(request.prompt),
            request.max_tokens,
            request.temperature)

        if not _validate_request(request, context):
            return

        tokens_generated: int = 0
        # Held across the yields: create_completion(stream=True) is lazy, so the
        # model is in use until the last chunk. A client cancel closes this
        # generator, and GeneratorExit unwinds the with and releases the lock.
        with self._llm_lock:
            stream = cast(
                Iterator[CreateCompletionStreamResponse],
                self.llm.create_completion(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=True,
                ),
            )
            for chunk in stream:
                choice = chunk["choices"][0]
                token_text: str = choice["text"]
                finished: bool = choice.get("finish_reason") is not None
                tokens_generated += 1

                yield inference_pb2.GenerateStreamResponse(
                    request_id=request.request_id,
                    token=token_text,
                    finished=finished,
                    tokens_generated=tokens_generated,
                )

                if finished:
                    break

        log.info(
            "GenerateStream done request_id=%s tokens=%d",
            request.request_id,
            tokens_generated,
        )

    def Engine(
            self, request_iterator: Iterator[inference_pb2.EngineRequest], context: grpc.ServicerContext
        ) -> Iterator[inference_pb2.EngineEvent]:
            # One llama_context, not thread safe. Exactly one stream at a time.
            if not self._engine_busy.acquire(blocking=False):
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "an Engine stream is already active; this worker serves one"
                )
                return

            inbox: queue.Queue = queue.Queue()

            def reader() -> None:
                # Decoupled from the step loop: run() cannot block on request
                # iterator and call llama_decode at same time
                try:
                    for req in request_iterator:
                        inbox.put(req)
                except Exception as e:
                    log.warning("engine reader stopped: %s", e)
                finally:
                    inbox.put(None)

            threading.Thread(target=reader, name="engine-reader", daemon=True).start()

            try:
                engine = self._get_engine()
            except BaseException:
                self._engine_busy.release()   # or the RPC is permanently wedged
                raise

            try:
                yield from engine.run(inbox)
            finally:
                engine.reset()
                self._engine_busy.release()

    def close(self) -> None:
            """Free the engine deterministically. The llama context must be released
            before the model, and interpreter-shutdown GC order does not guarantee
            that."""
            with self._engine_init_lock:
                if self._engine is not None:
                    self._engine.close()
                    self._engine = None


def serve() -> None:
    parser = argparse.ArgumentParser(description="Inference worker")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to GGUF model")
    parser.add_argument("--port", default=PORT, help="Port to listen on")
    parser.add_argument("--llm-n-ctx",type=int, default=512, help="context window for the unary/streaming Llama instance only")
    parser.add_argument("--n-slots", type=int, default=MAX_WORKERS, help="Concurrent request slots")
    parser.add_argument("--max-threads", type=int, default=MAX_WORKERS, help="gRPC thread pool size; must exceed the number of concurrent long-lived streams")
    parser.add_argument("--engine-slots", type=int, default=8, help="continous-batching slots (KV sequences)")
    parser.add_argument("--engine-per-seq-ctx", type=int, default=512, help="per-sequence context; engine n_ctx = slots * this")
    args = parser.parse_args()

    max_threads: int = max(args.max_threads, 2)
    server: grpc.Server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_threads))

    servicer = InferenceServicer(
        model_path=args.model_path,
        llm_n_ctx=args.llm_n_ctx,
        engine_slots=args.engine_slots,
        engine_per_seq_ctx=args.engine_per_seq_ctx,
    )
    inference_pb2_grpc.add_InferenceServicer_to_server(servicer, server)

    SERVICE_NAMES: tuple[str, ...] = (
        inference_pb2.DESCRIPTOR.services_by_name["Inference"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    log.info("worker listening on port %s (threads=%d engine_slots=%d per_seq_ctx=%d engine_n_ctx=%d)",
                 args.port, max_threads, args.engine_slots, args.engine_per_seq_ctx,
                 args.engine_slots * args.engine_per_seq_ctx)

    # --- Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum: int, frame: Optional[FrameType]) -> None:
        log.info("Shutdown signal received, stopping server...")
        server.stop(grace=5).wait()
        servicer.close()
        log.info("Server stopped.")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()

if __name__ == "__main__":
    serve()
