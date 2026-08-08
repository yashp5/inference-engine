from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "prompt", "max_tokens", "temperature")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    def __init__(self, request_id: _Optional[str] = ..., prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("request_id", "generated_text", "tokens_generated", "inference_time_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKENS_GENERATED_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    generated_text: str
    tokens_generated: int
    inference_time_ms: int
    def __init__(self, request_id: _Optional[str] = ..., generated_text: _Optional[str] = ..., tokens_generated: _Optional[int] = ..., inference_time_ms: _Optional[int] = ...) -> None: ...

class GenerateStreamResponse(_message.Message):
    __slots__ = ("request_id", "token", "finished", "tokens_generated")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    FINISHED_FIELD_NUMBER: _ClassVar[int]
    TOKENS_GENERATED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    token: str
    finished: bool
    tokens_generated: int
    def __init__(self, request_id: _Optional[str] = ..., token: _Optional[str] = ..., finished: bool = ..., tokens_generated: _Optional[int] = ...) -> None: ...
