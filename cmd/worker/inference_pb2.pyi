from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FinishReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FINISH_REASON_UNSPECIFIED: _ClassVar[FinishReason]
    FINISH_REASON_EOS: _ClassVar[FinishReason]
    FINSIH_REASON_LENGTH: _ClassVar[FinishReason]
    FINISH_REASON_CANCELLED: _ClassVar[FinishReason]
    FINISH_REASON_ERROR: _ClassVar[FinishReason]
FINISH_REASON_UNSPECIFIED: FinishReason
FINISH_REASON_EOS: FinishReason
FINSIH_REASON_LENGTH: FinishReason
FINISH_REASON_CANCELLED: FinishReason
FINISH_REASON_ERROR: FinishReason

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

class Admit(_message.Message):
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

class Cancel(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class EngineRequest(_message.Message):
    __slots__ = ("admit", "cancel")
    ADMIT_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    admit: Admit
    cancel: Cancel
    def __init__(self, admit: _Optional[_Union[Admit, _Mapping]] = ..., cancel: _Optional[_Union[Cancel, _Mapping]] = ...) -> None: ...

class Admitted(_message.Message):
    __slots__ = ("request_id", "slot_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SLOT_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    slot_id: int
    def __init__(self, request_id: _Optional[str] = ..., slot_id: _Optional[int] = ...) -> None: ...

class Token(_message.Message):
    __slots__ = ("request_id", "slot_id", "text", "token_id", "index")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SLOT_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    slot_id: int
    text: str
    token_id: int
    index: int
    def __init__(self, request_id: _Optional[str] = ..., slot_id: _Optional[int] = ..., text: _Optional[str] = ..., token_id: _Optional[int] = ..., index: _Optional[int] = ...) -> None: ...

class Finished(_message.Message):
    __slots__ = ("request_id", "slot_id", "reason", "tokens_generated")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SLOT_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    TOKENS_GENERATED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    slot_id: int
    reason: str
    tokens_generated: int
    def __init__(self, request_id: _Optional[str] = ..., slot_id: _Optional[int] = ..., reason: _Optional[str] = ..., tokens_generated: _Optional[int] = ...) -> None: ...

class Rejected(_message.Message):
    __slots__ = ("request_id", "reason")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    reason: str
    def __init__(self, request_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class StepStats(_message.Message):
    __slots__ = ("step", "active_slots", "free_slots", "waiting", "batch_tokens", "prefill_tokens", "step_time_us", "kv_used")
    STEP_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SLOTS_FIELD_NUMBER: _ClassVar[int]
    FREE_SLOTS_FIELD_NUMBER: _ClassVar[int]
    WAITING_FIELD_NUMBER: _ClassVar[int]
    BATCH_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PREFILL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    STEP_TIME_US_FIELD_NUMBER: _ClassVar[int]
    KV_USED_FIELD_NUMBER: _ClassVar[int]
    step: int
    active_slots: int
    free_slots: int
    waiting: int
    batch_tokens: int
    prefill_tokens: int
    step_time_us: int
    kv_used: int
    def __init__(self, step: _Optional[int] = ..., active_slots: _Optional[int] = ..., free_slots: _Optional[int] = ..., waiting: _Optional[int] = ..., batch_tokens: _Optional[int] = ..., prefill_tokens: _Optional[int] = ..., step_time_us: _Optional[int] = ..., kv_used: _Optional[int] = ...) -> None: ...

class EngineEvent(_message.Message):
    __slots__ = ("admitted", "token", "finished", "stats", "rejected")
    ADMITTED_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    FINISHED_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    REJECTED_FIELD_NUMBER: _ClassVar[int]
    admitted: Admitted
    token: Token
    finished: Finished
    stats: StepStats
    rejected: Rejected
    def __init__(self, admitted: _Optional[_Union[Admitted, _Mapping]] = ..., token: _Optional[_Union[Token, _Mapping]] = ..., finished: _Optional[_Union[Finished, _Mapping]] = ..., stats: _Optional[_Union[StepStats, _Mapping]] = ..., rejected: _Optional[_Union[Rejected, _Mapping]] = ...) -> None: ...
