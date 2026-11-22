"""Request object tracked by the engine and scheduler."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Optional


@dataclasses.dataclass
class Request:
    req_id: int
    prompt: str
    max_tokens: int
    token_queue: asyncio.Queue
    submitted_at: float = None
    started_at: float = None
    slot: int = -1
    past_len: int = 0
    next_token_id: Optional[int] = None
    generated_ids: list = dataclasses.field(default_factory=list)
    finished: bool = False
    # incremental detokenizer state
    _prefix_offset: int = 0
    _read_offset: int = 0
