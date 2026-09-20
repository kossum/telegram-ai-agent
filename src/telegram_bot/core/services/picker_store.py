"""Short-lived in-memory store for Telegram /resume picker callbacks."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from telegram_bot.core.services.resume_listing import SessionEntry
from telegram_bot.core.services.topic_config import Engine


@dataclass(frozen=True)
class PickerState:
    chat_id: int
    thread_id: int | None
    cwd: Path
    engine: Engine
    entries: tuple[SessionEntry, ...]
    created_at: float


class PickerStore:
    """In-memory picker state. TTL is enforced lazily on access."""

    def __init__(self, *, ttl_sec: float = 300.0, clock: object = time.time) -> None:
        self._ttl_sec = ttl_sec
        self._clock = clock
        self._states: dict[str, PickerState] = {}

    def put(self, state: PickerState) -> str:
        while True:
            token = secrets.token_hex(4)
            if token not in self._states:
                self._states[token] = state
                return token

    def get(self, token: str) -> PickerState | None:
        state = self._states.get(token)
        if state is None:
            return None
        now = self._clock()  # type: ignore[operator]
        if now - state.created_at > self._ttl_sec:
            self._states.pop(token, None)
            return None
        return state

    def drop(self, token: str) -> None:
        self._states.pop(token, None)

    def latest_for(
        self, chat_id: int, thread_id: int | None
    ) -> PickerState | None:
        """Most recent live state for one channel, or None.

        /resume N refers to the channel's latest /resume list, so no
        token is needed. Expired states are dropped lazily, like get().
        """
        now = self._clock()  # type: ignore[operator]
        best_token: str | None = None
        best_at = 0.0
        for token, state in list(self._states.items()):
            if (state.chat_id, state.thread_id) != (chat_id, thread_id):
                continue
            if now - state.created_at > self._ttl_sec:
                self._states.pop(token, None)
                continue
            if state.created_at >= best_at:
                best_at = state.created_at
                best_token = token
        if best_token is None:
            return None
        return self._states.get(best_token)
