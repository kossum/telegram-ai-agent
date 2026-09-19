"""Codex rollout helpers for /resend.

A codex session's history is an append-only ``.jsonl`` rollout, one JSON object
per line. Resending the last user message means: locate the rollout, find the
last ``role=user`` message, drop it (and the ``turn_context`` that opens its
turn), then replay the message as a fresh prompt. The turn_context that
immediately precedes a user message is the delimiter that marks where that
turn begins, so cutting at it is a clean, format-safe boundary.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from aiogram.exceptions import TelegramAPIError
from aiogram.methods.base import TelegramMethod
from aiogram.types import ChatIdUnion, Message

from telegram_bot.core.services.providers import _codex_sessions_root


class _GetMessage(TelegramMethod[Message]):
    """Raw Bot API ``getMessage`` — this aiogram build has no ``Bot.get_message``."""

    __returning__ = Message
    __api_method__ = "getMessage"

    chat_id: ChatIdUnion
    message_id: int


@dataclass(frozen=True)
class ReplayPoint:
    """A resend point: the text to replay and the line index to cut at."""

    text: str
    cut_index: int  # keep lines[0:cut_index]; the last user msg (+ its turn_context) drops


def find_rollout_path(session_id: str) -> Path | None:
    """Locate the on-disk codex rollout for a session, or None when not found.

    Matches on the ``session_meta`` id in the first line. Session ids are
    UUIDv7, so a single match is expected; zero or multiple fail closed to None.
    """
    root = _codex_sessions_root()
    matches: list[Path] = []
    for path in root.glob(f"**/*{session_id}*.jsonl"):
        try:
            first = path.read_text(errors="replace").splitlines()[0]
        except (OSError, IndexError):
            continue
        data = _line_json(first)
        if not data or data.get("type") != "session_meta":
            continue
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        if payload.get("id") == session_id:
            matches.append(path.resolve())
    return matches[0] if len(matches) == 1 else None


def _line_json(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _user_message_text(payload: dict) -> str | None:
    """Text of a ``role=user`` message payload, or None when not a user text message.

    Codex stores user content as a list of typed parts; only ``input_text``
    parts carry real text. Non-text content (images, etc.) yields None.
    """
    if payload.get("type") != "message" or payload.get("role") != "user":
        return None
    content = payload.get("content")
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "input_text"
        ]
        text = "".join(parts)
        return text if text.strip() else None
    return None


def _is_turn_context(line: str) -> bool:
    obj = _line_json(line)
    if obj is None:
        return False
    return obj.get("type") == "turn_context"


def locate_replay_point(path: Path) -> ReplayPoint | None:
    """Return the last user message and the cut index that drops it, or None.

    ``None`` when the file is missing, unreadable, or holds no user text message.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None

    last_user: tuple[int, str] | None = None
    for index in range(len(lines) - 1, -1, -1):
        obj = _line_json(lines[index])
        if obj is None:
            continue
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
        text = _user_message_text(payload)
        if text is not None:
            last_user = (index, text)
            break
    if last_user is None:
        return None

    index, text = last_user
    cut = index
    if index > 0 and _is_turn_context(lines[index - 1]):
        cut = index - 1
    return ReplayPoint(text=text, cut_index=cut)


def truncate_to(path: Path, cut_index: int) -> bool:
    """Atomically rewrite the rollout, keeping ``lines[0:cut_index]``.

    Uses a same-directory temp file + ``os.replace`` so a crash mid-write never
    leaves a truncated rollout. Returns False on I/O failure.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return False
    kept = lines[: max(0, cut_index)]
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            for line in kept:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return False
    return True


async def fetch_message_text(
    bot, chat_id: int, message_id: int | None
) -> tuple[Message | None, str | None]:
    """Re-fetch a user message's current (possibly edited) text via the Bot API.

    Returns ``(message, text)``. ``text`` is None when the message is deleted,
    unreachable, or has no text (e.g. a voice note). Callers fall back to the
    text stored in the rollout when ``text`` is None. ``message`` is the fetched
    object (or None) so the caller can reply-to-resume against the original.
    """
    if message_id is None:
        return None, None
    try:
        message = await bot(_GetMessage(chat_id=chat_id, message_id=message_id))
    except TelegramAPIError:
        return None, None
    except Exception:
        return None, None
    if message is None:
        return None, None
    text = (message.text or message.caption or "").strip()
    return message, (text if text else None)
