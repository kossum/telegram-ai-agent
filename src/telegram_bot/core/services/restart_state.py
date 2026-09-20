"""Persistent state for in-place bot restarts (os.execv, same PID).

The bot can't tell the user whether its own re-exec succeeded — it only
finds out once the *new* process is up and can talk again. So the old
process leaves a marker (kept in the bind-mounted bot-data dir, so it
survives both in-place and full container restarts) and the new process
reports back. The marker also counts bounces, so a startup crash-loop
becomes visible instead of silently looping until Docker gives up.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# A marker older than this is from a long-dead restart; drop it silently.
MAX_WINDOW_S = 180.0
# If the new process is still alive this long after coming up, the restart
# is stable enough to confirm and the marker can be cleared.
SURVIVE_S = 5.0


@dataclass
class RestartState:
    chat_ids: list[dict[str, object]] = field(default_factory=list)
    requested_at: float = 0.0
    attempts: int = 0


def path_for(mapping_path: Path) -> Path:
    """Marker path, kept next to the (persistent) session-mapping file."""
    return mapping_path.with_name("restart_state.json")


def load(path: Path) -> RestartState | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    chats: list[dict[str, object]] = []
    for item in raw.get("chat_ids", []):
        if isinstance(item, dict) and "chat_id" in item:
            chats.append(
                {
                    "chat_id": int(item["chat_id"]),
                    "thread_id": item.get("thread_id"),
                }
            )
    return RestartState(
        chat_ids=chats,
        requested_at=float(raw.get("requested_at", 0.0)),
        attempts=int(raw.get("attempts", 0)),
    )


def save(path: Path, state: RestartState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        {
            "chat_ids": state.chat_ids,
            "requested_at": state.requested_at,
            "attempts": state.attempts,
        }
    ).encode()
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except OSError:
        pass  # best-effort: a lost marker just means no post-restart note


def clear(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def is_stale(state: RestartState, now: float | None = None) -> bool:
    return (now if now is not None else time.time()) - state.requested_at > MAX_WINDOW_S


def record_request(
    state: RestartState | None,
    chat_id: int,
    thread_id: int | None,
    now: float,
) -> RestartState:
    """Start a fresh restart request: reset bounces, add this chat."""
    base = state if state is not None else RestartState()
    chats = list(base.chat_ids)
    entry: dict[str, object] = {"chat_id": chat_id, "thread_id": thread_id}
    if entry not in chats:
        chats.append(entry)
    return RestartState(chat_ids=chats, requested_at=now, attempts=0)
