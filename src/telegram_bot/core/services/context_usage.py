"""Current-session context-window usage for Codex.

Reads the Codex rollout jsonl for the active session and reports how many
tokens the current context occupies, plus cumulative turn and session
totals. The max context window is resolved from an explicit override, then
from the Codex config (``model_context_window``), else left unset.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_CODEX_SESSION_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}"
    r"-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_CONTEXT_WINDOW_RE = re.compile(
    r"^\s*model_context_window\s*="
    r"\s*(\d+)\s*(?:#.*)?$"
)
_SOFT_CAP_BYTES = 128 * 1024


@dataclass(frozen=True)
class ContextUsage:
    session_id: str
    model: str | None
    context_tokens: int
    turn_total_tokens: int
    thread_total_tokens: int
    last_turn_output_tokens: int
    timestamp: str | None
    compaction_count: int


def _codex_home(home: Path) -> Path:
    env_home = os.environ.get("CODEX_HOME", "").strip()
    return Path(env_home) if env_home else home / ".codex"


def _sessions_root(home: Path) -> Path:
    return _codex_home(home) / "sessions"


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _iter_tail(path: Path) -> Iterator[dict[str, object]]:
    """Yield parsed JSONL rows newest-first from the transcript tail."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > _SOFT_CAP_BYTES:
                handle.seek(size - _SOFT_CAP_BYTES)
                handle.readline()
            raw = handle.read()
    except OSError:
        return
    for line in reversed(raw.splitlines()):
        if not line.strip():
            continue
        try:
            data = json.loads(line.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


def _find_latest_model(path: Path) -> str | None:
    """Full-file fallback: newest turn_context model (tail may miss it)."""
    model: str | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"turn_context"' not in line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                payload = data.get("payload")
                if isinstance(payload, dict) and isinstance(payload.get("model"), str):
                    model = payload["model"]
    except OSError:
        pass
    return model


_COMPACT_KIND = "compaction.summary"


def _count_compactions(path: Path) -> int:
    """Count compaction summaries persisted in the rollout file."""
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if _COMPACT_KIND not in line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = data.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") != "message":
                    continue
                meta = payload.get("internal_chat_message_metadata_passthrough")
                if not isinstance(meta, dict):
                    continue
                kinds = meta.get("content_item_kinds")
                if isinstance(kinds, list) and _COMPACT_KIND in kinds:
                    count += 1
    except OSError:
        pass
    return count


def _find_rollout(session_id: str, home: Path) -> Path | None:
    """Locate the newest rollout jsonl whose name embeds the session id."""
    root = _sessions_root(home)
    if not root.exists():
        return None
    matches = list(root.glob(f"**/*{session_id}*.jsonl"))
    if not matches:
        return None
    return max(matches, key=_safe_mtime)


def get_codex_context_usage(session_id: str, home: Path | None = None) -> ContextUsage | None:
    """Return the current context-window usage for a Codex session, or None."""
    if not _CODEX_SESSION_ID_RE.fullmatch(session_id or ""):
        return None
    home = home or Path.home()
    path = _find_rollout(session_id, home)
    if path is None:
        return None

    model: str | None = None
    last_record: dict[str, object] | None = None
    for data in _iter_tail(path):
        record_type = data.get("type")
        if last_record is None and record_type == "token_usage_record":
            payload = data.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
                last_record = data
        if model is None and record_type == "turn_context":
            payload = data.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("model"), str):
                model = payload["model"]
        if last_record is not None and model is not None:
            break
    if last_record is None:
        return None
    if model is None:
        model = _find_latest_model(path)

    payload = last_record.get("payload")
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    turn = payload.get("turn_token_usage")
    thread = payload.get("thread_token_usage")
    turn = turn if isinstance(turn, dict) else {}
    thread = thread if isinstance(thread, dict) else {}
    raw_ts = last_record.get("timestamp")
    timestamp = raw_ts if isinstance(raw_ts, str) else None
    return ContextUsage(
        session_id=session_id,
        model=model,
        context_tokens=int(usage.get("input_tokens", 0)),
        turn_total_tokens=int(turn.get("total_tokens", 0)),
        thread_total_tokens=int(thread.get("total_tokens", 0)),
        last_turn_output_tokens=int(usage.get("output_tokens", 0)),
        timestamp=timestamp,
        compaction_count=_count_compactions(path),
    )


def resolve_max_context_tokens(override: int | None = None, home: Path | None = None) -> int | None:
    """Resolve the model max context window: override, then Codex config."""
    if override is not None and override > 0:
        return override
    home = home or Path.home()
    config = _codex_home(home) / "config.toml"
    try:
        text = config.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _CONTEXT_WINDOW_RE.match(line)
        if match:
            return int(match.group(1))
    return None


def format_usage(usage: ContextUsage, max_tokens: int | None) -> str:
    """Render the usage report as Telegram HTML."""
    from telegram_bot.core.messages import t

    lines = [t("ui.usage_header")]
    if usage.model:
        lines.append(t("ui.usage_model", model=usage.model))
    if max_tokens and max_tokens > 0:
        pct = usage.context_tokens * 100.0 / max_tokens
        key = (
            "ui.usage_context_over" if usage.context_tokens > max_tokens else "ui.usage_context_pct"
        )
        lines.append(
            t(
                key,
                used=f"{usage.context_tokens:,}",
                max=f"{max_tokens:,}",
                pct=f"{pct:.0f}%",
            )
        )
    else:
        lines.append(t("ui.usage_context_raw", used=f"{usage.context_tokens:,}"))
    lines.append(
        t(
            "ui.usage_totals",
            out=f"{usage.last_turn_output_tokens:,}",
            turn=f"{usage.turn_total_tokens:,}",
            session=f"{usage.thread_total_tokens:,}",
        )
    )
    lines.append(t("ui.usage_compactions", count=f"{usage.compaction_count}"))
    return "\n".join(lines)
