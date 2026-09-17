"""Current-session context-window usage for Codex and Claude Code.

Reads the transcript for the active session (Codex rollout jsonl or
Claude Code project jsonl) and reports how many tokens the current
context occupies, plus cumulative turn and session totals. The max
context window is resolved from an explicit override, then from the
Codex config (``model_context_window``) or a per-model default.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from telegram_bot.core.tui.paths import _SESSION_ID_RE, transcript_path

_CODEX_SESSION_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}"
    r"-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_CLAUDE_DEFAULT_WINDOW = 200_000
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
                if '"type":"compacted"' in line or '"type": "compacted"' in line:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict) and data.get("type") == "compacted":
                        count += 1
                    continue
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


_COMPACT_TURN_WINDOW_FLOOR = 49_152  # 48 KiB tokens


def resolve_compact_turn_window(override: int | None = None) -> int:
    """Window for the /compact turn: env override, floored at 48K."""
    if override is not None and override > 0:
        return max(override, _COMPACT_TURN_WINDOW_FLOOR)
    return _COMPACT_TURN_WINDOW_FLOOR


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


def _find_claude_transcript(
    session_id: str, cwd: str | None, home: Path
) -> Path | None:
    """Locate the CC transcript jsonl for a session id (newest first)."""
    if not _SESSION_ID_RE.fullmatch(session_id or ""):
        return None
    if cwd:
        path = transcript_path(cwd, session_id, home=home)
        if path.is_file():
            return path
    root = home / ".claude" / "projects"
    if not root.exists():
        return None
    matches = list(root.glob(f"*/{session_id}.jsonl"))
    if not matches:
        return None
    return max(matches, key=_safe_mtime)


def _claude_usage_of(data: dict[str, object]) -> tuple[int, int, str | None] | None:
    """(context_tokens, turn_total_tokens, model) from an assistant record."""
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    context = int(usage.get("input_tokens", 0))
    context += int(usage.get("cache_read_input_tokens", 0))
    context += int(usage.get("cache_creation_input_tokens", 0))
    output = int(usage.get("output_tokens", 0))
    model = message.get("model")
    model = model if isinstance(model, str) else None
    return context, context + output, model


def _count_claude_compactions(path: Path) -> int:
    """Count compact-summary entries persisted in the transcript."""
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"summary"' not in line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("type") == "summary":
                    count += 1
    except OSError:
        pass
    return count


def _sum_claude_thread_tokens(path: Path) -> int:
    """Cumulative assistant tokens across the whole transcript."""
    total = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"assistant"' not in line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = _claude_usage_of(data)
                if parsed is not None:
                    total += parsed[1]
    except OSError:
        pass
    return total


def get_claude_context_usage(
    session_id: str, cwd: str | None = None, home: Path | None = None
) -> ContextUsage | None:
    """Return the current context-window usage for a Claude Code session."""
    home = home or Path.home()
    path = _find_claude_transcript(session_id, cwd, home)
    if path is None:
        return None
    last_record: dict[str, object] | None = None
    last_usage: tuple[int, int, str | None] | None = None
    for data in _iter_tail(path):
        if data.get("type") != "assistant":
            continue
        parsed = _claude_usage_of(data)
        if parsed is not None:
            last_record, last_usage = data, parsed
        if last_record is not None:
            break
    if last_record is None or last_usage is None:
        return None
    context, turn_total, model = last_usage
    raw_ts = last_record.get("timestamp")
    timestamp = raw_ts if isinstance(raw_ts, str) else None
    return ContextUsage(
        session_id=session_id,
        model=model,
        context_tokens=context,
        turn_total_tokens=turn_total,
        thread_total_tokens=_sum_claude_thread_tokens(path),
        last_turn_output_tokens=turn_total - context,
        timestamp=timestamp,
        compaction_count=_count_claude_compactions(path),
    )


def resolve_claude_max_context_tokens(
    override: int | None = None, model: str | None = None
) -> int | None:
    """Resolve the Claude max context: override, then per-model default."""
    if override is not None and override > 0:
        return override
    if not model:
        return None
    lowered = model.lower()
    if any(key in lowered for key in ("sonnet", "opus", "haiku")):
        return _CLAUDE_DEFAULT_WINDOW
    return None
