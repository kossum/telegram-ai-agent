"""Command handlers for bot-owned slash commands except /tui and /tail."""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message, ReplyKeyboardRemove
from aiogram.types.inaccessible_message import InaccessibleMessage

from telegram_bot.core.config import Settings
from telegram_bot.core.handlers.forward import ForwardBatcher
from telegram_bot.core.handlers.streaming import (
    ensure_exec_mode_ready,
    send_to_tmux_if_active,
)
from telegram_bot.core.keyboards import (
    RESUME_PAGE_SIZE,
    _format_age,
    _format_size,
    engine_keyboard,
    exec_mode_keyboard,
    resume_keyboard,
    stream_mode_keyboard,
    topic_keyboard,
)
from telegram_bot.core.messages import reset_lang_cache, t
from telegram_bot.core.services import restart_state
from telegram_bot.core.services.claude import SessionData, SessionManager
from telegram_bot.core.services.codex_rollout import (
    fetch_message_text,
    find_rollout_path,
    locate_replay_point,
    truncate_to,
)
from telegram_bot.core.services.codex_update import CodexUpdateResult, CodexUpdateService
from telegram_bot.core.services.context_usage import (
    ContextUsage,
    format_usage,
    get_claude_context_usage,
    get_codex_context_usage,
    resolve_claude_max_context_tokens,
    resolve_compact_turn_window,
    resolve_max_context_tokens,
)
from telegram_bot.core.services.message_queue import MessageQueue
from telegram_bot.core.services.picker_store import PickerState, PickerStore
from telegram_bot.core.services.process_cleanup import (
    RuntimeDiagnostics,
    tagged_processes,
)
from telegram_bot.core.services.providers import (
    choose_available_engine,
    engine_display_name,
)
from telegram_bot.core.services.resume_listing import (
    SessionEntry,
    _same_cwd,
    get_last_assistant_message,
    list_recent,
    list_sessions,
)
from telegram_bot.core.services.telegram_utils import send_html_with_fallback
from telegram_bot.core.services.tmux_manager import TmuxManager
from telegram_bot.core.services.topic_config import (
    _VALID_ENGINES,
    _VALID_EXEC_MODES,
    _VALID_STREAM_MODES,
    Engine,
    TopicConfig,
)
from telegram_bot.core.services.topic_runtime import (
    BotDefaults,
    TopicRuntimeConfig,
    resolve_topic_runtime_config,
)
from telegram_bot.core.types import ChannelKey, channel_key
from telegram_bot.core.utils.telegram_html import split_html_message

logger = logging.getLogger(__name__)


def _exec_mode_label(mode: str) -> str:
    """Human-facing label per exec_mode.

    Raw "subprocess" must never leak into the "Mode: …" toast — the picker
    button text is the contract surface.
    """
    if mode == "subprocess":
        return t("ui.exec_mode_label_subprocess")
    if mode == "tmux":
        return t("ui.exec_mode_label_tmux")
    return mode


def _exec_mode_picker_caption(mode: str) -> str:
    return t("ui.exec_mode_picker_caption", current=_exec_mode_label(mode))


router = Router(name="commands")


def _format_codex_update_time(value: float | None) -> str:
    if value is None:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(value))


def _format_codex_update_output(output: str) -> str:
    return html.escape(output or "No output")


def _codex_update_active_check(
    tmux_manager: TmuxManager,
    session_manager: SessionManager,
) -> bool:
    return tmux_manager.has_live_provider("codex") or session_manager.has_active_provider_process(
        "codex"
    )


def _codex_update_result_message(result: CodexUpdateResult) -> str:
    if result.status == "success":
        return t("ui.codex_update_success", output=_format_codex_update_output(result.output))
    if result.status == "already_running":
        return t("ui.codex_update_already_running")
    if result.status == "blocked_active_sessions":
        return t("ui.codex_update_active_sessions")
    if result.status == "skipped_cooldown":
        return t("ui.codex_update_cooldown")
    return t(
        "ui.codex_update_failed",
        status=result.status,
        output=_format_codex_update_output(result.output),
    )


def _resume_caption(
    cwd: Path,
    *,
    page: int,
    total_pages: int,
    entries: tuple[SessionEntry, ...] = (),
    current_session_id: str | None = None,
) -> str:
    safe_cwd = html.escape(str(cwd))
    text = t("ui.resume_picker_caption_hdr", cwd=safe_cwd, page=page + 1, total=total_pages)
    if not entries:
        return text

    blocks: list[str] = []
    start = page * RESUME_PAGE_SIZE
    for idx, entry in enumerate(entries[start : start + RESUME_PAGE_SIZE], start=start):
        provider = engine_display_name(entry.provider)
        preview = html.escape(entry.preview)
        prefix = "✅ " if entry.session_id == current_session_id else ""
        parts = [
            f"{prefix}{idx + 1}. <b>{provider}</b>",
            _format_age(entry.mtime),
            _format_size(entry.size_bytes),
            f"<code>{html.escape(entry.session_id[:8])}</code>",
        ]
        if entry.session_id == current_session_id:
            parts.append(t("ui.resume_current_marker"))
        block_lines = [" · ".join(parts)]
        if preview:
            block_lines.append(f"   {preview}")
        blocks.append("\n".join(block_lines))
    return "\n\n".join([text, *blocks])


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    logger.debug("User %s started the bot", message.from_user and message.from_user.id)
    is_group = message.chat.type == ChatType.SUPERGROUP
    keyboard = topic_keyboard() if is_group else ReplyKeyboardRemove()
    await message.answer(
        text=t("ui.start_welcome"),
        reply_markup=keyboard,
    )


def _descendant_pids(self_pid: int) -> set[int]:
    """All descendant PIDs of self_pid via a /proc walk (no psutil)."""
    children: dict[int, list[int]] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text()
        except OSError:
            continue
        rest = stat.split(")", 1)
        if len(rest) != 2:
            continue
        fields = rest[1].split()
        # after ')' : fields[0]=state, fields[1]=ppid
        children.setdefault(int(fields[1]), []).append(int(proc.name))
    seen: set[int] = set()
    stack = list(children.get(self_pid, ()))
    while stack:
        pid = stack.pop()
        if pid in seen or pid == self_pid:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, ()))
    return seen


@router.message(Command("restart"))
async def handle_restart(message: Message, session_manager: SessionManager) -> None:
    """Restart this bot process in place via os.execv (same PID)."""
    state_path = session_manager.restart_state_path
    state = restart_state.load(state_path)
    state = restart_state.record_request(
        state,
        message.chat.id,
        message.message_thread_id,
        time.time(),
    )
    restart_state.save(state_path, state)

    await message.answer(t("ui.restart_running"))
    await asyncio.sleep(0.5)  # let Telegram deliver the note
    for pid in _descendant_pids(os.getpid()):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
    await asyncio.sleep(0.3)
    try:
        os.execv(sys.executable, [sys.executable, sys.argv[0], *sys.argv[1:]])
    except OSError as exc:
        # re-exec failed: the old process is still up on the old code.
        cur = restart_state.load(state_path) or state
        cur.attempts = 1
        restart_state.save(state_path, cur)
        await message.answer(t("ui.restart_failed", err=str(exc)))


@router.message(Command("language"))
async def handle_language(message: Message) -> None:
    """Show or switch bot UI language for the current process."""
    text = message.text or ""
    parts = text.split(maxsplit=1)
    current = os.environ.get("BOT_LANG", "en")
    if current not in {"en", "ru"}:
        current = "en"

    if len(parts) == 1:
        await message.answer(t("ui.language_current", lang=current))
        return

    lang = parts[1].strip().lower()
    if lang not in {"en", "ru"}:
        await message.answer(t("ui.language_invalid"))
        return

    os.environ["BOT_LANG"] = lang
    reset_lang_cache()
    await message.answer(t("ui.language_changed", lang=lang))


@router.message(Command("codex_update"))
async def handle_codex_update(
    message: Message,
    codex_update_service: CodexUpdateService,
    tmux_manager: TmuxManager,
    session_manager: SessionManager,
) -> None:
    """Run or inspect the bot-managed Codex CLI updater."""
    text = message.text or ""
    parts = text.split(maxsplit=1)
    if len(parts) > 1 and parts[1].strip().lower() == "status":
        state = codex_update_service.status()
        output = f"<pre>{_format_codex_update_output(state.last_output)}</pre>"
        await message.answer(
            t(
                "ui.codex_update_status",
                status=state.last_status or "never",
                last_success=_format_codex_update_time(state.last_success_at),
                output=output,
            ),
            parse_mode="HTML",
        )
        return

    running = await message.answer(t("ui.codex_update_running"))
    result = await codex_update_service.run_manual(
        active_check=lambda: _codex_update_active_check(tmux_manager, session_manager)
    )
    response = _codex_update_result_message(result)
    with contextlib.suppress(TelegramBadRequest):
        await running.edit_text(response, parse_mode="HTML")
        return
    await message.answer(response, parse_mode="HTML")


async def _reset_channel(
    message: Message,
    key: ChannelKey,
    session_manager: SessionManager,
    message_queue: MessageQueue,
    forward_batcher: ForwardBatcher,
    tmux_manager: TmuxManager,
    topic_config: TopicConfig,
) -> None:
    """Unified reset path for /new, /clear, and the "Новый чат" reply button.

    Live tmux → clear_context respawns a fresh TUI immediately.
    Dormant tmux → drop stale state and start a fresh TUI immediately.
    Otherwise → full subprocess reset + ui.new_session.
    """
    settings = topic_config.get_topic(key[1])
    if tmux_manager.is_active(key):
        # clear_context respawns the tmux session; _spawn_tmux can fail
        # (tmux server shutdown race, readiness timeout, etc.). Without a
        # catch here the RuntimeError reaches aiogram's error middleware
        # and the user sees nothing — "Новый чат" becomes a silent button.
        try:
            reset_live = await tmux_manager.clear_context(key, session_manager)
        except RuntimeError:
            logger.warning("clear_context failed for %s", key, exc_info=True)
            await message.answer(t("ui.reset_failed"))
            return
        if reset_live:
            session = session_manager._get_session(key)
            await message.answer(
                t("ui.tmux_started_engine", engine=engine_display_name(session.engine))
            )
            return
        logger.info("clear_context found no live tmux for %s; starting fresh", key)

    if settings.exec_mode == "tmux":
        await tmux_manager.kill(key)
        forward_batcher.clear(key)
        await message_queue.clear(key)
        await session_manager.kill_session(key)
        session = session_manager._get_session(key)
        try:
            started = await tmux_manager.start_session(
                key,
                mode=session.mode,
                cwd=session.cwd,
                mcp_config=session.mcp_config,
                chat_id=session.chat_id,
                session_manager=session_manager,
                provider=session.engine,
                model=session.model,
            )
        except RuntimeError:
            logger.warning("fresh tmux start failed for %s", key, exc_info=True)
            await message.answer(t("ui.reset_failed"))
            return
        if started:
            await message.answer(
                t("ui.tmux_started_engine", engine=engine_display_name(session.engine))
            )
        return

    forward_batcher.clear(key)
    await message_queue.clear(key)
    await session_manager.kill_session(key)
    await message.answer(t("ui.new_session"))


@router.message(Command("new"))
async def handle_new(
    message: Message,
    session_manager: SessionManager,
    message_queue: MessageQueue,
    forward_batcher: ForwardBatcher,
    tmux_manager: TmuxManager,
    topic_config: TopicConfig,
) -> None:
    key = channel_key(message)
    logger.debug("User %s requested new session", message.from_user and message.from_user.id)
    await _reset_channel(
        message, key, session_manager, message_queue, forward_batcher, tmux_manager, topic_config
    )


@router.message(Command("clear"))
async def handle_clear(
    message: Message,
    session_manager: SessionManager,
    message_queue: MessageQueue,
    forward_batcher: ForwardBatcher,
    tmux_manager: TmuxManager,
    topic_config: TopicConfig,
) -> None:
    key = channel_key(message)
    logger.debug("User %s requested clear", message.from_user and message.from_user.id)
    await _reset_channel(
        message, key, session_manager, message_queue, forward_batcher, tmux_manager, topic_config
    )


@router.message(Command("cancel"))
async def handle_cancel_command(
    message: Message,
    session_manager: SessionManager,
    message_queue: MessageQueue,
    tmux_manager: TmuxManager,
) -> None:
    key = channel_key(message)
    tmux_acted = tmux_manager.is_active(key)
    if tmux_acted:
        await tmux_manager.cancel(key)
    cancelled = await message_queue.cancel(key)
    if cancelled or tmux_acted:
        logger.debug("User cancelled CC processing (command) for %s", key)
        await message.answer(t("ui.cancelled"))
    else:
        await message.answer(t("ui.nothing_to_cancel"))


@router.message(Command("resend"))
async def handle_resend(
    message: Message,
    session_manager: SessionManager,
    message_queue: MessageQueue,
    tmux_manager: TmuxManager,
) -> None:
    """Replay the last user message from scratch.

    Cancels the in-flight turn, cuts the codex rollout at the last user message
    (dropping it and everything after), then re-sends that message — preferring
    the message's current Telegram text so an edited message replays the edit.
    """
    key = channel_key(message)
    session = session_manager._get_session(key)
    if not session.session_id:
        await message.answer(t("ui.resend_no_session"))
        return
    if session.engine != "codex" or tmux_manager.is_active(key):
        await message.answer(t("ui.resend_not_codex"))
        return

    last_message_id = session.last_user_message_id
    # Prefer the message's current (possibly edited) Telegram text.
    if message.bot is not None:
        orig_message, fresh_text = await fetch_message_text(message.bot, key[0], last_message_id)
    else:
        orig_message, fresh_text = None, None

    # 1. Cancel the in-flight turn (kills the codex process, clears the queue).
    await message_queue.cancel(key)

    # 2. Under the channel lock: cut the rollout at the last user message and
    #    capture its stored text as the replay fallback.
    prompt_text: str | None = None
    async with message_queue.lock_for(key):
        rollout_path = find_rollout_path(session.session_id)
        replay_point = locate_replay_point(rollout_path) if rollout_path else None
        if replay_point is not None and rollout_path is not None:
            truncate_to(rollout_path, replay_point.cut_index)
        # Freshness order: live Bot-API text, then the newest text we saw
        # for that message (original delivery or a message_edit we cached
        # — covers chats where getMessage 404s), then the stored rollout
        # text.
        if fresh_text is not None:
            prompt_text = fresh_text
        elif session.last_user_message_text is not None:
            prompt_text = session.last_user_message_text
        elif replay_point is not None:
            prompt_text = replay_point.text
        else:
            prompt_text = None

    if prompt_text is None:
        await message.answer(t("ui.resend_not_found"))
        return

    # 3. Replay. The source is the original message when still fetchable (so
    #    reply-to-resume threads to it); otherwise the /resend command itself.
    source = orig_message if orig_message is not None else message
    message_queue.enqueue(
        key,
        prompt_text,
        source.message_id,
        source,
        target_session_id=session.session_id,
        # start_new_session=True here only stops the enqueue from batch-merging
        # this replay into a concurrent same-session item; the target_session_id
        # above is what keeps it a resume (not a fresh session).
        start_new_session=True,
        resend=True,
        suppress_notification=True,
    )
    await message.answer(t("ui.resend_started"))


@router.message(Command("kill"))
async def handle_kill(message: Message, tmux_manager: TmuxManager) -> None:
    """Kill the tmux session in the current topic."""
    key = channel_key(message)
    if not tmux_manager.is_active(key):
        await message.answer(t("ui.tmux_not_active"))
        return
    logger.debug(
        "User %s killed tmux session for %s", message.from_user and message.from_user.id, key
    )
    await tmux_manager.kill(key)
    await message.answer(t("ui.tmux_killed"))


@router.message(Command("mcpstatus"))
async def handle_mcpstatus(
    message: Message,
    tmux_manager: TmuxManager,
    topic_config: TopicConfig,
    bot_defaults: BotDefaults,
) -> None:
    """Show redacted MCP process diagnostics for the current topic."""
    key = channel_key(message)
    runtime = resolve_topic_runtime_config(topic_config.get_topic(key[1]), bot_defaults)
    if runtime.exec_mode == "tmux":
        status = tmux_manager.mcp_status_text(key)
    else:
        status = _mcp_status_subprocess(runtime, key, tmux_manager)
    await message.answer(f"<pre>{html.escape(status)}</pre>", parse_mode="HTML")


def _mcp_status_subprocess(
    runtime: TopicRuntimeConfig, key: ChannelKey, tmux_manager: TmuxManager
) -> str:
    """MCP diagnostics for a subprocess-mode channel (no tmux pane)."""
    # Resolve the engine the way a session does (config may fall back, e.g.
    # an unset claude resolving to an installed codex).
    engine = choose_available_engine(runtime.engine) or runtime.engine
    configured = tmux_manager._configured_mcp_servers(runtime.mcp_config)
    procs = tagged_processes(channel_key=key, tmux_session=None, runtime_path=None)
    diag = RuntimeDiagnostics(
        pane_pid=None,
        pane_sid=None,
        sid_processes=(),
        tagged_processes=procs,
        configured_servers=configured,
    )
    dupes = diag.duplicate_generations
    duplicate_lines = (
        ", ".join(f"{name}={count}" for name, count in sorted(dupes.items()))
        if dupes
        else "none"
    )
    return "\n".join(
        [
            f"topic: {key[0]}:{key[1]}",
            "mode: subprocess",
            f"provider: {engine}",
            f"configured: {', '.join(configured) if configured else 'none'}",
            f"tagged_processes: {len(procs)}",
            f"mcp_counts: {duplicate_lines}",
            f"rss_mb: {diag.rss_kb / 1024:.1f}",
        ]
    )


def _resolve_usage_for_session(
    session: SessionData, settings: Settings
) -> tuple[ContextUsage | None, int | None]:
    """(usage, max_tokens) for the session's engine, or (None, None)."""
    if session.engine == "claude":
        if not session.session_id:
            return None, None
        usage = get_claude_context_usage(session.session_id, session.cwd or None)
        max_tokens = resolve_claude_max_context_tokens(
            settings.claude_context_window_max,
            usage.model if usage is not None else None,
        )
        return usage, max_tokens
    if session.engine == "codex":
        if not session.session_id:
            return None, None
        usage = get_codex_context_usage(session.session_id)
        max_tokens = resolve_max_context_tokens(settings.codex_context_window_max)
        return usage, max_tokens
    return None, None


@router.message(Command("usage"))
async def handle_usage(
    message: Message,
    session_manager: SessionManager,
    settings: Settings,
) -> None:
    """Report the current context-window usage of this chat's agent session."""
    key = channel_key(message)
    session = session_manager._get_session(key)
    usage, max_tokens = _resolve_usage_for_session(session, settings)
    if usage is None:
        await message.answer(
            t("ui.usage_no_session") if not session.session_id else t("ui.usage_not_found")
        )
        return
    text = format_usage(usage, max_tokens)
    text += "\n" + t(
        "ui.usage_session",
        engine=session.engine,
        session_id=session.session_id,
    )

    async def _send_html() -> object:
        return await message.answer(text, parse_mode="HTML")

    async def _send_plain() -> object:
        return await message.answer(text)

    await send_html_with_fallback(
        send_html=_send_html,
        send_plain=_send_plain,
        label=f"usage {key}",
    )


@router.message(Command("compact"))
async def handle_compact(
    message: Message,
    session_manager: SessionManager,
    message_queue: MessageQueue,
    tmux_manager: TmuxManager,
    topic_config: TopicConfig,
    settings: Settings,
) -> None:
    """Manually compact this chat's agent session context."""
    key = channel_key(message)
    session = session_manager._get_session(key)
    if not session.session_id:
        await message.answer(t("ui.usage_no_session"))
        return
    if message_queue.is_busy(key) or tmux_manager.is_processing(key):
        await message.answer(t("ui.exec_mode_busy"))
        return

    parts = (message.text or "").split(maxsplit=1)
    focus = parts[1].strip() if len(parts) > 1 else ""
    prompt = "/compact" + (f" {focus}" if focus else "")

    if topic_config.get_topic(key[1]).exec_mode == "tmux":
        if not await ensure_exec_mode_ready(
            key, topic_config, tmux_manager, session_manager, message
        ):
            return
        if await send_to_tmux_if_active(key, prompt, message, tmux_manager):
            return

    if session.engine == "codex":
        # Subprocess path: run this one turn under a small window override
        # so the engine's built-in compaction fires (context > window).
        window = resolve_compact_turn_window(settings.codex_compact_turn_window)
        usage = get_codex_context_usage(session.session_id)
        if usage is None or usage.context_tokens < window:
            await message.answer(t("ui.compact_under_threshold", threshold_k=window // 1024))
            return
        session.compact_window_override = window
        # `codex exec` has no slash-command layer, so the literal
        # "/compact" would reach the model as plain text; a small
        # local model then wanders off mid-task, and its long tool
        # calls are what poisoned the session. The window override
        # already forced the engine's auto-compaction, so ask the
        # model for a minimal confirmation instead.
        prompt = t("ui.compact_codex_prompt")
        if focus:
            prompt += f" (focus: {focus})"

    message_queue.enqueue(
        key,
        prompt,
        message.message_id,
        message,
        target_session_id=session.session_id,
        suppress_notification=tmux_manager.is_active(key),
    )
    await message.answer(t("ui.compact_started"))


@router.message(Command("recycle"))
async def handle_recycle(
    message: Message,
    tmux_manager: TmuxManager,
    session_manager: SessionManager,
    message_queue: MessageQueue,
) -> None:
    """Restart the current tmux runtime without intentionally clearing context."""
    key = channel_key(message)
    if not tmux_manager.is_active(key):
        await message.answer(t("ui.tmux_not_active"))
        return
    if tmux_manager.is_processing(key) or message_queue.is_busy(key):
        await message.answer(t("ui.exec_mode_busy"))
        return
    try:
        ok = await tmux_manager.recycle(key, session_manager)
    except RuntimeError:
        logger.warning("recycle failed for %s", key, exc_info=True)
        await message.answer(t("ui.recycle_failed"))
        return
    if ok:
        await message.answer(t("ui.recycle_done"))
    else:
        await message.answer(t("ui.tmux_not_active"))


def _command_arg(message: Message) -> str | None:
    """First space-separated argument of a slash command message, if any."""
    tokens = (message.text or "").split()
    if len(tokens) < 2:
        return None
    return tokens[1]


def _sessions_caption(
    entries: tuple[SessionEntry, ...],
    current_session_id: str | None,
    cwd: str,
) -> str:
    lines = [
        t("ui.sessions_header", cwd=html.escape(cwd)),
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        marker = (
            f" ({t('ui.resume_current_marker')})"
            if entry.session_id == current_session_id
            else ""
        )
        lines.append(f"{index}. {html.escape(entry.preview)}{marker}")
    lines.append("")
    lines.append(t("ui.sessions_hint", count=len(entries)))
    return "\n".join(lines)


async def _sessions_list(
    message: Message,
    runtime: TopicRuntimeConfig,
    key: ChannelKey,
    session_manager: SessionManager,
    picker_store: PickerStore,
) -> None:
    """Subprocess /resume: numbered list of the most recent sessions."""
    engine = choose_available_engine(runtime.engine) or runtime.engine
    entries = await asyncio.to_thread(list_recent, runtime.cwd, engine)
    if not entries:
        await message.answer(t("ui.resume_no_sessions"))
        return
    state = PickerState(
        chat_id=key[0],
        thread_id=key[1],
        cwd=runtime.cwd,
        engine=engine,
        entries=tuple(entries),
        created_at=time.time(),
    )
    picker_store.put(state)
    current = session_manager.get_current_session_id(key)
    await message.answer(
        _sessions_caption(tuple(entries), current, str(runtime.cwd)),
        parse_mode="HTML",
    )



async def _sessions_switch(
    message: Message,
    runtime: TopicRuntimeConfig,
    key: ChannelKey,
    arg: str,
    session_manager: SessionManager,
    picker_store: PickerStore,
    message_queue: MessageQueue,
    tmux_manager: TmuxManager,
) -> None:
    """Subprocess /resume N: switch the channel to session number N."""
    state = picker_store.latest_for(key[0], key[1])
    if state is None or not _same_cwd(runtime.cwd, state.cwd):
        await message.answer(t("ui.sessions_stale"))
        return
    try:
        index = int(arg) - 1
    except ValueError:
        index = -1
    if index < 0 or index >= len(state.entries):
        await message.answer(t("ui.sessions_bad_number", count=len(state.entries)))
        return
    if message_queue.is_busy(key) or tmux_manager.is_processing(key):
        await message.answer(t("ui.sessions_busy"))
        return
    entry = state.entries[index]
    await session_manager.override_session(key, entry.session_id)
    await message.answer(
        t("ui.sessions_switched", sid=entry.session_id[:8]),
        parse_mode="HTML",
    )


@router.message(Command("resume"))
async def handle_resume(
    message: Message,
    session_manager: SessionManager,
    topic_config: TopicConfig,
    tmux_manager: TmuxManager,
    picker_store: PickerStore,
    message_queue: MessageQueue,
    bot_defaults: BotDefaults,
) -> None:
    """List resumable sessions; in subprocess mode /resume N switches."""
    key = channel_key(message)
    runtime = resolve_topic_runtime_config(topic_config.get_topic(key[1]), bot_defaults)
    if runtime.exec_mode == "subprocess":
        arg = _command_arg(message)
        if arg is None:
            await _sessions_list(message, runtime, key, session_manager, picker_store)
        else:
            await _sessions_switch(
                message,
                runtime,
                key,
                arg,
                session_manager,
                picker_store,
                message_queue,
                tmux_manager,
            )
        return
    if key[1] is None:
        await message.answer(t("ui.resume_not_in_forum"))
        return

    entries = tuple(await asyncio.to_thread(list_sessions, runtime.cwd))
    if not entries:
        await message.answer(t("ui.resume_no_sessions"))
        return

    token = picker_store.put(
        PickerState(
            chat_id=key[0],
            thread_id=key[1],
            cwd=runtime.cwd,
            engine=runtime.engine,
            entries=entries,
            created_at=time.time(),
        )
    )
    total_pages = max(1, math.ceil(len(entries) / 8))
    current_session_id = tmux_manager.get_active_session_id(key)
    await message.answer(
        _resume_caption(
            runtime.cwd,
            page=0,
            total_pages=total_pages,
            entries=entries,
            current_session_id=current_session_id,
        ),
        reply_markup=resume_keyboard(
            entries,
            page=0,
            current_session_id=current_session_id,
            token=token,
        ),
        parse_mode="HTML",
    )


def _callback_key(callback: CallbackQuery) -> ChannelKey | None:
    if callback.message is None or isinstance(callback.message, InaccessibleMessage):
        return None
    return (callback.message.chat.id, callback.message.message_thread_id)


async def _stale_resume_picker(callback: CallbackQuery) -> None:
    if callback.message is not None and not isinstance(callback.message, InaccessibleMessage):
        with contextlib.suppress(TelegramBadRequest):
            await callback.message.edit_text(t("ui.resume_picker_stale"), reply_markup=None)
    await callback.answer(t("ui.resume_picker_stale"), show_alert=True)


async def _answer_callback_safely(
    callback: CallbackQuery, text: str | None = None, *, show_alert: bool = False
) -> None:
    with contextlib.suppress(TelegramBadRequest):
        await callback.answer(text, show_alert=show_alert)


async def _replay_last_assistant_message(
    message: Message,
    entry: SessionEntry,
    key: ChannelKey,
    session_manager: SessionManager,
) -> None:
    content = await asyncio.to_thread(
        get_last_assistant_message,
        entry.provider,
        entry.transcript_path,
    )
    if not content:
        return

    for chunk in split_html_message(content):

        async def _send_html(c: str = chunk) -> object:
            return await message.answer(c, parse_mode="HTML")

        async def _send_plain(c: str = chunk) -> object:
            return await message.answer(c)

        outcome = await send_html_with_fallback(
            send_html=_send_html,
            send_plain=_send_plain,
            label=f"resume replay {key}",
        )
        if outcome.message_id is not None:
            session_manager.record_message(
                outcome.message_id,
                entry.session_id,
                key,
                provider=entry.provider,
                model=None,
            )
        if outcome.fatal:
            return


@router.callback_query(F.data.startswith("rs:p:"))
async def on_resume_page(
    callback: CallbackQuery,
    picker_store: PickerStore,
    tmux_manager: TmuxManager,
) -> None:
    if callback.data is None or callback.message is None:
        await callback.answer()
        return
    if isinstance(callback.message, InaccessibleMessage):
        await callback.answer()
        return
    parts = callback.data.split(":")
    if len(parts) != 4:
        await _stale_resume_picker(callback)
        return
    _, _, token, raw_page = parts
    state = picker_store.get(token)
    key = _callback_key(callback)
    if state is None or key != (state.chat_id, state.thread_id):
        await _stale_resume_picker(callback)
        return
    try:
        page = int(raw_page)
    except ValueError:
        await _stale_resume_picker(callback)
        return
    total_pages = max(1, math.ceil(len(state.entries) / 8))
    page = max(0, min(page, total_pages - 1))
    try:
        await callback.message.edit_text(
            _resume_caption(
                state.cwd,
                page=page,
                total_pages=total_pages,
                entries=state.entries,
                current_session_id=tmux_manager.get_active_session_id(key),
            ),
            reply_markup=resume_keyboard(
                state.entries,
                page=page,
                current_session_id=tmux_manager.get_active_session_id(key),
                token=token,
            ),
            parse_mode="HTML",
        )
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise
    await callback.answer()


@router.callback_query(F.data.startswith("rs:s:"))
async def on_resume_pick(
    callback: CallbackQuery,
    session_manager: SessionManager,
    topic_config: TopicConfig,
    tmux_manager: TmuxManager,
    picker_store: PickerStore,
    bot_defaults: BotDefaults,
) -> None:
    if callback.data is None or callback.message is None:
        await callback.answer()
        return
    if isinstance(callback.message, InaccessibleMessage):
        await callback.answer()
        return
    parts = callback.data.split(":")
    if len(parts) != 4:
        await _stale_resume_picker(callback)
        return
    _, _, token, raw_idx = parts
    state = picker_store.get(token)
    key = _callback_key(callback)
    if state is None or key != (state.chat_id, state.thread_id):
        await _stale_resume_picker(callback)
        return
    runtime = resolve_topic_runtime_config(topic_config.get_topic(key[1]), bot_defaults)
    if not _same_cwd(runtime.cwd, state.cwd):
        await _stale_resume_picker(callback)
        return
    try:
        idx = int(raw_idx)
    except ValueError:
        await _stale_resume_picker(callback)
        return
    if idx < 0:
        await _stale_resume_picker(callback)
        return
    try:
        entry = state.entries[idx]
    except IndexError:
        await _stale_resume_picker(callback)
        return

    await _answer_callback_safely(callback, t("ui.resume_starting"))
    result = await tmux_manager.switch_or_start_session(
        key,
        entry.session_id,
        entry.provider,
        entry.transcript_path,
        session_manager=session_manager,
        topic_config=topic_config,
        defaults=bot_defaults,
    )
    if result.kind == "target_missing":
        await callback.message.edit_text(t("ui.resume_target_missing"), reply_markup=None)
        return
    if result.kind in {"invalid_id", "spawn_failed", "config_write_failed"}:
        key_name = (
            "ui.resume_spawn_failed_engine_changed"
            if result.kind == "spawn_failed" and result.engine_changed
            else f"ui.resume_{result.kind}"
        )
        await callback.message.edit_text(
            t(key_name, engine=entry.provider),
            reply_markup=None,
        )
        return

    picker_store.drop(token)
    if result.kind == "already_on_it":
        await callback.message.edit_text(t("ui.resume_already_on_it"), reply_markup=None)
        await _replay_last_assistant_message(callback.message, entry, key, session_manager)
        return

    message_key = "ui.resume_switched" if result.kind == "switched" else "ui.resume_started"
    text = t(message_key, sid=entry.session_id[:8])
    if result.engine_changed:
        text += "\n" + t("ui.resume_engine_switched", engine=entry.provider)
    await callback.message.edit_text(text, reply_markup=None, parse_mode="HTML")
    await _replay_last_assistant_message(callback.message, entry, key, session_manager)


@router.callback_query(F.data.startswith("rs:cancel:"))
async def on_resume_cancel(callback: CallbackQuery, picker_store: PickerStore) -> None:
    if callback.data is not None:
        picker_store.drop(callback.data.rsplit(":", 1)[-1])
    if callback.message is not None and not isinstance(callback.message, InaccessibleMessage):
        with contextlib.suppress(TelegramBadRequest):
            await callback.message.edit_text(t("ui.resume_cancelled"), reply_markup=None)
    await callback.answer()


@router.message(Command("stream"))
async def handle_stream_mode(message: Message, topic_config: TopicConfig) -> None:
    """Show a 3-button picker to switch stream_mode for the current topic."""
    _, thread_id = channel_key(message)
    if thread_id is None:
        await message.answer(t("ui.stream_mode_not_in_forum"))
        return
    current = topic_config.get_topic(thread_id).stream_mode
    await message.answer(
        t("ui.stream_mode_picker_caption", current=current),
        reply_markup=stream_mode_keyboard(current),
        parse_mode="HTML",
    )


@router.callback_query(F.data.startswith("stream_mode:"))
async def on_stream_mode_click(
    callback: CallbackQuery,
    topic_config: TopicConfig,
    tmux_manager: TmuxManager | None = None,
) -> None:
    """Apply a new stream_mode for the topic the picker was posted in."""
    if callback.data is None or callback.message is None:
        await callback.answer()
        return
    # InaccessibleMessage has no thread_id/edit methods — bail out if the
    # picker message is no longer reachable (e.g. deleted, chat lost).
    if isinstance(callback.message, InaccessibleMessage):
        await callback.answer()
        return
    _, _, mode = callback.data.partition(":")
    if mode not in _VALID_STREAM_MODES:
        await callback.answer(t("ui.stream_mode_invalid"), show_alert=True)
        return

    thread_id = callback.message.message_thread_id
    if thread_id is None:
        await callback.answer(
            t("ui.stream_mode_not_in_forum"),
            show_alert=True,
        )
        return

    previous_mode = topic_config.get_topic(thread_id).stream_mode
    ok = await topic_config.update_stream_mode(thread_id, mode)  # type: ignore[arg-type]
    if not ok:
        await callback.answer(t("ui.stream_mode_write_failed"), show_alert=True)
        return
    if previous_mode == "live" and mode != "live" and tmux_manager is not None:
        await tmux_manager.close_buffer(
            (callback.message.chat.id, thread_id),
        )

    # Refresh both caption and keyboard so the visible current value matches the checkmark.
    try:
        await callback.message.edit_text(
            t("ui.stream_mode_picker_caption", current=mode),
            reply_markup=stream_mode_keyboard(mode),
            parse_mode="HTML",
        )
    except Exception:
        logger.debug("Failed to refresh stream_mode picker", exc_info=True)
    await callback.answer(t("ui.stream_mode_changed", mode=mode))


@router.message(Command("mode"))
async def handle_mode_command(message: Message, topic_config: TopicConfig) -> None:
    """Show a 2-button picker to switch exec_mode for the current topic."""
    _, thread_id = channel_key(message)
    if thread_id is None:
        await message.answer(t("ui.exec_mode_not_in_forum"))
        return
    current = topic_config.get_topic(thread_id).exec_mode
    await message.answer(
        _exec_mode_picker_caption(current),
        reply_markup=exec_mode_keyboard(current),
        parse_mode="HTML",
    )


@router.callback_query(F.data.startswith("exec_mode:"))
async def on_exec_mode_click(
    callback: CallbackQuery,
    topic_config: TopicConfig,
    tmux_manager: TmuxManager,
    message_queue: MessageQueue,
) -> None:
    """Apply a new exec_mode for the topic the picker was posted in.

    Order matters: busy-check precedes any side-effect, and tmux.kill strictly
    precedes the config write on tmux→subprocess (Decision 2 — if we wrote
    first and crashed, the next message would race a still-running tmux
    session against a fresh subprocess under the new mode).
    """
    if callback.data is None or callback.message is None:
        await callback.answer()
        return
    # InaccessibleMessage has no thread_id / edit methods — bail out if the
    # picker message is no longer reachable.
    if isinstance(callback.message, InaccessibleMessage):
        await callback.answer()
        return

    _, _, new_mode = callback.data.partition(":")
    # Re-validate against the whitelist even though the keyboard only emits
    # two canonical values — raw callback.data is user-controlled.
    if new_mode not in _VALID_EXEC_MODES:
        await callback.answer(t("ui.exec_mode_invalid"), show_alert=True)
        return

    thread_id = callback.message.message_thread_id
    if thread_id is None:
        await callback.answer(t("ui.exec_mode_not_in_forum"), show_alert=True)
        return

    key = (callback.message.chat.id, thread_id)
    previous_mode = topic_config.get_topic(thread_id).exec_mode

    if new_mode == previous_mode:
        await callback.answer(t("ui.exec_mode_already", mode=_exec_mode_label(new_mode)))
        return

    # Busy-check covers both channels: tmux's own processing flag AND the
    # subprocess-path MessageQueue (lock held OR items pending). Either way
    # we refuse the switch without touching tmux state.
    if tmux_manager.is_processing(key) or message_queue.is_busy(key):
        await callback.answer(t("ui.exec_mode_busy"), show_alert=True)
        return

    # tmux→subprocess: kill first, then persist. Reverse order leaves an
    # orphan tmux session if the write fails.
    if previous_mode == "tmux" and new_mode == "subprocess":
        await tmux_manager.kill(key)

    ok = await topic_config.update_exec_mode(thread_id, new_mode)
    if not ok:
        await callback.answer(t("ui.exec_mode_write_failed"), show_alert=True)
        return

    user_id = callback.from_user.id if callback.from_user else None
    logger.info(
        "exec_mode switched: user_id=%s thread_id=%s previous_mode=%s new_mode=%s",
        user_id,
        thread_id,
        previous_mode,
        new_mode,
    )

    # Refresh both caption and keyboard so the visible current value matches the checkmark.
    try:
        await callback.message.edit_text(
            _exec_mode_picker_caption(new_mode),
            reply_markup=exec_mode_keyboard(new_mode),
            parse_mode="HTML",
        )
    except Exception:
        logger.debug("Failed to refresh exec_mode picker", exc_info=True)
    await callback.answer(t("ui.exec_mode_changed", mode=_exec_mode_label(new_mode)))


@router.message(Command("engine"))
async def handle_engine_command(message: Message, topic_config: TopicConfig) -> None:
    """Show provider engine picker for the current forum topic."""
    _, thread_id = channel_key(message)
    if thread_id is None:
        await message.answer(t("ui.engine_not_in_forum"))
        return
    settings = topic_config.get_topic(thread_id)
    await message.answer(
        t(
            "ui.engine_picker_caption",
            engine=engine_display_name(settings.engine),
        ),
        reply_markup=engine_keyboard(settings.engine),
        parse_mode="HTML",
    )


@router.callback_query(F.data.startswith("engine:"))
async def on_engine_click(
    callback: CallbackQuery,
    topic_config: TopicConfig,
    tmux_manager: TmuxManager,
    message_queue: MessageQueue,
    session_manager: SessionManager,
) -> None:
    """Apply provider engine changes for the picker topic."""
    if callback.data is None or callback.message is None:
        await callback.answer()
        return
    if isinstance(callback.message, InaccessibleMessage):
        await callback.answer()
        return

    _, _, raw_value = callback.data.partition(":")
    thread_id = callback.message.message_thread_id
    if thread_id is None:
        await callback.answer(t("ui.engine_not_in_forum"), show_alert=True)
        return
    key = (callback.message.chat.id, thread_id)
    current = topic_config.get_topic(thread_id)

    if tmux_manager.is_processing(key) or message_queue.is_busy(key):
        await callback.answer(t("ui.exec_mode_busy"), show_alert=True)
        return

    if raw_value not in _VALID_ENGINES:
        await callback.answer(t("ui.engine_invalid"), show_alert=True)
        return
    new_engine: Engine = "claude" if raw_value == "claude" else "codex"

    if new_engine == current.engine:
        await callback.answer(t("ui.engine_already"))
        return

    if current.models:
        ok = await topic_config.update_engine(thread_id, new_engine)
    else:
        ok = await topic_config.update_engine_model(thread_id, new_engine, None)
    if not ok:
        await callback.answer(t("ui.engine_write_failed"), show_alert=True)
        return

    if tmux_manager.is_active(key):
        await tmux_manager.kill(key)
    await session_manager.clear_provider_session(key)

    logger.info(
        "engine switched: user_id=%s thread_id=%s previous=%s new=%s model=%s",
        callback.from_user.id if callback.from_user else None,
        thread_id,
        current.engine,
        new_engine,
        current.models.get(new_engine, current.model),
    )
    engine_name = engine_display_name(new_engine)
    try:
        await callback.message.edit_text(
            t("ui.engine_picker_caption", engine=engine_name),
            reply_markup=engine_keyboard(new_engine),
            parse_mode="HTML",
        )
    except Exception:
        logger.debug("Failed to refresh engine picker", exc_info=True)
    await callback.answer(t("ui.engine_changed", engine=engine_name))
    await callback.message.answer(t("ui.engine_changed_new_session", engine=engine_name))
