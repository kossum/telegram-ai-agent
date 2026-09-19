from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from telegram_bot.core.config import Settings
from telegram_bot.core.env_file import read_exact_env_file
from telegram_bot.core.handlers import commands
from telegram_bot.core.handlers.tail import handle_tail_command
from telegram_bot.core.services import cc_modes
from telegram_bot.core.services.bot_commands import build_bot_commands
from telegram_bot.core.services.cc_events import mcp_server_event_by_name
from telegram_bot.core.services.claude import SessionManager
from telegram_bot.core.services.providers import (
    CODEX_ADAPTER,
    CodexTranscriptParser,
    agent_process_env,
    choose_available_engine,
)
from telegram_bot.core.services.rich_sender import detect_rich_send
from telegram_bot.core.services.tmux_spawn import (
    sanitized_tmux_environment,
    tmux_pane_inherits_disallowed_environment,
)
from telegram_bot.core.services.topic_config import (
    TopicConfig,
    TopicSettings,
)
from telegram_bot.core.services.topic_runtime import BotDefaults, resolve_topic_runtime_config
from telegram_bot.core.tui.transcript import ClaudeTranscriptParser


def test_public_entrypoint_imports() -> None:
    entrypoint = importlib.import_module("telegram_bot.__main__")

    assert callable(entrypoint.main)
    assert callable(entrypoint.make_recovery_on_event)


def test_public_entrypoint_selects_dedicated_tmux_server(tmp_path: Path, monkeypatch) -> None:
    entrypoint = importlib.import_module("telegram_bot.__main__")
    monkeypatch.delenv("TMUX_TMPDIR", raising=False)
    monkeypatch.setenv("TMUX", "/tmp/custom/default,123,0")
    monkeypatch.setattr(
        entrypoint.subprocess,
        "run",
        MagicMock(return_value=MagicMock(returncode=1, stdout="")),
    )

    runtime_dir = entrypoint._ensure_dedicated_tmux_tmpdir(tmp_path, tmp_path / "tmux_sessions")

    assert runtime_dir == tmp_path / ".telegram-bot-tmux"
    assert runtime_dir.stat().st_mode & 0o777 == 0o700
    assert entrypoint.os.environ["TMUX_TMPDIR"] == str(runtime_dir)


def test_public_entrypoint_migrates_only_state_owned_legacy_tmux_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    entrypoint = importlib.import_module("telegram_bot.__main__")
    monkeypatch.delenv("TMUX_TMPDIR", raising=False)
    monkeypatch.setenv("TMUX", "/tmp/custom/default,123,0")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    sessions_dir = tmp_path / "tmux_sessions"
    sessions_dir.mkdir()
    (sessions_dir / "state.json").write_text(
        json.dumps({"1:None": {"session_name": "cc-1-0"}}),
        encoding="utf-8",
    )

    def _run(command, **_kwargs):
        if command[:2] == ["tmux", "list-sessions"]:
            return MagicMock(returncode=0, stdout="cc-1-0\nunrelated\n")
        return MagicMock(returncode=0, stdout="")

    run = MagicMock(side_effect=_run)
    monkeypatch.setattr(entrypoint.subprocess, "run", run)

    runtime_dir = entrypoint._ensure_dedicated_tmux_tmpdir(tmp_path, sessions_dir)

    commands = [call.args[0] for call in run.call_args_list]
    assert all("TMUX" not in call.kwargs["env"] for call in run.call_args_list)
    assert ["tmux", "kill-session", "-t", "=cc-1-0"] in commands
    assert not any(command == ["tmux", "kill-session", "-t", "=unrelated"] for command in commands)
    assert ["tmux", "set-environment", "-gu", "TELEGRAM_BOT_TOKEN"] in commands
    assert (runtime_dir / ".legacy-default-migrated").read_text() == "migrated\n"


def test_public_settings_default_cwd_is_generic(monkeypatch) -> None:
    for name in ("BOT_LANG", "PROJECT_ROOT", "DEFAULT_CWD", "TOPIC_CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None, telegram_bot_token="test-token")

    assert settings.bot_lang == "en"
    assert settings.project_root == "."
    assert settings.default_cwd == "."
    assert settings.topic_config_path == "./topic_config.json"


def test_public_dotenv_values_are_not_interpolated(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=token-${SHOULD_NOT_EXPAND} # deployment\n"
        'DEEPGRAM_API_KEY="key-$ALSO_LITERAL"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SHOULD_NOT_EXPAND", "secret")
    monkeypatch.setenv("ALSO_LITERAL", "secret")

    exact = read_exact_env_file(env_path)
    settings = Settings(_env_file=env_path)

    assert exact["TELEGRAM_BOT_TOKEN"] == "token-${SHOULD_NOT_EXPAND}"
    assert settings.telegram_bot_token == "token-${SHOULD_NOT_EXPAND}"
    assert settings.deepgram_api_key == "key-$ALSO_LITERAL"


def test_public_tmux_server_environment_drops_bot_secrets(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    monkeypatch.setenv("TMUX_TMPDIR", "/run/public-bot-tmux")
    run = MagicMock(
        side_effect=[
            MagicMock(returncode=0, stdout="TELEGRAM_BOT_TOKEN=secret\n"),
            MagicMock(returncode=0, stdout=""),
        ]
    )

    env = sanitized_tmux_environment(run=run)

    assert "TELEGRAM_BOT_TOKEN" not in env
    assert env["TMUX_TMPDIR"] == "/run/public-bot-tmux"
    assert run.call_args_list[1].args[0] == [
        "tmux",
        "set-environment",
        "-gu",
        "TELEGRAM_BOT_TOKEN",
    ]


def test_public_detects_legacy_tmux_pane_with_service_secret(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"TELEGRAM_BOT_TOKEN=secret\0")
    run = MagicMock(return_value=MagicMock(returncode=0, stdout="123\n"))

    assert tmux_pane_inherits_disallowed_environment("cc-1-0", run=run) is True


def test_public_settings_support_split_app_and_workspace_roots(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    workspace_root = tmp_path / "workspace"
    settings = Settings(
        _env_file=None,
        telegram_bot_token="test-token",
        project_root=str(tmp_path / "legacy"),
        app_root=str(app_root),
        agent_workspace_root=str(workspace_root),
    )

    assert settings.app_root_path == app_root
    assert settings.workspace_root_path == workspace_root
    assert settings.resolve_app_path("config.json") == app_root / "config.json"
    assert settings.resolve_workspace_path("topic_config.json") == (
        workspace_root / "topic_config.json"
    )


def test_public_relative_file_cache_dir_is_agent_readable(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    settings = Settings(
        _env_file=None,
        telegram_bot_token="test-token",
        project_root=str(root),
        default_cwd=".",
        file_cache_dir="./data",
    )
    session_manager = SessionManager(settings)

    assert session_manager.file_cache_dir == str((root / "data").resolve())


def test_public_start_wires_live_buffer_before_restore_all() -> None:
    source = Path("src/telegram_bot/__main__.py").read_text(encoding="utf-8")

    assert "tmux_manager.wire_live_buffer(bot=bot, topic_config=topic_config)" in source
    assert source.index("tmux_manager.wire_live_buffer") < source.index("tmux_manager.restore_all")
    assert "tmux_manager.restore_all(session_manager)" in source


def test_public_start_wires_codex_update_and_tail_runtime() -> None:
    source = Path("src/telegram_bot/__main__.py").read_text(encoding="utf-8")

    assert "codex_update_service = CodexUpdateService(" in source
    assert "tmux_manager.wire_codex_update_service(codex_update_service)" in source
    assert 'dp["codex_update_service"] = codex_update_service' in source
    assert "ForwardBatcher(bot=bot, transcriber=transcriber)" in source
    assert "dp.include_router(tail_router)" in source
    assert source.index("dp.include_router(tail_router)") < source.index(
        "dp.include_router(text_router)"
    )
    assert "recovery_factory = make_recovery_on_event(" in source
    assert "await tmux_manager.resume_tails(recovery_factory)" in source
    assert "tmux_manager.start_modal_watchdog()" in source
    assert "tmux_manager.start_transcript_watchdog()" in source
    assert "await tmux_manager.stop_transcript_watchdog()" in source
    assert "await tmux_manager.stop_modal_watchdog()" in source
    assert "picker_store = PickerStore()" in source
    assert "bot_defaults = BotDefaults(" in source
    assert 'dp["picker_store"] = picker_store' in source
    assert 'dp["bot_defaults"] = bot_defaults' in source
    assert source.index("await tmux_manager.resume_tails(recovery_factory)") < source.index(
        "await dp.start_polling"
    )


def test_public_entrypoint_uses_workspace_root_for_runtime_state() -> None:
    source = Path("src/telegram_bot/__main__.py").read_text(encoding="utf-8")

    assert "settings.resolve_workspace_path(settings.topic_config_path)" in source
    assert "settings.resolve_workspace_path(settings.tmux_sessions_dir)" in source
    assert "settings.resolve_workspace_path(settings.default_cwd)" in source
    assert "TopicConfig(str(topic_config_path), str(workspace_root))" in source
    assert "TmuxManager(\n        sessions_dir=tmux_sessions_dir" in source
    assert 'state_path=tmux_sessions_dir / "codex_update.json"' in source
    assert "cwd=settings.resolve_workspace_path(settings.default_cwd)" in source


def test_public_prompt_modes_are_available() -> None:
    prompts_dir = Path("src/telegram_bot/prompts")
    assert {path.name for path in prompts_dir.glob("*.md")} == {
        "default.md",
        "task-manager.md",
    }
    for mode in ("free", "task"):
        assert cc_modes._get_mode_prompt(mode)
    assert cc_modes._get_mode_prompt("task") == (prompts_dir / "task-manager.md").read_text()
    assert set(cc_modes._MODE_TOOLS) == {"free", "task"}


def test_public_runtime_rejects_unregistered_prompt_mode(tmp_path: Path) -> None:
    runtime = resolve_topic_runtime_config(
        TopicSettings(
            name="Private",
            type="project",
            mode="private",
            cwd=None,
            mcp_config=None,
        ),
        BotDefaults(cwd=tmp_path, mcp_config=tmp_path / ".mcp.json"),
    )

    assert runtime.mode == "free"


def test_public_agent_environment_is_sanitized() -> None:
    env = agent_process_env(
        base_env={
            "HOME": "/home/test",
            "PATH": "/usr/bin",
            "APP_ROOT": "/srv/bot",
            "AGENT_WORKSPACE_ROOT": "/srv/workspace",
            "PROJECT_ROOT": "/srv/legacy",
            "TELEGRAM_BOT_TOKEN": "must-not-leak",
            "DEEPGRAM_API_KEY": "must-not-leak",
            "UNRELATED_SECRET": "must-not-leak",
        }
    )

    assert env["APP_ROOT"] == "/srv/bot"
    assert env["AGENT_WORKSPACE_ROOT"] == "/srv/workspace"
    assert env["PROJECT_ROOT"] == "/srv/legacy"
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert "DEEPGRAM_API_KEY" not in env
    assert "UNRELATED_SECRET" not in env


def test_public_agent_environment_honors_extra_env_names() -> None:
    base = {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "TZ": "America/Phoenix",
        "UNRELATED_SECRET": "must-not-leak",
    }
    assert "TZ" not in agent_process_env(base_env=base)
    extra = agent_process_env(base_env={**base, "AGENT_EXTRA_ENV": "TZ"})
    assert extra["TZ"] == "America/Phoenix"
    assert "UNRELATED_SECRET" not in extra


def test_public_prompt_modes_have_bot_mcp_tools() -> None:
    required = {
        "mcp__bot__send_message",
        "mcp__bot__send_image",
        "mcp__bot__send_image_gallery",
        "mcp__bot__send_document",
    }

    for mode in ("free", "task"):
        tools = set(cc_modes._MODE_TOOLS[mode].split(","))
        assert required <= tools
        assert "mcp__bot__send_file" not in tools


def test_free_mode_allows_skill_for_topic_setup() -> None:
    tools = set(cc_modes._MODE_TOOLS["free"].split(","))

    assert "Skill" in tools


def test_public_prompt_modes_allow_context7_docs_tools() -> None:
    required = {
        "mcp__context7__resolve-library-id",
        "mcp__context7__query-docs",
        "mcp__context7__get-library-docs",
    }

    for mode in ("free", "task"):
        tools = set(cc_modes._MODE_TOOLS[mode].split(","))
        assert required <= tools


def test_engine_selection_falls_back_to_available_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "telegram_bot.core.services.providers.is_engine_available",
        lambda engine: engine == "codex",
    )

    assert choose_available_engine("claude") == "codex"


def test_topic_config_parses_public_runtime_fields(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    mcp_config = project / ".mcp.json"
    mcp_config.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "topic_config.json"
    config_path.write_text(
        json.dumps(
            {
                "topics": {
                    "42": {
                        "name": "Demo",
                        "type": "project",
                        "mode": "free",
                        "cwd": str(project),
                        "mcp_config": str(mcp_config),
                        "stream_mode": "minimal",
                        "exec_mode": "tmux",
                        "engine": "codex",
                        "model": "legacy-model",
                        "models": {
                            "claude": "claude-model",
                            "codex": "codex-model",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    topic = TopicConfig(str(config_path), ".").get_topic(42)

    assert topic.name == "Demo"
    assert topic.mode == "free"
    assert topic.cwd == str(project)
    assert topic.mcp_config == str(mcp_config)
    assert topic.stream_mode == "minimal"
    assert topic.exec_mode == "tmux"
    assert topic.engine == "codex"
    assert topic.model == "legacy-model"
    assert topic.models == {
        "claude": "claude-model",
        "codex": "codex-model",
    }


def test_topic_config_normalizes_and_filters_model_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "topic_config.json"
    config_path.write_text(
        json.dumps(
            {
                "topics": {
                    "1": {
                        "mode": "free",
                        "model": "  legacy-model  ",
                        "models": {
                            "claude": "  claude-model  ",
                            "codex": "bad model with spaces",
                            "unknown": "private-model",
                        },
                    },
                    "2": {
                        "mode": "free",
                        "model": {"invalid": "type"},
                        "models": ["invalid", "type"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    config = TopicConfig(str(config_path), ".")

    first = config.get_topic(1)
    second = config.get_topic(2)

    assert first.model == "legacy-model"
    assert first.models == {"claude": "claude-model"}
    assert second.model is None
    assert second.models == {}


def test_codex_provider_parser_smoke() -> None:
    parsed = CODEX_ADAPTER.parse_exec_event('{"type":"thread.started","thread_id":"abc"}')

    assert parsed.session_id == "abc"
    assert parsed.events == []


def test_codex_0147_items_keep_clarification_in_the_same_turn() -> None:
    parser = CodexTranscriptParser()
    records = [
        '{"type":"turn_context","payload":{"turn_id":"turn-147"}}',
        (
            '{"type":"event_msg","payload":{"type":"item_completed","item":'
            '{"type":"AgentMessage","phase":"commentary","content":'
            '[{"type":"Text","text":"working"}]}}}'
        ),
        (
            '{"type":"event_msg","payload":{"type":"item_completed","item":'
            '{"type":"AgentMessage","phase":"final_answer","content":'
            '[{"type":"Text","text":"first answer"}]}}}'
        ),
        (
            '{"type":"event_msg","payload":{"type":"item_completed","item":'
            '{"type":"UserMessage","content":[{"type":"text","text":"clarify"}]}}}'
        ),
        (
            '{"type":"event_msg","payload":{"type":"item_completed","item":'
            '{"type":"AgentMessage","phase":"final_answer","content":'
            '[{"type":"Text","text":"second answer"}]}}}'
        ),
        '{"type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-147"}}',
    ]

    events = [event for raw in records for event in parser.parse(raw).events]

    assert [(event.type, event.content) for event in events] == [
        ("turn_start", ""),
        ("text", "working"),
        ("result_message", "first answer"),
        ("result_message", "second answer"),
        ("turn_end", ""),
    ]
    assert {event.turn_id for event in events} == {"turn-147"}


def test_claude_local_command_records_do_not_open_a_turn() -> None:
    parser = ClaudeTranscriptParser()
    boundary_parser = ClaudeTranscriptParser()
    records = [
        (
            "<local-command-caveat>Local command metadata.</local-command-caveat>",
            True,
        ),
        (
            "<command-name>/model</command-name>\n"
            "<command-message>model</command-message>\n"
            "<command-args></command-args>",
            None,
        ),
        ("<local-command-stdout>Set model to Opus.</local-command-stdout>", None),
    ]

    for content, is_meta in records:
        record: dict[str, object] = {
            "type": "user",
            "promptId": "local-command",
            "message": {"role": "user", "content": content},
        }
        if is_meta is not None:
            record["isMeta"] = is_meta
        raw = json.dumps(record)
        assert parser.parse(raw)[0] == []
        assert boundary_parser.is_turn_boundary(raw) is False

    assert parser.current_turn_id is None


def test_public_command_handlers_are_wired() -> None:
    assert commands.handle_resume is not None
    assert commands.handle_stream_mode is not None
    assert commands.handle_mode_command is not None
    assert commands.handle_engine_command is not None
    assert commands.handle_recycle is not None
    assert commands.handle_mcpstatus is not None
    assert handle_tail_command is not None


def test_public_bot_command_menu_is_public_only() -> None:
    command_names = {command.command for command in build_bot_commands("ru")}

    assert "clear" in command_names
    assert "codex_update" in command_names
    assert "recycle" in command_names
    assert "mcpstatus" in command_names
    assert "tui" in command_names
    assert "tail" in command_names
    assert "new" not in command_names
    assert "day" not in command_names


def test_public_start_registers_bot_commands() -> None:
    source = Path("src/telegram_bot/__main__.py").read_text(encoding="utf-8")

    assert "setup_bot_commands(bot)" in source
    assert source.index("setup_bot_commands(bot)") < source.index("dp.start_polling")
    assert "_stop_polling_when_started(dp)" in source


def test_mcp_bot_server_imports() -> None:
    path = Path("mcp-servers/bot/server.py")
    spec = importlib.util.spec_from_file_location("public_bot_mcp_server", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "send_message")
    assert hasattr(module, "send_image")
    assert hasattr(module, "send_image_gallery")
    assert hasattr(module, "send_document")
    assert not hasattr(module, "send_file")

    assert module._normalize_parse_mode("html") == ("HTML", None)
    assert module._normalize_parse_mode("MarkdownV2") == ("MarkdownV2", None)
    invalid_mode, error = module._normalize_parse_mode("Markdown")
    assert invalid_mode is None
    assert error is not None


def test_public_rich_final_answer_detection_requires_table() -> None:
    plain = detect_rich_send("Final answer without a table.")
    table = detect_rich_send("| Feature | Status |\n| --- | --- |\n| Tables | work |")

    assert not plain.eligible
    assert plain.reason == "plain-no-rich-structure"
    assert table.eligible
    assert table.input_rich_message is not None


def test_public_rich_sender_falls_back_when_ordered_list_restarts() -> None:
    item = "   Context line for the plan item.\n\n"
    contacts = "Contacts:\n\n" + "".join(f"{n}. Contact {n}\n{item}" for n in range(1, 4))
    tasks = "Tasks:\n\n" + "".join(f"{n}. Task {n}\n{item}" for n in range(4, 8))
    text = contacts + tasks + ("body " * 900)

    decision = detect_rich_send(text)

    assert decision.eligible is False
    assert decision.reason == "ordered-list-restart"
    assert decision.fallback_text == text


def test_usage_command_is_public() -> None:
    names = {c.command for c in build_bot_commands("en")}
    assert "usage" in names


def test_usage_handler_exists() -> None:
    assert commands.handle_usage is not None


def test_compact_command_is_public() -> None:
    names = {c.command for c in build_bot_commands("en")}
    assert "compact" in names


def test_compact_handler_exists() -> None:
    assert commands.handle_compact is not None


def test_compact_routes_to_bot_not_tui() -> None:
    from telegram_bot.core.tui.routing import route_slash_command

    assert route_slash_command("/compact") == "bot"
    assert route_slash_command("/compact keep only the API notes") == "bot"
    assert route_slash_command("/model sonnet") == "tui"


def test_compact_turn_window_floor() -> None:
    from telegram_bot.core.services.context_usage import resolve_compact_turn_window

    assert resolve_compact_turn_window(None) == 49152
    assert resolve_compact_turn_window(30000) == 49152
    assert resolve_compact_turn_window(-5) == 49152
    assert resolve_compact_turn_window(65536) == 65536


def test_compaction_counter_counts_compacted_records(tmp_path) -> None:
    from telegram_bot.core.services.context_usage import _count_compactions

    rollout = tmp_path / "rollout.jsonl"
    rec1 = json.dumps({"type": "compacted", "payload": {"message": "summary"}})
    rec2 = json.dumps({"type": "response_item", "payload": {"type": "message"}})
    rollout.write_text(rec1 + "\n" + rec2 + "\n")
    assert _count_compactions(rollout) == 1


def test_context_usage_from_rollout(tmp_path, monkeypatch) -> None:
    from telegram_bot.core.services.context_usage import (
        format_usage,
        get_codex_context_usage,
        resolve_max_context_tokens,
    )

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    sid = "01a0a4c2-6fe5-7fd2-ba62-23f21c5b8bf3"
    root = tmp_path / "sessions" / "2026" / "09" / "15"
    root.mkdir(parents=True)
    rollout = root / f"rollout-2026-09-15T11-10-01-{sid}.jsonl"
    turn_ctx = {"type": "turn_context", "payload": {"model": "qwen3.8-27b"}}
    usage_rec = {
        "type": "token_usage_record",
        "payload": {
            "session_id": sid,
            "timestamp": "2026-09-15T11:11:00Z",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
            },
            "turn_token_usage": {"total_tokens": 1100},
            "thread_token_usage": {"total_tokens": 1100},
        },
    }
    compact_rec = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Summary of prior turns"}],
            "internal_chat_message_metadata_passthrough": {
                "turn_id": "t1",
                "content_item_kinds": ["compaction.summary"],
            },
        },
    }
    data = "".join(json.dumps(r) + "\n" for r in (turn_ctx, usage_rec, compact_rec))
    rollout.write_text(data)

    usage = get_codex_context_usage(sid, home=tmp_path)
    assert usage is not None
    assert usage.context_tokens == 1000
    assert usage.model == "qwen3.8-27b"
    assert usage.turn_total_tokens == 1100
    assert usage.thread_total_tokens == 1100
    assert usage.compaction_count == 1

    (tmp_path / "config.toml").write_text("model_context_window = 32768\n")
    assert resolve_max_context_tokens(None, home=tmp_path) == 32768
    assert resolve_max_context_tokens(2000, home=tmp_path) == 2000

    text = format_usage(usage, 2000)
    assert "50%" in text
    assert "1,000" in text
    assert "2,000" in text
    assert "Compactions: 1" in text


def test_claude_context_usage_from_transcript(tmp_path) -> None:
    from telegram_bot.core.services.context_usage import (
        format_usage,
        get_claude_context_usage,
        resolve_claude_max_context_tokens,
    )

    sid = "0f2f1a3c-1234-4a67-8b9c-0d1e2f3a4b5c"
    proj = tmp_path / ".claude" / "projects" / "-app-workspace"
    proj.mkdir(parents=True)
    transcript = proj / f"{sid}.jsonl"
    user_rec = {"type": "user", "message": {"role": "user", "content": "hi"}}
    asst1 = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 50,
                "output_tokens": 20,
            },
        },
    }
    summary_rec = {"type": "summary", "summary": "compacted prior turns"}
    asst2 = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "usage": {
                "input_tokens": 200,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 100,
                "output_tokens": 40,
            },
        },
        "timestamp": "2026-09-16T10:00:00Z",
    }
    lines = "".join(json.dumps(r) + "\n" for r in (user_rec, asst1, summary_rec, asst2))
    transcript.write_text(lines)

    usage = get_claude_context_usage(sid, cwd="/app/workspace", home=tmp_path)
    assert usage is not None
    assert usage.session_id == sid
    assert usage.model == "claude-sonnet-4-20250514"
    assert usage.context_tokens == 1100
    assert usage.last_turn_output_tokens == 40
    assert usage.turn_total_tokens == 1140
    assert usage.thread_total_tokens == 1710
    assert usage.compaction_count == 1

    assert resolve_claude_max_context_tokens(None, usage.model) == 200000
    assert resolve_claude_max_context_tokens(1000, usage.model) == 1000
    assert resolve_claude_max_context_tokens(None, None) is None

    text = format_usage(usage, 2000)
    assert "55%" in text
    assert "1,100" in text


def test_repair_poisoned_rollout(tmp_path, monkeypatch) -> None:
    from telegram_bot.core.services.context_usage import repair_poisoned_rollout

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    sid = "01a0a4c2-6fe5-7fd2-ba62-23f21c5b8bf3"
    root = tmp_path / "sessions" / "2026" / "09" / "15"
    root.mkdir(parents=True)
    rollout = root / f"rollout-2026-09-15T11-10-01-{sid}.jsonl"

    def msg(turn_id: str, text: str) -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
                "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
            },
        }

    def call(turn_id: str, call_id: str, args: str) -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": call_id,
                "arguments": args,
                "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
            },
        }

    def out(turn_id: str, call_id: str) -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "ok",
                "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
            },
        }

    def done(turn_id: str, error: dict | None) -> dict:
        payload: dict = {"type": "task_complete", "turn_id": turn_id}
        if error is not None:
            payload["error"] = error
        return {"type": "event_msg", "payload": payload}

    err_400 = {
        "message": json.dumps(
            {
                "error": {
                    "message": "At most 1 image(s) may be provided in one prompt.",
                    "type": "BadRequestError",
                    "code": 400,
                }
            }
        ),
        "codex_error_info": "other",
    }
    err_500 = {
        "message": json.dumps(
            {
                "error": {
                    "message": "internal server error",
                    "type": "InternalServerError",
                    "code": 500,
                }
            }
        ),
    }
    records = (
        msg("t-good", "hi"),
        call("t-good", "call_good", '{"cmd": "ls"}'),
        out("t-good", "call_good"),
        done("t-good", None),
        # Failed turn: 4xx -> the model never accepted it; drop its calls.
        msg("t-bad", "stuck?"),
        call("t-bad", "call_bad", '{"images": ["a.png", "b.png"]}'),
        out("t-bad", "call_bad"),
        done("t-bad", err_400),
        # Server 5xx does not poison a session; its calls must survive.
        msg("t-500", "boom"),
        call("t-500", "call_500", '{"cmd": "sleep 1"}'),
        out("t-500", "call_500"),
        done("t-500", err_500),
    )
    rollout.write_text("".join(json.dumps(r) + "\n" for r in records))

    assert repair_poisoned_rollout(sid, home=tmp_path) == 2

    kept = [json.loads(line) for line in rollout.read_text().splitlines()]
    assert len(kept) == 10
    turns = [
        p.get("internal_chat_message_metadata_passthrough", {}).get("turn_id") or p.get("turn_id")
        for p in (k["payload"] for k in kept)
    ]
    # t-bad's user message+done survive; its calls are gone. t-500 is intact.
    assert turns == ["t-good"] * 4 + ["t-bad"] * 2 + ["t-500"] * 4
    payloads = [k["payload"] for k in kept]
    assert not any(p.get("call_id") == "call_bad" for p in payloads)
    assert any(p.get("call_id") == "call_good" for p in payloads)
    assert any(p.get("call_id") == "call_500" for p in payloads)

    assert repair_poisoned_rollout(sid, home=tmp_path) == 0
    assert repair_poisoned_rollout("not-a-session-id", home=tmp_path) == 0


async def test_stream_retries_after_repairing_poisoned_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    """CCProcessError carrying a 4xx codex error repairs the rollout before retry."""
    from telegram_bot.core.services.claude import (
        NOOP_STREAM_EVENT,
        CCProcessError,
        SessionManager,
    )

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    settings = Settings(
        _env_file=None,
        telegram_bot_token="test-token",
        project_root=str(tmp_path),
    )
    mgr = SessionManager(settings)

    sid = "01a0a4c2-6fe5-7fd2-ba62-23f21c5b8bf3"
    root = tmp_path / "sessions" / "2026" / "09" / "15"
    root.mkdir(parents=True)
    rollout = root / f"rollout-2026-09-15T11-10-01-{sid}.jsonl"

    def msg(turn_id: str, text: str) -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
                "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
            },
        }

    def call(turn_id: str, call_id: str, args: str) -> dict:
        return {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": call_id,
                "arguments": args,
                "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
            },
        }

    def done(turn_id: str, error: dict | None) -> dict:
        payload: dict = {"type": "task_complete", "turn_id": turn_id}
        if error is not None:
            payload["error"] = error
        return {"type": "event_msg", "payload": payload}

    err_400 = {
        "message": json.dumps(
            {
                "error": {
                    "message": "Unterminated string starting at: line 1 column 9 (char 8)",
                    "type": "BadRequestError",
                    "code": 400,
                }
            }
        ),
        "codex_error_info": "other",
    }
    records = (
        msg("t-good", "hi"),
        call("t-good", "call_good", '{"cmd": "ls"}'),
        done("t-good", None),
        msg("t-bad", "stuck?"),
        call("t-bad", "call_bad", '{"cmd": "stat -c \'%y\' /app/x.py'),
        done("t-bad", err_400),
    )
    rollout.write_text("".join(json.dumps(r) + "\n" for r in records))

    channel_key = (5371020261, None)
    session = mgr._get_session(channel_key)
    session.engine = "codex"
    session.session_id = sid

    calls = 0

    async def fake_run_cc_stream(prompt: str, sess, on_event) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            sess.last_codex_error = err_400["message"]
            raise CCProcessError(1)
        return "fixed reply"

    monkeypatch.setattr(mgr, "_run_cc_stream", fake_run_cc_stream)

    result = await mgr.send_stream(channel_key, "stuck?", NOOP_STREAM_EVENT)

    assert result == "fixed reply"
    assert calls == 2
    kept = [json.loads(line) for line in rollout.read_text().splitlines()]
    payloads = [k["payload"] for k in kept]
    assert not any(p.get("call_id") == "call_bad" for p in payloads)
    assert any(p.get("call_id") == "call_good" for p in payloads)
    assert any(
        p.get("type") == "message" and "stuck?" in json.dumps(p.get("content"))
        for p in payloads
    )


def test_codex_error_text_status_prefix() -> None:
    from telegram_bot.core.services.cc_events import _codex_error_text

    e400 = {
        "message": json.dumps(
            {
                "error": {
                    "message": "Unterminated string starting at: line 1 column 9 (char 8)",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            }
        )
    }
    assert _codex_error_text(e400) == (
        "\u26a0\ufe0f (400) Unterminated string starting at: line 1 column 9 (char 8)"
    )

    no_status = {"message": "failed to reach model server"}
    assert _codex_error_text(no_status) == "\u26a0\ufe0f failed to reach model server"

    type_only = {
        "message": json.dumps(
            {"error": {"message": "boom", "type": "InternalServerError", "code": None}}
        )
    }
    assert _codex_error_text(type_only) == "\u26a0\ufe0f (500) boom"

    assert _codex_error_text(None) == ""
    assert _codex_error_text({"message": ""}) == ""


def test_codex_mcp_tool_call_emits_mcp_notice() -> None:
    assert mcp_server_event_by_name("blender").content == "🔌 MCP: blender"

    started = json.dumps(
        {
            "type": "item.started",
            "item": {
                "id": "item_x",
                "type": "mcp_tool_call",
                "server": "blender",
                "tool": "get_scene_info",
                "arguments": {},
                "status": "in_progress",
            },
        }
    )
    evs = CODEX_ADAPTER.parse_exec_event(started).events
    assert [e.content for e in evs if e.type == "mcp"] == ["🔌 MCP: blender"]

    completed = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "id": "item_x",
                "type": "mcp_tool_call",
                "server": "blender",
                "tool": "get_scene_info",
                "status": "completed",
            },
        }
    )
    assert [e for e in CODEX_ADAPTER.parse_exec_event(completed).events if e.type == "mcp"] == []

    collab = json.dumps(
        {
            "type": "item.started",
            "item": {
                "id": "item_y",
                "type": "collab_tool_call",
                "server": None,
                "status": "in_progress",
            },
        }
    )
    assert [e for e in CODEX_ADAPTER.parse_exec_event(collab).events if e.type == "mcp"] == []


async def test_typing_keepalive_repings_until_stopped(monkeypatch) -> None:
    """The keepalive re-sends 'typing' repeatedly and stops on the event."""
    from telegram_bot.core.handlers import streaming

    monkeypatch.setattr(streaming, "_TYPING_KEEPALIVE_SEC", 0.01)

    bot = MagicMock()
    bot.send_chat_action = AsyncMock()
    stop = asyncio.Event()

    task = asyncio.create_task(
        streaming._typing_keepalive(bot, 42, None, stop)
    )
    await asyncio.sleep(0.05)  # let it re-ping more than once
    stop.set()
    await task

    assert bot.send_chat_action.call_count >= 2
    _, kwargs = bot.send_chat_action.call_args
    assert kwargs["action"] == "typing"
    assert kwargs["chat_id"] == 42
    assert kwargs["message_thread_id"] is None
