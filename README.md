# agent-voice

Local-first voice layer for terminal coding agents.

`agent-voice` is not another hosted voice assistant. It is a thin local layer
that sits on top of terminal coding agents like Codex, Pi, and Claude Code
without modifying them.

The long-term target is a voice operating layer for coding agents:

```text
Mic
  -> Smart Turn
  -> Whisper
  -> Agent Adapter
  -> Codex / Pi / Claude Code
  -> Voice Presenter
  -> Kokoro
  -> Speaker
```

The first working target is Codex.

## Current Status

This repo currently contains the Codex MVP core:

- `PexpectAgent`: spawns a terminal agent and submits transcript text as input
- `VoicePresenter`: turns noisy agent output into short speech summaries
- `InterruptManager`: detects stop phrases while the system is speaking
- `VoiceSession`: tracks `LISTENING -> THINKING -> SPEAKING -> INTERRUPTED`
- `agent-voice codex`: starts one persistent Codex session and keeps sending
  commands to that same process

Audio capture, Smart Turn, Whisper, and Kokoro are intentionally next-step
modules. The first commit keeps the hard agent boundary stable before adding
device/audio complexity.

## Install

```bash
uv sync
```

## Run

Start a persistent Codex session:

```bash
uv run agent-voice codex
```

Use a different Codex command:

```bash
uv run agent-voice codex --agent-command "codex --model gpt-5"
```

### Connect to Pi

`agent-voice` does not have a dedicated `pi` subcommand yet. The current
fallback is to reuse the pexpect adapter and point it at the Pi CLI:

```bash
uv run agent-voice codex --agent-command pi
```

To continue the most recent Pi session, pass Pi's own session flag:

```bash
uv run agent-voice codex --agent-command "pi -c"
```

This keeps one long-lived `pi` process running and sends each transcript as
terminal input, the same way the Codex MVP works. The intended stable interface
is:

```bash
uv run agent-voice pi
uv run agent-voice pi --continue
```

That dedicated Pi adapter is not implemented yet. It should prefer Pi's RPC or
JSON event modes instead of scraping TUI text, because structured events are a
better fit for voice summaries, interrupt handling, and "what are you doing
now?" state answers.

Send one command for smoke tests or automation:

```bash
uv run agent-voice codex --once "auth 버그 고쳐"
```

## Why This Exists

Existing projects mostly solve adjacent problems:

- remote/mobile control of coding agents
- STT or TTS wrappers
- Claude Code-specific voice plugins
- output narration companions

`agent-voice` is scoped around the missing local layer:

- local-first by default
- no Realtime API dependency
- no agent internals patching
- shared adapters for Codex, Pi, and Claude Code
- Voice Presenter as a first-class module
- interrupt/barge-in as part of the core state machine

## Development

```bash
uv run pytest
```

Project notes:

- Design: `docs/superpowers/specs/2026-06-08-agent-voice-design.md`
- Implementation plan: `docs/superpowers/plans/2026-06-08-agent-voice-mvp.md`
- Handoff: `docs/handoff.md`
