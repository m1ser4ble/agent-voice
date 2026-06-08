# agent-voice

Local-first voice layer for terminal coding agents.

`agent-voice` is not another hosted voice assistant. It is a thin local layer
that sits on top of terminal coding agents like Codex, Pi, and Claude Code
without modifying them.

Part of the value is assembly: choosing compatible local components, wiring
them together, and providing sane defaults for Smart Turn/VAD, Whisper, Kokoro,
and terminal-agent adapters so users do not have to hand-build a voice stack.

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
- `agent-voice codex --text`: starts one persistent Codex session and keeps
  sending keyboard text commands to that same process

Audio capture, Smart Turn, Whisper, and Kokoro are intentionally next-step
modules. The first commit keeps the hard agent boundary stable before adding
device/audio complexity.

## Install

```bash
uv sync
```

## Run

Voice mode entrypoint:

```bash
uv run agent-voice codex
```

Full voice mode is the intended default for this command, but the voice loop is
not implemented yet. The command currently exits with guidance to use `--text`.

Start the current persistent text-mode Codex session:

```bash
uv run agent-voice codex --text
```

Use a different Codex command in text mode:

```bash
uv run agent-voice codex --text --agent-command "codex --model gpt-5"
```

### Connect to Pi

`agent-voice` does not have a dedicated `pi` subcommand yet. The current
fallback is to reuse the pexpect adapter and point it at the Pi CLI:

```bash
uv run agent-voice codex --text --agent-command pi
```

To continue the most recent Pi session, pass Pi's own session flag:

```bash
uv run agent-voice codex --text --agent-command "pi -c"
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
uv run agent-voice codex --text --once "auth 버그 고쳐"
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
- assembled local voice stack with sane defaults
- shared adapters for Codex, Pi, and Claude Code
- Voice Presenter as a first-class module
- interrupt/barge-in as part of the core state machine

## Development

```bash
uv run pytest
```

Run the real local provider smoke:

```bash
uv run --extra voice-onnx python scripts/provider_smoke.py
```

This installs the verified ONNX/light provider set and runs:

```text
Kokoro ONNX -> WAV -> faster-whisper -> Pipecat Smart Turn v3
```

The smoke downloads Kokoro model assets into `.cache/provider-smoke/` and uses
`faster-whisper` `tiny.en` on CPU. It is not part of the default test suite
because it downloads model files.

Project notes:

- Architecture: `docs/architecture.md`
- Design: `docs/superpowers/specs/2026-06-08-agent-voice-design.md`
- Implementation plan: `docs/superpowers/plans/2026-06-08-agent-voice-mvp.md`
- Handoff: `docs/handoff.md`
