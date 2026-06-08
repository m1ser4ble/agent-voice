# agent-voice Handoff

## Project Goal

Build an advanced local voice layer for terminal coding agents.

Target UX:

```text
"야 codex"
"auth 버그 고쳐"
"지금 뭐하고 있어?"
"테스트는?"
"잠깐"
"그 수정 취소"
```

No keyboard should be required for common coding-agent control.

## Positioning

This is not primarily STT or TTS. The important product surface is:

- `Agent Adapter`: voice transcript becomes terminal input without modifying Codex
- `Voice Presenter`: terminal output becomes a short spoken summary
- `Interrupt Manager`: user can stop speech and issue a new command
- `Agent State Awareness`: later, answer "what are you doing now?"

## Architecture

Detailed component architecture: `docs/architecture.md`

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

Current code implements the core non-audio boundary:

- `src/agent_voice/adapter.py`
- `src/agent_voice/presenter.py`
- `src/agent_voice/interrupt.py`
- `src/agent_voice/loop.py`
- `src/agent_voice/cli.py`

## MVP Scope

Implemented:

- Codex adapter via `pexpect`
- transcript-to-terminal input emulation
- Korean/English rule-based speech summaries
- session state machine
- interrupt phrase detection
- `VoiceLoop` contract with `TranscriptSource` and `Speaker` protocols
- `VoiceLoop.run_once()` command and interrupt paths covered by tests
- `VoiceLoop.run_until_idle()` batch multi-turn path covered by tests
- `VoiceLoop.run_forever()` keeps polling during user silence and exits only on
  explicit exit intent
- Runtime exit intent covered by tests: `이제 그만`, `종료`, `exit`, `quit`
- Barge-in interrupt loop covered by tests: while `speaker.say()` is playing,
  `VoiceLoop` keeps polling transcripts and calls `speaker.stop()` on `잠깐`
- Non-interrupt transcripts during `SPEAKING` are ignored to avoid echoing the
  spoken summary back into the agent as a command
- `agent-voice codex` reserved as the default voice-mode entrypoint
- `agent-voice codex --text` persistent text session
- `agent-voice codex --text --once` for smoke tests and automation
- test coverage for all current modules

Not implemented yet:

- mic-backed `TranscriptSource`
- Pipecat Smart Turn command-boundary integration
- Whisper-backed `TranscriptSource`
- Kokoro-backed `Speaker`
- `agent-voice codex` full voice-mode wiring
- true mic/Kokoro-backed barge-in E2E during audio playback
- Pi / Claude Code adapters
- agent state inspection

## Provider Smoke Result

The target local provider combination has been tested once in this repo:

```bash
uv run --extra voice-onnx python scripts/provider_smoke.py
```

Verified stack:

- Kokoro ONNX generated `.cache/provider-smoke/kokoro-smoke.wav`.
- faster-whisper `tiny.en` transcribed the generated audio as:
  `Authentication bug fixed. One test passed.`
- Pipecat Smart Turn v3.2 CPU ONNX returned `EndOfTurnState.COMPLETE`.
- Observed Smart Turn probability: `0.947`.
- Observed Smart Turn latency: tens of milliseconds in the project venv.

This is provider-level smoke, not full product E2E. The repo now has
`TranscriptSource` and `Speaker` protocols plus a test-backed `VoiceLoop`.
The loop now models runtime silence, explicit exit, and interrupt polling during
speech playback, but still needs real mic/Whisper/Kokoro implementations before
mic-to-agent voice E2E is meaningful.

## Connecting Agents

Codex is the current first-class path:

```bash
uv run agent-voice codex
```

This is now the voice-mode entrypoint. It currently reports that the voice loop
is not implemented yet. Use text mode for the currently working path:

```bash
uv run agent-voice codex --text
```

The current Pi fallback is:

```bash
uv run agent-voice codex --text --agent-command pi
uv run agent-voice codex --text --agent-command "pi -c"
```

This works because the existing `PexpectAgent` can spawn any terminal command
that accepts text input. It is a compatibility path, not the final Pi design.

The intended Pi interface is:

```bash
uv run agent-voice pi
uv run agent-voice pi --continue
```

Implement that with a Pi-specific adapter. Prefer Pi's structured RPC or JSON
event modes over TUI scraping so the voice layer can derive agent state from
events instead of raw terminal text.

## Next Session Task

Recommended next task:

1. Add a text-backed `TranscriptSource` implementation and wire `--text` through `VoiceLoop`.
2. Add a Kokoro-backed `Speaker` implementation.
3. Add a Whisper-backed `TranscriptSource`.
4. Add Smart Turn command-boundary integration.
5. Wire `agent-voice codex` to the full voice-mode path.

Keep the public contract local-first and avoid Realtime API dependency.
