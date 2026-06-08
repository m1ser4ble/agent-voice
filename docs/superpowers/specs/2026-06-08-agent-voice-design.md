# agent-voice Design

## Goal

Create a local-first voice layer for terminal coding agents. The initial target
is Codex, with Pi and Claude Code added through the same adapter boundary later.

The system should allow commands like:

```text
야 codex
auth 버그 고쳐
지금 뭐하고 있어?
테스트는?
잠깐
그 수정 취소
```

## Non-Goals

- Do not depend on the OpenAI Realtime API.
- Do not patch Codex, Pi, or Claude Code internals.
- Do not start with a mobile/remote-control product.
- Do not make STT/TTS the main architectural boundary.

## Architecture

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

## Components

### Agent Adapter

The adapter owns terminal-agent process control. Codex is launched through
`pexpect`, and transcript text is submitted with:

```python
child.send(text)
child.send("\r")
```

The adapter exposes `start`, `submit`, `read_available`, and `stop`.

### Voice Presenter

The presenter converts terminal output into speech-ready summaries. It does not
read raw diffs, stack traces, or bullet lists aloud. The first implementation is
rule-based and recognizes common output such as modified files and pytest
results.

Example:

```text
Modified:
- auth.py
- login.py

Tests:
18 passed
```

becomes:

```text
파일 2개를 수정했고, 테스트 18개는 모두 통과했습니다.
```

### Interrupt Manager

Interrupt detection runs while the system is speaking. If a stop phrase such as
`잠깐`, `멈춰`, `stop`, or `pause` is detected in `SPEAKING` state, speech should
stop and the session returns to listening.

### State Machine

```text
LISTENING
  -> THINKING
  -> SPEAKING
  -> LISTENING
```

Interrupt path:

```text
SPEAKING
  -> INTERRUPTED
  -> LISTENING
```

## MVP

The MVP proves the non-audio boundary first:

- Codex adapter
- terminal input emulation
- output collection until idle
- voice summary generation
- interrupt state primitives
- persistent CLI entry point with one long-lived Codex child process
- tests

Audio modules come after this boundary is stable.

## V1

Add:

- mic capture
- Pipecat Smart Turn
- Whisper transcription
- Kokoro TTS
- true speech interruption
- Pi and Claude Code adapters

## V2

Add agent state awareness:

```text
지금 뭐하고 있어?
```

should return a concise state answer such as:

```text
현재 테스트 실행 중입니다.
```
