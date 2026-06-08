# agent-voice MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Codex-focused core for a local voice layer.

**Architecture:** Keep audio providers outside the first boundary. Implement the
terminal agent adapter, presenter, interrupt state machine, and CLI so audio can
be attached later without changing agent control semantics.

**Tech Stack:** Python 3.12+, `pexpect`, `pytest`, `uv`.

---

### Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `src/agent_voice/__init__.py`

- [x] Create a Python package named `agent-voice`.
- [x] Add the console script `agent-voice = agent_voice.cli:main`.
- [x] Add `pytest` as the dev dependency.

### Task 2: Core Tests

**Files:**
- Create: `tests/test_adapter.py`
- Create: `tests/test_presenter.py`
- Create: `tests/test_interrupt.py`
- Create: `tests/test_cli.py`

- [x] Test that `PexpectAgent.submit()` sends text and `\r`.
- [x] Test that `VoicePresenter` summarizes modified files and passing tests.
- [x] Test that interrupts only trigger while speaking.
- [x] Test that CLI `codex --once` submits a command and prints a summary.
- [x] Test that CLI `codex` keeps one agent session for multiple commands.

### Task 3: Core Implementation

**Files:**
- Create: `src/agent_voice/adapter.py`
- Create: `src/agent_voice/presenter.py`
- Create: `src/agent_voice/interrupt.py`
- Create: `src/agent_voice/cli.py`

- [x] Implement `PexpectAgent`.
- [x] Implement `VoicePresenter`.
- [x] Implement `InterruptManager` and `VoiceSession`.
- [x] Implement `agent-voice codex --once`.
- [x] Implement `agent-voice codex` as the persistent default loop.
- [x] Collect agent output until idle instead of reading once.

### Task 4: Verification

**Files:**
- Test: all tests

- [ ] Run `uv run pytest`.
- [ ] Run `uv run agent-voice --help`.
- [ ] Run `uv run agent-voice codex --help`.

### Task 5: Next Implementation Plan

Recommended next plan:

- [ ] Add `src/agent_voice/audio.py` with provider protocols.
- [ ] Add `src/agent_voice/loop.py` with a fake transcript-to-speaker test.
- [ ] Add Kokoro as an optional speaker after the protocol test is green.
- [ ] Add Whisper as an optional transcriber after the transcript boundary is stable.
