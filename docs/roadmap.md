# Roadmap

This document captures future project candidates. These are not current MVP
commitments; they are issue-sized directions that can be promoted into GitHub
issues or project milestones later.

## Guiding Constraint

`agent-voice` should stay local-first by default. API-backed modes can exist,
but they should be explicit optional profiles, not the default product path.

The current ONNX-friendly provider stack suggests a broader direction: the
voice layer can become a portable agent voice runtime, not only a wrapper around
terminal coding agents.

## Research Notes

Checked sources on 2026-06-08:

- LiteLLM docs: https://docs.litellm.ai/
- ONNX Runtime mobile docs: https://onnxruntime.ai/docs/tutorials/mobile/
- ONNX Runtime NNAPI EP docs:
  https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html
- ONNX Runtime QNN EP docs:
  https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Android XR docs: https://developer.android.com/develop/xr
- Android XR quality guidelines:
  https://developer.android.com/docs/quality-guidelines/android-xr
- Samsung Galaxy XR launch article:
  https://www.samsungmobilepress.com/articles/galaxy-xr-opening-new-worlds

### LiteLLM Findings

LiteLLM is a reasonable candidate for a generic one-shot agent adapter because
it exposes many providers through OpenAI-style input/output, supports both a
Python SDK and a proxy server, supports streaming, maps provider errors into a
more consistent exception shape, and has proxy-level auth, logging, rate-limit,
and cost-management features.

Things to decide before implementation:

- SDK first or proxy first:
  - SDK is simpler and fits a local CLI.
  - Proxy is better if mobile/XR clients or multiple devices will share one
    model gateway.
- One-shot only or short memory:
  - One-shot is easier to reason about and test.
  - Short memory makes the voice assistant more useful but introduces context
    truncation and privacy decisions.
- Stream or non-stream:
  - Non-stream is easier to fit into the current `read_available()` contract.
  - Streaming would improve perceived latency but needs presenter/speaker
    support for partial output.
- Local endpoints first or hosted providers first:
  - Local OpenAI-compatible endpoints preserve the local-first project stance.
  - Hosted providers broaden usefulness but require explicit API-key handling.
- Presenter strategy:
  - Coding-agent output needs summarization.
  - General LLM output may already be speech-like, so it might need a
    pass-through or lighter presenter profile.
- Failure semantics:
  - Decide how auth errors, rate limits, provider timeouts, and model-not-found
    errors become spoken responses.

Research tasks:

- Verify LiteLLM behavior with a fake/local OpenAI-compatible server before
  adding real hosted-provider examples.
- Check whether LiteLLM streaming chunks can be cleanly adapted to future
  incremental TTS.
- Define how model names and credentials are loaded from config profiles.
- Decide whether cost/usage callbacks belong in the core runtime or only in the
  LiteLLM profile.

### ONNX / Mobile Runtime Findings

ONNX Runtime mobile is relevant because it separates model execution from the
hardware backend through Execution Providers. The mobile docs indicate CPU is
the universal baseline; Android can use NNAPI and XNNPACK; iOS can use CoreML
and XNNPACK. The docs also call out that mobile validation must measure binary
size, model size, latency, and power consumption.

NNAPI is Android-specific and requires Android 8.1 or higher, with Android 9+
recommended for better performance. It must be explicitly registered in the
session options and has operator/support limitations. Unsupported operators can
force fallback or partition the graph, which can hurt performance.

QNN is the Qualcomm path for Snapdragon-class acceleration on Android and
Windows. It is attractive for XR/mobile hardware, but it adds SDK, build,
quantization, backend, and device-specific compatibility questions.

Things to decide before implementation:

- Baseline provider order:
  - CPU/XNNPACK first for correctness and portability.
  - NNAPI after model compatibility is known.
  - QNN only after there is a specific Snapdragon device target.
- Model packaging:
  - Keep model download/cache behavior on workstation first.
  - For mobile, decide whether models are app-bundled, downloaded on first run,
    or served by the workstation.
- Quantization target:
  - Quantization is probably required for device-local mode.
  - It must be verified per model; do not assume every provider model benefits.
- Provider health checks:
  - Check model load time, inference latency, memory, and whether hardware EP
    actually took the graph.
- Fallback behavior:
  - Decide whether mobile profiles allow CPU fallback or fail loudly when
    hardware acceleration is unavailable.

Research tasks:

- Build a provider compatibility matrix for Kokoro ONNX, Smart Turn v3 ONNX,
  and Whisper-compatible transcribers across CPU, XNNPACK, NNAPI, and QNN.
- Measure provider smoke latency and memory on at least one Android device
  before promising device-local XR.
- Investigate whether faster-whisper remains desktop-only for our purposes and
  whether an ONNX or whisper.cpp transcriber is the better mobile target.
- Define a model artifact manifest: name, version, checksum, license, expected
  sample rate, memory budget, and compatible execution providers.

### Android XR / Galaxy XR Findings

Android XR is a plausible client target because Google documents an Android XR
SDK path, familiar tooling options such as Jetpack XR SDK, Unity, Godot, Unreal,
OpenXR, and WebXR, and compatibility paths for existing Android apps on XR
headsets and wired XR glasses.

Samsung positions Galaxy XR as the first product on Android XR, built with
Google and Qualcomm, with multimodal interaction through voice, vision, and
gesture. That aligns with this project's long-term direction, but it does not
mean the full agent runtime should move onto the headset first.

Things to decide before implementation:

- Client shape:
  - Start as an Android app/panel that works on Android XR.
  - Delay fully spatial UI until the transport and voice loop are stable.
- Runtime placement:
  - Companion mode first: device handles mic/speaker/UI, workstation handles
    repo, agent process, auth, and heavy inference.
  - Device-local mode later: only after mobile ONNX profiles are validated.
- Input model:
  - Voice is primary.
  - Hand input should cover push-to-talk, stop, retry, and mute controls.
  - Gaze/gesture should not be required for the first useful version.
- Response output:
  - Decide whether the device receives synthesized audio, text summary, or both.
  - Text fallback matters for noisy environments and debugging.
- Network/privacy boundary:
  - Decide whether raw audio ever leaves the device.
  - Companion v1 can send transcripts only if on-device STT is available, or
    send audio chunks if workstation STT is the first target.

Research tasks:

- Verify Android XR microphone permissions, background audio behavior, and
  websocket reliability in headset/panel mode.
- Define the minimum mobile client: connect, wake word/push-to-talk, transcript
  event, interrupt event, state display, response playback.
- Decide whether `agent-voice` should expose a local websocket server or a
  small HTTP event API first.
- Measure acceptable round-trip latency for "잠깐" interruption from device to
  workstation to speaker stop.
- Check whether Galaxy XR-specific APIs are needed, or whether generic Android
  XR / Android app APIs are enough for the first companion.

## Decisions To Make Before Promoting To Issues

| Area | Decision | Recommended default |
| --- | --- | --- |
| LiteLLM adapter | SDK or proxy first | SDK first for CLI; proxy later for shared/mobile gateway |
| LiteLLM behavior | One-shot or short memory | One-shot first |
| LiteLLM output | Presenter or pass-through | Add a lighter presenter profile |
| API stance | Hosted APIs in default setup | No; explicit optional profile only |
| Mobile runtime | Companion or device-local first | Companion first |
| Mobile transport | WebSocket or HTTP events | WebSocket for low-latency interrupt/state events |
| Mobile STT | Device STT or workstation STT | Workstation STT first, then device transcript mode |
| ONNX mobile baseline | CPU/XNNPACK, NNAPI, or QNN | CPU/XNNPACK first; NNAPI/QNN after measurements |
| Model artifacts | Ad hoc cache or manifest | Manifest with version/checksum/license |
| Interruption | Stop TTS only or cancel agent too | Stop TTS only; agent cancel as separate command |

## Project Candidate: LiteLLM One-Shot Agent Adapter

### Idea

Add an agent adapter that sends each completed transcript to a LiteLLM-backed
chat completion instead of a persistent terminal agent such as Codex or Pi.

Target flow:

```text
Mic
  -> Smart Turn / VAD
  -> Whisper
  -> LiteLLM Agent Adapter
  -> Voice Presenter
  -> Kokoro
  -> Speaker
```

This mode would make `agent-voice` useful as a general voice interface for any
LiteLLM-compatible provider:

- local OpenAI-compatible servers
- Ollama or vLLM-style local endpoints
- hosted LLM APIs when the user explicitly configures keys
- smaller one-shot task assistants where persistent terminal sessions are not
  needed

### Why It Fits

The existing `VoiceLoop` already depends on an `Agent` protocol rather than
Codex internals. A LiteLLM adapter can satisfy the same boundary:

```python
class Agent:
    def start(self) -> None: ...
    def submit(self, text: str) -> None: ...
    def read_available(self) -> str: ...
    def stop(self) -> None: ...
```

For one-shot mode, `submit(text)` can call LiteLLM and store the response for
`read_available()`. The presenter and speaker do not need to know whether the
response came from Codex, Pi, Claude Code, or an API-backed model.

### Product Shape

Possible commands:

```bash
uv run agent-voice llm --model ollama/qwen2.5-coder
uv run agent-voice llm --model openai/<model>
uv run agent-voice llm --model anthropic/<model>
```

API-backed providers should require explicit configuration. They should not
change the default Codex/Pi/Claude Code voice layer.

### Open Questions

- Should LiteLLM mode share the same `VoicePresenter`, or should it use a
  simpler presenter because API responses are already conversational?
- Should it be one-shot only at first, or maintain short conversation history?
- Should local OpenAI-compatible endpoints be prioritized before hosted APIs?
- How should API-key configuration be documented without making hosted API use
  feel required?

### Acceptance Criteria

- `VoiceLoop` can run with a LiteLLM-backed `Agent` implementation.
- One-shot prompts return spoken summaries through the same presenter/speaker
  path.
- API configuration is optional and isolated from the default local-first
  Codex/Pi path.
- Tests cover the adapter with a fake LiteLLM client, not real network calls.

## Project Candidate: Portable ONNX Provider Profiles

### Idea

Turn the current ONNX-friendly provider smoke into explicit runtime profiles.
Kokoro ONNX and Pipecat Smart Turn v3 already point in this direction: the
voice stack can be packaged around replaceable local model artifacts rather
than hardcoded Python implementations.

Possible profiles:

- `local-cpu`: CPU-only desktop/laptop baseline
- `apple-silicon`: local acceleration where available
- `cuda`: workstation GPU profile
- `mobile-xr-companion`: mobile/XR device as mic/speaker/UI, workstation as
  agent/model runtime
- `mobile-xr-local`: device-local inference if ONNX acceleration is viable

### Why It Fits

The project value is partly assembly. Provider setup should be able to choose
compatible combinations of:

- Smart Turn / VAD provider
- Whisper or Whisper-compatible transcriber
- Kokoro or another speaker
- agent adapter
- transport layer when input/output runs on another device

### Acceptance Criteria

- Provider metadata can describe model files, capabilities, setup hints, and
  health checks.
- `agent-voice doctor` can verify the selected profile beyond the current
  baseline package, command, audio-device, and Kokoro-cache checks.
- Provider smoke tests can run per profile without requiring hosted services.

## Project Candidate: Mobile / Galaxy XR Companion Mode

### Idea

Support mobile or XR devices, such as Galaxy XR-class hardware, as a voice
front-end for the same agent voice runtime.

The recommended first version is companion mode:

```text
Galaxy XR / mobile device
  -> mic capture
  -> optional local VAD
  -> WebSocket/HTTP transport
  -> workstation agent-voice runtime
  -> agent response summary
  -> audio or text response back to device
```

This keeps the coding agent, repository checkout, auth, and heavier models on
the workstation while allowing the user to control the agent from a wearable or
mobile form factor.

Device-local mode can come later:

```text
Galaxy XR / mobile device
  -> local Smart Turn / VAD
  -> local Whisper-compatible transcriber
  -> local or remote agent adapter
  -> local Kokoro-compatible TTS
```

### Why Companion First

Companion mode avoids the hardest early constraints:

- mobile package complexity
- model file size
- thermal and battery limits
- Android audio permission and playback edge cases
- coding-agent auth and repository access on-device

It still validates the important product question: can the user comfortably
control a coding agent by voice from a mobile/XR device?

### Acceptance Criteria

- Define a transport contract for transcript events, interrupt events, state
  updates, and spoken responses.
- Add a workstation server mode that accepts remote transcript/interrupt events.
- Keep agent execution on the workstation for the first version.
- Document latency budget and privacy boundaries.
- Add fake transport tests before any real mobile client work.

## Candidate GitHub Issues

These can be copied into GitHub issues when ready:

1. Add `OneShotAgent` contract support for API-backed agents.
2. Implement `LiteLLMAgent` behind the existing `Agent` protocol.
3. Decide LiteLLM SDK vs Proxy mode for CLI and mobile/XR profiles.
4. Define LiteLLM credential and model config loading.
5. Add provider profile metadata for ONNX/local runtime combinations.
6. Add a model artifact manifest with version, checksum, license, and provider
   compatibility.
7. Build an ONNX mobile compatibility matrix for Kokoro, Smart Turn, and
   Whisper-compatible transcription.
8. Extend `agent-voice doctor` checks for selected provider profiles.
9. Define mobile/XR transport events for transcript, interrupt, state, and
   response output.
10. Prototype `mobile-xr-companion` mode with fake client tests first.
11. Measure end-to-end interruption latency for a remote mobile/XR client.

## Priority Recommendation

Do not start with mobile UI. First make the core runtime provider boundaries
real:

1. Kokoro-backed `Speaker`
2. Whisper-backed `TranscriptSource`
3. CLI voice-mode wiring for Codex
4. LiteLLM one-shot adapter
5. Transport contract for mobile/XR companion mode

This order keeps the project grounded: mobile and XR become deployment targets
for a working voice runtime instead of separate products.
