from __future__ import annotations

import re
from dataclasses import dataclass


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_ESCAPE_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORKING_STATUS_RE = re.compile(
    r"(?:[\u2800-\u28ff]\s*)?\bWorking(?:\.\.\.)?",
    re.IGNORECASE,
)
VISUAL_SYMBOL_RE = re.compile(
    "["
    "\u2800-\u28ff"
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0000fe0e-\U0000fe0f"
    "]+"
)
PASSED_RE = re.compile(r"\b(\d+)\s+passed\b", re.IGNORECASE)
FAILED_RE = re.compile(r"\b(\d+)\s+failed\b", re.IGNORECASE)
TERMINAL_UI_PATTERNS = (
    re.compile(r"^pi\s+v\d", re.IGNORECASE),
    re.compile(r"^escape interrupt\b", re.IGNORECASE),
    re.compile(r"^ctrl[+/\w-]*\b", re.IGNORECASE),
    re.compile(r"^/[ \w-]*commands\b", re.IGNORECASE),
    re.compile(r"^!\s*bash\b", re.IGNORECASE),
    re.compile(r"^more$", re.IGNORECASE),
    re.compile(r"^press ctrl", re.IGNORECASE),
    re.compile(r"^pi can explain\b", re.IGNORECASE),
    re.compile(r"^extend pi\.?$", re.IGNORECASE),
    re.compile(r"^\[extensions\]$", re.IGNORECASE),
    re.compile(r"^pi-[\w-]+:", re.IGNORECASE),
    re.compile(r"^~[/\w.-]", re.IGNORECASE),
)


@dataclass(frozen=True)
class VoicePresenter:
    language: str = "ko"
    max_conversation_chars: int = 600

    def summarize(self, output: str, *, prompt: str | None = None) -> str:
        clean = self._strip_ansi(output).strip()
        if not clean:
            return ""

        modified_files = self._modified_files(clean)
        passed = self._first_int(PASSED_RE, clean)
        failed = self._first_int(FAILED_RE, clean)

        if self.language == "ko":
            return self._summarize_ko(clean, len(modified_files), passed, failed, prompt)
        return self._summarize_en(clean, len(modified_files), passed, failed, prompt)

    def _summarize_ko(
        self,
        clean: str,
        modified_count: int,
        passed: int | None,
        failed: int | None,
        prompt: str | None,
    ) -> str:
        if failed is not None and failed > 0:
            return f"테스트 {failed}개가 실패했습니다."
        if modified_count and passed is not None:
            return (
                f"파일 {modified_count}개를 수정했고, "
                f"테스트 {passed}개는 모두 통과했습니다."
            )
        if modified_count:
            return f"파일 {modified_count}개를 수정했습니다."
        if passed is not None:
            return f"테스트 {passed}개는 모두 통과했습니다."
        return self._fallback_summary(clean, prompt=prompt)

    def _summarize_en(
        self,
        clean: str,
        modified_count: int,
        passed: int | None,
        failed: int | None,
        prompt: str | None,
    ) -> str:
        if failed is not None and failed > 0:
            return f"{failed} tests failed."
        if modified_count and passed is not None:
            return f"Modified {modified_count} files and all {passed} tests passed."
        if modified_count:
            return f"Modified {modified_count} files."
        if passed is not None:
            return f"All {passed} tests passed."
        return self._fallback_summary(clean, prompt=prompt)

    def _modified_files(self, clean: str) -> list[str]:
        lines = clean.splitlines()
        files: list[str] = []
        collecting = False

        for line in lines:
            stripped = line.strip()
            if stripped.lower() == "modified:":
                collecting = True
                continue
            if collecting and stripped.startswith("- "):
                files.append(stripped[2:].strip())
                continue
            if collecting and stripped:
                break

        return files

    def _fallback_summary(self, clean: str, *, prompt: str | None = None) -> str:
        normalized_prompt = self._normalize_echo_line(prompt) if prompt else None
        blocks: list[list[str]] = []
        current_block: list[str] = []

        for line in clean.splitlines():
            stripped = line.strip()
            if not stripped:
                self._append_block(blocks, current_block)
                current_block = []
                continue
            if self._is_ignored_fallback_line(stripped, normalized_prompt):
                self._append_block(blocks, current_block)
                current_block = []
                continue

            cleaned_line = self._clean_speech_text(stripped)
            if not cleaned_line:
                continue
            if self._is_ignored_fallback_line(cleaned_line, normalized_prompt):
                self._append_block(blocks, current_block)
                current_block = []
                continue
            self._append_speech_line(current_block, cleaned_line)

        self._append_block(blocks, current_block)

        if normalized_prompt is not None:
            return self._render_conversation_blocks(blocks, normalized_prompt)

        candidate_blocks = blocks
        for block in candidate_blocks:
            summary = self._clean_speech_text(" ".join(block))
            summary = self._dedupe_streaming_repeat(summary)
            if self._is_ignored_fallback_line(summary, normalized_prompt):
                continue
            if summary:
                return summary[:220]
        return ""

    def _render_conversation_blocks(
        self,
        blocks: list[list[str]],
        normalized_prompt: str,
    ) -> str:
        rendered_blocks: list[str] = []
        for block in blocks:
            summary = self._clean_speech_text(" ".join(block))
            summary = self._dedupe_streaming_repeat(summary)
            if self._is_ignored_fallback_line(summary, normalized_prompt):
                continue
            if summary:
                rendered_blocks.append(summary)
        return self._clean_speech_text(" ".join(rendered_blocks))[
            : self.max_conversation_chars
        ]

    def _append_block(self, blocks: list[list[str]], block: list[str]) -> None:
        if block:
            blocks.append(block.copy())

    def _append_speech_line(self, block: list[str], line: str) -> None:
        if not block:
            block.append(line)
            return

        normalized_line = self._normalize_echo_line(line)
        normalized_previous = self._normalize_echo_line(block[-1])
        if normalized_line == normalized_previous:
            return
        if normalized_line.startswith(normalized_previous):
            block[-1] = line
            return
        if normalized_previous.startswith(normalized_line):
            return

        block.append(line)

    def _is_ignored_fallback_line(
        self,
        line: str,
        normalized_prompt: str | None,
    ) -> bool:
        if self._is_separator_line(line):
            return True
        if (
            normalized_prompt is not None
            and self._normalize_echo_line(line) == normalized_prompt
        ):
            return True
        if "openai-codex" in line.casefold():
            return True
        if any(pattern.search(line) for pattern in TERMINAL_UI_PATTERNS):
            return True
        return False

    def _is_separator_line(self, line: str) -> bool:
        return len(line) >= 8 and not (set(line) - set("─━-_= "))

    def _clean_speech_text(self, text: str) -> str:
        text = WORKING_STATUS_RE.sub("", text)
        text = VISUAL_SYMBOL_RE.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    def _dedupe_streaming_repeat(self, text: str) -> str:
        text = text.strip()
        for index in range(1, len(text)):
            left = text[:index].strip()
            right = text[index:].strip()
            if len(left) < 8 or len(right) <= len(left):
                continue
            if right.startswith(left):
                return right
        return text

    def _normalize_echo_line(self, line: str | None) -> str:
        if line is None:
            return ""
        return re.sub(r"\s+", " ", line.strip()).casefold()

    def _strip_ansi(self, output: str) -> str:
        output = OSC_ESCAPE_RE.sub("", output)
        output = ANSI_ESCAPE_RE.sub("", output)
        output = output.replace("\r", "\n")
        return CONTROL_RE.sub("", output)

    def _first_int(self, pattern: re.Pattern[str], clean: str) -> int | None:
        match = pattern.search(clean)
        if match is None:
            return None
        return int(match.group(1))
