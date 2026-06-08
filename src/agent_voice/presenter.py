from __future__ import annotations

import re
from dataclasses import dataclass


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PASSED_RE = re.compile(r"\b(\d+)\s+passed\b", re.IGNORECASE)
FAILED_RE = re.compile(r"\b(\d+)\s+failed\b", re.IGNORECASE)


@dataclass(frozen=True)
class VoicePresenter:
    language: str = "ko"

    def summarize(self, output: str) -> str:
        clean = self._strip_ansi(output).strip()
        if not clean:
            return ""

        modified_files = self._modified_files(clean)
        passed = self._first_int(PASSED_RE, clean)
        failed = self._first_int(FAILED_RE, clean)

        if self.language == "ko":
            return self._summarize_ko(clean, len(modified_files), passed, failed)
        return self._summarize_en(clean, len(modified_files), passed, failed)

    def _summarize_ko(
        self, clean: str, modified_count: int, passed: int | None, failed: int | None
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
        return self._first_useful_line(clean)

    def _summarize_en(
        self, clean: str, modified_count: int, passed: int | None, failed: int | None
    ) -> str:
        if failed is not None and failed > 0:
            return f"{failed} tests failed."
        if modified_count and passed is not None:
            return f"Modified {modified_count} files and all {passed} tests passed."
        if modified_count:
            return f"Modified {modified_count} files."
        if passed is not None:
            return f"All {passed} tests passed."
        return self._first_useful_line(clean)

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

    def _first_useful_line(self, clean: str) -> str:
        for line in clean.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:220]
        return ""

    def _strip_ansi(self, output: str) -> str:
        return ANSI_ESCAPE_RE.sub("", output)

    def _first_int(self, pattern: re.Pattern[str], clean: str) -> int | None:
        match = pattern.search(clean)
        if match is None:
            return None
        return int(match.group(1))
