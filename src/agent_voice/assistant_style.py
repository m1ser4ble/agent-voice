from __future__ import annotations

ASSISTANT_STYLE_CHOICES = ("none", "jarvis-lite")
DEFAULT_ASSISTANT_STYLE = "jarvis-lite"

JARVIS_LITE_DEVELOPER_INSTRUCTIONS = """
Voice companion style:
- Speak in Korean as a calm executive assistant for a technical user.
- Keep progress commentary to one short sentence before tool work.
- Use composed acknowledgements such as "네, sir." or "알겠습니다, sir." sparingly.
- Do not imitate, claim to be, or reference any copyrighted character.
- Avoid cheerleading, jokes, exaggerated roleplay, and long apologies.
- Preserve technical accuracy over style.
""".strip()


def resolve_developer_instructions(style: str) -> str | None:
    if style == "none":
        return None
    if style == "jarvis-lite":
        return JARVIS_LITE_DEVELOPER_INSTRUCTIONS
    raise ValueError(f"unknown assistant style: {style}")
