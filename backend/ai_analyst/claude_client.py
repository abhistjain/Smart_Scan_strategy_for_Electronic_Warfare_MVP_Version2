"""Thin Anthropic Claude wrapper (v3 add-on Section 3.1).

Design goals:
- Single `generate(prompt, max_tokens, system)` entry point.
- Model configurable via env (default: a small, fast Claude model, since this is
  narration/summarisation, not the app's core algorithm).
- FAIL GRACEFULLY: if `ANTHROPIC_API_KEY` is unset or the import/call fails, the
  rest of the app keeps working; callers get a clear "unavailable" signal and the
  UI shows "AI narration unavailable" instead of breaking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Small, fast default for narration/summarisation. Overridable via env.
DEFAULT_MODEL = "claude-haiku-4-5"


@dataclass
class AIResult:
    available: bool
    text: str
    error: str | None = None


class ClaudeClient:
    def __init__(self) -> None:
        self._client = None
        self._init_error: str | None = None
        # Read the model at instantiation so a .env loaded after import is honoured.
        self.model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._try_init()

    def _try_init(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._init_error = "ANTHROPIC_API_KEY not set"
            return
        try:
            import anthropic  # lazy import so the app runs without the package

            self._client = anthropic.Anthropic(api_key=api_key)
            self._init_error = None
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"anthropic init failed: {exc!r}"
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def status(self) -> dict:
        return {
            "available": self.available,
            "model": self.model if self.available else None,
            "reason": None if self.available else self._init_error,
        }

    def generate(
        self, prompt: str, max_tokens: int = 400, system: str | None = None
    ) -> AIResult:
        if not self.available:
            self._try_init()
        if not self.available:
            return AIResult(False, "AI narration unavailable", self._init_error)
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            # Concatenate any text blocks in the response.
            parts = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return AIResult(True, "".join(parts).strip() or "(empty response)")
        except Exception as exc:  # noqa: BLE001
            return AIResult(False, "AI narration unavailable", str(exc))


# Module-level singleton (re-reads env on first use).
_client: ClaudeClient | None = None


def get_client() -> ClaudeClient:
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client
