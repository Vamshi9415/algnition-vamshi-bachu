"""Gemini client helpers with round-robin API key rotation."""
from __future__ import annotations

import itertools
import os
from typing import Any

from loguru import logger
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)

try:
    from google import genai
    from google.genai import types as genai_types

    GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the dependency is absent
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    GENAI_AVAILABLE = False

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiError(RuntimeError):
    """Raised when Gemini requests cannot be completed."""


class RoundRobinKeyManager:
    """Cycle through Gemini keys in round-robin order."""

    def __init__(self, keys: list[str] | None = None) -> None:
        if keys:
            resolved = []
            seen: set[str] = set()
            for key in keys:
                stripped = key.strip()
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    resolved.append(stripped)
        else:
            resolved = self._resolve_from_env()

        if not resolved:
            raise ValueError(
                "No Gemini API keys found. Set GEMINI_API_KEY, GEMINI_API_KEYS, "
                "or GEMINI_API_KEY_1 / GEMINI_API_KEY_2 / ..."
            )

        self._keys = resolved
        self._cycle = itertools.cycle(self._keys)
        self._current = next(self._cycle)

        logger.info(
            "RoundRobinKeyManager initialised with {} key(s): [{}]",
            len(self._keys),
            ", ".join(f"...{key[-4:]}" for key in self._keys),
        )

    @staticmethod
    def _resolve_from_env() -> list[str]:
        seen: set[str] = set()
        keys: list[str] = []

        def add(value: str) -> None:
            stripped = value.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                keys.append(stripped)

        index = 1
        while True:
            value = os.getenv(f"GEMINI_API_KEY_{index}")
            if not value:
                break
            add(value)
            index += 1

        raw_keys = os.getenv("GEMINI_API_KEYS", "")
        for part in raw_keys.replace("\n", ",").split(","):
            key = part.strip()
            if key:
                add(key)

        add(os.getenv("GEMINI_API_KEY", ""))
        return keys

    @property
    def count(self) -> int:
        return len(self._keys)

    def current(self) -> str:
        return self._current

    def next(self) -> str:
        self._current = next(self._cycle)
        return self._current

    def mark_exhausted(self, key: str) -> str:
        logger.warning("Key ...{} hit quota/rate-limit - rotating to next key.", key[-4:])
        return self.next()


class GeminiLLM:
    """Thin Gemini wrapper that retries on quota errors and rotates keys."""

    def __init__(
        self,
        keys: list[str] | None = None,
        model: str | None = None,
        system_instruction: str | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.system_instruction = system_instruction
        self._clients: dict[str, Any] = {}
        self._key_manager: RoundRobinKeyManager | None = None

        if GENAI_AVAILABLE:
            self._key_manager = RoundRobinKeyManager(keys)
        elif keys:
            raise GeminiError("google-genai is not installed, but explicit keys were provided")

    def _client_for(self, key: str) -> Any:
        if key not in self._clients:
            if not GENAI_AVAILABLE:
                raise GeminiError("google-genai is not available in this environment")
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]

    def generate(self, user_content: str, system_prompt: str | None = None) -> str:
        if not self._key_manager:
            raise GeminiError("Gemini client is unavailable because no API key was configured")

        config_kwargs: dict[str, Any] = {}
        resolved_system_instruction = system_prompt or self.system_instruction
        if resolved_system_instruction:
            config_kwargs["system_instruction"] = resolved_system_instruction

        config = (
            genai_types.GenerateContentConfig(**config_kwargs)
            if config_kwargs and GENAI_AVAILABLE
            else None
        )

        max_attempts = self._key_manager.count
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            key = self._key_manager.current()
            client = self._client_for(key)

            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_content,
                    config=config,
                )
                text = getattr(response, "text", None)
                if text:
                    return text.strip()

                candidate = response.candidates[0] if getattr(response, "candidates", None) else None
                if candidate and getattr(candidate, "content", None):
                    parts = []
                    for part in candidate.content.parts:
                        if getattr(part, "text", None):
                            parts.append(part.text)
                    if parts:
                        return "".join(parts).strip()
                return ""
            except Exception as exc:  # pragma: no cover - exercised via retry tests
                if not self._is_quota_error(exc):
                    raise GeminiError(f"Gemini request failed: {exc}") from exc

                last_error = exc
                if attempt < max_attempts - 1:
                    self._key_manager.mark_exhausted(key)
                    continue

                break

        raise GeminiError(
            f"All {max_attempts} Gemini API key(s) are rate-limited. Last error: {last_error}"
        )

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status == 429:
            return True
        return "ResourceExhausted" in type(exc).__name__

    def close(self) -> None:
        for client in self._clients.values():
            close = getattr(getattr(client, "aio", None), "aclose", None)
            if callable(close):
                close()
        self._clients.clear()


GeminiClient = GeminiLLM

