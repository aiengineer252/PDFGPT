"""
LLM Manager - Unified interface for Groq and Ollama LLM providers.

Supports:
- Groq API (free tier) with streaming
- Ollama (local) with streaming
- Automatic retry and fallback handling
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

import httpx
from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM interactions with support for Groq and Ollama."""

    _instance: Optional["LLMManager"] = None

    def __new__(cls) -> "LLMManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.provider = settings.LLM_PROVIDER
        self._groq_client: Optional[AsyncGroq] = None
        self._http_client: Optional[httpx.AsyncClient] = None

        if self.provider == "groq":
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
                logger.warning(
                    "Groq API key not set. Please set GROQ_API_KEY in .env file. "
                    "Get a free key at https://console.groq.com"
                )
            else:
                self._groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        elif self.provider == "ollama":
            self._http_client = httpx.AsyncClient(
                base_url=settings.OLLAMA_BASE_URL,
                timeout=120.0,
            )

        logger.info(f"LLM Manager initialized with provider: {self.provider}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that answers questions based on the provided context.",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a complete response from the LLM."""
        if self.provider == "groq":
            return await self._groq_generate(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "ollama":
            return await self._ollama_generate(prompt, system_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that answers questions based on the provided context.",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from the LLM."""
        if self.provider == "groq":
            async for token in self._groq_stream(prompt, system_prompt, temperature, max_tokens):
                yield token
        elif self.provider == "ollama":
            async for token in self._ollama_stream(prompt, system_prompt, temperature, max_tokens):
                yield token
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    # ── Groq Implementation ──────────────────────────────────────

    async def _groq_generate(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        if not self._groq_client:
            raise RuntimeError("Groq client not initialized. Check your GROQ_API_KEY.")

        for attempt in range(3):
            try:
                response = await self._groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Groq attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        return ""

    async def _groq_stream(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        if not self._groq_client:
            raise RuntimeError("Groq client not initialized. Check your GROQ_API_KEY.")

        stream = await self._groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ── Ollama Implementation ─────────────────────────────────────

    async def _ollama_generate(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        if not self._http_client:
            raise RuntimeError("Ollama HTTP client not initialized.")

        response = await self._http_client.post(
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def _ollama_stream(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        if not self._http_client:
            raise RuntimeError("Ollama HTTP client not initialized.")

        async with self._http_client.stream(
            "POST",
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        ) as response:
            import json
            async for line in response.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
