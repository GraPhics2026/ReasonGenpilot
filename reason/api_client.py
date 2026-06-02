"""OpenAI-compatible MLLM client used by gen/edit/hybrid routes."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_env_file(path: str | Path = "config/.env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.lstrip("\ufeff").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(slots=True)
class MLLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "MLLMConfig":
        load_env_file()
        return cls(
            api_key=os.getenv("MLLM_API_KEY", ""),
            base_url=os.getenv("MLLM_BASE_URL", "").rstrip("/"),
            model=os.getenv("MLLM_MODEL", "gemini-2.0-flash"),
            timeout=float(os.getenv("MLLM_TIMEOUT", "60")),
        )


class MLLMClient:
    """Small wrapper around the chat-completions API.

    The base URL should point to an OpenAI-compatible root, for example:
    https://generativelanguage.googleapis.com/v1beta/openai/
    """

    def __init__(self, config: MLLMConfig | None = None) -> None:
        self.config = config or MLLMConfig.from_env()

    @property
    def configured(self) -> bool:
        return bool(self.config.api_key and self.config.base_url)

    def chat_text(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._chat(messages, temperature=temperature)

    def chat_vision(
        self,
        image_path: str | Path,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                ],
            }
        )
        return self._chat(messages, temperature=temperature)

    def _chat(self, messages: list[dict[str, Any]], temperature: float) -> str:
        if not self.configured:
            raise RuntimeError(
                "MLLM is not configured. Copy config/.env.example to config/.env "
                "and fill MLLM_API_KEY plus MLLM_BASE_URL."
            )
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx before calling the real MLLM API.") from exc

        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"MLLM API request failed: {response.status_code} {response.text[:500]}"
                ) from exc
            data = response.json()
        return data["choices"][0]["message"]["content"]


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction for model responses."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])
