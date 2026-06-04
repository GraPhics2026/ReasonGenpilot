"""OpenAI-compatible MLLM client used by gen/edit/hybrid routes."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    vision_timeout: float = 120.0
    vision_max_side: int = 1024
    vision_detail: str = "auto"
    http_referer: str = ""
    app_name: str = ""

    @classmethod
    def from_env(cls) -> "MLLMConfig":
        load_env_file()
        return cls(
            api_key=os.getenv("MLLM_API_KEY", ""),
            base_url=os.getenv("MLLM_BASE_URL", "").rstrip("/"),
            model=os.getenv("MLLM_MODEL", "gemini-2.0-flash"),
            timeout=float(os.getenv("MLLM_TIMEOUT", "60")),
            vision_timeout=float(os.getenv("MLLM_VISION_TIMEOUT", "120")),
            vision_max_side=int(os.getenv("MLLM_VISION_MAX_SIDE", "1024")),
            vision_detail=os.getenv("MLLM_VISION_DETAIL", "auto"),
            http_referer=os.getenv("MLLM_HTTP_REFERER", ""),
            app_name=os.getenv("MLLM_APP_NAME", "ReasonGenPilot"),
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
        image_url: dict[str, Any] = {
            "url": image_to_data_url(image_path, max_side=self.config.vision_max_side),
        }
        if self.config.vision_detail:
            image_url["detail"] = self.config.vision_detail
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        )
        return self._chat(messages, temperature=temperature, use_vision_timeout=True)

    def _chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        use_vision_timeout: bool = False,
    ) -> str:
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
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name
        timeout = self.config.vision_timeout if use_vision_timeout else self.config.timeout
        t0 = time.perf_counter()
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"MLLM API request failed: {response.status_code} {response.text[:500]}"
                ) from exc
            data = response.json()
        elapsed = time.perf_counter() - t0
        logger.info("MLLM %s completed in %.2fs", self.config.model, elapsed)
        return extract_message_content(data["choices"][0]["message"])


def extract_message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        ]
        if text_parts:
            return "\n".join(text_parts).strip()
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError("MLLM returned empty content.")


def image_to_data_url(image_path: str | Path, max_side: int = 1024) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    image_bytes = path.read_bytes()
    if max_side > 0:
        image_bytes = _maybe_resize_image(image_bytes, max_side=max_side, mime_type=mime_type)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _maybe_resize_image(image_bytes: bytes, max_side: int, mime_type: str) -> bytes:
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return image_bytes

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        if max(width, height) <= max_side:
            return image_bytes
        scale = max_side / float(max(width, height))
        resized = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
        if resized.mode not in ("RGB", "RGBA"):
            resized = resized.convert("RGB")
            save_mime = "image/jpeg"
        else:
            save_mime = mime_type if mime_type in {"image/jpeg", "image/png", "image/webp"} else "image/jpeg"
        buffer = BytesIO()
        save_format = "PNG" if save_mime == "image/png" else "JPEG"
        if save_format == "JPEG" and resized.mode == "RGBA":
            resized = resized.convert("RGB")
        resized.save(buffer, format=save_format, quality=85)
        return buffer.getvalue()


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
