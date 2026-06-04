"""Image edit client for the edit route (instruction editing, no mask)."""

from __future__ import annotations

import html
import json
import logging
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Any

from .api_client import load_env_file
from .schemas import EditBackend

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = _non_empty_env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class EditConfig:
    backend: EditBackend = "dashscope"
    api_key: str = ""
    base_url: str = ""
    model: str = "qwen-image-2.0"
    prompt_extend: bool = False
    negative_prompt: str = "noise, grain, blur, low quality, artifacts, oversharpened"
    size: str = ""  # empty = match source image dimensions

    @classmethod
    def from_env(cls) -> "EditConfig":
        load_env_file()
        backend_raw = _non_empty_env("EDIT_BACKEND") or _non_empty_env("T2I_BACKEND") or "dashscope"
        backend = backend_raw.lower()
        if backend not in {"dry_run", "dashscope"}:
            backend = "dashscope"
        return cls(
            backend=backend,  # type: ignore[arg-type]
            api_key=_non_empty_env("EDIT_API_KEY") or _non_empty_env("T2I_API_KEY"),
            base_url=(
                _non_empty_env("EDIT_BASE_URL")
                or _non_empty_env("T2I_BASE_URL")
                or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            ).rstrip("/"),
            model=_non_empty_env("EDIT_MODEL") or _non_empty_env("T2I_MODEL") or "qwen-image-2.0",
            prompt_extend=_env_bool("EDIT_PROMPT_EXTEND", False),
            negative_prompt=_non_empty_env("EDIT_NEGATIVE_PROMPT")
            or "noise, grain, blur, low quality, artifacts, oversharpened",
            size=_non_empty_env("EDIT_SIZE"),
        )


def _non_empty_env(name: str) -> str:
    return os.getenv(name, "").strip()


class EditClient:
    def __init__(self, config: EditConfig | None = None) -> None:
        self.config = config or EditConfig.from_env()

    def edit(
        self,
        image_path: str | Path,
        prompt: str,
        output_path: str | Path,
        seed: int | None = None,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        source = Path(image_path)
        if not source.exists():
            raise FileNotFoundError(f"Edit source image not found: {source}")

        if self.config.backend == "dry_run":
            return self._edit_placeholder(source, prompt, output, seed)
        if self.config.backend == "dashscope":
            return self._edit_dashscope(source, prompt, output, seed)
        raise ValueError(f"Unsupported EDIT_BACKEND: {self.config.backend}")

    def _edit_placeholder(self, source: Path, prompt: str, output: Path, seed: int | None) -> Path:
        if output.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            output = output.with_suffix(".png")
        if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            shutil.copy2(source, output)
            return output

        lines = wrap(prompt, width=52)[:6]
        escaped_lines = [html.escape(line) for line in lines]
        text_nodes = "\n".join(
            f'<text x="48" y="{520 + i * 28}" font-size="18" fill="#1f2937">{line}</text>'
            for i, line in enumerate(escaped_lines)
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="768" viewBox="0 0 1024 768">
  <rect width="1024" height="768" fill="#eef2ff"/>
  <text x="48" y="64" font-family="Arial, sans-serif" font-size="28" font-weight="700">ReasonGenPilot edit dry-run</text>
  <text x="48" y="96" font-family="Arial, sans-serif" font-size="16" fill="#64748b">seed={seed if seed is not None else "none"}</text>
  <g font-family="Arial, sans-serif">{text_nodes}</g>
</svg>
"""
        output = output.with_suffix(".svg")
        output.write_text(svg, encoding="utf-8")
        return output

    def _edit_dashscope(self, source: Path, prompt: str, output: Path, seed: int | None) -> Path:
        if not self.config.api_key or not self.config.base_url:
            raise RuntimeError("Set EDIT_API_KEY/T2I_API_KEY and EDIT_BASE_URL/T2I_BASE_URL for dashscope edit.")

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx before calling the real edit API.") from exc

        image_data = image_to_data_url(source)
        parameters: dict[str, Any] = {
            "n": 1,
            "prompt_extend": self.config.prompt_extend,
            "watermark": False,
        }
        size = self.config.size or infer_edit_size(source)
        if size:
            parameters["size"] = size
        if self.config.negative_prompt.strip():
            parameters["negative_prompt"] = self.config.negative_prompt.strip()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_data},
                            {"text": compact_edit_prompt_for_api(prompt)},
                        ],
                    }
                ]
            },
            "parameters": parameters,
        }
        if seed is not None:
            payload["parameters"]["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        data: dict[str, Any] = {}
        t0 = time.perf_counter()
        for _ in range(2):
            try:
                with httpx.Client(timeout=180, trust_env=False) as client:
                    response = client.post(self.config.base_url, headers=headers, json=payload)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise RuntimeError(
                            f"DashScope edit request failed: {response.status_code} {response.text[:500]}"
                        ) from exc
                    data = response.json()
                    break
            except httpx.TransportError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"DashScope edit network error: {last_error}") from last_error
        elapsed = time.perf_counter() - t0
        logger.info("Edit dashscope completed in %.2fs", elapsed)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".response.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        image_url = extract_image_url(data)
        if not image_url:
            raise RuntimeError(f"DashScope edit response did not contain an image URL: {str(data)[:500]}")

        t_img = time.perf_counter()
        with httpx.Client(timeout=180, trust_env=False) as client:
            image_response = client.get(image_url)
            image_response.raise_for_status()
            output.write_bytes(image_response.content)
        elapsed = time.perf_counter() - t_img
        logger.info("Edit image download completed in %.2fs", elapsed)
        return output


def compact_edit_prompt_for_api(prompt: str) -> str:
    """Send a shorter prompt to the edit API to reduce global re-generation."""

    text = prompt.strip()
    if not text:
        return text
    for marker in (" Keep unchanged:", " Expected outcome:", " Fix these issues:", " Refine further:"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].rstrip(". ")
            break
    if len(text) > 400:
        text = text[:397].rstrip() + "..."
    return text


def infer_edit_size(image_path: Path) -> str:
    """Match output resolution to the source image (DashScope default behavior)."""

    try:
        from PIL import Image
    except ImportError:
        return "1024*1024"

    with Image.open(image_path) as image:
        width, height = image.size
    width = max(512, min(2048, width))
    height = max(512, min(2048, height))
    while width * height > 2048 * 2048:
        width = max(512, width * 9 // 10)
        height = max(512, height * 9 // 10)
    while width * height < 512 * 512:
        width = min(2048, width * 11 // 10)
        height = min(2048, height * 11 // 10)
    return f"{width}*{height}"


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = __import__("base64").b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_image_url(data: dict[str, Any]) -> str | None:
    choices = data.get("output", {}).get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                return str(item["image"])
    return None
