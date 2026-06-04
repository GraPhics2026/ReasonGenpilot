"""Text-to-image client boundary for gen route.

The dry-run backend deliberately avoids network/GPU dependency and writes a
simple SVG image. It gives members 2-4 a stable pipeline contract while the
real T2I backend is being configured.
"""

from __future__ import annotations

import html
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Any

from .api_client import load_env_file
from .schemas import GenerationBackend

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class T2IConfig:
    backend: GenerationBackend = "dry_run"
    api_key: str = ""
    base_url: str = ""
    model: str = "black-forest-labs/FLUX.1-schnell"
    comfyui_url: str = "http://127.0.0.1:8188"

    @classmethod
    def from_env(cls) -> "T2IConfig":
        load_env_file()
        return cls(
            backend=os.getenv("T2I_BACKEND", "dry_run").lower(),  # type: ignore[arg-type]
            api_key=os.getenv("T2I_API_KEY", ""),
            base_url=os.getenv("T2I_BASE_URL", "").rstrip("/"),
            model=os.getenv("T2I_MODEL", "black-forest-labs/FLUX.1-schnell"),
            comfyui_url=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/"),
        )


class T2IClient:
    def __init__(self, config: T2IConfig | None = None) -> None:
        self.config = config or T2IConfig.from_env()

    def generate(self, prompt: str, output_path: str | Path, seed: int | None = None) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.config.backend == "dry_run":
            return self._generate_placeholder(prompt, output, seed)
        if self.config.backend == "siliconflow":
            return self._generate_siliconflow(prompt, output, seed)
        if self.config.backend == "dashscope":
            return self._generate_dashscope(prompt, output, seed)
        if self.config.backend == "comfyui":
            return self._generate_comfyui(prompt, output, seed)
        raise ValueError(f"Unsupported T2I_BACKEND: {self.config.backend}")

    def _generate_placeholder(self, prompt: str, output: Path, seed: int | None) -> Path:
        if output.suffix.lower() != ".svg":
            output = output.with_suffix(".svg")
        lines = wrap(prompt, width=52)[:8]
        escaped_lines = [html.escape(line) for line in lines]
        text_nodes = "\n".join(
            f'<text x="48" y="{140 + i * 32}" font-size="22" fill="#1f2937">{line}</text>'
            for i, line in enumerate(escaped_lines)
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="768" viewBox="0 0 1024 768">
  <rect width="1024" height="768" fill="#f7f3ea"/>
  <rect x="32" y="32" width="960" height="704" rx="24" fill="#ffffff" stroke="#334155" stroke-width="3"/>
  <text x="48" y="82" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#0f172a">ReasonGenPilot dry-run image</text>
  <text x="48" y="112" font-family="Arial, sans-serif" font-size="16" fill="#64748b">seed={seed if seed is not None else "none"}</text>
  <g font-family="Arial, sans-serif">{text_nodes}</g>
</svg>
"""
        output.write_text(svg, encoding="utf-8")
        return output

    def _generate_siliconflow(self, prompt: str, output: Path, seed: int | None) -> Path:
        if not self.config.api_key or not self.config.base_url:
            raise RuntimeError("Set T2I_API_KEY and T2I_BASE_URL for siliconflow backend.")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx before calling the real T2I API.") from exc

        url = f"{self.config.base_url}/images/generations"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "size": "1024x1024",
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.perf_counter()
        with httpx.Client(timeout=120) as client:
            response = client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"T2I API request failed: {response.status_code} {response.text[:500]}"
                ) from exc
            data = response.json()
        elapsed = time.perf_counter() - t0
        logger.info("T2I siliconflow completed in %.2fs", elapsed)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".response.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        image_url = data["data"][0].get("url")
        image_b64 = data["data"][0].get("b64_json")
        if image_b64:
            import base64

            output.write_bytes(base64.b64decode(image_b64))
            return output
        if image_url:
            with httpx.Client(timeout=120) as client:
                image_response = client.get(image_url)
                image_response.raise_for_status()
                output.write_bytes(image_response.content)
            return output
        raise RuntimeError("T2I response did not contain url or b64_json.")

    def _generate_dashscope(self, prompt: str, output: Path, seed: int | None) -> Path:
        if not self.config.api_key or not self.config.base_url:
            raise RuntimeError("Set T2I_API_KEY and T2I_BASE_URL for dashscope backend.")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install httpx before calling the real T2I API.") from exc

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": {
                "size": "1024*1024",
                "n": 1,
                "prompt_extend": True,
            },
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
                            f"DashScope T2I request failed: {response.status_code} {response.text[:500]}"
                        ) from exc
                    data = response.json()
                    break
            except httpx.TransportError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"DashScope T2I network error: {last_error}") from last_error
        elapsed = time.perf_counter() - t0
        logger.info("T2I dashscope completed in %.2fs", elapsed)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".response.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        image_url = None
        choices = data.get("output", {}).get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", [])
            for item in content:
                if isinstance(item, dict) and item.get("image"):
                    image_url = item["image"]
                    break
        if not image_url:
            raise RuntimeError(f"DashScope response did not contain an image URL: {str(data)[:500]}")

        t0 = time.perf_counter()
        with httpx.Client(timeout=180, trust_env=False) as client:
            image_response = client.get(image_url)
            image_response.raise_for_status()
            output.write_bytes(image_response.content)
        elapsed = time.perf_counter() - t0
        logger.info("T2I image download completed in %.2fs", elapsed)
        return output

    def _generate_comfyui(self, prompt: str, output: Path, seed: int | None) -> Path:
        raise NotImplementedError(
            "ComfyUI workflow differs by local graph. Keep this adapter boundary "
            "and fill it once your ComfyUI FLUX workflow JSON is ready."
        )
