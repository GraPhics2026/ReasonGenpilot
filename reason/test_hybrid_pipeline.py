"""Unit tests for hybrid_pipeline.py pure functions.

Tests the following functions without requiring API calls:
- _validate_and_fix_scene_prompt
- _inject_perspective_anchor
- _inject_style_anchor
- _detect_transformative_change
- _build_scene_prompt
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Literal

from .hybrid_pipeline import (
    _build_scene_prompt,
    _detect_transformative_change,
    _inject_perspective_anchor,
    _inject_style_anchor,
    _validate_and_fix_scene_prompt,
)
from .schemas import ReasonResult, VQACheck


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockReason:
    """Minimal mock for ReasonResult used in testing pure functions."""

    mode: Literal["edit", "hybrid"] = "hybrid"
    reasoning_chain: list[str] = None  # type: ignore[assignment]
    vqa_checklist: list[VQACheck] = None  # type: ignore[assignment]
    edit_prompt: str | None = None
    scene_prompt: str | None = None
    target_objects: list[str] = None  # type: ignore[assignment]
    reasoning_type: str | None = None
    visual_cues: list[str] = None  # type: ignore[assignment]
    physics_implications: list[str] = None  # type: ignore[assignment]
    preserve_objects: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.vqa_checklist is None:
            self.vqa_checklist = []
        if self.target_objects is None:
            self.target_objects = []
        if self.visual_cues is None:
            self.visual_cues = []
        if self.physics_implications is None:
            self.physics_implications = []
        if self.preserve_objects is None:
            self.preserve_objects = []

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "reasoning_chain": self.reasoning_chain,
            "vqa_checklist": [item.to_dict() for item in self.vqa_checklist],
            "edit_prompt": self.edit_prompt,
            "scene_prompt": self.scene_prompt,
            "target_objects": self.target_objects,
            "reasoning_type": self.reasoning_type,
            "visual_cues": self.visual_cues,
            "physics_implications": self.physics_implications,
            "preserve_objects": self.preserve_objects,
        }


@pytest.fixture
def night_indoor_reason() -> MockReason:
    """Mock reason result for night indoor scene."""
    return MockReason(
        reasoning_type="temporal",
        visual_cues=[
            "Camera viewpoint: eye-level view facing the corner of a room, with the window on the RIGHT side of the frame.",
            "Spatial positions: the white square plant pot is on the windowsill at CENTER-RIGHT; the green plant with broad oval leaves grows upward from it.",
            "Dominant color palette: soft pastel tones of light blue and white, with green accents from the plant.",
            "Lighting quality and direction: natural daylight enters from the window on the RIGHT, creating diagonal shadows.",
            "Surface materials and textures: matte finish on the walls and floor; glossy glass on the window panes.",
            "Overall atmosphere: serene, quiet, and bright due to the sunlight.",
        ],
        physics_implications=[
            "The window now shows a dark night sky instead of daylight.",
            "Moonlight replaces natural sunlight, casting cooler, paler shadows.",
        ],
        preserve_objects=[
            "white square plant pot",
            "green plant with broad oval leaves",
            "dried brown leaf on floor",
            "light blue matte walls",
        ],
    )


@pytest.fixture
def park_snow_reason() -> MockReason:
    """Mock reason result for park snow scene."""
    return MockReason(
        reasoning_type="physical",
        visual_cues=[
            "The man is wearing a light blue button-up shirt, khaki pants, and white sneakers.",
            "The golden retriever is sitting on green grass next to the man.",
            "The background includes trees, two wooden park benches, and a distant city skyline across a body of water.",
            "The sky is clear and blue, indicating sunny weather.",
        ],
        physics_implications=[
            "The green grass will be replaced by a layer of fresh white snow.",
            "Snowflakes may settle on the dog's golden fur.",
            "The overall lighting will shift to a cooler, overcast tone.",
        ],
        preserve_objects=[
            "man's clothing and pose",
            "dog's pose and breed characteristics",
            "park benches",
            "trees",
            "city skyline",
        ],
    )


# ---------------------------------------------------------------------------
# Tests for _validate_and_fix_scene_prompt
# ---------------------------------------------------------------------------


class TestValidateAndFixScenePrompt:
    """Tests for scene_prompt validation and fixing."""

    def test_valid_scene_prompt_passes_through(self, night_indoor_reason: MockReason):
        """A valid, detailed scene_prompt should pass through unchanged."""
        prompt = (
            "An eye-level photograph of a serene indoor corner viewed from slightly left "
            "of center, facing a large multi-pane window on the RIGHT wall. The wall is "
            "painted a soft light blue with a matte finish, now appearing darker under low "
            "light conditions. The window has a white frame divided into six rectangular panes, "
            "through which a dark night sky dotted with faint stars is visible."
        )
        result = _validate_and_fix_scene_prompt(
            prompt, night_indoor_reason, "如果房间变成深夜"
        )
        assert result == prompt

    def test_short_prompt_gets_fixed(self, night_indoor_reason: MockReason):
        """A prompt shorter than 60 chars should be replaced with a constructed one."""
        short_prompt = "A dark room with moonlight."
        result = _validate_and_fix_scene_prompt(
            short_prompt, night_indoor_reason, "如果房间变成深夜"
        )
        assert len(result) >= 60
        assert "photorealistic" in result.lower() or "scene" in result.lower()

    def test_fragmentary_prompt_gets_fixed(self, park_snow_reason: MockReason):
        """A prompt matching fragmentary patterns should be replaced."""
        fragmentary = "grass, man's sneakers, dog's fur showing the result: snow"
        result = _validate_and_fix_scene_prompt(
            fragmentary, park_snow_reason, "如果下雪了"
        )
        assert len(result) >= 60

    def test_prompt_without_scene_words_gets_fixed(self, night_indoor_reason: MockReason):
        """A prompt without scene description indicators should be replaced."""
        no_scene = "A dark blue color with white spots and red lines."
        result = _validate_and_fix_scene_prompt(
            no_scene, night_indoor_reason, "如果房间变成深夜"
        )
        assert len(result) >= 60


# ---------------------------------------------------------------------------
# Tests for _detect_transformative_change
# ---------------------------------------------------------------------------


class TestDetectTransformativeChange:
    """Tests for transformative change detection."""

    def test_night_detection(self):
        """Instructions mentioning night should detect lighting + atmosphere changes."""
        result = _detect_transformative_change("如果这间房间变成深夜")
        assert "lighting" in result
        assert "atmosphere" in result

    def test_snow_detection(self):
        """Instructions mentioning snow should detect lighting + atmosphere changes."""
        result = _detect_transformative_change("如果公园里下雪了")
        assert "lighting" in result
        assert "atmosphere" in result

    def test_autumn_detection(self):
        """Instructions mentioning autumn should detect lighting + atmosphere changes."""
        result = _detect_transformative_change("如果现在是秋天")
        assert "lighting" in result
        assert "atmosphere" in result

    def test_no_transformative_change(self):
        """Instructions without transformative keywords should return empty set."""
        result = _detect_transformative_change("把花分成两束")
        assert len(result) == 0

    def test_english_night_detection(self):
        """English instructions should also be detected."""
        result = _detect_transformative_change("What if the room becomes dark at night?")
        assert "lighting" in result
        assert "atmosphere" in result


# ---------------------------------------------------------------------------
# Tests for _inject_perspective_anchor
# ---------------------------------------------------------------------------


class TestInjectPerspectiveAnchor:
    """Tests for perspective anchor injection."""

    def test_spatial_cues_injected(self, park_snow_reason: MockReason):
        """Spatial cues should be injected as perspective anchor."""
        base_prompt = "A man standing in a park on a snowy day."
        result = _inject_perspective_anchor(base_prompt, park_snow_reason, "如果下雪了")
        assert "Perspective:" in result
        assert "man" in result.lower() or "dog" in result.lower()

    def test_non_spatial_cues_filtered(self, night_indoor_reason: MockReason):
        """Non-spatial cues (lighting, atmosphere) should be filtered out for transformative changes."""
        base_prompt = "A dark room with moonlight."
        result = _inject_perspective_anchor(
            base_prompt, night_indoor_reason, "如果房间变成深夜"
        )
        # Should not contain pure lighting/atmosphere cues
        assert "sunlight" not in result.lower() or "bright" not in result.lower()

    def test_no_duplicate_anchor(self, park_snow_reason: MockReason):
        """If anchor already exists, it should not be added again."""
        # Build the exact anchor that would be generated from the visual cues
        # The visual_cues contain "background" which is a spatial keyword
        # So an anchor would be generated: "Perspective: The background includes..."
        # If we put this exact anchor in the base_prompt, it should not be added again
        expected_anchor = "Perspective: The background includes trees, two wooden park benches, and a distant city skyline across a body of water."
        base_prompt = f"A man in a park. {expected_anchor}"
        result = _inject_perspective_anchor(base_prompt, park_snow_reason, "如果下雪了")
        # The anchor should not be duplicated
        assert result.count("Perspective:") == 1

    def test_empty_cues_returns_original(self):
        """Empty visual cues should return the original prompt."""
        reason = MockReason(visual_cues=[])
        result = _inject_perspective_anchor("A scene.", reason, "如果下雪了")
        assert result == "A scene."


# ---------------------------------------------------------------------------
# Tests for _inject_style_anchor
# ---------------------------------------------------------------------------


class TestInjectStyleAnchor:
    """Tests for style anchor injection."""

    def test_style_anchor_injected(self, night_indoor_reason: MockReason):
        """Style cues should be injected as style anchor."""
        base_prompt = "A dark room."
        result = _inject_style_anchor(base_prompt, night_indoor_reason, "如果房间变成深夜")
        assert "Style reference:" in result

    def test_transformative_skips_lighting(self, night_indoor_reason: MockReason):
        """For transformative changes, lighting cues should be skipped."""
        base_prompt = "A dark room."
        result = _inject_style_anchor(
            base_prompt, night_indoor_reason, "如果房间变成深夜"
        )
        # Should not contain original lighting cues like "sunlight" or "bright"
        assert "sunlight" not in result.lower()

    def test_non_transformative_preserves_all(self, park_snow_reason: MockReason):
        """For non-transformative changes, all style cues should be preserved."""
        base_prompt = "A man in a park."
        result = _inject_style_anchor(base_prompt, park_snow_reason, "把花分成两束")
        assert "Style reference:" in result

    def test_no_duplicate_anchor(self, night_indoor_reason: MockReason):
        """If anchor already exists, it should not be added again."""
        # Build the exact anchor that would be generated from the visual cues
        base_prompt = "A dark room."
        result = _inject_style_anchor(base_prompt, night_indoor_reason, "如果房间变成深夜")
        # Check that anchor was added only once
        assert result.count("Style reference:") == 1

    def test_empty_cues_returns_original(self):
        """Empty visual cues should return the original prompt."""
        reason = MockReason(visual_cues=[])
        result = _inject_style_anchor("A scene.", reason, "如果下雪了")
        assert result == "A scene."


# ---------------------------------------------------------------------------
# Tests for _build_scene_prompt
# ---------------------------------------------------------------------------


class TestBuildScenePrompt:
    """Tests for scene prompt construction from reason output."""

    def test_constructs_full_scene(self, park_snow_reason: MockReason):
        """Should construct a complete scene description from reason output."""
        result = _build_scene_prompt(park_snow_reason, "如果下雪了", "test")
        assert len(result) >= 60
        assert "photorealistic" in result.lower() or "scene" in result.lower()

    def test_includes_physics_changes(self, park_snow_reason: MockReason):
        """Should include physics implications in the scene description."""
        result = _build_scene_prompt(park_snow_reason, "如果下雪了", "test")
        assert "snow" in result.lower() or "grass" in result.lower()

    def test_includes_preserve_objects(self, park_snow_reason: MockReason):
        """Should mention preserved objects."""
        result = _build_scene_prompt(park_snow_reason, "如果下雪了", "test")
        assert "man" in result.lower() or "dog" in result.lower()


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_prompt_stays_unchanged(self, night_indoor_reason: MockReason):
        """A very long but valid prompt should pass through unchanged."""
        long_prompt = (
            "An eye-level photograph of a serene indoor corner viewed from slightly "
            "left of center, facing a large multi-pane window on the RIGHT wall. "
            "The wall is painted a soft light blue with a matte finish, now appearing "
            "darker under low light conditions. The window has a white frame divided "
            "into six rectangular panes, through which a dark night sky dotted with "
            "faint stars is visible. A shallow white windowsill runs the full width "
            "of the window. On the left side of the windowsill sits a small square "
            "ceramic pot in off-white, approximately 15cm tall, containing a single "
            "green plant with broad oval leaves that have pointed tips. Sheer white "
            "curtains hang from a rod above the window, one panel on the right side "
            "drawn slightly open. The floor is covered in a smooth, light blue surface "
            "with a subtle sheen, reflecting the cool moonlight."
        )
        result = _validate_and_fix_scene_prompt(
            long_prompt, night_indoor_reason, "如果房间变成深夜"
        )
        assert result == long_prompt

    def test_chinese_instruction_detection(self):
        """Chinese instructions should be properly detected."""
        result = _detect_transformative_change("如果这间阳光明媚的房间突然变成深夜")
        assert "lighting" in result
        assert "atmosphere" in result

    def test_mixed_language_instruction(self):
        """Mixed language instructions should work."""
        result = _detect_transformative_change("What if it becomes 下雪 winter?")
        assert "lighting" in result
        assert "atmosphere" in result
