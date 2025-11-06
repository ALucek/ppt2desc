from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import pytest
from PIL import Image

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class StubLLMClient:
    """Minimal stand-in for `LLMClient` implementations."""

    def __init__(self, responses: Optional[Iterable[str]] = None, error: Optional[Exception] = None):
        self._responses = deque(responses or ["stub-response"])
        self._error = error
        self.model_name = "stub-llm"

    def generate(self, prompt: str, image_path: Path) -> str:
        if self._error:
            raise self._error
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if self._responses:
            return self._responses[0] if len(self._responses) == 1 else self._responses.popleft()
        return ""


@pytest.fixture
def stub_llm_client() -> StubLLMClient:
    """Return a simple stub LLM client with a default response."""

    return StubLLMClient()


@pytest.fixture
def make_image(tmp_path):
    """Factory that creates simple PNG images for slide processing tests."""

    def _make_image(filename: str = "slide_1.png", size: tuple[int, int] = (64, 64)) -> Path:
        image_path = tmp_path / filename
        Image.new("RGB", size, (255, 255, 255)).save(image_path)
        return image_path

    return _make_image

