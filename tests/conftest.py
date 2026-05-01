"""Shared pytest fixtures for ml-automation-cv."""
from __future__ import annotations

import pytest


@pytest.fixture
def mock_llm_response() -> dict:
    """Return an LLM-shaped dict with content field."""
    return {
        "content": "test response content"
    }


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Return 10 dict rows with id, feature_a, feature_b, and label."""
    return [
        {"id": i, "feature_a": i * 1.5, "feature_b": i * 2.0, "label": i % 2}
        for i in range(10)
    ]


@pytest.fixture
def temp_workspace(tmp_path):
    """Return a temporary workspace directory."""
    return tmp_path
