"""Tests for core.config module."""

from __future__ import annotations

import os
import pytest
from core.config import Settings, load_settings


def test_default_settings():
    settings = load_settings(config_path="nonexistent.yaml")
    assert settings.pipeline.random_seed == 42
    assert settings.llm.model == "cerebras/llama3.1-8b"
    assert settings.splitting.train_ratio + settings.splitting.val_ratio + settings.splitting.test_ratio == pytest.approx(1.0)


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = load_settings(config_path="nonexistent.yaml")
    assert settings.llm.model == "openai/gpt-4o-mini"
    assert settings.logging.level == "DEBUG"


def test_settings_programmatic_override():
    settings = load_settings(overrides={"pipeline": {"random_seed": 123}})
    assert settings.pipeline.random_seed == 123


def test_invalid_split_ratios():
    with pytest.raises(Exception):
        load_settings(overrides={"splitting": {"train_ratio": 0.5, "val_ratio": 0.5, "test_ratio": 0.5}})


def test_temperature_validation():
    with pytest.raises(Exception):
        load_settings(overrides={"llm": {"temperature": 5.0}})
