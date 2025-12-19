"""
Tests for configuration module.

Tests settings loading, validation, and environment variable overrides.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from web_intel.config import (
    Settings,
    BrowserSettings,
    CrawlerSettings,
    StorageSettings,
    LocalLLMSettings,
    load_config,
)


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings_valid(self):
        """Default settings should be valid."""
        settings = Settings()

        assert settings.browser.headless is True
        assert settings.crawler.max_pages == 500
        assert settings.storage.wal_mode is True
        assert settings.local_llm.enabled is True

    def test_browser_settings_validation(self):
        """Browser settings should validate constraints."""
        # Valid settings
        browser = BrowserSettings(timeout_ms=30000, viewport_width=1920)
        assert browser.timeout_ms == 30000
        assert browser.viewport_width == 1920

        # Invalid timeout should raise
        with pytest.raises(ValueError):
            BrowserSettings(timeout_ms=1000)  # Below minimum

    def test_crawler_settings_validation(self):
        """Crawler settings should validate constraints."""
        crawler = CrawlerSettings(max_pages=100, max_depth=5)
        assert crawler.max_pages == 100
        assert crawler.max_depth == 5

        with pytest.raises(ValueError):
            CrawlerSettings(max_pages=0)  # Must be positive

    def test_storage_settings_defaults(self):
        """Storage settings should have sensible defaults."""
        storage = StorageSettings()

        assert storage.wal_mode is True
        assert storage.vector_dimensions == 384
        assert storage.cache_size_mb > 0

    def test_settings_nested_override(self):
        """Nested settings can be overridden."""
        settings = Settings(
            browser={"headless": False, "timeout_ms": 45000},
            crawler={"max_pages": 200},
        )

        assert settings.browser.headless is False
        assert settings.browser.timeout_ms == 45000
        assert settings.crawler.max_pages == 200
        # Non-overridden should keep defaults
        assert settings.crawler.max_depth == 3


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_config_defaults(self):
        """Loading without file should use defaults."""
        settings = load_config(config_path=None)

        assert isinstance(settings, Settings)
        assert settings.browser.headless is True

    def test_load_config_from_yaml(self, temp_dir: Path):
        """Configuration should load from YAML file."""
        config_path = temp_dir / "config.yaml"
        config_data = {
            "browser": {"headless": False},
            "crawler": {"max_pages": 250},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        settings = load_config(config_path)

        assert settings.browser.headless is False
        assert settings.crawler.max_pages == 250

    def test_load_config_env_override(self, monkeypatch):
        """Environment variables should override file settings."""
        monkeypatch.setenv("WEB_INTEL__CRAWLER__MAX_PAGES", "999")

        settings = load_config(config_path=None)

        assert settings.crawler.max_pages == 999

    def test_load_config_invalid_yaml(self, temp_dir: Path):
        """Invalid YAML should raise appropriate error."""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text("{ invalid yaml content")

        with pytest.raises(Exception):  # yaml.YAMLError
            load_config(config_path)


class TestLocalLLMSettings:
    """Tests for local LLM configuration."""

    def test_default_model_name(self):
        """Default model should be Qwen."""
        llm = LocalLLMSettings()
        assert "Qwen" in llm.model_name

    def test_temperature_validation(self):
        """Temperature should be between 0 and 2."""
        llm = LocalLLMSettings(temperature=0.5)
        assert llm.temperature == 0.5

        with pytest.raises(ValueError):
            LocalLLMSettings(temperature=3.0)

    def test_max_tokens_validation(self):
        """Max tokens should be within valid range."""
        llm = LocalLLMSettings(max_new_tokens=1024)
        assert llm.max_new_tokens == 1024

        with pytest.raises(ValueError):
            LocalLLMSettings(max_new_tokens=50)  # Below minimum
