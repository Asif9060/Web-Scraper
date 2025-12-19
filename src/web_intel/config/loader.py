"""
Configuration loader with YAML file support and environment variable overrides.

Supports loading from:
1. Default values (defined in settings.py)
2. YAML configuration file
3. Environment variables (highest priority)

Environment variables use the pattern: WEB_INTEL__{SECTION}__{KEY}
Example: WEB_INTEL__CRAWLER__MAX_PAGES=1000
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from web_intel.config.settings import Settings


# Module-level settings cache
_settings_instance: Settings | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary to merge into
        override: Dictionary with values to override

    Returns:
        Merged dictionary with override values taking precedence
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _parse_env_value(value: str) -> Any:
    """
    Parse environment variable string into appropriate Python type.

    Args:
        value: String value from environment variable

    Returns:
        Parsed value (bool, int, float, or string)
    """
    # Handle booleans
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # Handle None
    if value.lower() in ("none", "null", ""):
        return None

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def _load_env_overrides(prefix: str = "WEB_INTEL") -> dict[str, Any]:
    """
    Load configuration overrides from environment variables.

    Environment variables should follow the pattern:
    {PREFIX}__{SECTION}__{KEY}

    For example:
    - WEB_INTEL__CRAWLER__MAX_PAGES=1000
    - WEB_INTEL__LOGGING__LEVEL=DEBUG

    Args:
        prefix: Environment variable prefix to look for

    Returns:
        Dictionary of configuration overrides
    """
    overrides: dict[str, Any] = {}
    prefix_with_sep = f"{prefix}__"

    for key, value in os.environ.items():
        if not key.startswith(prefix_with_sep):
            continue

        # Remove prefix and split into parts
        key_path = key[len(prefix_with_sep):].lower().split("__")

        if len(key_path) < 2:
            continue

        # Build nested dictionary
        current = overrides
        for part in key_path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value
        current[key_path[-1]] = _parse_env_value(value)

    return overrides


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Dictionary of configuration values

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    # Handle empty files
    if content is None:
        return {}

    if not isinstance(content, dict):
        raise ValueError(
            f"Configuration file must contain a mapping, got: {type(content)}")

    return content


def load_config(
    config_path: Path | str | None = None,
    env_prefix: str = "WEB_INTEL",
) -> Settings:
    """
    Load configuration from YAML file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables
    2. YAML configuration file
    3. Default values

    Args:
        config_path: Path to YAML configuration file. If None, uses defaults only.
        env_prefix: Prefix for environment variables

    Returns:
        Validated Settings instance

    Raises:
        FileNotFoundError: If config_path is specified but doesn't exist
        ValidationError: If configuration values are invalid
    """
    config_data: dict[str, Any] = {}

    # Load from YAML file if provided
    if config_path is not None:
        path = Path(config_path) if isinstance(
            config_path, str) else config_path
        yaml_config = _load_yaml_file(path)
        config_data = _deep_merge(config_data, yaml_config)

    # Load environment variable overrides
    env_overrides = _load_env_overrides(env_prefix)
    config_data = _deep_merge(config_data, env_overrides)

    # Create and validate settings
    return Settings(**config_data)


def get_settings(
    config_path: Path | str | None = None,
    reload: bool = False,
) -> Settings:
    """
    Get the global Settings instance, loading it if necessary.

    This function provides a singleton-like access to configuration,
    caching the settings after first load.

    Args:
        config_path: Path to YAML configuration file (only used on first load or reload)
        reload: If True, force reload configuration from file

    Returns:
        Global Settings instance
    """
    global _settings_instance

    if _settings_instance is None or reload:
        _settings_instance = load_config(config_path)

    return _settings_instance


def reset_settings() -> None:
    """
    Reset the cached settings instance.

    Useful for testing or when configuration needs to be reloaded.
    """
    global _settings_instance
    _settings_instance = None


@lru_cache(maxsize=1)
def get_default_config_path() -> Path | None:
    """
    Find the default configuration file path.

    Searches for config.yaml in common locations:
    1. Current working directory
    2. Project root (if identifiable)
    3. User's home directory/.web_intel/

    Returns:
        Path to configuration file if found, None otherwise
    """
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config" / "config.yaml",
        Path.home() / ".web_intel" / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None
