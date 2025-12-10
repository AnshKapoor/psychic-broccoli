"""Configuration helpers for the backbone tracks pipeline.

Provides YAML loading and small utilities for accessing nested configuration
values with defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_nested(config: Dict[str, Any], keys: list[str], default: Any) -> Any:
    """Retrieve a nested value from a config dict with a default."""

    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
