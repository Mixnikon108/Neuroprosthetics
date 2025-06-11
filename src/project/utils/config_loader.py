"""
config_loader.py

Provides a simple utility to load and cache configuration parameters
from a YAML file (e.g., config.yml) for use throughout the application.

Why this exists:
- Centralizes settings like file paths, hyperparameters, or system flags.
- Avoids hardcoding configuration inside .py files.
- Loads the config only once (caching), improving performance and consistency.
"""

import yaml                                   # PyYAML library to parse YAML files
from pathlib import Path                      # For OS-independent path handling
from typing import Any, Dict, Optional, Union # Type hints for function and global variable

# ───────────────────────── Internal cache ─────────────────────────

_CFG: Dict[str, Any] | None = None
# Global variable to store the loaded config dictionary
# Once loaded, reused in future calls to avoid reopening the file

# ───────────────────────── Public function ─────────────────────────

def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    Load & cache configuration from YAML.

    If path is None, looks in <repo_root>/config/config.yml.
    Otherwise loads the given file.
    """
    global _CFG
    if _CFG is None:
        # Determinar la ruta al config.yml
        if path is None:
            # __file__ = .../NPD3/src/project/utils/config_loader.py
            repo_root = Path(__file__).resolve().parents[3]
            config_file = repo_root / "config" / "config.yml"
        else:
            config_file = Path(path)

        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            _CFG = yaml.safe_load(f)

    return _CFG
