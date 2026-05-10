"""
Preset Manager — save and load named simulation configurations as JSON files.

Presets are stored in a `presets/` directory next to dashboard.py.
Each preset is a single JSON file named `<preset_name>.json`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Presets directory: <project_root>/presets/
PRESETS_DIR = Path(__file__).parent / "presets"


def ensure_presets_dir() -> None:
    PRESETS_DIR.mkdir(exist_ok=True)


def _path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()
    return PRESETS_DIR / f"{safe}.json"


def list_presets() -> List[str]:
    """Return sorted list of saved preset names (without extension)."""
    ensure_presets_dir()
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def save_preset(name: str, data: Dict[str, Any]) -> None:
    """
    Persist a config dict to disk.

    Parameters
    ----------
    name : str
        Human-readable preset name.
    data : dict
        Flat dict of sidebar widget values to serialise.
    """
    ensure_presets_dir()
    if not name.strip():
        raise ValueError("Preset name cannot be empty.")
    path = _path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "config": data}, f, indent=2)


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Load a preset by name.

    Returns
    -------
    dict | None
        The saved config dict, or None if not found.
    """
    path = _path(name)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("config", {})


def delete_preset(name: str) -> bool:
    """Delete a preset file. Returns True if deleted, False if not found."""
    path = _path(name)
    if path.exists():
        path.unlink()
        return True
    return False
