from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_CONFIG: Dict[str, Any] = {
    "output_dir": "results",
    "qa_low_threshold": 6.0,
    "study_guide": {
        "max_qa_hotspots": 20,
    },
    "models": {
        "mistakes": "gemini-2.0-flash",
        "qa": "gemini-2.0-flash",
    },
    "chunking": {
        # По умолчанию выключено, чтобы сохранить прежнее поведение
        "mistakes": {
            "enabled": False,
            "chunk_duration_sec": 480,
            "overlap_sec": 20,
            "max_chunks": None,
            # Если указать число, добавится мягкий хинт на лимит элементов в ответе модели
            "max_issues_hint": None,
        },
        "qa": {
            "enabled": False,
            "chunk_duration_sec": 480,
            "overlap_sec": 20,
            "max_chunks": None,
            "max_items_hint": None,
        },
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)  # type: ignore
        else:
            base[k] = v
    return base


def _read_config_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        if path.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        # Fallback to JSON
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        cwd = Path.cwd()
        candidates.extend([
            cwd / "config.yaml",
            cwd / "config.yml",
            cwd / "config.json",
        ])
    for p in candidates:
        data = _read_config_file(p)
        if data:
            _deep_update(cfg, data)
            break
    return cfg
