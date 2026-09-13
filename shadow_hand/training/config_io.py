from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


def dump_run_config(config: Any, run_dir: Path) -> None:
    path = run_dir / "config.json"
    path.write_text(json.dumps(dataclasses.asdict(config), indent=2, default=str))
    print(f"Run config written to {path}")


def apply_saved_config(config: Any, saved: dict, _path: str = "config") -> None:
    for f in dataclasses.fields(config):
        if f.name not in saved:
            continue
        cur = getattr(config, f.name)
        new = saved[f.name]
        if dataclasses.is_dataclass(cur):
            apply_saved_config(cur, new, f"{_path}.{f.name}")
            continue

        if isinstance(cur, tuple) and isinstance(new, list):
            new = tuple(new)
        elif isinstance(cur, list) and cur and isinstance(cur[0], tuple):
            new = [tuple(x) for x in new]

        if new != cur:
            print(f"[resume] {_path}.{f.name}: default {cur!r} -> saved {new!r}")
        setattr(config, f.name, new)


def load_saved_config(config: Any, model_path: Path) -> None:
    for candidate in (model_path.parent, model_path.parent.parent):
        saved_path = candidate / "config.json"
        if saved_path.exists():
            apply_saved_config(config, json.loads(saved_path.read_text()))
            print(f"[resume] applied saved run config from {saved_path}")
            return

    print("[resume] WARNING: no config.json next to the model; using dataclass defaults.")
