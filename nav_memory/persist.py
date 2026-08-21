"""Save / load memory store as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .store import MemoryStore
    from .vocab import SemanticVocab


def save_memory(
    path: Union[str, Path], store: "MemoryStore", vocab: "SemanticVocab" = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "memories": store.to_serializable(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_memory(path: Union[str, Path], store: "MemoryStore") -> None:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    version = int(payload.get("version", 1))
    if version < 2:
        print(
            "[memory] WARNING legacy persistence v1: radius is treated as "
            "world-unit radius; anchor_scale_ref initializes from current pose"
        )
    store.load_serializable(payload.get("memories", []), version=version)
