"""Semantic vocabulary: int32 ids ↔ strings with aliases."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from .types import SID_DUST_BIN, SID_NONE, SID_WALKED


class SemanticVocab:
    def __init__(self):
        self._id_to_name: Dict[int, str] = {SID_NONE: "none"}
        self._name_to_id: Dict[str, int] = {"none": SID_NONE}
        self._next_id = 1

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SemanticVocab":
        vocab = cls()
        path = Path(path)
        if not path.exists():
            vocab._add_entry(SID_WALKED, "walked", ["walk", "footprint"])
            # object memory 尚未启用；保留实现，暂不注册入口。
            # vocab._add_entry(
            #     SID_DUST_BIN, "dust_bin", ["dustbin", "trash_can", "垃圾桶"]
            # )
            vocab._next_id = max(vocab._id_to_name.keys()) + 1
            return vocab

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for entry in data.get("entries", []):
            sid = int(entry["id"])
            name = str(entry["name"])
            aliases = entry.get("aliases") or []
            vocab._add_entry(sid, name, aliases)

        if vocab._id_to_name.keys():
            vocab._next_id = max(vocab._id_to_name.keys()) + 1
        return vocab

    def _normalize(self, name: str) -> str:
        return name.strip().lower().replace("-", "_").replace(" ", "_")

    def _add_entry(self, sid: int, name: str, aliases) -> None:
        key = self._normalize(name)
        self._id_to_name[sid] = key
        self._name_to_id[key] = sid
        for a in aliases:
            self._name_to_id[self._normalize(str(a))] = sid

    def resolve(self, semantic: Union[str, int]) -> Optional[int]:
        if isinstance(semantic, int):
            return semantic if semantic in self._id_to_name else None
        key = self._normalize(str(semantic))
        return self._name_to_id.get(key)

    def name(self, sid: int) -> str:
        return self._id_to_name.get(int(sid), f"unknown_{sid}")

    def register(self, name: str) -> int:
        key = self._normalize(name)
        if key in self._name_to_id:
            return self._name_to_id[key]
        sid = self._next_id
        self._next_id += 1
        self._add_entry(sid, key, [])
        return sid

    def __contains__(self, semantic: Union[str, int]) -> bool:
        return self.resolve(semantic) is not None
