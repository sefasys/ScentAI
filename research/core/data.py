from __future__ import annotations

import json
from pathlib import Path
from typing import Any


Perfume = dict[str, Any]


def load_perfumes(path: Path) -> list[Perfume]:
    perfumes: list[Perfume] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                perfume = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
            enrich_perfume(perfume)
            perfumes.append(perfume)
    return perfumes


def enrich_perfume(perfume: Perfume) -> Perfume:
    meta = perfume.setdefault("metadata", {})
    meta["_gender_low"] = str(meta.get("gender") or "").lower()
    meta["_seasons_set"] = {s.lower() for s in (meta.get("best_seasons") or [])}
    meta["_times_set"] = {t.lower() for t in (meta.get("time_profile") or [])}
    meta["_accords_set"] = {a.lower() for a in (meta.get("accords_list") or [])}
    meta["_notes_set"] = {n.lower() for n in (meta.get("notes_list") or [])}
    return perfume


def perfume_key(perfume: Perfume) -> str:
    return f"{perfume.get('name', '').strip().lower()}::{perfume.get('brand', '').strip().lower()}"


def build_id_index(perfumes: list[Perfume]) -> dict[str, Perfume]:
    return {p["id"]: p for p in perfumes if p.get("id")}
