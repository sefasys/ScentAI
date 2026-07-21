from __future__ import annotations

import re

from research.core.data import Perfume


def parse_card_accords(card_text: str) -> list[str]:
    for line in card_text.splitlines():
        if line.startswith("Accords:"):
            accords = []
            for part in line.replace("Accords:", "", 1).split(","):
                name = re.split(r"\s*\(", part.strip())[0].strip()
                if name:
                    accords.append(name)
            return accords
    return []


def parse_card_note_groups(card_text: str) -> dict[str, list[str]]:
    groups = {"top": [], "middle": [], "base": [], "flat": []}
    mapping = {
        "Top Notes:": "top",
        "Middle Notes:": "middle",
        "Base Notes:": "base",
    }

    for line in card_text.splitlines():
        for prefix, key in mapping.items():
            if prefix in line:
                segment = line.split(prefix, 1)[1].split("|", 1)[0].strip()
                groups[key] = [n.strip() for n in segment.split(",") if n.strip()]

        if line.startswith("Notes:"):
            segment = line.replace("Notes:", "", 1).split("|", 1)[0].strip()
            groups["flat"] = [n.strip() for n in segment.split(",") if n.strip()]

    return groups


def parse_card_notes(card_text: str) -> list[str]:
    groups = parse_card_note_groups(card_text)
    return groups["top"] + groups["middle"] + groups["base"] + groups["flat"]


def parse_card_metrics(card_text: str) -> tuple[float | None, float | None, float | None]:
    for line in card_text.splitlines():
        if not line.startswith("Rating:"):
            continue
        longevity = _extract_metric(line, r"Longevity:\s*([\d.]+)/5")
        sillage = _extract_metric(line, r"Sillage:\s*([\d.]+)/4")
        value = _extract_metric(line, r"Value:\s*([\d.]+)/5")
        return longevity, sillage, value
    return None, None, None


def build_perfume_context(perfumes: list[Perfume]) -> str | None:
    cards = [p.get("card_text", "") for p in perfumes if p.get("card_text")]
    if not cards:
        return None
    return "[PERFUMES]\n" + "\n\n".join(cards) + "\n[/PERFUMES]\n\n"


def _extract_metric(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None
