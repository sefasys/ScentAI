from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TEMPLATE_LIKE_PATTERNS = [
    re.compile(r"<think|</think", re.I),
    re.compile(r"\bthink okay\b", re.I),
    re.compile(r"\bi'?m looking for\b", re.I),
    re.compile(r"\bperfect for everyday wear\b", re.I),
    re.compile(r"\bcan you suggest\b", re.I),
    re.compile(r"\bcan you recommend\b", re.I),
    re.compile(r"\bcould you suggest\b", re.I),
    re.compile(r"\bcould you recommend\b", re.I),
    re.compile(r"\blately i(?:['’]|\s)?ve been\b", re.I),
    re.compile(r"\blately i have been\b", re.I),
    re.compile(r"\blately i(?:'ve| have) been craving\b", re.I),
    re.compile(r"\bi want something\b", re.I),
]


def analyze_query_quality(path: Path) -> int:
    rows = load_rows(path)
    queries = [extract_query(row) for row in rows]
    sources = Counter((row.get("_meta") or {}).get("query_source", "no_meta") for row in rows)
    categories = Counter((row.get("_meta") or {}).get("category", "unknown") for row in rows)

    duplicate_queries = {query: count for query, count in Counter(queries).items() if count > 1}
    openings3 = Counter(opening(query, 3) for query in queries)
    openings4 = Counter(opening(query, 4) for query in queries)
    lengths = [len(query.split()) for query in queries]
    template_hits = Counter()
    for query in queries:
        for pattern in TEMPLATE_LIKE_PATTERNS:
            if pattern.search(query):
                template_hits[pattern.pattern] += 1

    print("Query quality report")
    print(f"File                  : {path}")
    print(f"Total records         : {len(rows)}")
    print(f"Duplicate queries     : {sum(count - 1 for count in duplicate_queries.values())}")
    if lengths:
        print(f"Query words           : min {min(lengths)} / avg {sum(lengths) / len(lengths):.1f} / max {max(lengths)}")

    print("\nSources:")
    for source, count in sources.most_common():
        print(f"  {source:32s}: {count}")

    print("\nCategories:")
    for category, count in categories.most_common():
        print(f"  {category:32s}: {count}")

    print("\nTop 3-word openings:")
    for text, count in openings3.most_common(12):
        print(f"  {count:4d}  {text}")

    print("\nTop 4-word openings:")
    for text, count in openings4.most_common(12):
        print(f"  {count:4d}  {text}")

    print("\nTemplate-like hits:")
    if template_hits:
        for pattern, count in template_hits.most_common():
            print(f"  {count:4d}  {pattern}")
    else:
        print("  none")

    if duplicate_queries:
        print("\nDuplicate examples:")
        for query, count in list(duplicate_queries.items())[:10]:
            print(f"  {count}x {query}")

    print("\nBy source:")
    print_source_breakdowns(rows)

    return 0


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {path}") from exc
    return rows


def extract_query(row: dict) -> str:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return ""
    user = messages[1].get("content", "")
    if "[/PERFUMES]" in user:
        return user.split("[/PERFUMES]", 1)[1].strip()
    if "[/USER PROFILE]" in user:
        return user.split("[/USER PROFILE]", 1)[1].strip()
    return user.strip()


def opening(query: str, n: int) -> str:
    words = re.findall(r"[A-Za-z']+", query.lower())[:n]
    return " ".join(words) if words else ""


def print_source_breakdowns(rows: list[dict]) -> None:
    by_source: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        source = (row.get("_meta") or {}).get("query_source", "no_meta")
        by_source[source].append(extract_query(row))

    for source, queries in sorted(by_source.items()):
        openings = Counter(opening(query, 4) for query in queries)
        duplicates = sum(count - 1 for count in Counter(queries).values() if count > 1)
        lengths = [len(query.split()) for query in queries]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        print(f"  {source}: {len(queries)} queries, duplicates={duplicates}, avg_words={avg_len:.1f}")
        for text, count in openings.most_common(5):
            print(f"    {count:4d}  {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated user query quality.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    raise SystemExit(analyze_query_quality(args.path))


if __name__ == "__main__":
    main()
