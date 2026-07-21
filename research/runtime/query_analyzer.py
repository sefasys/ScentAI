from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryAnalysis:
    raw_query: str
    model_intent: str | None = None
    planner_confidence: float | None = None
    gender: str | None = None
    season: str | None = None
    time_profile: str | None = None
    wanted_accords: tuple[str, ...] = ()
    wanted_notes: tuple[str, ...] = ()
    negative_accords: tuple[str, ...] = ()
    negative_notes: tuple[str, ...] = ()
    excluded_entities: tuple[str, ...] = ()
    requested_brand: str | None = None
    reference_perfume: str | None = None
    reference_perfumes: tuple[str, ...] = ()
    reference_relation: str | None = None
    resolved_reference: str | None = None
    resolved_references: tuple[str, ...] = ()
    comparison_perfumes: tuple[str, ...] = ()
    target_perfumes: tuple[str, ...] = ()
    owned_perfumes: tuple[str, ...] = ()
    collection_gap_request: bool = False
    requested_count: int | None = None
    sort_by: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    contradictions: tuple[str, ...] = ()
    min_rating: float | None = None
    min_popularity: int | None = 100
    niche_mode: bool = False
    debug: dict[str, object] = field(default_factory=dict)


GENDER_PATTERNS = {
    "male": re.compile(r"\b(male|men|man|masculine|him|his|guy|erkek|bay)\b", re.I),
    "female": re.compile(r"\b(female|women|woman|feminine|her|hers|girl|kadın|kadin|bayan)\b", re.I),
    "unisex": re.compile(r"\b(unisex|genderless|everyone|herkes)\b", re.I),
}

SEASON_PATTERNS = {
    "spring": re.compile(r"\b(spring|ilkbahar)\b", re.I),
    "summer": re.compile(r"\b(summer|hot|heat|beach|yaz|sıcak|sicak|plaj)\b", re.I),
    "autumn": re.compile(r"\b(autumn|fall|sonbahar)\b", re.I),
    "winter": re.compile(r"\b(winter|cold|cozy|kış|kis|soğuk|soguk)\b", re.I),
}

TIME_PATTERNS = {
    "day": re.compile(r"\b(day|daily|office|work|gym|gündüz|gunduz|ofis|iş|is)\b", re.I),
    "night": re.compile(r"\b(night|evening|date|club|bar|gece|akşam|aksam|randevu)\b", re.I),
}

VIBE_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "fresh": {"accords": ("fresh", "citrus", "aromatic", "green", "aquatic"), "notes": ("bergamot", "lemon", "mint", "grapefruit")},
    "clean": {"accords": ("fresh", "musky", "soapy", "powdery", "citrus"), "notes": ("musk", "bergamot", "lavender", "iris")},
    "office": {"accords": ("fresh", "musky", "woody", "powdery", "citrus"), "avoid": ("animalic", "oud", "smoky")},
    "date": {"accords": ("vanilla", "amber", "sweet", "warm spicy", "woody"), "notes": ("vanilla", "tonka bean", "jasmine", "amber")},
    "sexy": {"accords": ("amber", "vanilla", "warm spicy", "musky", "animalic")},
    "cozy": {"accords": ("vanilla", "amber", "warm spicy", "sweet", "woody"), "notes": ("vanilla", "tonka bean", "sandalwood")},
    "dark": {"accords": ("woody", "smoky", "leather", "oud", "amber", "earthy"), "notes": ("patchouli", "incense", "oud", "tobacco")},
    "elegant": {"accords": ("iris", "powdery", "woody", "musky", "floral"), "notes": ("iris", "rose", "musk", "sandalwood")},
    "sporty": {"accords": ("citrus", "aromatic", "fresh", "aquatic", "green")},
    "sweet": {"accords": ("sweet", "vanilla", "amber", "fruity"), "notes": ("vanilla", "tonka bean", "caramel")},
    "vanilla": {"accords": ("vanilla", "sweet", "balsamic"), "notes": ("vanilla", "tonka bean")},
    "citrus": {"accords": ("citrus", "fresh", "aromatic"), "notes": ("bergamot", "lemon", "orange", "grapefruit")},
    "woody": {"accords": ("woody", "aromatic", "earthy"), "notes": ("cedar", "sandalwood", "vetiver")},
    "fougere": {"accords": ("aromatic", "lavender", "fresh spicy", "mossy", "green"), "notes": ("lavender", "oakmoss", "geranium")},
}

VIBE_ALIASES = {
    "fresh": ("fresh", "ferah", "refreshing"),
    "clean": ("clean", "temiz", "sabunsu", "soapy"),
    "office": ("office", "ofis", "work", "iş", "is"),
    "date": ("date", "date night", "randevu"),
    "sexy": ("sexy", "çekici", "cekici"),
    "cozy": ("cozy", "warm", "sıcak", "sicak"),
    "dark": ("dark", "mysterious", "karanlık", "karanlik"),
    "elegant": ("elegant", "classy", "sophisticated", "şık", "sik"),
    "sporty": ("sporty", "gym", "spor"),
    "sweet": ("sweet", "tatlı", "tatli"),
    "vanilla": ("vanilla", "vanilya", "vanilyalı", "vanilyali"),
    "citrus": ("citrus", "turunçgil", "turuncgil", "narenciye", "bergamot", "lemon", "limon"),
    "woody": ("woody", "odunsu", "wood"),
    "fougere": ("fougere", "fougère", "füjer", "fujer"),
}

KNOWN_ACCORDS = {
    "amber",
    "animalic",
    "aquatic",
    "aromatic",
    "balsamic",
    "citrus",
    "earthy",
    "floral",
    "fresh",
    "fresh spicy",
    "fruity",
    "green",
    "iris",
    "lavender",
    "leather",
    "musky",
    "oud",
    "patchouli",
    "powdery",
    "rose",
    "smoky",
    "soapy",
    "sweet",
    "tobacco",
    "vanilla",
    "warm spicy",
    "white floral",
    "woody",
}

KNOWN_NOTES = {
    "amber",
    "bergamot",
    "caramel",
    "cardamom",
    "cedar",
    "cinnamon",
    "coconut",
    "grapefruit",
    "incense",
    "iris",
    "jasmine",
    "lavender",
    "leather",
    "lemon",
    "musk",
    "oakmoss",
    "orange",
    "mandarin orange",
    "oud",
    "patchouli",
    "pink pepper",
    "pineapple",
    "rose",
    "sandalwood",
    "suede",
    "tobacco",
    "tonka bean",
    "vanilla",
    "vetiver",
}

# Some catalog brands are also ordinary scent, occasion, note, or season words.
# They must not become implicit brand requests merely because the word appears in prose.
AMBIGUOUS_BRAND_TERMS = {
    *KNOWN_ACCORDS,
    *KNOWN_NOTES,
    *VIBE_ALIASES,
    *(alias for aliases in VIBE_ALIASES.values() for alias in aliases),
    *SEASON_PATTERNS,
    *TIME_PATTERNS,
    "office",
    "date",
    "night",
    "day",
}

NEGATIVE_PATTERNS = [
    re.compile(r"\b(?:without|no|not|avoid|hate|dislike|excluding)\s+([a-zA-ZğüşöçıİĞÜŞÖÇ\s-]{2,60})", re.I),
    re.compile(r"\b(?:less|not too|not so|too much)\s+([a-zA-ZğüşöçıİĞÜŞÖÇ\s-]{2,60})", re.I),
    re.compile(r"\b([a-zA-ZğüşöçıİĞÜŞÖÇ\s-]{2,60})\s+(?:olmasın|istemiyorum|sevmiyorum)\b", re.I),
    re.compile(r"\b(?:daha az|fazla olmayan)\s+([a-zA-ZğüşöçıİĞÜŞÖÇ\s-]{2,60})", re.I),
]

ENTITY_EXCLUSION_PATTERNS = [
    re.compile(
        r"\b(?:exclude|avoid|skip|omit|without|no)\s+"
        r"(?:(?:all|any|anything)\s+)?"
        r"(?:(?:kinds?|perfumes?|fragrances?|scents?)\s+)?"
        r"(?:(?:of|from|by)\s+)?"
        r"([^.!?;\n]{2,80})",
        re.I,
    ),
    re.compile(
        r"\b([^.!?;\n]{2,80}?)\s+"
        r"(?:markasından|markasindan)(?:\s+(?:parfümleri|parfumleri|parfümlerini|parfumlerini))?\s+"
        r"(?:çıkar|cikar|önerme|onerme)",
        re.I,
    ),
]

ENTITY_NOISE_WORDS = {
    "all", "any", "anything", "brand", "brands", "kind", "kinds", "perfume", "perfumes",
    "fragrance", "fragrances", "scent", "scents", "please", "products", "options", "the",
    "a", "an", "from", "by", "of", "too", "much", "listed", "accord", "accords", "note",
    "notes", "parfum", "parfüm", "parfumler", "parfümler", "marka", "markası", "markasi",
}

BRAND_REQUEST_PATTERNS = [
    re.compile(
        r"\b([a-zA-Z0-9&.'’ğüşöçıİĞÜŞÖÇ -]{2,60}?)"
        r"(?:['’]?(?:den|dan|ten|tan))\s+"
        r"(?:(?:erkek|kadın|kadin|unisex)\s+)?"
        r"(?:parfüm(?:ü|leri|lerini)?|parfum(?:u|leri|lerini)?|koku(?:su|ları|lari)?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:perfumes?|fragrances?|scents?)\s+(?:from|by)\s+([^,.!?;\n]{2,60})",
        re.I,
    ),
    re.compile(
        r"\b(?:from|by)\s+([^,.!?;\n]{2,60}?)\s+(?:perfumes?|fragrances?|scents?)\b",
        re.I,
    ),
]

BRAND_HINT_NOISE_WORDS = {
    "a", "an", "any", "bana", "bir", "few", "give", "goster", "göster", "me", "oner",
    "öner", "please", "recommend", "some", "show", "suggest",
}

REFERENCE_PATTERNS = [
    re.compile(
        r"\b(?:(?:recommend|find|show me|give me)\s+)?"
        r"(?:something|anything|a perfume|a fragrance|a scent|perfumes?|fragrances?|scents?)\s+"
        r"(?:similar to|like|with the same vibe as)\s+"
        r"(.+?)(?=\s+(?:but|without|with less|with more|less|more|except)\b|[,.!?;\n]|$)",
        re.I,
    ),
    re.compile(
        r"\b(?:similar to|same vibe as|an alternative to|alternative to|dupe for|dupe of)\s+"
        r"(.+?)(?=\s+(?:but|without|with less|with more|less|more|except)\b|[,.!?;\n]|$)",
        re.I,
    ),
    re.compile(
        r"\b(?:if i like|i like|i love)\s+"
        r"(.+?)(?=\s+(?:what|but|and want|without|with less|with more)\b|[,.!?;\n]|$)",
        re.I,
    ),
    re.compile(
        r"\b(.+?)(?:['’]?(?:a|e|ya|ye))?\s+"
        r"(?:benzeyen|benzer|benzeri|gibi|alternatif|alternatifi|tarzı|tarzi)\b",
        re.I,
    ),
]

REFERENCE_NOISE_WORDS = {
    "a", "an", "anything", "bana", "bir", "fragrance", "koku", "parfum", "parfüm",
    "perfume", "scent", "something",
}

COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "bir": 1, "iki": 2, "üç": 3, "uc": 3, "dört": 4, "dort": 4, "beş": 5, "bes": 5,
}

SORT_PATTERNS = [
    ("rating", re.compile(r"\b(?:highest|best|top)[ -]rated\b|\ben yüksek puanlı\b", re.I)),
    ("popularity", re.compile(r"\bmost popular\b|\ben popüler\b|\ben populer\b", re.I)),
    ("year", re.compile(r"\b(?:newest|latest|most recent|new releases?)\b|\ben yeni\b", re.I)),
    ("longevity", re.compile(r"\b(?:longest[ -]lasting|best longevity|long[ -]lasting)\b|\ben kalıcı\b|\bkalıcılığı yüksek\b", re.I)),
    ("sillage", re.compile(r"\b(?:strongest sillage|best sillage|projects? (?:the )?most|strong projection)\b|\byayılımı güçlü\b|\ben güçlü yayılım\b", re.I)),
    ("value_score", re.compile(r"\b(?:best value|highest value score)\b|\bfiyat performans\b|\ben iyi değer\b", re.I)),
]

MULTI_REFERENCE_PATTERNS = [
    re.compile(
        r"\b(?:something\s+)?(?:between|a mix of|a blend of|the middle ground between)\s+"
        r"(.+?)\s+(?:and|with)\s+(.+?)(?=\s+(?:but|without|that|which)\b|[?.!;]|$)",
        re.I,
    ),
    re.compile(
        r"\b(.+?)\s+(?:ile|ve)\s+(.+?)\s+(?:arası|arasi|ortası|ortasi|karışımı|karisimi)\b",
        re.I,
    ),
]

COLLECTION_GAP_PATTERN = re.compile(
    r"\b(?:what(?:'s| is) missing from my collection|what should i add to my collection|"
    r"complete my collection|fill (?:a|the) gap in my collection|"
    r"koleksiyonumda ne eksik|koleksiyonuma ne eklemeliyim|koleksiyonumu tamamla)\b",
    re.I,
)

OWNED_PERFUME_PATTERNS = [
    re.compile(r"\b(?:my collection(?: includes| contains| has)?|owned perfumes?)\s*:\s*([^\n.!?]+)", re.I),
    re.compile(r"\bi (?:own|have)\s+([^\n.!?]+?)(?=\s+(?:what|which|recommend|suggest|and i want)\b|[.!?]|$)", re.I),
    re.compile(r"\bkoleksiyonumda\s+([^\n.!?]+?)\s+(?:var|bulunuyor)\b", re.I),
]


def analyze_query(query: str) -> QueryAnalysis:
    text = query.strip()
    lowered = text.lower()
    gender = _first_match(GENDER_PATTERNS, lowered)
    season = _first_match(SEASON_PATTERNS, lowered)
    time_profile = _first_match(TIME_PATTERNS, lowered)

    wanted_accords: set[str] = set()
    wanted_notes: set[str] = set()
    negative_accords: set[str] = set()
    negative_notes: set[str] = set()
    matched_vibes: list[str] = []
    extracted_negatives = _extract_negative_terms(lowered)
    extracted_entities = _extract_excluded_entities(lowered, extracted_negatives)
    requested_brand = _extract_requested_brand(lowered)
    reference_perfumes = _extract_multi_reference_perfumes(lowered)
    reference_perfume = None if reference_perfumes else _extract_reference_perfume(lowered)
    reference_relation = _extract_reference_relation(lowered, reference_perfume, reference_perfumes)
    comparison_perfumes = _extract_comparison_perfumes(lowered)
    owned_perfumes = extract_owned_perfumes(lowered)
    collection_gap_request = bool(COLLECTION_GAP_PATTERN.search(lowered))
    requested_count = _extract_requested_count(lowered)
    sort_by = _extract_sort_by(lowered)
    year_min, year_max = _extract_year_bounds(lowered)

    for vibe, aliases in VIBE_ALIASES.items():
        if vibe in extracted_negatives:
            continue
        if any(_contains_phrase(lowered, alias) for alias in aliases):
            matched_vibes.append(vibe)
            profile = VIBE_MAP[vibe]
            wanted_accords.update(profile.get("accords", ()))
            wanted_notes.update(profile.get("notes", ()))
            negative_accords.update(profile.get("avoid", ()))

    wanted_accords.update(term for term in KNOWN_ACCORDS if _contains_phrase(lowered, term))
    wanted_notes.update(term for term in KNOWN_NOTES if _contains_phrase(lowered, term))

    for term in extracted_negatives:
        _route_term(term, negative_accords, negative_notes)
    contradictions = _detect_contradictions(lowered, extracted_negatives)

    niche_mode = bool(re.search(r"\b(niche|obscure|unique|rare|underrated|az bilinen|niş|nis)\b", lowered, re.I))
    min_popularity = 0 if niche_mode else 100

    min_rating = None
    rating_match = re.search(r"\b(?:above|over|higher than|rating)\s+([34](?:\.\d)?)", lowered, re.I)
    if rating_match:
        min_rating = float(rating_match.group(1))

    return QueryAnalysis(
        raw_query=text,
        gender=gender,
        season=season,
        time_profile=time_profile,
        wanted_accords=tuple(sorted(wanted_accords - negative_accords)),
        wanted_notes=tuple(sorted(wanted_notes - negative_notes)),
        negative_accords=tuple(sorted(negative_accords)),
        negative_notes=tuple(sorted(negative_notes)),
        excluded_entities=tuple(sorted(extracted_entities)),
        requested_brand=requested_brand,
        reference_perfume=reference_perfume,
        reference_perfumes=tuple(reference_perfumes),
        reference_relation=reference_relation,
        comparison_perfumes=tuple(comparison_perfumes),
        owned_perfumes=tuple(owned_perfumes),
        collection_gap_request=collection_gap_request,
        requested_count=requested_count,
        sort_by=sort_by,
        year_min=year_min,
        year_max=year_max,
        contradictions=tuple(sorted(contradictions)),
        min_rating=min_rating,
        min_popularity=min_popularity,
        niche_mode=niche_mode,
        debug={
            "matched_vibes": matched_vibes,
            "reference_perfume": reference_perfume,
            "reference_perfumes": reference_perfumes,
            "reference_relation": reference_relation,
            "comparison_perfumes": comparison_perfumes,
            "owned_perfumes": owned_perfumes,
            "collection_gap_request": collection_gap_request,
            "requested_count": requested_count,
            "sort_by": sort_by,
            "year_min": year_min,
            "year_max": year_max,
            "contradictions": sorted(contradictions),
            "extracted_negatives": sorted(extracted_negatives),
            "excluded_entities": sorted(extracted_entities),
            "requested_brand": requested_brand,
        },
    )


def _first_match(patterns: dict[str, re.Pattern], text: str) -> str | None:
    for value, pattern in patterns.items():
        if pattern.search(text):
            return value
    return None


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text, re.I))


def _extract_negative_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for pattern in NEGATIVE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1).strip(" .,;:!?")
            raw = re.split(r"\b(?:but|ama)\b", raw, maxsplit=1, flags=re.I)[0]
            raw = re.sub(r"\b(?:too|çok|cok|bir|a|an|with|and|ve|ama|but)\b", " ", raw, flags=re.I)
            raw = re.sub(r"\s+", " ", raw).strip()
            for term in sorted(KNOWN_ACCORDS | KNOWN_NOTES, key=len, reverse=True):
                if _contains_phrase(raw, term):
                    terms.add(term)
            if raw in KNOWN_ACCORDS or raw in KNOWN_NOTES:
                terms.add(raw)
    return terms


def _route_term(term: str, negative_accords: set[str], negative_notes: set[str]) -> None:
    if term in KNOWN_ACCORDS:
        negative_accords.add(term)
    if term in KNOWN_NOTES:
        negative_notes.add(term)


def _extract_excluded_entities(text: str, known_negatives: set[str]) -> set[str]:
    """Extract arbitrary excluded brands/perfumes without a hardcoded brand list."""
    entities: set[str] = set()
    for pattern in ENTITY_EXCLUSION_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1).strip(" ,:-")
            raw = re.split(r"\b(?:but|ama|except|ancak|fakat)\b", raw, maxsplit=1, flags=re.I)[0]
            for chunk in re.split(r"\s*(?:,|\bor\b|\band\b|\bveya\b|\bya da\b|\bve\b)\s*", raw, flags=re.I):
                entity = _clean_entity(chunk)
                if not entity:
                    continue
                normalized = normalize_match_text(entity)
                if any(entity_matches(term, normalized) for term in known_negatives):
                    continue
                if normalized in KNOWN_ACCORDS or normalized in KNOWN_NOTES:
                    continue
                entities.add(entity)
    return entities


def _clean_entity(value: str) -> str:
    words = [word for word in value.strip().split() if normalize_match_text(word) not in ENTITY_NOISE_WORDS]
    return " ".join(words).strip(" ,:-")


def _extract_requested_brand(text: str) -> str | None:
    for pattern in BRAND_REQUEST_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        words = [
            word for word in match.group(1).strip().split()
            if normalize_match_text(word) not in BRAND_HINT_NOISE_WORDS
        ]
        hint = " ".join(words).strip(" ,:-'’")
        if hint:
            return hint
    return None


def _extract_reference_perfume(text: str) -> str | None:
    for pattern in REFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        words = [
            word for word in match.group(1).strip().split()
            if normalize_match_text(word) not in REFERENCE_NOISE_WORDS
        ]
        hint = " ".join(words).strip(" ,:-'’")
        if hint:
            return hint
    return None


def _extract_reference_relation(
    text: str,
    reference_perfume: str | None,
    reference_perfumes: list[str] | None = None,
) -> str | None:
    if not reference_perfume and not reference_perfumes:
        return None
    if re.search(r"\b(dupe|clone|alternative|what else|another brand|muadil|alternatif|başka|baska)\b", text, re.I):
        return "alternative"
    return "similar"


def _extract_multi_reference_perfumes(text: str) -> list[str]:
    for pattern in MULTI_REFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        values = [_clean_comparison_hint(match.group(index)) for index in (1, 2)]
        if all(values) and normalize_match_text(values[0]) != normalize_match_text(values[1]):
            return values
    return []


def extract_owned_perfumes(text: str) -> list[str]:
    """Parse only explicitly delimited collection lists; never guess names from prose."""
    for pattern in OWNED_PERFUME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1)
        chunks = re.split(r"\s*(?:,|;|\band\b|\bve\b)\s*", raw, flags=re.I)
        values = []
        for chunk in chunks:
            cleaned = _clean_comparison_hint(chunk)
            cleaned = re.sub(r"^(?:the|a|an|bir)\s+", "", cleaned, flags=re.I).strip()
            if cleaned and len(normalize_match_text(cleaned)) >= 3:
                values.append(cleaned)
        return values[:20]
    return []


def _extract_comparison_perfumes(text: str) -> list[str]:
    patterns = [
        re.compile(r"\bcompare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)(?:[?.!]|$)", re.I),
        re.compile(r"\bdifference between\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)", re.I),
        re.compile(r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|$)", re.I),
        re.compile(r"\b(.+?)\s+ile\s+(.+?)\s+(?:karşılaştır|karsilastir|karşılaştırır|karsilastirir)", re.I),
        re.compile(r"\b(.+?)\s+ve\s+(.+?)\s+arasındaki\s+(?:fark|farklar)", re.I),
        re.compile(r"\bwhich\s+(?:lasts longer|has (?:stronger|better) sillage|is rated higher|is better)\s*,?\s*(.+?)\s+or\s+(.+?)(?:[?.!]|$)", re.I),
        re.compile(r"\b(.+?)\s+m[ıiuü]\s+(.+?)\s+m[ıiuü]\s+(?:daha kalıcı|daha iyi|daha güçlü)", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            values = [_clean_comparison_hint(match.group(index)) for index in (1, 2)]
            if all(values):
                return values
    return []


def _clean_comparison_hint(value: str) -> str:
    value = re.sub(r"\b(?:perfume|fragrance|scent|parfüm|parfum)\b", " ", value, flags=re.I)
    return " ".join(value.strip(" ,:-").split())


def _extract_requested_count(text: str) -> int | None:
    token = r"(\d{1,2}|one|two|three|four|five|bir|iki|üç|uc|dört|dort|beş|bes)"
    patterns = [
        re.compile(rf"\b(?:recommend|suggest|give me|show me|list)\s+(?:exactly\s+)?{token}\b", re.I),
        re.compile(rf"\b(?:exactly|tam olarak)\s+{token}\b", re.I),
        re.compile(rf"\b{token}\s+(?:perfumes?|fragrances?|scents?|parfüm|parfum|koku)\b", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            raw = match.group(1).lower()
            value = int(raw) if raw.isdigit() else COUNT_WORDS.get(raw)
            if value is not None:
                return min(max(value, 1), 10)
    return None


def _extract_sort_by(text: str) -> str | None:
    for field, pattern in SORT_PATTERNS:
        if pattern.search(text):
            return field
    return None


def _extract_year_bounds(text: str) -> tuple[int | None, int | None]:
    between = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", text, re.I)
    if between:
        left, right = int(between.group(1)), int(between.group(2))
        return min(left, right), max(left, right)
    after = re.search(r"\b(?:after|since|newer than)\s+((?:19|20)\d{2})\b|((?:19|20)\d{2})\s+(?:sonrası|sonrasi)", text, re.I)
    before = re.search(r"\b(?:before|older than)\s+((?:19|20)\d{2})\b|((?:19|20)\d{2})\s+(?:öncesi|oncesi)", text, re.I)
    minimum = int(next(group for group in after.groups() if group)) + 1 if after else None
    maximum = int(next(group for group in before.groups() if group)) - 1 if before else None
    return minimum, maximum


def _detect_contradictions(text: str, negatives: set[str]) -> set[str]:
    first_negative = re.search(r"\b(?:without|no|avoid|excluding|less|not too|olmasın|istemiyorum|daha az)\b", text, re.I)
    positive_segment = text[:first_negative.start()] if first_negative else ""
    return {term for term in negatives if _contains_phrase(positive_segment, term)}


def normalize_match_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def entity_matches(entity: str, candidate_text: str, *, threshold: float = 0.82) -> bool:
    """Match user-entered brands/perfumes, including small spelling mistakes."""
    left = normalize_match_text(entity)
    right = normalize_match_text(candidate_text)
    if not left or not right:
        return False
    if left in right or right in left:
        return True

    left_tokens = left.split()
    right_tokens = right.split()
    if len(left_tokens) == 1:
        return any(difflib.SequenceMatcher(None, left, token).ratio() >= threshold for token in right_tokens)

    window_size = len(left_tokens)
    windows = [" ".join(right_tokens[index:index + window_size]) for index in range(max(1, len(right_tokens) - window_size + 1))]
    return any(difflib.SequenceMatcher(None, left, window).ratio() >= threshold for window in windows)
