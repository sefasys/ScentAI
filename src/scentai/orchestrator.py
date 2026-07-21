from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any


PLANNER_INTENTS = {
    "recommendation",
    "similarity",
    "alternative",
    "comparison",
    "perfume_profile",
    "exact_lookup",
    "unsupported_price",
    "unsupported_availability",
    "unsupported_medical",
    "unsupported_social_claim",
    "unsupported_layering",
}

UNSUPPORTED_ANSWERS = {
    "unsupported_price": "I do not have a live price source, so I cannot safely answer current price or budget questions.",
    "unsupported_availability": "I do not have a live stock source, so I cannot verify current availability or discontinued status.",
    "unsupported_medical": "I cannot guarantee allergy or medical safety from perfume database fields. Check the ingredient label and consult a qualified professional when needed.",
    "unsupported_social_claim": "The database does not record compliment outcomes, so I cannot make a grounded compliment claim.",
    "unsupported_layering": "The database does not contain verified layering guidance, so I cannot make a grounded layering claim.",
}

UNSUPPORTED_ANSWERS_TR = {
    "unsupported_price": "Canlı bir fiyat kaynağım olmadığı için güncel fiyat veya bütçe sorularını güvenilir biçimde yanıtlayamam.",
    "unsupported_availability": "Canlı stok kaynağım olmadığı için güncel bulunabilirlik veya üretimden kalkma durumunu doğrulayamam.",
    "unsupported_medical": "Parfüm veritabanı alanlarından alerji veya tıbbi güvenlik garantisi veremem. Gerektiğinde içerik etiketini kontrol edin ve yetkin bir uzmana danışın.",
    "unsupported_social_claim": "Veritabanı iltifat sonucunu kaydetmediği için buna dayalı güvenilir bir iddiada bulunamam.",
    "unsupported_layering": "Veritabanında doğrulanmış katmanlama bilgisi bulunmadığı için güvenilir bir katmanlama önerisi veremem.",
}

STATIC_MESSAGES = {
    "retrieval_error": {
        "en": "The perfume retrieval service could not complete this request. No recommendation was generated.",
        "tr": "Parfüm arama servisi bu isteği tamamlayamadı; bu nedenle bir öneri üretilmedi.",
    },
    "comparison_unresolved": {
        "en": "I could not resolve two distinct perfume records for a grounded comparison.",
        "tr": "Veriye dayalı bir karşılaştırma için iki farklı parfüm kaydını güvenilir biçimde eşleştiremedim.",
    },
    "no_safe_match": {
        "en": "I could not find a safe grounded match after applying the requested constraints.",
        "tr": "İstenen koşulları uyguladıktan sonra veriye dayalı güvenli bir eşleşme bulamadım.",
    },
}

PLANNER_PROMPT = """You are the semantic planner for ScentAI, a grounded perfume assistant.
Read the user's meaning rather than routing by fixed keywords. Return exactly one compact JSON object.

Schema:
{
  "intent": "recommendation|similarity|alternative|comparison|perfume_profile|exact_lookup|unsupported_price|unsupported_availability|unsupported_medical|unsupported_social_claim|unsupported_layering",
  "confidence": 0.0,
  "semantic_query": "concise English occasion, mood, and wear-context search description",
  "perfumes": [{"value": "catalog name or user wording", "evidence": "exact quote"}],
  "requested_brand": {"value": "brand", "evidence": "exact quote"},
  "gender": {"value": "male|female|unisex", "evidence": "exact quote"},
  "season": {"value": "spring|summer|autumn|winter", "evidence": "exact quote"},
  "time_profile": {"value": "day|night", "evidence": "exact quote"},
  "wanted_terms": [{"value": "accord or note", "evidence": "exact quote"}],
  "required_terms": [{"value": "required accord or note", "evidence": "exact quote"}],
  "excluded_terms": [{"value": "accord, note, brand, or perfume", "evidence": "exact quote"}],
  "requested_count": {"value": 3, "evidence": "exact quote"},
  "requested_fields": [{"value": "rating|popularity|year|longevity|sillage|value|accords|notes|seasons|time", "evidence": "exact quote"}],
  "discovery_mode": {"value": "balanced|mainstream|niche", "evidence": "exact quote"},
  "conversation_action": "new_request|more_options|refine_previous"
}

Rules:
- Omit empty fields.
- semantic_query is a model-written retrieval query, not a hard constraint. Keep occasion, mood, audience, season, and wear context; remove requested counts, perfume/brand names, required traits, and excluded traits. Example: 'date night fragrance with vanilla and warm spice' becomes 'romantic date night fragrance'.
- Every populated constraint needs an exact quote copied from the user.
- Term values must use the English database taxonomy even when evidence is in another language. For example, Turkish 'vanilya' maps to 'vanilla'; a warm/date-night use of 'baharatlı' maps to 'warm spicy'.
- Put a trait in required_terms when the user says it must be present (for example 'olsun', 'must have', or 'with vanilla'). Use wanted_terms for softer preferences.
- A noun-category request such as 'rose fragrances', 'coconut perfumes', or 'iris scents' makes that trait required. Turkish '-li/-lı' trait wording and 'X karakterli' also make it required unless the wording is explicitly soft.
- Mood and style descriptions such as clean, unusual, skin scent, softer amber, polished, or sporty are retrieval preferences, not required database traits unless the user explicitly says that exact accord or note must be present.
- A negative form such as without vanilla, less leathery, misk icermeyen, or vanilyasiz must never also appear in required_terms.
- Extract every item in a coordinated exclusion list. For 'no vanilla, oud, or smoke', return all three exclusions rather than only the first.
- discovery_mode is mainstream only when the user explicitly prioritizes famous, popular, recognizable, or mass-appeal choices; niche only when they explicitly request hidden gems, obscure, unusual, or niche discovery. Otherwise omit it and balanced will be used.
- Use conversation_action only when conversation context is supplied. more_options means the user wants different recommendations under the previous constraints. refine_previous means the user modifies the previous request. Otherwise use new_request.
- For more_options or refine_previous, extract only constraints explicitly present in the current message; the application will safely inherit the remaining previous constraints.
- Never invent a perfume, brand, or hard exclusion.
- Set requested_brand only when the wording clearly identifies a perfume house or brand. A style word such as clean, fresh, rose, musk, or office is not a brand by itself.
- similarity means smell/profile similarity; alternative means a replacement or dupe.
- comparison requires two perfumes and asks for differences, choice, or contrast.
- perfume_profile asks what one perfume is like, its vibe, performance, versatility, or wear situations.
- exact_lookup asks for literal database fields rather than interpretation.
- Any request that asks which perfume is guaranteed/certain to receive compliments or attract someone is unsupported_social_claim. Ordinary date-night or popular-fragrance advice is still recommendation.
- Do not use Markdown or explanatory text outside the JSON object."""

LEGACY_ANSWER_PROMPT = """You are ScentAI, a warm and perceptive perfume consultant.
Answer in the same language as the user. Use only the supplied database cards.

Grounding rules:
- Recommend or discuss only perfume names printed in the supplied cards.
- Never invent notes, accords, ratings, votes, years, performance, seasons, or use cases.
- Treat character and vibe as careful interpretations of the printed accords, notes, season, time, longevity, and sillage. Phrase interpretations naturally, not as certainty beyond the card.
- Obey every excluded term. Do not mention an excluded candidate even to say it was excluded.
- For recommendations, explain character, suitable setting, practical wear, and a meaningful tradeoff. Do not write like a search engine.
- For comparisons, discuss both perfumes separately, explain their different character and likely use, then summarize the practical choice. Do not choose a winner unless the user asks.
- Avoid repetitive templates such as 'it matches through', 'I would include it for', or a bare list of database fields.
- Do not use rigid section labels such as 'Neden', 'Karakter', 'Pratik Kullanım', or 'Önemli Bir Uzlaşı'. Give each recommendation one natural, compact paragraph.
- Keep the answer concise. Use numbered recommendations when recommending multiple perfumes.
- When a requested number is supplied, recommend exactly that many distinct perfumes."""


ADVISOR_ANSWER_PROMPT = """You are ScentAI, a perceptive perfume consultant helping a real person make a choice.
Answer in the same language as the user. The database cards are a verified shortlist, not a ranking you must copy.

Your job:
- First understand the experience the user is seeking: mood, identity, setting, practicality, and tradeoffs.
- Select the strongest options from the cards using holistic fit. A lower-ranked card may be the best recommendation.
- Give each recommendation a distinct role. Explain how it feels different from the other choices and who or what situation it suits.
- Turn accords, notes, season, time, longevity, and sillage into useful advice. Do not merely recite fields.
- Treat each card's longevity and sillage calibration as authoritative. Never describe a strong or very strong score as moderate, light, weak, restrained, or intimate; never describe a light or restrained score as strong or room-filling.
- If a performance value is not recorded, do not infer longevity or projection. Never convert a score into promised hours, guaranteed room projection, compliments, or social outcomes.
- Offer real decision support: indicate the safer choice, the more characterful direction, or the key compromise when those distinctions are supported by the cards.
- Use careful interpretive language for vibe and character. These are grounded readings of the recorded profile, not objective facts.
- Sound natural and engaged. Vary sentence structure and avoid repeating a fixed paragraph template.
- Write one cohesive consultation, not three database-card summaries. Later recommendations should contrast with or build on earlier ones instead of restarting the same explanation.
- Use a different grammatical opening and rhetorical purpose for every numbered item. Vary sentence count and rhythm as well as vocabulary.
- Select only the decisive evidence for each option. Do not enumerate accords and notes in every paragraph.
- Avoid stock constructions such as 'I included this because', 'this option stands out', 'bu secenegi buraya koyuyorum', 'bir adim one cikar', 'notalarla desteklenen bir yapi', and repeated 'profil ciziyor' sentences.
- In Turkish, use natural Turkish descriptions instead of calling accords 'kodlar' or ingredients 'malzemeler'. Keep exact perfume names unchanged.

Grounding boundaries:
- Recommend or discuss only exact perfume names printed in the supplied cards.
- Never invent notes, accords, ratings, votes, years, performance, seasons, availability, price, or social outcomes.
- Obey every required and excluded term. Never mention an excluded candidate, even as a warning.
- Do not assume retrieval order equals recommendation order.
- If the cards do not support a claim, leave it out.
- When comparing, discuss both perfumes' character and likely use, then explain the practical difference. Do not choose a winner unless asked.
- When recommending multiple perfumes, use a numbered list with exactly the requested number of distinct perfumes.
- Give each perfume one compact, natural paragraph. Do not use rigid labels such as Why, Character, Practical Use, Tradeoff, Neden, Karakter, or Best pick.
- End with a short, situational decision cue when useful; do not repeat the recommendation list.
- Keep a three-recommendation answer under roughly 320 words. Do not add a greeting unless the conversation genuinely calls for one."""


ANSWER_PROMPT = ADVISOR_ANSWER_PROMPT


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


TURKISH_LANGUAGE_MARKERS = {
    "aksam", "ama", "anlat", "bana", "baska", "ben", "bir", "bisey", "bu",
    "daha", "dusun", "dusunuyorsun", "erkek", "ferah", "gibi", "gunduz",
    "hakkinda", "icin", "ile", "istiyorum", "kadar", "kadin", "kis", "mi",
    "miyim", "nasil", "ne", "neden", "olan", "olarak", "olsun", "oner",
    "oneri", "sence", "tane", "ve", "veya", "yaz",
}
ENGLISH_LANGUAGE_MARKERS = {
    "a", "about", "and", "another", "are", "best", "compare", "for", "give",
    "how", "i", "in", "is", "like", "me", "more", "of", "recommend", "scent",
    "something", "than", "the", "this", "to", "what", "which", "with", "without",
}


def language_marker_scores(value: Any) -> tuple[int, int]:
    raw = str(value or "").lower()
    tokens = normalize_text(raw).split()
    turkish_score = sum(token in TURKISH_LANGUAGE_MARKERS for token in tokens)
    english_score = sum(token in ENGLISH_LANGUAGE_MARKERS for token in tokens)
    if re.search(r"[çğıöşü]", raw):
        turkish_score += 3
    return turkish_score, english_score


def detect_response_language(query: str, context: dict[str, Any] | None = None) -> str:
    """Resolve Turkish/English output locally so prompt examples cannot choose it."""
    normalized = normalize_text(query)
    if re.search(r"\b(?:answer|respond|reply) in english\b", normalized) or re.search(
        r"\bingilizce (?:cevap|yanit|yanitla|yaz)\b", normalized
    ):
        return "en"
    if re.search(r"\b(?:answer|respond|reply) in turkish\b", normalized) or re.search(
        r"\bturkce (?:cevap|yanit|yanitla|yaz)\b", normalized
    ):
        return "tr"
    turkish_score, english_score = language_marker_scores(query)
    if turkish_score > english_score:
        return "tr"
    if english_score > turkish_score:
        return "en"
    previous_language = str((context or {}).get("previous_response_language") or "").lower()
    return previous_language if previous_language in {"en", "tr"} else "en"


def output_language_instruction(language: str) -> str:
    name = "Turkish" if language == "tr" else "English"
    return (
        f"{name} ({language}) is the required response language. "
        f"Write every explanatory sentence in {name}. The language of database cards, "
        "earlier responses, examples, or taxonomy terms must not override this requirement. "
        "Keep perfume and brand names exactly as printed in the cards."
    )


def response_language_matches(answer: str, expected_language: str) -> bool:
    """Reject only clear language mismatches; catalog names may remain multilingual."""
    if len(normalize_text(answer).split()) < 8:
        return True
    turkish_score, english_score = language_marker_scores(answer)
    if expected_language == "tr":
        return not (english_score >= 4 and english_score >= turkish_score + 3)
    return not (turkish_score >= 4 and turkish_score >= english_score + 2)


def static_message(key: str, language: str) -> str:
    return STATIC_MESSAGES[key]["tr" if language == "tr" else "en"]


def unsupported_answer(intent: str, language: str) -> str | None:
    answers = UNSUPPORTED_ANSWERS_TR if language == "tr" else UNSUPPORTED_ANSWERS
    return answers.get(intent)


def parse_json_object(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    if start < 0:
        raise ValueError("Planner returned no JSON object")
    payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(payload, dict):
        raise ValueError("Planner JSON must be an object")
    return payload


def evidence_supported(evidence: Any, query: str) -> bool:
    evidence_norm = normalize_text(evidence)
    query_norm = normalize_text(query)
    return bool(evidence_norm and f" {evidence_norm} " in f" {query_norm} ")


def evidenced_values(raw: Any, query: str, *, limit: int = 20) -> list[Any]:
    values = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    output: list[Any] = []
    for item in values:
        if not isinstance(item, dict) or not evidence_supported(item.get("evidence"), query):
            continue
        value = item.get("value")
        if value is None or value == "":
            continue
        output.append(value)
        if len(output) >= limit:
            break
    return output


def first_evidenced_value(raw: Any, query: str) -> Any | None:
    values = evidenced_values(raw, query, limit=1)
    return values[0] if values else None


def explicit_brand_context(query: str, brand: str, evidence: str) -> bool:
    query_norm = normalize_text(query)
    brand_norm = normalize_text(brand)
    evidence_norm = normalize_text(evidence)
    if not brand_norm or not evidence_norm or evidence_norm not in query_norm:
        return False
    escaped = re.escape(brand_norm)
    patterns = (
        rf"\b(?:from|by|brand|house|house of)\s+{escaped}\b",
        rf"\b{escaped}\s+(?:brand|house|perfume|perfumes|fragrance|fragrances|cologne|colognes|scent|scents|parfum|parfumu|parfumleri)\b",
        rf"\b{escaped}\s+(?:den|dan|ten|tan)\b",
    )
    return any(re.search(pattern, query_norm) for pattern in patterns)


def canonical_trait_term(value: Any) -> str:
    normalized = normalize_text(value)
    aliases = {
        "vanilya": "vanilla",
        "vanila": "vanilla",
        "baharatli": "spicy",
        "baharatl": "spicy",
        "baharat": "spicy",
        "tutun": "tobacco",
        "deri": "leather",
        "misk": "musk",
        "duman": "smoky",
        "dumanli": "smoky",
        "gul": "rose",
        "tatli": "sweet",
        "paculi": "patchouli",
        "tutsu": "incense",
        "kahve": "coffee",
        "karamel": "caramel",
        "ananas": "pineapple",
        "hindistan cevizi": "coconut",
        "kehribar": "amber",
        "aromatik": "aromatic",
        "hayvansi": "animalic",
        "tarcin": "cinnamon",
        "pudrali": "powdery",
        "yesil": "green",
        "ciceksi": "floral",
        "odunsu": "woody",
        "narenciye": "citrus",
        "leathery": "leather",
        "musky": "musk",
        "smoke": "smoky",
        "agarwood": "oud",
        "aoud": "oud",
    }
    return aliases.get(normalized, normalized)


def hard_trait_requested(query: str, evidence: Any, trait: Any = None) -> bool:
    """Recover explicit hard-positive wording without making soft tastes strict."""
    query_norm = normalize_text(query)
    evidence_norm = normalize_text(evidence)
    trait_norm = canonical_trait_term(trait if trait is not None else evidence)
    if not evidence_norm or evidence_norm not in query_norm:
        return False
    negative_patterns = (
        r"\b(?:without|avoid|avoiding|exclude|excluding|less|olmayan|icermeyen|icermesin)\b",
        r"\b\w+s[ıiuu]z\b",
    )
    if any(re.search(pattern, evidence_norm) for pattern in negative_patterns):
        return False
    if any(token.endswith(("li", "lu")) for token in evidence_norm.split()):
        return True
    if re.search(r"\b(?:must|must have|required|requires|containing|contains|mutlaka|olsun|iceren|bulunan|karakterli|agirlikli)\b", evidence_norm):
        return True

    style_descriptors = {
        "clean", "unusual", "skin scent", "soft amber", "sporty",
        "professional", "polished", "versatile", "characterful",
    }

    escaped = re.escape(evidence_norm)
    for match in re.finditer(rf"\b{escaped}\b", query_norm):
        before = " ".join(query_norm[:match.start()].split()[-4:])
        after = " ".join(query_norm[match.end():].split()[:7])
        if re.search(r"\b(?:prefer|preferably|ideally|maybe|perhaps|tercihen|mumkunse)\b", before):
            continue
        if re.search(r"\b(?:softer|lighter|less|daha yumusak|daha hafif)\b", f"{before} {evidence_norm} {after}"):
            continue
        if re.search(r"\b(?:must have|must include|must be|with|containing|contains|requires|required)\s+(?:a|an|the)?\s*$", before):
            return True
        if re.match(r"^(?:must|required|mutlaka|olsun|iceren|bulunan|karakterli|agirlikli)\b", after):
            return True
        # A trailing hard marker scopes over coordinated traits:
        # "vanilla and spice must be present" / "vanilya ve baharat mutlaka bulunan".
        if re.match(
            r"^(?:(?:and|ve|ile)\s+){1,2}.{0,50}\b"
            r"(?:must|required|mutlaka|olsun|iceren|bulunan|karakterli|agirlikli)\b",
            after,
        ):
            return True
        if trait_norm in style_descriptors:
            continue
        if re.match(r"^(?:fragrance|fragrances|perfume|perfumes|scent|scents|parfum|parfumu|parfumler|koku)\b", after):
            return True
        # Planner evidence may include the category noun ("rose fragrances")
        # instead of returning only the trait token.
        if re.search(
            r"\b(?:fragrance|fragrances|perfume|perfumes|scent|scents|parfum|parfumu|parfumler|koku)\b",
            evidence_norm,
        ):
            return True
    return False


def explicit_unsupported_intent(query: str) -> str | None:
    """Deterministic safety recovery for claims the catalog cannot support."""
    normalized = f" {normalize_text(query)} "
    social_terms = (" compliment ", " compliments ", " iltifat ", " attract ", " attraction ")
    if any(term in normalized for term in social_terms):
        return "unsupported_social_claim"
    return None


def infer_explicit_audience(query: str) -> str | None:
    normalized = f" {normalize_text(query)} "
    if any(term in normalized for term in (" unisex ", " gender neutral ", " cinsiyetsiz ")):
        return "unisex"
    if any(term in normalized for term in (" women ", " women s ", " woman ", " female ", " kadin ", " kadinlar ")):
        return "female"
    if any(term in normalized for term in (" men ", " men s ", " man ", " male ", " erkek ", " erkekler ")):
        return "male"
    return None


def infer_explicit_season(query: str) -> str | None:
    normalized = f" {normalize_text(query)} "
    aliases = {
        "spring": (" spring ", " ilkbahar "),
        "summer": (" summer ", " yaz "),
        "autumn": (" autumn ", " fall ", " sonbahar "),
        "winter": (" winter ", " kis "),
    }
    matches = [season for season, terms in aliases.items() if any(term in normalized for term in terms)]
    return matches[0] if len(matches) == 1 else None


def infer_explicit_time_profile(query: str) -> str | None:
    normalized = f" {normalize_text(query)} "
    if any(term in normalized for term in (" night ", " nighttime ", " evening ", " date ", " gece ", " aksam ")):
        return "night"
    if any(term in normalized for term in (" daytime ", " day wear ", " office ", " gym ", " gunduz ", " ofis ")):
        return "day"
    return None


def infer_coordinated_exclusions(query: str) -> list[str]:
    """Recover explicit English negative lists without a fixed trait vocabulary."""
    normalized = str(query or "").lower().replace("ı", "i")
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9,.;]+", " ", normalized)
    normalized = " ".join(normalized.split())
    match = re.search(
        r"\b(?:without|no|avoid|avoiding|exclude|excluding)\b\s+(.{1,120}?)"
        r"(?:\b(?:accord|accords|note|notes)\b|[.;]|$)",
        normalized,
    )
    if not match:
        return []
    phrase = match.group(1).strip()
    # Stop at a new imperative clause instead of treating the command as a
    # scent trait: "exclude tobacco and show three different choices".
    phrase = re.split(
        r"(?:\b(?:and|or)\s+|[,;]\s*)(?=(?:show|recommend|give|suggest|list|find|provide|keep|make|tell)\b)",
        phrase,
        maxsplit=1,
    )[0].strip()
    if "," not in phrase and not re.search(r"\b(?:and|or)\b", phrase):
        return []
    parts = re.split(r"\s*,\s*|\b(?:and|or)\b", phrase)
    noise = {"a", "all", "any", "anything", "kind", "kinds", "of", "the"}
    output: list[str] = []
    for part in parts:
        tokens = [token for token in normalize_text(part).split() if token not in noise]
        if not tokens or len(tokens) > 4:
            continue
        term = canonical_trait_term(" ".join(tokens))
        if term:
            output.append(term)
    return list(dict.fromkeys(output))


def infer_explicit_exclusions(query: str) -> list[str]:
    """Recover simple explicit negatives when the planner omits them.

    The extraction is grammar based rather than tied to a fixed fragrance-trait
    vocabulary, so new catalog terms receive the same treatment.
    """
    normalized = normalize_text(query)
    if not normalized:
        return []

    output: list[str] = []

    def add(value: str) -> None:
        tokens = [
            token for token in normalize_text(value).split()
            if token not in {
                "a", "an", "any", "the", "accord", "accords", "note", "notes",
                "akor", "nota", "pls", "please", "all", "kind", "kinds", "type", "types", "of",
                "heavy", "strong", "prominent", "dominant", "noticeable",
            }
        ]
        if not tokens or len(tokens) > 3:
            return
        if tokens[0].endswith("ing") or tokens[0] in {
            "declaring", "choosing", "naming", "recommending", "showing", "giving",
            "need", "preference", "more", "fewer", "doubt",
        }:
            return
        if any(
            token in {
                "winner", "choice", "choices", "option", "options", "price", "budget",
                "longevity", "projection", "sillage", "performance",
            }
            or token.isdigit()
            for token in tokens
        ):
            return
        term = canonical_trait_term(" ".join(tokens))
        if term:
            output.append(term)

    # English: "not sweet", "without vanilla", "must not contain musk".
    for match in re.finditer(
        r"\b(?:without|no)\s+"
        r"([a-z0-9]+(?:\s+[a-z0-9]+){0,2}?)"
        r"(?=\s+(?:accord|accords|note|notes|fragrance|fragrances|perfume|perfumes|scent|scents|and|or|but|with|for)|[,.;]|$)",
        normalized,
    ):
        add(match.group(1))
    for match in re.finditer(
        r"\b(?:must\s+)?not\s+(?:contain|include|have)\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,2}?)"
        r"(?=\s+(?:accord|accords|note|notes|and|or|but|with|for)|[,.;]|$)",
        normalized,
    ):
        add(match.group(1))
    for match in re.finditer(
        r"\bnot\s+(?!too\b|very\b|overly\b)([a-z0-9]+)"
        r"(?=\s+(?:accord|accords|note|notes|fragrance|fragrances|perfume|perfumes|scent|scents|and|or|but|with|for)|[,.;]|$)",
        normalized,
    ):
        add(match.group(1))
    for match in re.finditer(r"\b(?:less|except)\s+([a-z0-9]+)\b", normalized):
        add(match.group(1))
    for match in re.finditer(r"\b([a-z0-9]+(?:\s+[a-z0-9]+){0,1})\s+free\b", normalized):
        add(match.group(1))
    for match in re.finditer(
        r"\b(?:avoid|exclude|excluding)\s+"
        r"(?:(?:anything\s+with|all\s+(?:kinds?|types?)\s+of)\s+)?([a-z0-9]+)\b",
        normalized,
    ):
        add(match.group(1))
    for match in re.finditer(
        r"\b(?:do\s+not|dont|don\s+t)\s+want\s+(?:a\s+|an\s+|any\s+)?"
        r"([a-z0-9]+(?:\s+[a-z0-9]+){0,1})\b",
        normalized,
    ):
        add(match.group(1))
    for match in re.finditer(r"\bnothing\s+([a-z0-9]+)\b", normalized):
        add(match.group(1))

    # Turkish and mixed Turkish/English: "sweet olmayan", "misk icermeyen".
    for match in re.finditer(
        r"\b([a-z0-9]+)\s+(?:olmayan|icermeyen|icermesin|barindirmayan|barindirmasin|istemiyorum)\b",
        normalized,
    ):
        add(match.group(1))
    for match in re.finditer(r"\b([a-z0-9]+)\s+haric\b", normalized):
        add(match.group(1))

    # Productive Turkish privative suffixes after ASCII normalization.
    for token in normalized.split():
        match = re.fullmatch(r"([a-z0-9]{3,}?)(?:siz|suz)", token)
        if match:
            add(match.group(1))

    unique = list(dict.fromkeys(output))
    atomic = {term for term in unique if " " not in term}
    return [
        term for term in unique
        if " " not in term or not all(part in atomic for part in term.split())
    ]


def infer_explicit_requirements(query: str) -> list[str]:
    """Recover only unambiguously hard-positive grammar omitted by the planner."""
    normalized = normalize_text(query)
    if not normalized:
        return []
    output: list[str] = []

    def add(value: str) -> None:
        tokens = [
            token for token in normalize_text(value).split()
            if token not in {
                "akor", "accord", "nota", "note", "icin", "for",
                "ve", "ile", "ama", "daha",
            }
        ]
        term = canonical_trait_term(" ".join(tokens))
        if term:
            output.append(term)

    for match in re.finditer(
        r"\bmust\s+(?:have|include|contain|feature)\s+"
        r"(?:a\s+|an\s+|the\s+)?([a-z0-9]+(?:\s+[a-z0-9]+){0,1}?)"
        r"(?=\s+(?:accord|accords|note|notes|and|or|but|without|for)|$)",
        normalized,
    ):
        add(match.group(1))

    for match in re.finditer(
        r"\b([a-z0-9]+(?:\s+[a-z0-9]+)?)(?:\s+(?:ve|ile)\s+([a-z0-9]+(?:\s+[a-z0-9]+)?))?"
        r"\s+(?:akor\s+|nota\s+)?mutlaka\s+bulunan\b",
        normalized,
    ):
        add(match.group(1))
        if match.group(2):
            add(match.group(2))

    for match in re.finditer(
        r"\b([a-z0-9]+(?:\s+[a-z0-9]+)?)\s+(?:iceren|karakterli|agirlikli)\b",
        normalized,
    ):
        add(match.group(1))
    for match in re.finditer(r"\b([a-z0-9]+)\s+(?:olsun|icersin|barindirsin)\b", normalized):
        add(match.group(1))

    return list(dict.fromkeys(output))


def normalize_plan(raw_plan: dict[str, Any], query: str) -> dict[str, Any]:
    intent = normalize_text(raw_plan.get("intent")).replace(" ", "_")
    if intent not in PLANNER_INTENTS:
        intent = "recommendation"
    try:
        confidence = min(max(float(raw_plan.get("confidence") or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0

    required_items = raw_plan.get("required_terms")
    required_items = required_items if isinstance(required_items, list) else []
    accepted_required: list[str] = []
    downgraded_required: list[str] = []
    for item in required_items:
        if not isinstance(item, dict) or not evidence_supported(item.get("evidence"), query):
            continue
        term = canonical_trait_term(item.get("value"))
        if not term:
            continue
        target = accepted_required if hard_trait_requested(query, item.get("evidence"), term) else downgraded_required
        target.append(term)

    plan: dict[str, Any] = {
        "intent": intent,
        "confidence": confidence,
        "perfumes": [str(value).strip() for value in evidenced_values(raw_plan.get("perfumes"), query, limit=3)],
        "wanted_terms": [canonical_trait_term(value) for value in evidenced_values(raw_plan.get("wanted_terms"), query)],
        "required_terms": accepted_required,
        "excluded_terms": [canonical_trait_term(value) for value in evidenced_values(raw_plan.get("excluded_terms"), query)],
        "requested_fields": [normalize_text(value) for value in evidenced_values(raw_plan.get("requested_fields"), query)],
    }
    wanted_items = raw_plan.get("wanted_terms")
    wanted_items = wanted_items if isinstance(wanted_items, list) else []
    promoted_required = [
        canonical_trait_term(item.get("value"))
        for item in wanted_items
        if isinstance(item, dict)
        and evidence_supported(item.get("evidence"), query)
        and hard_trait_requested(query, item.get("evidence"), item.get("value"))
    ]
    plan["required_terms"] = list(dict.fromkeys([
        *plan["required_terms"],
        *filter(None, promoted_required),
        *infer_explicit_requirements(query),
    ]))
    if any(term.endswith(" spicy") for term in plan["required_terms"]):
        plan["required_terms"] = [term for term in plan["required_terms"] if term != "spicy"]
    plan["wanted_terms"] = list(dict.fromkeys([
        *plan["wanted_terms"],
        *downgraded_required,
    ]))
    plan["wanted_terms"] = [
        term for term in plan["wanted_terms"] if term not in plan["required_terms"]
    ]
    plan["excluded_terms"] = list(dict.fromkeys([
        *plan["excluded_terms"],
        *infer_coordinated_exclusions(query),
        *infer_explicit_exclusions(query),
    ]))
    atomic_exclusions = {term for term in plan["excluded_terms"] if " " not in term}
    plan["excluded_terms"] = [
        term for term in plan["excluded_terms"]
        if " " not in term or not all(part in atomic_exclusions for part in term.split())
    ]
    semantic_query = str(raw_plan.get("semantic_query") or "").strip()
    if 3 <= len(semantic_query) <= 240 and "\n" not in semantic_query:
        plan["semantic_query"] = semantic_query
    for field, allowed in {
        "gender": {"male", "female", "unisex"},
        "season": {"spring", "summer", "autumn", "winter"},
        "time_profile": {"day", "night"},
    }.items():
        value = normalize_text(first_evidenced_value(raw_plan.get(field), query))
        if value in allowed:
            plan[field] = value
    inferred_constraints = {
        "gender": infer_explicit_audience(query),
        "season": infer_explicit_season(query),
        "time_profile": infer_explicit_time_profile(query),
    }
    for field, value in inferred_constraints.items():
        if field not in plan and value:
            plan[field] = value

    count = first_evidenced_value(raw_plan.get("requested_count"), query)
    try:
        count_value = int(count)
    except (TypeError, ValueError):
        count_value = 0
    if 1 <= count_value <= 5:
        plan["requested_count"] = count_value
    elif intent in {"recommendation", "similarity", "alternative"}:
        inferred_count = infer_requested_count(query)
        if inferred_count is not None:
            plan["requested_count"] = inferred_count

    brand_item = raw_plan.get("requested_brand")
    if isinstance(brand_item, dict) and evidence_supported(brand_item.get("evidence"), query):
        brand = str(brand_item.get("value") or "").strip()
        evidence = str(brand_item.get("evidence") or "")
        if explicit_brand_context(query, brand, evidence):
            brand_field = "reference_brand" if intent in {"similarity", "alternative"} and plan["perfumes"] else "requested_brand"
            plan[brand_field] = brand

    plan["wanted_terms"] = list(dict.fromkeys(filter(None, plan["wanted_terms"])))
    plan["required_terms"] = list(dict.fromkeys(filter(None, plan["required_terms"])))
    plan["excluded_terms"] = list(dict.fromkeys(filter(None, plan["excluded_terms"])))
    excluded = set(plan["excluded_terms"])
    plan["required_terms"] = [term for term in plan["required_terms"] if term not in excluded]
    plan["wanted_terms"] = [term for term in plan["wanted_terms"] if term not in excluded]
    plan["requested_fields"] = list(dict.fromkeys(filter(None, plan["requested_fields"])))
    discovery_item = raw_plan.get("discovery_mode")
    if isinstance(discovery_item, dict) and evidence_supported(discovery_item.get("evidence"), query):
        discovery_mode = normalize_text(discovery_item.get("value"))
        if discovery_mode in {"balanced", "mainstream", "niche"}:
            plan["discovery_mode"] = discovery_mode
    if "discovery_mode" not in plan:
        inferred_discovery_mode = infer_discovery_mode(query)
        if inferred_discovery_mode:
            plan["discovery_mode"] = inferred_discovery_mode
    conversation_action = normalize_text(raw_plan.get("conversation_action")).replace(" ", "_")
    if conversation_action in {"new_request", "more_options", "refine_previous"}:
        plan["conversation_action"] = conversation_action
    else:
        plan["conversation_action"] = "new_request"
    unsupported_intent = explicit_unsupported_intent(query)
    if unsupported_intent:
        plan["intent"] = unsupported_intent
        plan["perfumes"] = []
    return plan


def inherit_conversation_plan(plan: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
    """Apply model-decided follow-up semantics without keyword routing."""
    if not context or plan.get("conversation_action") not in {"more_options", "refine_previous"}:
        return plan
    previous = context.get("previous_plan")
    if not isinstance(previous, dict):
        return plan
    merged = dict(plan)
    action = merged["conversation_action"]
    if action == "more_options" and previous.get("semantic_query"):
        # Phrases such as "other options" carry no new scent semantics; keep
        # the previous retrieval meaning even if the planner emitted a generic
        # paraphrase for the follow-up sentence.
        merged["semantic_query"] = previous["semantic_query"]
    for key in (
        "semantic_query",
        "requested_brand",
        "reference_brand",
        "gender",
        "season",
        "time_profile",
        "requested_count",
        "discovery_mode",
    ):
        if not merged.get(key) and previous.get(key):
            merged[key] = previous[key]
    current_required = set(plan.get("required_terms", []))
    current_excluded = set(plan.get("excluded_terms", []))
    for key in ("wanted_terms", "required_terms", "excluded_terms"):
        merged[key] = list(dict.fromkeys([
            *previous.get(key, []),
            *merged.get(key, []),
        ]))
    # A new negative instruction overrides an old positive preference and a
    # newly required trait overrides an old exclusion.
    merged["wanted_terms"] = [term for term in merged["wanted_terms"] if term not in current_excluded]
    merged["required_terms"] = [term for term in merged["required_terms"] if term not in current_excluded]
    merged["excluded_terms"] = [term for term in merged["excluded_terms"] if term not in current_required]
    if merged.get("intent") == "recommendation" and previous.get("intent") in {
        "recommendation", "similarity", "alternative",
    }:
        merged["intent"] = previous["intent"]
        if not merged.get("perfumes") and previous.get("perfumes"):
            merged["perfumes"] = list(previous["perfumes"])
    merged["exclude_candidate_ids"] = list(dict.fromkeys(
        int(value)
        for value in context.get("previous_recommendation_ids", [])
        if str(value).isdigit() and int(value) > 0
    ))
    merged["inherited_previous_constraints"] = True
    return merged


def infer_requested_count(query: str) -> int | None:
    normalized = normalize_text(query)
    number_tokens = {
        "1": 1, "one": 1, "a": 1, "bir": 1,
        "2": 2, "two": 2, "iki": 2,
        "3": 3, "three": 3, "uc": 3,
        "4": 4, "four": 4, "dort": 4,
        "5": 5, "five": 5, "bes": 5,
    }
    token_pattern = "|".join(sorted((re.escape(token) for token in number_tokens), key=len, reverse=True))
    noun = r"(?:perfume|perfumes|fragrance|fragrances|scent|scents|option|options|parfum|parfumu|koku|secenek)"
    action = r"(?:recommend|suggest|give|show|list|provide|oner|tavsiye)"
    patterns = (
        rf"\b{action}\b.{{0,35}}?\b({token_pattern})\b(?:.{{0,20}}?\b{noun}\b)?",
        rf"\b({token_pattern})\b(?:\s+adet)?\s+{noun}\b.{{0,25}}?\b{action}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return number_tokens.get(match.group(1))
    return None


def infer_discovery_mode(query: str) -> str | None:
    """Recover an explicit discovery preference when planner JSON omits it."""
    normalized = f" {normalize_text(query)} "
    niche_phrases = (
        " niche ", " obscure ", " hidden gem ", " hidden gems ",
        " underrated ", " less known ", " not popular ",
        " nis ", " az bilinen ", " populer olmayan ", " kesif ",
    )
    mainstream_phrases = (
        " popular ", " mainstream ", " famous ", " well known ",
        " recognizable ", " mass appeal ", " crowd pleasing ",
        " populer ", " unlu ", " cok bilinen ", " herkesin bildigi ",
    )
    if any(phrase in normalized for phrase in niche_phrases):
        return "niche"
    if any(phrase in normalized for phrase in mainstream_phrases):
        return "mainstream"
    return None


def semantic_search_query(query: str, plan: dict[str, Any]) -> str:
    """Remove output-format quantities that distort embedding retrieval."""
    planner_query = str(plan.get("semantic_query") or "").strip()
    if planner_query:
        return planner_query
    count = int(plan.get("requested_count") or 0)
    if not count:
        return query.strip()
    number_words = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    }
    tokens = [str(count)]
    if count in number_words:
        tokens.append(number_words[count])
    cleaned = query
    for token in tokens:
        cleaned = re.sub(rf"\bexactly\s+{re.escape(token)}\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(rf"\b{re.escape(token)}\s+(?:options?|recommendations?|choices?)\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(rf"\b{re.escape(token)}\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\bexactly\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\b(?:recommend|suggest|list|show|provide|give)(?:\s+me)?\s*[.!?]*$",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned or query.strip()


class JsonHttpClient:
    def __init__(self, base_url: str, *, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc


class VLLMClient:
    def __init__(self, http: JsonHttpClient) -> None:
        self.http = http

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        structured_mode_fallback = False
        try:
            response = self.http.post("/v1/chat/completions", payload)
        except RuntimeError:
            if not json_mode:
                raise
            payload.pop("response_format", None)
            structured_mode_fallback = True
            response = self.http.post("/v1/chat/completions", payload)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {response}")
        answer = str(choices[0].get("message", {}).get("content") or "").strip()
        if not answer:
            raise RuntimeError("vLLM returned an empty message")
        return answer, {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "usage": response.get("usage") or {},
            "finish_reason": choices[0].get("finish_reason"),
            "structured_mode_fallback": structured_mode_fallback,
        }


class RetrievalClient:
    def __init__(self, http: JsonHttpClient) -> None:
        self.http = http

    def health(self) -> dict[str, Any]:
        return self.http.get("/health")

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.http.post("/search", payload)

    def resolve(self, hint: str) -> dict[str, Any] | None:
        return self.http.post("/resolve", {"hint": hint}).get("resolved")

    def similar(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.http.post("/similar", payload)


def candidate_record(item: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    merged = {**metadata, **{key: value for key, value in item.items() if key != "metadata"}}
    merged["perfume_id"] = int(merged.get("perfume_id") or 0)
    merged["name"] = str(merged.get("name") or "")
    merged["brand"] = str(merged.get("brand") or "")
    merged["label"] = str(merged.get("label") or f"{merged['name']} by {merged['brand']}")
    return merged


PERFORMANCE_CALIBRATION = {
    # Thresholds are catalog percentiles from the current 131,930-record snapshot.
    "longevity": {
        "scale": 5,
        "bands": (
            (2.31, "light; bottom catalog quartile"),
            (3.01, "moderate-light; below catalog median"),
            (3.57, "moderate; typical catalog range"),
            (4.30, "strong; top catalog quartile"),
            (float("inf"), "very strong; approximately top 5% of the catalog"),
        ),
    },
    "sillage": {
        "scale": 4,
        "bands": (
            (1.68, "restrained; bottom catalog quartile"),
            (2.15, "moderate-restrained; below catalog median"),
            (2.57, "moderate; typical catalog range"),
            (3.15, "noticeable/strong; top catalog quartile"),
            (float("inf"), "very strong; approximately top 5% of the catalog"),
        ),
    },
}


def calibrated_performance(metric: str, value: Any) -> str:
    config = PERFORMANCE_CALIBRATION[metric]
    if value in (None, ""):
        return "not recorded; do not infer performance"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "not recorded; do not infer performance"
    band = next(label for upper_bound, label in config["bands"] if score < upper_bound)
    return f"{score:g}/{config['scale']} ({band})"


def card_text(candidate: dict[str, Any]) -> str:
    fields = [
        f"Name: {candidate['label']}",
        f"Gender: {candidate.get('gender') or 'not recorded'}",
        f"Year: {candidate.get('year') if candidate.get('year') is not None else 'not recorded'}",
        f"Accords: {candidate.get('accords_csv') or 'not recorded'}",
        f"Notes: {candidate.get('notes_csv') or 'not recorded'}",
        f"Rating: {candidate.get('rating') if candidate.get('rating') is not None else 'not recorded'}",
        f"Votes: {candidate.get('popularity') if candidate.get('popularity') is not None else 'not recorded'}",
        f"Longevity: {calibrated_performance('longevity', candidate.get('longevity'))}",
        f"Sillage: {calibrated_performance('sillage', candidate.get('sillage'))}",
        f"Value: {candidate.get('value_score') if candidate.get('value_score') is not None else 'not recorded'}",
        f"Seasons: {candidate.get('seasons_csv') or 'not recorded'}",
        f"Time: {candidate.get('time_profile_csv') or 'not recorded'}",
    ]
    return "\n".join(fields)


def extract_numbered_recommendations(answer: str, candidates: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    known = [(candidate["label"], normalize_text(candidate["label"]), normalize_text(candidate["name"])) for candidate in candidates]
    selected: list[str] = []
    unknown: list[str] = []
    for line in answer.splitlines():
        match = re.match(r"^\s*(?:\d{1,2}[.)]|[-*])\s+(.+?)\s*$", line)
        if not match:
            continue
        # Keep punctuation inside catalog names (for example A*Men); normalizing
        # it away as formatting changes the identity tokens.
        heading = match.group(1).replace("`", "").strip()
        heading_norm = normalize_text(heading)
        matches: list[tuple[int, int, str]] = []
        padded_heading = f" {heading_norm} "
        for label, label_norm, name_norm in known:
            phrases = ((label_norm, 2), (name_norm, 1))
            for phrase, phrase_priority in phrases:
                if not phrase:
                    continue
                if heading_norm == phrase:
                    match_class = 4
                elif heading_norm.startswith(phrase + " "):
                    match_class = 3
                elif f" {phrase} " in padded_heading:
                    match_class = phrase_priority
                else:
                    continue
                matches.append((match_class, len(phrase), label))
        found = max(matches, default=(0, 0, None))[2]
        if not found and " by " in heading_norm:
            heading_name, heading_brand = heading_norm.rsplit(" by ", 1)
            scoped_aliases = [
                candidate["label"]
                for candidate in candidates
                if normalize_text(candidate.get("brand")) == heading_brand
                and (
                    normalize_text(candidate.get("name")) == heading_name
                    or normalize_text(candidate.get("name")).endswith(f" {heading_name}")
                )
            ]
            if len(scoped_aliases) == 1:
                found = scoped_aliases[0]
        if found:
            if found not in selected:
                selected.append(found)
        elif " by " in f" {heading_norm} ":
            unknown.append(heading)
    return selected, unknown


def candidate_mentions(answer: str, candidates: list[dict[str, Any]]) -> list[str]:
    answer_norm = normalize_text(answer)
    occurrences: list[tuple[int, int, str]] = []
    for candidate in candidates:
        phrases = {normalize_text(candidate["name"]), normalize_text(candidate["label"])} - {""}
        for phrase in phrases:
            occurrences.extend(
                (match.start(), match.end(), candidate["label"])
                for match in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", answer_norm)
            )

    output: list[str] = []
    for start, end, label in occurrences:
        # "Eros" inside "Eros Flame" is not evidence that both perfumes were
        # discussed. Keep shorter identities only when they also occur alone.
        covered_by_other = any(
            other_label != label
            and other_start <= start
            and other_end >= end
            and (other_end - other_start) > (end - start)
            for other_start, other_end, other_label in occurrences
        )
        if not covered_by_other and label not in output:
            output.append(label)
    return output


def candidate_has_term(candidate: dict[str, Any], term: str) -> bool:
    searchable = normalize_text(" ".join(str(candidate.get(key) or "") for key in ("name", "brand", "accords_csv", "notes_csv")))
    term_norm = canonical_trait_term(term)
    if f" {term_norm} " in f" {searchable} ":
        return True
    variants = {
        "musk": {"musk", "musky", "white musk"},
        "leather": {"leather", "leathery", "suede"},
        "oud": {"oud", "aoud", "agarwood"},
        "smoky": {"smoke", "smoky"},
    }.get(term_norm, {term_norm})
    traits = {
        normalize_text(part)
        for key in ("accords_csv", "notes_csv")
        for part in str(candidate.get(key) or "").split(",")
        if normalize_text(part)
    }
    if traits & variants:
        return True
    if term_norm in {"spicy", "floral"}:
        return any(term_norm in trait.split() for trait in traits)
    return False


def keep_required_candidates(candidates: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    required = plan.get("required_terms", [])
    if not required:
        return candidates
    return [
        candidate
        for candidate in candidates
        if all(candidate_has_term(candidate, term) for term in required)
    ]


def unique_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate catalog aliases/duplicate rows while preserving rank order."""
    output: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_labels: set[str] = set()
    for candidate in candidates:
        perfume_id = int(candidate.get("perfume_id") or 0)
        label = normalize_text(candidate.get("label"))
        if (perfume_id and perfume_id in seen_ids) or (label and label in seen_labels):
            continue
        output.append(candidate)
        if perfume_id:
            seen_ids.add(perfume_id)
        if label:
            seen_labels.add(label)
    return output


def candidate_answer_sections(answer: str, candidates: list[dict[str, Any]]) -> dict[str, str]:
    if len(candidates) == 1:
        return {candidates[0]["label"]: answer}
    sections: dict[str, list[str]] = {}
    active_label: str | None = None
    for line in answer.splitlines():
        if re.match(r"^\s*\d{1,2}[.)]\s+", line):
            line_mentions = candidate_mentions(line, candidates)
            active_label = line_mentions[0] if line_mentions else None
            if active_label:
                sections.setdefault(active_label, [])
        if active_label:
            sections[active_label].append(line)
    if sections:
        return {label: "\n".join(lines) for label, lines in sections.items()}

    # Comparisons are usually written as natural paragraphs instead of numbered
    # recommendations. Attribute only unambiguous paragraphs/sentences so a
    # statement about one perfume cannot leak into the other perfume's checks.
    for paragraph in re.split(r"\n\s*\n+", str(answer or "")):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_mentions = candidate_mentions(paragraph, candidates)
        if len(paragraph_mentions) == 1:
            sections.setdefault(paragraph_mentions[0], []).append(paragraph)
            continue
        if len(paragraph_mentions) > 1:
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                sentence_mentions = candidate_mentions(sentence, candidates)
                if len(sentence_mentions) == 1:
                    sections.setdefault(sentence_mentions[0], []).append(sentence.strip())
    return {label: "\n".join(parts) for label, parts in sections.items()}


def explicit_performance_labels(text: str, metric: str) -> set[str]:
    clauses = [normalize_text(part) for part in re.split(r"[,;.\n]+", str(text or ""))]
    clauses = [
        re.sub(
            r"\b(?:more|less|daha)\s+(?:noticeable|pronounced|strong|light|weak|restrained|belirgin|guclu|hafif|zayif)\b",
            "relative",
            re.sub(
                r"\b(?:moderate restrained|moderate light|orta kisitli|orta hafif)\b",
                "moderate",
                clause,
            ),
        )
        for clause in clauses
    ]
    noun_patterns = {
        "longevity": r"(?:longevity|lasting power|kalicilik|kaliciligi|kaliciliga)",
        "sillage": r"(?:sillage|projection|yayilim|yayilimi|yayilima)",
    }
    nouns = noun_patterns[metric]
    low_extra = r"|short lived" if metric == "longevity" else r"|soft|subtle|close to skin|yumusak"
    high_extra = r"|long lasting|excellent" if metric == "longevity" else r"|noticeable|pronounced|projecting|belirgin"
    labels = {
        "low": rf"(?:light|weak|low|short|restrained|intimate|hafif|zayif|dusuk|kisa|tene yakin{low_extra})",
        "moderate": r"(?:moderate|average|medium|orta|ortalama)",
        "high": rf"(?:very strong|strong|high|powerful|exceptional|cok guclu|guclu|yuksek{high_extra})",
    }
    observed: set[str] = set()
    for group, adjective in labels.items():
        connector = (
            r"(?: is classified as| is rated as| is considered| can be described as|"
            r" is| feels| remains| stays| olarak tanimlanabilir| olarak siniflandirilabilir|"
            r" seviyesindedir| seviyesi| ise| degeri)?"
        )
        patterns = (
            rf"{nouns}{connector}(?: at| bir)?(?: quite| rather| relatively| oldukca| hayli| biraz)? {adjective}\b",
            rf"{adjective}(?: level| seviyede| seviyedeki| bir)? {nouns}\b",
        )
        if any(re.search(pattern, clause) for clause in clauses for pattern in patterns):
            observed.add(group)

    # Natural prose frequently inserts particles and qualifiers that rigid
    # grammar patterns cannot enumerate (for example, "yayilimi da guclu bir
    # seviyede"). Assign each class adjective to the nearest performance noun,
    # within a small window, to avoid leaking a sillage class into longevity or
    # one comparison subject into another.
    for clause in clauses:
        noun_hits: list[tuple[int, int, str]] = []
        for noun_metric, noun_pattern in noun_patterns.items():
            noun_hits.extend(
                (match.start(), match.end(), noun_metric)
                for match in re.finditer(rf"\b{noun_pattern}\b", clause)
            )
        if not noun_hits:
            continue
        for group, adjective in labels.items():
            for adjective_match in re.finditer(rf"\b{adjective}\b", clause):
                prefix = clause[max(0, adjective_match.start() - 28):adjective_match.start()]
                if re.search(r"(?:\bnot|\bno|\bdegil|\bisn t|\bwasn t)\s+(?:particularly\s+)?$", prefix):
                    continue
                nearby: list[tuple[int, int, str]] = []
                for noun_start, noun_end, noun_metric in noun_hits:
                    between = (
                        clause[noun_end:adjective_match.start()]
                        if noun_end <= adjective_match.start()
                        else clause[adjective_match.end():noun_start]
                    )
                    word_distance = len(between.split())
                    if word_distance <= 5:
                        char_distance = max(
                            noun_start - adjective_match.end(),
                            adjective_match.start() - noun_end,
                            0,
                        )
                        nearby.append((word_distance, char_distance, noun_metric))
                if nearby:
                    best_distance = min(item[:2] for item in nearby)
                    nearest_metrics = {
                        noun_metric
                        for word_distance, char_distance, noun_metric in nearby
                        if (word_distance, char_distance) == best_distance
                    }
                    if nearest_metrics == {metric}:
                        observed.add(group)

    # "Noticeable" alone can describe top-quartile sillage, but natural advice
    # also uses "moderate sillage ... noticeable but not overwhelming". In that
    # construction the explicit calibrated class must win over the softer prose
    # gloss; genuinely strong/high/pronounced claims remain contradictory.
    if metric == "sillage" and {"moderate", "high"}.issubset(observed):
        normalized = " ".join(clauses)
        explicit_moderate = re.search(
            rf"(?:\bmoderate\b|\bmedium\b|\baverage\b|\borta\b).{{0,24}}\b{nouns}\b|"
            rf"\b{nouns}\b.{{0,24}}(?:\bmoderate\b|\bmedium\b|\baverage\b|\borta\b)",
            normalized,
        )
        decisive_high = re.search(
            rf"(?:\bstrong\b|\bhigh\b|\bpowerful\b|\bexceptional\b|\bpronounced\b|\bprojecting\b|\bguclu\b|\byuksek\b|\bbelirgin\b).{{0,24}}\b{nouns}\b|"
            rf"\b{nouns}\b.{{0,24}}(?:\bstrong\b|\bhigh\b|\bpowerful\b|\bexceptional\b|\bpronounced\b|\bprojecting\b|\bguclu\b|\byuksek\b|\bbelirgin\b)",
            normalized,
        )
        if explicit_moderate and not decisive_high:
            observed.discard("high")
    return observed


def expected_performance_group(metric: str, value: Any) -> str:
    if value in (None, ""):
        return "unknown"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if metric == "longevity":
        return "low" if score < 2.31 else "moderate" if score < 3.57 else "high"
    return "low" if score < 1.68 else "moderate" if score < 2.57 else "high"


def performance_claim_violations(answer: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections = candidate_answer_sections(answer, candidates)
    violations: list[dict[str, Any]] = []
    for candidate in candidates:
        section = sections.get(candidate["label"])
        if not section:
            continue
        for metric in ("longevity", "sillage"):
            observed = explicit_performance_labels(section, metric)
            if not observed:
                continue
            expected = expected_performance_group(metric, candidate.get(metric))
            if expected == "unknown" or observed != {expected}:
                violations.append({
                    "candidate": candidate["label"],
                    "metric": metric,
                    "expected": expected,
                    "observed": sorted(observed),
                })
    return violations


def validate_answer(
    answer: str,
    plan: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    response_language: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    mentions = candidate_mentions(answer, candidates)
    numbered, unknown = extract_numbered_recommendations(answer, candidates)
    if not answer.strip():
        reasons.append("empty_answer")
    if unknown:
        reasons.append("unsupported_numbered_perfume")
    intent = plan["intent"]
    if intent == "comparison":
        missing = [candidate["label"] for candidate in candidates[:2] if candidate["label"] not in mentions]
        if missing:
            reasons.append("comparison_missing_perfume")
    elif intent != "exact_lookup" and not mentions:
        reasons.append("no_context_perfume_mentioned")

    expected_count = int(plan.get("requested_count") or 0)
    if expected_count and len(numbered) != expected_count:
        reasons.append("requested_count_mismatch")

    violated = []
    recommendation_labels = numbered or mentions
    by_label = {candidate["label"]: candidate for candidate in candidates}
    for label in recommendation_labels:
        candidate = by_label.get(label)
        if candidate and any(candidate_has_term(candidate, term) for term in plan.get("excluded_terms", [])):
            violated.append(label)
    if violated:
        reasons.append("strict_filter_violation")

    missing_required = []
    for label in recommendation_labels:
        candidate = by_label.get(label)
        if candidate and any(not candidate_has_term(candidate, term) for term in plan.get("required_terms", [])):
            missing_required.append(label)
    if missing_required:
        reasons.append("required_trait_violation")

    performance_violations = performance_claim_violations(answer, candidates)
    if performance_violations:
        reasons.append("performance_calibration_violation")

    generic_patterns = (
        r"\bit matches through\b",
        r"\bi would include it for\b",
        r"\bit earns a place here with\b",
        r"\bthe fit comes from\b",
        r"\bbu parfumu buraya dahil etme sebebim\b",
        r"\bnotalari ve akorlari uzerinden\b",
        r"\bonemli bir uzlasi\b",
        r"\buygun bir hava ciziyor\b",
    )
    normalized_answer = normalize_text(answer)
    if any(re.search(pattern, normalized_answer, re.I) for pattern in generic_patterns):
        reasons.append("mechanical_template_language")
    if response_language and not response_language_matches(answer, response_language):
        reasons.append("wrong_response_language")
    return {
        "pass": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "mentioned_candidates": mentions,
        "numbered_recommendations": numbered,
        "unsupported_numbered_items": unknown,
        "strict_filter_violations": violated,
        "required_trait_violations": missing_required,
        "performance_calibration_violations": performance_violations,
        "response_language": response_language,
    }


def exact_lookup_answer(
    candidate: dict[str, Any],
    requested_fields: list[str],
    *,
    response_language: str = "en",
) -> str:
    field_map = OrderedDict([
        ("rating", candidate.get("rating")),
        ("popularity", candidate.get("popularity")),
        ("year", candidate.get("year")),
        ("longevity", candidate.get("longevity")),
        ("sillage", candidate.get("sillage")),
        ("value", candidate.get("value_score")),
        ("accords", candidate.get("accords_csv")),
        ("notes", candidate.get("notes_csv")),
        ("seasons", candidate.get("seasons_csv")),
        ("time", candidate.get("time_profile_csv")),
    ])
    selected = requested_fields or list(field_map)
    labels_tr = {
        "rating": "Puan",
        "popularity": "Oy sayısı",
        "year": "Yıl",
        "longevity": "Kalıcılık",
        "sillage": "Yayılım",
        "value": "Fiyat-performans",
        "accords": "Akorlar",
        "notes": "Notalar",
        "seasons": "Mevsimler",
        "time": "Kullanım zamanı",
    }
    lines = [
        f"{candidate['label']} için veritabanı kaydı:"
        if response_language == "tr"
        else f"Database record for {candidate['label']}:"
    ]
    for key in selected:
        if key in field_map:
            value = field_map[key]
            if response_language == "tr":
                lines.append(f"- {labels_tr[key]}: {value if value not in (None, '') else 'kayıtlı değil'}")
            else:
                lines.append(f"- {key.title()}: {value if value not in (None, '') else 'not recorded'}")
    return "\n".join(lines)


def fallback_answer(
    plan: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    query: str = "",
    response_language: str | None = None,
) -> str:
    language = response_language or detect_response_language(query)
    if not candidates:
        return static_message("no_safe_match", language)
    if plan["intent"] == "exact_lookup":
        return exact_lookup_answer(
            candidates[0],
            plan.get("requested_fields", []),
            response_language=language,
        )
    if plan["intent"] == "comparison" and len(candidates) >= 2:
        left, right = candidates[:2]
        left_traits = [part.strip() for part in str(left.get("accords_csv") or "").split(",") if part.strip()]
        right_traits = [part.strip() for part in str(right.get("accords_csv") or "").split(",") if part.strip()]
        shared = [trait for trait in left_traits if normalize_text(trait) in {normalize_text(item) for item in right_traits}]
        left_distinct = [trait for trait in left_traits if normalize_text(trait) not in {normalize_text(item) for item in right_traits}]
        right_distinct = [trait for trait in right_traits if normalize_text(trait) not in {normalize_text(item) for item in left_traits}]

        def performance_word(metric: str, candidate: dict[str, Any]) -> str:
            group = expected_performance_group(metric, candidate.get(metric))
            if language == "tr":
                return {"low": "hafif", "moderate": "orta", "high": "güçlü"}.get(group, "kayıtlı değil")
            return {"low": "light", "moderate": "moderate", "high": "strong"}.get(group, "not recorded")

        def localized_use(value: Any) -> str:
            raw_parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
            if language != "tr":
                return ", ".join(raw_parts) or "not recorded"
            translations = {
                "spring": "ilkbahar",
                "summer": "yaz",
                "autumn": "sonbahar",
                "winter": "kış",
                "day": "gündüz",
                "night": "gece",
            }
            return ", ".join(translations.get(normalize_text(part), part) for part in raw_parts) or "belirtilmemiş"

        sections = ["Kayıtlara dayalı karşılaştırma:" if language == "tr" else "Grounded comparison:"]
        for candidate in candidates[:2]:
            profile = ", ".join(
                part.strip()
                for part in str(candidate.get("accords_csv") or "").split(",")[:4]
                if part.strip()
            )
            if language == "tr":
                sections.extend([
                    f"\n{candidate['label']}",
                    f"{profile or 'Kayıtlı akor bulunmayan'} ekseninde bir karakter sunuyor. Kayıtlı kullanım aralığı {localized_use(candidate.get('seasons_csv'))}; {localized_use(candidate.get('time_profile_csv'))}.",
                    f"Kalıcılığı {performance_word('longevity', candidate)}, yayılımı {performance_word('sillage', candidate)} seviyede.",
                ])
            else:
                sections.extend([
                    f"\n{candidate['label']}",
                    f"Its character centers on {profile or 'unrecorded accords'}. Its recorded wear range is {localized_use(candidate.get('seasons_csv'))}; {localized_use(candidate.get('time_profile_csv'))}.",
                    f"Longevity is {performance_word('longevity', candidate)} and sillage is {performance_word('sillage', candidate)}.",
                ])
        if language == "tr":
            sections.append(
                f"\nPratik fark: Ortak zeminleri {', '.join(shared[:3]) or 'sınırlı'}; "
                f"{left['name']} tarafında {', '.join(left_distinct[:3]) or 'benzer ana karakter'}, "
                f"{right['name']} tarafında ise {', '.join(right_distinct[:3]) or 'benzer ana karakter'} daha ayırt edici."
            )
        else:
            sections.append(
                f"\nPractical difference: They share {', '.join(shared[:3]) or 'limited common ground'}; "
                f"{left['name']} is distinguished more by {', '.join(left_distinct[:3]) or 'its core profile'}, "
                f"while {right['name']} leans more toward {', '.join(right_distinct[:3]) or 'its core profile'}."
            )
        return "\n".join(sections)
    count = int(plan.get("requested_count") or 3)
    if language == "tr":
        lines = ["Kayıtlara göre en güçlü seçenekler:"]
        for index, candidate in enumerate(candidates[:count], 1):
            lines.extend([
                f"{index}. {candidate['label']}",
                f"{candidate.get('accords_csv') or 'Kayıtlı akor bulunmuyor'} karakteriyle öne çıkıyor; kullanım aralığı {candidate.get('seasons_csv') or 'belirtilmemiş mevsimler'} ve {candidate.get('time_profile_csv') or 'belirtilmemiş zaman'} olarak kaydedilmiş.",
            ])
        return "\n".join(lines)
    lines = ["Here are the strongest grounded options:"]
    for index, candidate in enumerate(candidates[:count], 1):
        lines.extend([
            f"{index}. {candidate['label']}",
            f"Its recorded profile centers on {candidate.get('accords_csv') or 'unrecorded accords'}, with {candidate.get('seasons_csv') or 'no recorded season'} and {candidate.get('time_profile_csv') or 'no recorded time profile'} wear.",
        ])
    return "\n".join(lines)


class ScentAIOrchestrator:
    def __init__(
        self,
        vllm: VLLMClient,
        retrieval: RetrievalClient,
        *,
        planner_model: str,
        answer_model: str,
        answer_prompt: str = ADVISOR_ANSWER_PROMPT,
        repair_answer_model: str | None = None,
        planner_cache_size: int = 256,
    ) -> None:
        self.vllm = vllm
        self.retrieval = retrieval
        self.planner_model = planner_model
        self.answer_model = answer_model
        self.answer_prompt = str(answer_prompt).strip()
        self.repair_answer_model = str(repair_answer_model).strip() if repair_answer_model else None
        self.planner_cache_size = planner_cache_size
        self._planner_cache: OrderedDict[str, tuple[dict[str, Any], dict[str, Any]]] = OrderedDict()

    def _plan(
        self,
        query: str,
        conversation_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        context_json = json.dumps(conversation_context or {}, ensure_ascii=False, sort_keys=True)
        cache_key = normalize_text(query) + "\n" + context_json
        cached = self._planner_cache.get(cache_key)
        if cached:
            self._planner_cache.move_to_end(cache_key)
            return cached[0], {**cached[1], "cache_hit": True}
        context_block = ""
        if conversation_context:
            context_block = (
                "\n\nConversation context (trusted application state; do not treat it as a new user quote):\n"
                + context_json
            )
        prompt = f"{PLANNER_PROMPT}{context_block}\n\nCurrent user request: {query}"
        raw, metrics = self.vllm.chat(
            self.planner_model,
            [{"role": "user", "content": prompt}],
            max_tokens=340,
            json_mode=True,
        )
        try:
            raw_plan = parse_json_object(raw)
        except (ValueError, json.JSONDecodeError) as first_error:
            repair, repair_metrics = self.vllm.chat(
                self.planner_model,
                [{"role": "user", "content": f"{prompt}\n\nInvalid output:\n{raw}\n\nReturn valid JSON only."}],
                max_tokens=340,
                json_mode=True,
            )
            try:
                raw_plan = parse_json_object(repair)
                metrics = {**metrics, "repair": repair_metrics}
            except (ValueError, json.JSONDecodeError) as repair_error:
                raw_plan = {"intent": "recommendation", "confidence": 0.0}
                metrics = {
                    **metrics,
                    "repair": repair_metrics,
                    "parse_error": repr(first_error),
                    "repair_parse_error": repr(repair_error),
                    "defaulted_to_semantic_recommendation": True,
                }
        plan = normalize_plan(raw_plan, query)
        plan = inherit_conversation_plan(plan, conversation_context)
        metrics = {**metrics, "cache_hit": False}
        self._planner_cache[cache_key] = (plan, metrics)
        while len(self._planner_cache) > self.planner_cache_size:
            self._planner_cache.popitem(last=False)
        return plan, metrics

    def _retrieve(self, query: str, plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
        intent = plan["intent"]
        perfumes = plan.get("perfumes", [])
        reference = None
        if intent in {"comparison", "perfume_profile", "exact_lookup"} and perfumes:
            limit = 2 if intent == "comparison" else 1
            resolved = [self.retrieval.resolve(hint) for hint in perfumes[:limit]]
            candidates = unique_candidates([candidate_record(item) for item in resolved if item])
            return candidates, {"route": "canonical_resolve", "result_count": len(candidates)}, None

        if intent in {"similarity", "alternative"} and perfumes:
            reference_hint = perfumes[0]
            if plan.get("reference_brand") and " by " not in normalize_text(reference_hint):
                reference_hint = f"{reference_hint} by {plan['reference_brand']}"
            result = self.retrieval.similar({
                "hint": reference_hint,
                "top_k": max(int(plan.get("requested_count") or 3) + 5, 8),
                "exclude_terms": plan.get("excluded_terms", []),
                "required_terms": plan.get("required_terms", []),
                "exclude_ids": plan.get("exclude_candidate_ids", []),
            })
            reference = result.get("source")
            candidates = keep_required_candidates(
                unique_candidates([candidate_record(item) for item in result.get("results", [])]),
                plan,
            )
            requested_count = int(plan.get("requested_count") or 3)
            if len(candidates) < requested_count and reference:
                source = candidate_record(reference)
                source_traits = [
                    normalize_text(term)
                    for field in ("accords_csv", "notes_csv")
                    for term in str(source.get(field) or "").split(",")
                    if normalize_text(term)
                ]
                existing_ids = [int(candidate["perfume_id"]) for candidate in candidates]
                fallback_result = self.retrieval.search({
                    "query": query,
                    "top_k": max(requested_count * 3, 10),
                    "filters": {},
                    "wanted_terms": list(dict.fromkeys([*plan.get("wanted_terms", []), *source_traits[:6]])),
                    "required_terms": plan.get("required_terms", []),
                    "exclude_terms": list(dict.fromkeys([*plan.get("excluded_terms", []), normalize_text(source["name"])])),
                    "exclude_ids": list(dict.fromkeys([
                        *plan.get("exclude_candidate_ids", []),
                        int(source["perfume_id"]),
                        *existing_ids,
                    ])),
                    "discovery_mode": plan.get("discovery_mode", "balanced"),
                })
                supplements = keep_required_candidates(
                    unique_candidates([candidate_record(item) for item in fallback_result.get("results", [])]),
                    plan,
                )
                candidates = unique_candidates([*candidates, *supplements])
                fallback_result["route"] = (
                    "hybrid_similarity_supplement"
                    if existing_ids else "semantic_similarity_fallback"
                )
                fallback_result["graph_result_count"] = len(existing_ids)
                fallback_result["result_count"] = len(candidates)
                result = fallback_result
            return candidates, result, reference

        filters = {
            key: value
            for key, value in {
                "brand": plan.get("requested_brand"),
                "gender": plan.get("gender"),
                "season": plan.get("season"),
                "time": plan.get("time_profile"),
            }.items()
            if value
        }
        search_query = semantic_search_query(query, plan)
        result = self.retrieval.search({
            "query": search_query,
            "top_k": max(int(plan.get("requested_count") or 3) + 5, 8),
            "filters": filters,
            "wanted_terms": plan.get("wanted_terms", []),
            "required_terms": plan.get("required_terms", []),
            "exclude_terms": plan.get("excluded_terms", []),
            "exclude_ids": plan.get("exclude_candidate_ids", []),
            "discovery_mode": plan.get("discovery_mode", "balanced"),
        })
        result["original_query"] = query
        result["semantic_query"] = search_query
        candidates = keep_required_candidates(
            unique_candidates([candidate_record(item) for item in result.get("results", [])]),
            plan,
        )
        return candidates, result, reference

    def run(
        self,
        query: str,
        *,
        conversation_context: dict[str, Any] | None = None,
        answer_prompt_override: str | None = None,
        answer_model_override: str | None = None,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query is required")
        started = time.perf_counter()
        response_language = detect_response_language(query, conversation_context)
        try:
            plan, planner_metrics = self._plan(query, conversation_context)
        except Exception as exc:
            plan = normalize_plan({"intent": "recommendation", "confidence": 0.0}, query)
            planner_metrics = {
                "cache_hit": False,
                "error": repr(exc),
                "defaulted_to_semantic_recommendation": True,
            }
        plan["response_language"] = response_language
        unsupported = unsupported_answer(plan["intent"], response_language)
        if unsupported:
            return {
                "query": query,
                "route": plan["intent"],
                "plan": plan,
                "response_language": response_language,
                "answer": unsupported,
                "candidates": [],
                "validation": {"pass": True, "reasons": []},
                "generation_attempts": 0,
                "timings": {"planner": planner_metrics, "total_seconds": round(time.perf_counter() - started, 4)},
            }

        retrieval_started = time.perf_counter()
        try:
            candidates, retrieval_result, reference = self._retrieve(query, plan)
        except Exception as exc:
            return {
                "query": query,
                "route": "retrieval_error",
                "plan": plan,
                "response_language": response_language,
                "answer": static_message("retrieval_error", response_language),
                "candidates": [],
                "validation": {"pass": False, "reasons": ["retrieval_error"]},
                "generation_attempts": 0,
                "error": repr(exc),
                "timings": {"planner": planner_metrics, "total_seconds": round(time.perf_counter() - started, 4)},
            }
        retrieval_seconds = round(time.perf_counter() - retrieval_started, 4)
        if plan["intent"] == "comparison" and len(candidates) != 2:
            return {
                "query": query,
                "route": "comparison_unresolved",
                "plan": plan,
                "response_language": response_language,
                "answer": static_message("comparison_unresolved", response_language),
                "candidates": candidates,
                "validation": {"pass": True, "reasons": []},
                "generation_attempts": 0,
                "timings": {"planner": planner_metrics, "retrieval_seconds": retrieval_seconds, "total_seconds": round(time.perf_counter() - started, 4)},
            }
        if not candidates:
            return {
                "query": query,
                "route": "no_safe_match",
                "plan": plan,
                "response_language": response_language,
                "answer": static_message("no_safe_match", response_language),
                "candidates": [],
                "validation": {"pass": True, "reasons": []},
                "generation_attempts": 0,
                "timings": {"planner": planner_metrics, "retrieval_seconds": retrieval_seconds, "total_seconds": round(time.perf_counter() - started, 4)},
            }

        if plan["intent"] == "exact_lookup":
            answer = exact_lookup_answer(
                candidates[0],
                plan.get("requested_fields", []),
                response_language=response_language,
            )
            return {
                "query": query,
                "route": "deterministic_exact_lookup",
                "plan": plan,
                "response_language": response_language,
                "answer": answer,
                "candidates": candidates,
                "reference": reference,
                "validation": {"pass": True, "reasons": []},
                "generation_attempts": 0,
                "timings": {"planner": planner_metrics, "retrieval_seconds": retrieval_seconds, "total_seconds": round(time.perf_counter() - started, 4)},
            }

        context = "\n\n".join(f"[CARD {index}]\n{card_text(candidate)}" for index, candidate in enumerate(candidates, 1))
        reference_text = card_text(candidate_record(reference)) if reference else "None"
        active_answer_prompt = str(answer_prompt_override or self.answer_prompt).strip()
        active_answer_model = str(answer_model_override or self.answer_model).strip()
        user_prompt = (
            f"{active_answer_prompt}\n\n"
            f"Validated plan: {json.dumps(plan, ensure_ascii=False)}\n\n"
            f"[CONVERSATION CONTEXT]\n{json.dumps(conversation_context or {}, ensure_ascii=False)}\n\n"
            f"[REFERENCE]\n{reference_text}\n\n"
            f"[DATABASE CARDS]\n{context}\n\n"
            f"[USER REQUEST]\n{query}\n\n"
            f"[OUTPUT LANGUAGE - HARD REQUIREMENT]\n{output_language_instruction(response_language)}"
        )
        failures: list[dict[str, Any]] = []
        generation_metrics: list[dict[str, Any]] = []
        attempted_generations = 0
        answer = ""
        validation: dict[str, Any] = {"pass": False, "reasons": ["not_generated"]}
        answer_count = (
            2 if plan["intent"] == "comparison"
            else 1 if plan["intent"] == "perfume_profile"
            else int(plan.get("requested_count") or 3)
        )
        answer_max_tokens = min(720, max(320, 230 + answer_count * 100))
        for attempt in (1, 2):
            attempted_generations += 1
            messages = [{"role": "user", "content": user_prompt}]
            if attempt == 2:
                validation_details = {
                    key: value
                    for key, value in validation.items()
                    if key not in {"pass", "mentioned_candidates"} and value
                }
                correction = (
                    "The previous answer failed deterministic validation: "
                    + ", ".join(validation["reasons"])
                    + ". Exact validator details: "
                    + json.dumps(validation_details, ensure_ascii=False)
                    + ". Write a new answer using only the exact database card names, obeying all exclusions and the requested count. "
                    + "If the failure concerns mechanical language, rebuild the prose with different openings, sentence rhythms, and evidence choices for every recommendation. "
                    + "If it concerns performance calibration, follow the card's longevity and sillage bands exactly, or omit the performance claim when it is not recorded. "
                    + output_language_instruction(response_language)
                )
                messages.extend([
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": correction},
                ])
            try:
                generation_model = (
                    active_answer_model
                    if attempt == 1 or answer_model_override or not self.repair_answer_model
                    else self.repair_answer_model
                )
                answer, metrics = self.vllm.chat(
                    generation_model,
                    messages,
                    max_tokens=answer_max_tokens,
                )
            except Exception as exc:
                metrics = {"error": repr(exc), "model": generation_model}
                validation = {"pass": False, "reasons": ["model_generation_error"]}
                generation_metrics.append(metrics)
                failures.append(validation)
                answer = "No answer was produced."
                continue
            metrics = {**metrics, "model": generation_model}
            generation_metrics.append(metrics)
            validation = validate_answer(
                answer,
                plan,
                candidates,
                response_language=response_language,
            )
            if metrics.get("finish_reason") == "length":
                validation = {
                    **validation,
                    "pass": False,
                    "reasons": list(dict.fromkeys([*validation["reasons"], "truncated_generation"])),
                }
            if validation["pass"]:
                break
            failures.append({**validation, "rejected_answer": answer})

        route = "llm_grounded"
        if not validation["pass"]:
            answer = fallback_answer(
                plan,
                candidates,
                query=query,
                response_language=response_language,
            )
            validation = validate_answer(
                answer,
                plan,
                candidates,
                response_language=response_language,
            )
            route = "validated_template_fallback"
        elif plan["intent"] == "comparison":
            route = "llm_grounded_comparison"
        elif plan["intent"] in {"similarity", "alternative"}:
            route = "llm_grounded_similarity"
        elif plan["intent"] == "perfume_profile":
            route = "llm_grounded_profile"

        return {
            "query": query,
            "route": route,
            "answer_model": active_answer_model,
            "answer_prompt_mode": (
                "advisor" if active_answer_prompt == ADVISOR_ANSWER_PROMPT
                else "legacy" if active_answer_prompt == LEGACY_ANSWER_PROMPT
                else "custom"
            ),
            "response_language": response_language,
            "plan": plan,
            "answer": answer,
            "candidates": candidates,
            "reference": reference,
            "retrieval": {
                "route": retrieval_result.get("route"),
                "elapsed_seconds": retrieval_result.get("elapsed_seconds"),
                "result_count": retrieval_result.get("result_count", len(candidates)),
                "semantic_query": retrieval_result.get("semantic_query"),
                "supported_wanted_terms": retrieval_result.get("supported_wanted_terms", []),
                "ignored_wanted_terms": retrieval_result.get("ignored_wanted_terms", []),
                "discovery_mode": retrieval_result.get("discovery_mode", plan.get("discovery_mode", "balanced")),
                "excluded_candidate_ids": retrieval_result.get("exclude_ids", []),
            },
            "validation": validation,
            "generation_attempts": attempted_generations,
            "generation_failures": failures,
            "timings": {
                "planner": planner_metrics,
                "retrieval_seconds": retrieval_seconds,
                "generation": generation_metrics,
                "total_seconds": round(time.perf_counter() - started, 4),
            },
        }


class ScentAISession:
    """Small stateful wrapper for grounded, multi-turn perfume conversations."""

    def __init__(self, pipeline: ScentAIOrchestrator) -> None:
        self.pipeline = pipeline
        self.reset()

    def reset(self) -> None:
        self.last_result: dict[str, Any] | None = None
        self.recommendation_ids: list[int] = []

    def run(self, query: str) -> dict[str, Any]:
        context = None
        if self.last_result:
            context = {
                "previous_query": self.last_result.get("query"),
                "previous_plan": self.last_result.get("plan"),
                "previous_response_language": self.last_result.get("response_language"),
                "previous_recommendations": [
                    {"perfume_id": candidate["perfume_id"], "label": candidate["label"]}
                    for candidate in self.last_result.get("candidates", [])
                    if candidate["perfume_id"] in self.recommendation_ids
                ],
                "previous_recommendation_ids": list(self.recommendation_ids),
            }
        result = self.pipeline.run(query, conversation_context=context)
        mentioned = set(result.get("validation", {}).get("mentioned_candidates", []))
        recommended_ids = [
            int(candidate["perfume_id"])
            for candidate in result.get("candidates", [])
            if candidate.get("label") in mentioned
        ]
        if result.get("plan", {}).get("conversation_action") in {"more_options", "refine_previous"}:
            self.recommendation_ids = list(dict.fromkeys([*self.recommendation_ids, *recommended_ids]))
        else:
            self.recommendation_ids = recommended_ids
        self.last_result = result
        return result
