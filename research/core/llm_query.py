from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from research.core.semantic import SemanticProfile


QueryProvider = Literal["auto", "gemini", "compat", "pool", "fallback"]
FallbackPolicy = Literal["template", "fail"]


def load_provider_pool(path: Path | str) -> list[dict[str, Any]]:
    pool_path = Path(path)
    raw = json.loads(pool_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Provider pool must be a list: {pool_path}")

    providers: list[dict[str, Any]] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Provider pool item #{index} must be an object")
        name = str(item.get("name") or "").strip()
        base_url = str(item.get("base_url") or "").strip().rstrip("/")
        api_key_env = str(item.get("api_key_env") or "").strip()
        model = str(item.get("model") or "").strip()
        weight = int(item.get("weight") or 1)
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if not name or not base_url or not model:
            raise ValueError(f"Provider pool item #{index} is missing name/base_url/model")
        if not api_key:
            print(f"WARNING: {api_key_env} is not set; skipping provider {name}.")
            continue
        providers.append(
            {
                "name": name,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "weight": max(weight, 1),
            }
        )
    return providers


def weighted_provider_order(providers: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    remaining = list(providers)
    ordered = []
    while remaining:
        total = sum(max(provider["weight"], 1) for provider in remaining)
        pick = rng.uniform(0, total)
        cursor = 0.0
        for index, provider in enumerate(remaining):
            cursor += max(provider["weight"], 1)
            if cursor >= pick:
                ordered.append(provider)
                del remaining[index]
                break
    return ordered


class QueryGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        sleep_seconds: float = 4.0,
        use_api: bool = True,
        provider: QueryProvider = "auto",
        gemini_model: str = "gemini-2.5-flash-lite",
        compat_base_url: str | None = None,
        compat_api_key: str | None = None,
        compat_model: str = "llama-3.3-70b-versatile",
        provider_pool_path: Path | str | None = None,
        max_retries: int = 3,
        fallback_policy: FallbackPolicy = "template",
    ) -> None:
        self.provider = "fallback" if not use_api else provider
        self.gemini_model = model or gemini_model
        self.compat_base_url = (compat_base_url or os.environ.get("OPENAI_COMPAT_BASE_URL") or "").rstrip("/")
        self.compat_api_key = compat_api_key or os.environ.get("OPENAI_COMPAT_API_KEY")
        self.compat_model = model or compat_model or os.environ.get("OPENAI_COMPAT_MODEL") or "llama-3.3-70b-versatile"
        self.provider_pool = load_provider_pool(provider_pool_path) if provider_pool_path else []
        self.disabled_pool_providers: set[str] = set()
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.fallback_policy = fallback_policy
        self.gemini_client = None
        self.compat_enabled = False
        self._gemini_disabled = False
        self._compat_disabled = False
        self.last_source = "fallback"
        self.seen_queries: set[str] = set()
        self.duplicate_retries = 2

        if self.provider == "pool":
            if not self.provider_pool:
                print("WARNING: provider pool is empty, using fallback queries.")
                self.provider = "fallback"

        if self.provider in ("auto", "compat"):
            if self.compat_base_url and self.compat_api_key:
                self.compat_enabled = True
                self.provider = "compat"
            elif self.provider == "compat":
                print("WARNING: OpenAI-compatible provider is not configured, using fallback queries.")

        if self.provider in ("auto", "gemini") and not self.compat_enabled:
            gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                try:
                    from google import genai

                    self.gemini_client = genai.Client(api_key=gemini_key)
                    self.provider = "gemini"
                except Exception as exc:
                    print(f"WARNING: Gemini client unavailable, using fallback queries: {exc}")
            elif self.provider == "gemini":
                print("WARNING: GEMINI_API_KEY is not set, using fallback queries.")

        if self.provider != "pool" and self.gemini_client is None and not self.compat_enabled:
            self.provider = "fallback"

    def generate(self, category: str, profile: SemanticProfile, rng: random.Random, messy: bool = False) -> str:
        for attempt in range(self.duplicate_retries + 1):
            query = self._generate_api_query(category, profile, rng, messy)
            if not query:
                break
            if self._remember_query(query):
                return query
            if attempt < self.duplicate_retries:
                print(f"Duplicate query generated; retrying ({attempt + 1}/{self.duplicate_retries})")

        self.last_source = "fallback"
        if self.fallback_policy == "fail":
            raise RuntimeError("LLM query generation failed and fallback_policy=fail")
        query = fallback_query(category, profile, rng, messy)
        self._remember_query(query)
        return query

    def _generate_api_query(
        self,
        category: str,
        profile: SemanticProfile,
        rng: random.Random,
        messy: bool,
    ) -> str | None:
        if self.provider == "pool" and self.provider_pool:
            query = self._generate_with_pool(category, profile, messy, rng)
            if query:
                return query
        if self.gemini_client and not self._gemini_disabled:
            query = self._generate_with_gemini(category, profile, messy, rng)
            if query:
                self.last_source = "gemini"
                return query
        if self.compat_enabled and not self._compat_disabled:
            query = self._generate_with_compat(category, profile, messy, rng)
            if query:
                self.last_source = "compat"
                return query
        return None

    def _remember_query(self, query: str) -> bool:
        normalized = " ".join(query.lower().split())
        if normalized in self.seen_queries:
            return False
        self.seen_queries.add(normalized)
        return True

    def _generate_with_gemini(
        self,
        category: str,
        profile: SemanticProfile,
        messy: bool,
        rng: random.Random,
    ) -> str | None:
        from google.genai import types

        prompt = _adapt_prompt_for_model(build_prompt(category, profile, messy, rng), self.compat_model, self.compat_base_url)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.85, max_output_tokens=90),
                )
                time.sleep(self.sleep_seconds)
                text = _clean_query_text(response.text or "")
                if not text or _is_bad_query(text):
                    if attempt < self.max_retries:
                        continue
                    return None
                return text
            except Exception as exc:
                if _is_quota_error(exc):
                    print(f"Gemini quota exceeded, disabling Gemini and using fallback: {exc}")
                    self._gemini_disabled = True
                    self.gemini_client = None
                    self.provider = "fallback"
                    return None
                if _is_transient_error(exc) and attempt < self.max_retries:
                    delay = min(10 * (attempt + 1), 30)
                    print(f"Gemini temporarily unavailable; retrying in {delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                print(f"Gemini query generation failed, using fallback: {exc}")
                return None

    def _generate_with_compat(
        self,
        category: str,
        profile: SemanticProfile,
        messy: bool,
        rng: random.Random,
    ) -> str | None:
        prompt = _adapt_prompt_for_model(
            build_prompt(category, profile, messy, rng),
            self.compat_model,
            self.compat_base_url,
        )
        url = f"{self.compat_base_url}/chat/completions"
        payload = {
            "model": self.compat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.85,
            "max_tokens": 90,
        }
        _add_reasoning_controls(payload, self.compat_model, self.compat_base_url)
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.compat_api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "ScentAI-Dataset-Generator/0.1",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                text = _clean_query_text(text)
                if not text or _is_bad_query(text):
                    if attempt < self.max_retries:
                        continue
                    return None
                time.sleep(self.sleep_seconds)
                return text
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error_text = f"{exc.code} {body}"
                if "429" in error_text and attempt < self.max_retries:
                    delay = min(10 * (attempt + 1), 30)
                    print(f"OpenAI-compatible provider rate-limited; retrying in {delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                print(f"OpenAI-compatible query generation failed, using fallback: {error_text}")
                if "429" in error_text or "403" in error_text:
                    self._compat_disabled = True
                    self.compat_enabled = False
                return None
            except Exception as exc:
                if _is_transient_error(exc) and attempt < self.max_retries:
                    delay = min(10 * (attempt + 1), 30)
                    print(f"OpenAI-compatible provider temporarily unavailable; retrying in {delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                print(f"OpenAI-compatible query generation failed, using fallback: {exc}")
                return None

    def _generate_with_pool(
        self,
        category: str,
        profile: SemanticProfile,
        messy: bool,
        rng: random.Random,
    ) -> str | None:
        providers = [p for p in self.provider_pool if p["name"] not in self.disabled_pool_providers]
        providers = weighted_provider_order(providers, rng)

        for provider in providers:
            query = self._generate_with_compat_provider(category, profile, messy, provider, rng)
            if query:
                self.last_source = f"pool:{provider['name']}"
                return query
        return None

    def _generate_with_compat_provider(
        self,
        category: str,
        profile: SemanticProfile,
        messy: bool,
        provider: dict[str, Any],
        rng: random.Random,
    ) -> str | None:
        prompt = _adapt_prompt_for_model(
            build_prompt(category, profile, messy, rng),
            provider["model"],
            provider["base_url"],
        )
        url = f"{provider['base_url']}/chat/completions"
        payload = {
            "model": provider["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.85,
            "max_tokens": 90,
        }
        _add_reasoning_controls(payload, provider["model"], provider["base_url"])
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {provider['api_key']}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "ScentAI-Dataset-Generator/0.1",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                text = _clean_query_text(text)
                if not text or _is_bad_query(text):
                    if attempt < self.max_retries:
                        continue
                    return None
                time.sleep(self.sleep_seconds)
                return text
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error_text = f"{exc.code} {body}"
                if "429" in error_text and attempt < self.max_retries:
                    delay = min(10 * (attempt + 1), 30)
                    print(
                        f"Provider {provider['name']} rate-limited; "
                        f"retrying in {delay}s ({attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue
                print(f"Provider {provider['name']} query generation failed: {error_text}")
                if "429" in error_text or "403" in error_text:
                    self.disabled_pool_providers.add(provider["name"])
                return None
            except Exception as exc:
                if _is_transient_error(exc) and attempt < self.max_retries:
                    delay = min(10 * (attempt + 1), 30)
                    print(
                        f"Provider {provider['name']} temporarily unavailable; "
                        f"retrying in {delay}s ({attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue
                print(f"Provider {provider['name']} query generation failed: {exc}")
                return None


def _extract_response_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _clean_query_text(text: str | None) -> str:
    text = (text or "").strip().strip('"').strip("'").strip()
    text = re.sub(r"(?is)<think>\s*</think>", "", text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    return " ".join(text.split())


def _is_bad_query(text: str) -> bool:
    words = text.split()
    lowered = text.lower()
    if len(text) < 25 or len(words) < 5:
        return True
    if text.endswith(("'", '"', ",", ":")):
        return True
    if lowered in {"i", "i'm", "i'm looking", "hey, can", "i'"}:
        return True
    if "<think" in lowered or "</think" in lowered or lowered.startswith("think "):
        return True
    if "do you love" in lowered or "are you more into" in lowered:
        return True
    if _has_banned_opening(text):
        return True
    return False


def _add_reasoning_controls(payload: dict[str, Any], model: str, base_url: str) -> None:
    model_id = model.lower()
    base = base_url.lower()
    if "api.groq.com" in base and "openai/gpt-oss" in model_id:
        payload["reasoning_format"] = "hidden"
    if "openrouter.ai" in base and "deepseek/deepseek-v4-flash" in model_id:
        payload["reasoning"] = {"enabled": False}


def _adapt_prompt_for_model(prompt: str, model: str, base_url: str) -> str:
    model_id = model.lower()
    base = base_url.lower()
    if "api.groq.com" in base and "qwen/qwen3" in model_id:
        return f"{prompt}\n/no_think"
    return prompt


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc)
    return "503" in text or "UNAVAILABLE" in text or "high demand" in text.lower()


def build_prompt(category: str, profile: SemanticProfile, messy: bool, rng: random.Random | None = None) -> str:
    style = (
        "Messy casual English; minor typo/slang is OK."
        if messy
        else "Natural English, 1-2 short sentences."
    )
    category_instruction = _category_instruction(category)
    opening_instruction = _opening_instruction(category, rng)
    profile_text = _compact_profile_text(profile)
    return (
        "Write only a perfume user's request. No answer.\n"
        f"cat={category}\n"
        f"profile={profile_text}\n"
        f"style={style}\n"
        f"task={category_instruction}\n"
        f"opening={opening_instruction}\n"
        "Rules: >=8 words; vary openings; "
        "do not start with 'Could you suggest', 'Could you recommend', 'Can you suggest', "
        "'Can you recommend', \"I'm looking for\", 'I want something', 'I need something', "
        "or 'Lately I have been'; "
        "avoid 'perfect for everyday wear'; no words accord/metadata/criteria/profile; "
        "do not ask the assistant about its own taste; "
        "do not list hidden fields. Output only the query."
    )


def _has_banned_opening(text: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z']+", text.lower().replace("’", "'"))[:5])
    banned = (
        "could you suggest",
        "could you recommend",
        "can you suggest",
        "can you recommend",
        "i'm looking for",
        "im looking for",
    )
    return normalized.startswith(banned)


def _opening_instruction(category: str, rng: random.Random | None) -> str:
    chooser = rng or random
    general = [
        "start with a short personal situation, not a polite assistant request",
        "start with an occasion or time phrase, then ask naturally",
        "write it like a concise app search request with no question opener",
        "start with a taste description such as 'Lately...' or 'I usually...'",
        "start with a constraint or avoidance phrase, then explain what is wanted",
        "make it sound like a real shopper comparing needs, not a form field",
    ]
    by_category = {
        "empty_profile": [
            "start with the situation first, because there is no stored taste yet",
            "write a simple first-time request without mentioning saved preferences",
        ],
        "profile_likes": [
            "start from remembered taste or a liked style without listing the profile",
            "start with 'I usually...' or 'My taste leans...' in natural language",
        ],
        "profile_dislikes": [
            "start with what should be avoided, then ask for the target mood",
            "start with 'I am trying to avoid...' or a similar natural constraint",
        ],
        "profile_likes_and_dislikes": [
            "start by balancing what the user likes and avoids",
            "start with a taste tradeoff, not a generic recommendation phrase",
        ],
        "avoid_previous_recommendations": [
            "start with not wanting repeats from earlier suggestions",
            "start with a freshness/new-option request before the occasion",
        ],
        "profile_query_conflict": [
            "start by admitting the request is a little outside usual taste",
            "start with curiosity about trying a different direction",
        ],
        "profile_update_request": [
            "start with changed taste, recent shift, or no longer avoiding something",
            "start with a preference update before asking for a scent",
        ],
        "low_confidence_profile": [
            "start with uncertainty about taste and ask for a gentle recommendation",
            "start by saying the user is still figuring out what works",
        ],
    }
    options = by_category.get(category, []) + general
    return chooser.choice(options)


def _compact_profile_text(profile: SemanticProfile) -> str:
    parts = [f"label:{profile.label}"]
    _add_part(parts, "gender", profile.gender_any_of)
    _add_part(parts, "season", profile.hard_seasons)
    _add_part(parts, "time", profile.hard_times)
    _add_part(parts, "want_a", profile.wanted_accords)
    _add_part(parts, "want_n", profile.wanted_notes)
    _add_part(parts, "soft_a", profile.soft_accords)
    _add_part(parts, "soft_n", profile.soft_notes)
    _add_part(parts, "avoid_a", profile.avoid_accords)
    _add_part(parts, "avoid_n", profile.avoid_notes)
    if profile.reference_name and profile.reference_brand:
        parts.append(f"ref:{profile.reference_name} by {profile.reference_brand}")
    options = profile.extra.get("options")
    if options:
        parts.append(f"options:{_join(tuple(options))}")
    collection = profile.extra.get("collection")
    if collection:
        parts.append(f"collection:{_join(tuple(collection))}")
    return "; ".join(parts)


def _add_part(parts: list[str], key: str, values: tuple[str, ...]) -> None:
    if values:
        parts.append(f"{key}:{_join(values)}")


def _category_instruction(category: str) -> str:
    instructions = {
        "casual_vibe": "Express a mood or vibe casually, with varied phrasing.",
        "occasion": "Mention the situation or occasion naturally, not as a filter list.",
        "likes_dislikes": "Mention likes and dislikes conversationally.",
        "negative_preference": "Make the avoidance sound explicit but natural.",
        "reference_similarity": "Mention the reference perfume and ask for a similar alternative.",
        "messy_query": "Use lowercase, casual phrasing, minor typos, or incomplete but understandable wording.",
        "conceptual_contradiction": "Express the two opposing qualities in a natural way.",
        "persona_upgrade": "Mention a current perfume or habit and ask for a more fitting upgrade or variation.",
        "occasion_with_tradeoffs": "Mention an occasion plus a realistic tradeoff or constraint.",
        "compare_and_decide": "Ask the assistant to choose between the named options for a specific scenario.",
        "avoidance_reasoning": "Mention explicit dislikes and ask for a recommendation that avoids them.",
        "style_translation": "Express an abstract style or aesthetic in natural user language.",
        "collection_gap": "Mention a small current collection and ask what role to add next.",
        "no_strong_match": "Ask whether the available options fit a specific request, allowing that none may be perfect.",
        "empty_profile": "Ask a normal perfume recommendation request without implying stored preferences.",
        "profile_likes": "Ask for a recommendation where the stored likes can help, but do not list profile fields.",
        "profile_dislikes": "Ask for a recommendation where the stored dislikes should be respected.",
        "profile_likes_and_dislikes": "Ask naturally for a recommendation that can use both liked and disliked taste signals.",
        "avoid_previous_recommendations": "Ask for something new or different, implying previous suggestions should not repeat.",
        "profile_query_conflict": "Ask for something that partially conflicts with the stored profile.",
        "profile_update_request": "State that a previous preference has changed and ask for a recommendation.",
        "low_confidence_profile": "Ask naturally while leaving room for the assistant not to over-personalize.",
    }
    return instructions.get(category, "Write a varied natural user request.")


def fallback_query(category: str, profile: SemanticProfile, rng: random.Random, messy: bool = False) -> str:
    label = profile.label.replace("_", " ")
    gender = _gender_phrase(profile)
    gender_tail = _gender_tail(profile)
    avoid = _avoid_phrase(profile)
    reference = _reference_phrase(profile)

    templates = {
        "casual_vibe": [
            "I want something that feels {label}{gender}{avoid}.",
            "Can you suggest a perfume with a {label} kind of mood{gender}{avoid}?",
            "Looking for a scent that gives {label} energy{gender}{avoid}.",
        ],
        "occasion": [
            "I need a perfume for {label}{gender}{avoid}.",
            "What should I wear for {label}{gender}{avoid}?",
            "Find me something that works for {label}{gender}{avoid}.",
        ],
        "likes_dislikes": [
            "I like {likes} but I want to avoid {avoids}{gender_tail}.",
            "Can you find something with {likes}, just no {avoids}{gender_tail}?",
        ],
        "negative_preference": [
            "I want a {label} scent{gender}, but no {avoids}.",
            "Something {label}{gender} please, without {avoids}.",
        ],
        "reference_similarity": [
            "I love {reference}, but I want to try something with a similar vibe.",
            "Can you suggest something like {reference}, maybe a little different?",
        ],
        "messy_query": [
            "need smth {label} {gender} {avoid} pls",
            "yo looking for a {label} smell {gender} {avoid} nothing too much",
        ],
        "conceptual_contradiction": [
            "I want something {label}{gender}{avoid}.",
            "Can you find a perfume that feels {label}{gender}{avoid}?",
        ],
        "persona_upgrade": [
            "I usually wear {reference}, but I want something better suited to {label}{gender_tail}.",
            "I like {reference}, but I need a more polished option for {label}{gender_tail}.",
        ],
        "occasion_with_tradeoffs": [
            "I need something for {label}, but I still want it to feel balanced and wearable{gender_tail}.",
            "Can you help me pick a {label} scent that is noticeable without being too much{gender_tail}?",
        ],
        "compare_and_decide": [
            "Between {options}, which one would you choose for {label} and why?",
            "Can you compare {options} and tell me the best pick for {label}?",
        ],
        "avoidance_reasoning": [
            "I want something {label}{gender}, but I really want to avoid {avoids}.",
            "Can you recommend something in a {label} direction{gender_tail} that does not lean into {avoids}?",
        ],
        "style_translation": [
            "I want a perfume that feels {label}, but still easy to wear{gender_tail}.",
            "Can you suggest something with a {label} style that does not feel forced{gender_tail}?",
        ],
        "collection_gap": [
            "I already own {collection}. What should I add next for {label}?",
            "My small collection is {collection}; what kind of {label} scent would fill the gap?",
        ],
        "no_strong_match": [
            "Do any of these really fit a {label} scent{gender}{avoid}, or is there no strong match?",
            "From these options, is anything actually good for {label}{gender}{avoid}?",
        ],
        "empty_profile": [
            "Can you recommend something for {label}{gender}{avoid}?",
            "I need a perfume that works for {label}{gender}{avoid}.",
        ],
        "profile_likes": [
            "Can you recommend something for {label}{gender} based on what I usually enjoy?",
            "I want a {label} scent{gender}; use my taste if it helps.",
        ],
        "profile_dislikes": [
            "Can you recommend something for {label}{gender} while avoiding what I usually dislike?",
            "I need a {label} scent{gender}, but please keep my dislikes in mind.",
        ],
        "profile_likes_and_dislikes": [
            "Can you find a {label} scent{gender} that fits my taste without hitting my dislikes?",
            "I want something for {label}{gender}; please balance what I like and avoid.",
        ],
        "avoid_previous_recommendations": [
            "Can you suggest something new for {label}{gender}, not the same things as before?",
            "I want another option for {label}{gender}; avoid repeating previous recommendations.",
        ],
        "profile_query_conflict": [
            "I know this is a bit outside my usual taste, but can you suggest something for {label}{gender}?",
            "Can you help me try a {label} scent{gender} without going too far from what I like?",
        ],
        "profile_update_request": [
            "My taste has changed a bit; can you recommend something for {label}{gender}?",
            "I used to avoid this style, but now I want to try a {label} scent{gender}.",
        ],
        "low_confidence_profile": [
            "Can you recommend something for {label}{gender}, but do not assume too much about my taste yet?",
            "I am still figuring out what I like; suggest something for {label}{gender}.",
        ],
    }
    template = rng.choice(templates.get(category, templates["casual_vibe"]))
    query = template.format(
        label=label,
        gender=gender,
        avoid=avoid,
        avoids=_join(profile.avoid_accords + profile.avoid_notes) or "the notes I dislike",
        likes=_join_unique(profile.wanted_accords + profile.wanted_notes + profile.soft_accords[:2]) or label,
        reference=reference or "my current perfume",
        options=_join(profile.extra.get("options", ())) or "these options",
        collection=_join(profile.extra.get("collection", ())) or "a few daily scents",
        gender_tail=gender_tail,
    )
    query = _fix_query_grammar(query)
    if messy:
        query = query.lower().replace("something", "smth").replace("please", "pls")
    return " ".join(query.split())


def _gender_phrase(profile: SemanticProfile) -> str:
    genders = set(profile.gender_any_of)
    if genders == {"male", "unisex"}:
        return " for men"
    if genders == {"female", "unisex"}:
        return " for women"
    if genders == {"unisex"}:
        return " that feels unisex"
    return ""


def _gender_tail(profile: SemanticProfile) -> str:
    genders = set(profile.gender_any_of)
    if genders == {"male", "unisex"}:
        return " for men"
    if genders == {"female", "unisex"}:
        return " for women"
    if genders == {"unisex"}:
        return " with unisex options"
    return ""


def _avoid_phrase(profile: SemanticProfile) -> str:
    terms = profile.avoid_accords + profile.avoid_notes
    return f", without {_join(terms)}" if terms else ""


def _reference_phrase(profile: SemanticProfile) -> str:
    if profile.reference_name and profile.reference_brand:
        return f"{profile.reference_name} by {profile.reference_brand}"
    return ""


def _join(items: tuple[str, ...]) -> str:
    return _join_unique(items)


def _join_unique(items: tuple[str, ...]) -> str:
    seen = set()
    unique = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return ", ".join(unique)


def _fix_query_grammar(query: str) -> str:
    replacements = {
        " a elegant ": " an elegant ",
        " a expensive ": " an expensive ",
        " a understated ": " an understated ",
        " a office ": " an office ",
        " scent scent": " scent",
    }
    fixed = f" {query} "
    for old, new in replacements.items():
        fixed = fixed.replace(old, new)
    return fixed.strip()
