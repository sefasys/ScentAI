from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "full_pipeline_colab.ipynb"
INFERENCE_NOTEBOOK = ROOT / "notebooks" / "inference_colab.ipynb"
RETRIEVAL_NOTEBOOK = ROOT / "notebooks" / "retrieval_colab.ipynb"
ORCHESTRATOR_SOURCE = (ROOT / "src" / "scentai" / "orchestrator.py").read_text(encoding="utf-8")
FINAL_EVAL_SET_SOURCE = (ROOT / "research" / "evaluation" / "final_eval_v1.jsonl").read_text(encoding="utf-8")
FINAL_EVAL_RUNTIME_SOURCE = (ROOT / "research" / "evaluation" / "final_eval_runtime.py").read_text(encoding="utf-8")


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


inference = nbformat.read(INFERENCE_NOTEBOOK, as_version=4)
retrieval = nbformat.read(RETRIEVAL_NOTEBOOK, as_version=4)

install_uv = inference.cells[1].source
install_inference = inference.cells[3].source
probe_inference = inference.cells[5].source
install_retrieval = retrieval.cells[3].source
probe_retrieval = retrieval.cells[5].source
start_retrieval = retrieval.cells[9].source
start_vllm = inference.cells[9].source
inference_smoke = inference.cells[11].source


cells = [
    markdown(
        """
# ScentAI Stage 3 - Full Grounded Pipeline

This notebook connects the two independently validated boundaries:

`free-form query -> Gemma planner -> BGE-M3/Chroma/catalog -> ScentAI LoRA -> validator`

The vLLM/CUDA stack and CPU retrieval stack remain in separate uv environments.
The Colab kernel imports neither stack and only coordinates their localhost HTTP
APIs. Use a fresh **A100 High-RAM** runtime and choose **Run all**.
"""
    ),
    markdown("## Install the dependency-free uv launcher"),
    code(install_uv),
    markdown("## Create the isolated vLLM environment"),
    code(install_inference),
    markdown("## Create the isolated CPU retrieval environment"),
    code(install_retrieval),
    markdown("## Verify the vLLM/CUDA binary graph"),
    code(probe_inference),
    markdown("## Verify the CPU retrieval dependency graph"),
    code(probe_retrieval),
    markdown("## Mount Drive, validate assets, and stage retrieval data locally"),
    code(
        r'''
import os
import time
from google.colab import drive, userdata

drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/Perfume-Dataset")
MODEL_NAME = "google/gemma-4-12B-it"
LORA_NAME = "scentai"
FULL_ADAPTER_DIR = PROJECT_DIR / "models" / "scentai-gemma-4-12b-it-full-fastmodel-lora" / "best_lora_adapter"
PILOT_ADAPTER_DIR = PROJECT_DIR / "models" / "scentai-gemma-4-12b-it-pilot-fastmodel-lora" / "best_lora_adapter"
DRIVE_CHROMA_DIR = PROJECT_DIR / "chroma_db_bge_m3"
DRIVE_CATALOG = PROJECT_DIR / "scentai_catalog.sqlite3"
LOCAL_CHROMA_DIR = Path("/content/scentai_data/chroma_db_bge_m3")
LOCAL_CATALOG = Path("/content/scentai_data/scentai_catalog.sqlite3")

assert (DRIVE_CHROMA_DIR / "chroma.sqlite3").exists(), f"Missing Chroma DB: {DRIVE_CHROMA_DIR}"
assert DRIVE_CATALOG.exists(), f"Missing catalog: {DRIVE_CATALOG}"

expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

adapter_rejections = []
ADAPTER_DIR = None
adapter_config = None
for candidate in (FULL_ADAPTER_DIR, PILOT_ADAPTER_DIR):
    config_path = candidate / "adapter_config.json"
    weights_path = candidate / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        adapter_rejections.append({"path": str(candidate), "reason": "missing files"})
        continue
    candidate_config = json.loads(config_path.read_text(encoding="utf-8"))
    problems = []
    if candidate_config.get("base_model_name_or_path") != MODEL_NAME:
        problems.append("base model mismatch")
    if int(candidate_config.get("r") or 0) != 16:
        problems.append("rank is not 16")
    if candidate_config.get("use_dora", False):
        problems.append("DoRA is not enabled in this vLLM path")
    if set(candidate_config.get("target_modules") or []) != expected_targets:
        problems.append("target modules differ")
    if problems:
        adapter_rejections.append({"path": str(candidate), "reason": ", ".join(problems)})
        continue
    ADAPTER_DIR = candidate
    adapter_config = candidate_config
    adapter_config_path = config_path
    adapter_weights_path = weights_path
    break

assert ADAPTER_DIR is not None, {"message": "No vLLM-compatible ScentAI adapter found", "rejections": adapter_rejections}
assert adapter_config is not None

try:
    HF_TOKEN = userdata.get("HF_TOKEN") or ""
except Exception:
    HF_TOKEN = ""

copy_started = time.perf_counter()
if LOCAL_CHROMA_DIR.parent.exists():
    shutil.rmtree(LOCAL_CHROMA_DIR.parent)
LOCAL_CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_CHROMA_DIR, LOCAL_CHROMA_DIR)
shutil.copy2(DRIVE_CATALOG, LOCAL_CATALOG)
copy_elapsed = time.perf_counter() - copy_started

chroma_bytes = sum(path.stat().st_size for path in LOCAL_CHROMA_DIR.rglob("*") if path.is_file())
print("Adapter:", ADAPTER_DIR)
print("Adapter selection:", "full" if ADAPTER_DIR == FULL_ADAPTER_DIR else "pilot fallback")
if adapter_rejections:
    print("Skipped adapters:", json.dumps(adapter_rejections, indent=2))
print(f"Local Chroma: {chroma_bytes / (1024 ** 3):.2f} GB")
print(f"Local catalog: {LOCAL_CATALOG.stat().st_size / (1024 ** 2):.1f} MB")
print(f"Data staging: {copy_elapsed:.1f}s")
print("HF token:", "available" if HF_TOKEN else "not set; public model access will be attempted")
'''
    ),
    markdown("## Start the isolated BGE-M3 retrieval service"),
    code(start_retrieval),
    markdown("## Start the isolated Gemma 4 + ScentAI LoRA vLLM service"),
    code(start_vllm),
    markdown("## Prove the base and LoRA model endpoints"),
    code(inference_smoke),
    markdown("## Load the dependency-free Stage 3 orchestrator"),
    code(
        f'''
ORCHESTRATOR_SOURCE = {ORCHESTRATOR_SOURCE!r}
ORCHESTRATOR_PATH = Path("/content/scentai_orchestrator.py")
ORCHESTRATOR_PATH.write_text(ORCHESTRATOR_SOURCE, encoding="utf-8")
compile(ORCHESTRATOR_SOURCE, str(ORCHESTRATOR_PATH), "exec")
exec(compile(ORCHESTRATOR_SOURCE, str(ORCHESTRATOR_PATH), "exec"), globals())

vllm_client = VLLMClient(JsonHttpClient(BASE_URL, timeout=600))
retrieval_client = RetrievalClient(JsonHttpClient(RETRIEVAL_URL, timeout=180))
pipeline = ScentAIOrchestrator(
    vllm_client,
    retrieval_client,
    planner_model=MODEL_NAME,
    answer_model=MODEL_NAME,
    repair_answer_model=LORA_NAME,
)

assert retrieval_client.health()["status"] == "ok"
served_model_ids = {{item["id"] for item in get_json(f"{{BASE_URL}}/v1/models").get("data", [])}}
assert {{MODEL_NAME, LORA_NAME}}.issubset(served_model_ids), served_model_ids
print("Stage 3 orchestrator ready.")
'''
    ),
    markdown(
        """
## End-to-end contract tests

These cases cover the failure modes found during earlier experiments: style words
mistaken for brands, explicit brand requests, similarity with a negative trait,
and ambiguous family-name resolution in comparisons. Failed model generations
are retried once and then replaced by a grounded fallback.
"""
    ),
    code(
        r'''
from datetime import datetime, timezone

contract_cases = [
    {
        "name": "clean_office_not_clean_brand",
        "query": "I need a clean office scent without vanilla. Recommend exactly 3.",
        "expected_language": "en",
    },
    {
        "name": "explicit_versace_brand",
        "query": "Recommend exactly 5 men's fragrances from Versace.",
        "expected_language": "en",
    },
    {
        "name": "aventus_similarity_less_smoky",
        "query": "I want something similar to Aventus but less smoky. Give me exactly 3 options.",
        "expected_language": "en",
    },
    {
        "name": "canonical_comparison",
        "query": "Compare Club de Nuit by Armaf with Team Five by Adidas. Explain their vibe and best use cases.",
        "expected_language": "en",
    },
    {
        "name": "turkish_profile_language",
        "query": "Turkish Leather by Pryn Parfum nasıl bir parfüm? Karakterini, kalıcılığını, yayılımını ve kullanım alanını anlat.",
        "expected_language": "tr",
    },
]

contract_results = []
for case in contract_cases:
    print("\n" + "=" * 110)
    print(case["name"], "|", case["query"])
    result = pipeline.run(case["query"])
    contract_results.append({"name": case["name"], **result})
    print("Route:", result["route"])
    print("Plan:", json.dumps(result["plan"], ensure_ascii=False))
    print("Candidates:", [candidate["label"] for candidate in result["candidates"]])
    print("Validation:", result["validation"])
    if result.get("generation_failures"):
        print("Rejected generations:", result["generation_failures"])
    print("Total seconds:", result["timings"]["total_seconds"])
    print("\n" + result["answer"])
    assert result["validation"]["pass"], result
    assert result["response_language"] == case["expected_language"], result
    assert response_language_matches(result["answer"], case["expected_language"]), result

by_name = {row["name"]: row for row in contract_results}
clean_case = by_name["clean_office_not_clean_brand"]
assert "requested_brand" not in clean_case["plan"], clean_case["plan"]
assert clean_case["retrieval"]["semantic_query"], clean_case["retrieval"]
assert "3" not in clean_case["retrieval"]["semantic_query"], clean_case["retrieval"]
assert sum(candidate["brand"] == "Clean" for candidate in clean_case["candidates"]) <= 1
assert all(not candidate_has_term(candidate, "vanilla") for candidate in clean_case["candidates"])

versace_case = by_name["explicit_versace_brand"]
assert versace_case["plan"].get("requested_brand") == "Versace", versace_case["plan"]
assert all(candidate["brand"] == "Versace" for candidate in versace_case["candidates"])

similarity_case = by_name["aventus_similarity_less_smoky"]
assert similarity_case["plan"]["intent"] in {"similarity", "alternative"}
assert all(not candidate_has_term(candidate, "smoky") for candidate in similarity_case["candidates"])

comparison_case = by_name["canonical_comparison"]
assert comparison_case["plan"]["intent"] == "comparison", comparison_case["plan"]
assert len(comparison_case["candidates"]) == 2, comparison_case["candidates"]

turkish_profile_case = by_name["turkish_profile_language"]
turkish_leather = turkish_profile_case["candidates"][0]
assert "very strong" in calibrated_performance("longevity", turkish_leather["longevity"])
assert "noticeable/strong" in calibrated_performance("sillage", turkish_leather["sillage"])
assert not turkish_profile_case["validation"]["performance_calibration_violations"], turkish_profile_case["validation"]

report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "stage3_full_grounded_pipeline",
    "inference_models": sorted(served_model_ids),
    "adapter_dir": str(ADAPTER_DIR),
    "retrieval_health": retrieval_client.health(),
    "all_contracts_passed": True,
    "results": contract_results,
}
REPORT_PATH = PROJECT_DIR / "runs" / "stage3_pipeline_report.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nSTAGE 3 END-TO-END CONTRACT TEST: PASSED")
print("Saved:", REPORT_PATH)
'''
    ),
    markdown(
        """
## Longevity and sillage calibration contracts

These cases force the advisor to interpret performance across very strong,
moderate, and light catalog bands in both Turkish and English. The report keeps
the parsed performance labels per candidate so a passing validator cannot hide
an omitted or vague performance discussion.
"""
    ),
    code(
        r'''
from datetime import datetime, timezone

PERFORMANCE_CASES = [
    {
        "name": "tr_very_strong_profile",
        "query": "Turkish Leather by Pryn Parfum'un kalıcılığı ve yayılımı nasıl? İkisini de düşük, orta veya güçlü olarak yorumla.",
        "expected_language": "tr",
        "expected": {
            "Turkish Leather by Pryn Parfum": {"longevity": "high", "sillage": "high"},
        },
    },
    {
        "name": "en_light_profile",
        "query": "How are the longevity and projection of Team Five by Adidas? Classify both as light, moderate, or strong.",
        "expected_language": "en",
        "expected": {
            "Team Five by Adidas": {"longevity": "low", "sillage": "low"},
        },
    },
    {
        "name": "en_moderate_profile",
        "query": "How are the longevity and projection of Versace Pour Homme by Versace? Classify both as light, moderate, or strong.",
        "expected_language": "en",
        "expected": {
            "Versace Pour Homme by Versace": {"longevity": "moderate", "sillage": "moderate"},
        },
    },
    {
        "name": "en_high_vs_low_comparison",
        "query": "Compare the longevity and projection of Tobacco Vanille by Tom Ford with Team Five by Adidas. State whether each is light, moderate, or strong.",
        "expected_language": "en",
        "expected": {
            "Tobacco Vanille by Tom Ford": {"longevity": "high", "sillage": "high"},
            "Team Five by Adidas": {"longevity": "low", "sillage": "low"},
        },
    },
]

performance_results = []
for case in PERFORMANCE_CASES:
    print("\n" + "=" * 110)
    print(case["name"], "|", case["query"])
    result = pipeline.run(case["query"])
    sections = candidate_answer_sections(result["answer"], result["candidates"])
    observed = {
        candidate["label"]: {
            metric: sorted(explicit_performance_labels(sections.get(candidate["label"], ""), metric))
            for metric in ("longevity", "sillage")
        }
        for candidate in result["candidates"]
        if candidate["label"] in case["expected"]
    }
    calibration = {
        candidate["label"]: {
            metric: {
                "score": candidate.get(metric),
                "card_text": calibrated_performance(metric, candidate.get(metric)),
                "expected_group": expected_performance_group(metric, candidate.get(metric)),
            }
            for metric in ("longevity", "sillage")
        }
        for candidate in result["candidates"]
        if candidate["label"] in case["expected"]
    }
    row = {
        "name": case["name"],
        "expected": case["expected"],
        "observed_explicit_labels": observed,
        "calibration": calibration,
        **result,
    }
    performance_results.append(row)
    print("Route:", result["route"])
    print("Language:", result["response_language"])
    print("Calibration:", json.dumps(calibration, ensure_ascii=False, indent=2))
    print("Observed labels:", json.dumps(observed, ensure_ascii=False, indent=2))
    print("Validation:", result["validation"])
    print("Attempts:", result["generation_attempts"])
    print("Seconds:", result["timings"]["total_seconds"])
    print("\n" + result["answer"])

    assert result["response_language"] == case["expected_language"], result
    assert result["validation"]["pass"], result
    assert not result["validation"]["performance_calibration_violations"], result["validation"]
    assert set(calibration) == set(case["expected"]), calibration
    for label, expected_metrics in case["expected"].items():
        for metric, expected_group in expected_metrics.items():
            assert calibration[label][metric]["expected_group"] == expected_group, calibration[label]
            assert observed[label][metric] == [expected_group], {
                "message": "The answer did not state one clear calibrated performance class.",
                "case": case["name"],
                "candidate": label,
                "metric": metric,
                "expected": expected_group,
                "observed": observed[label][metric],
                "answer": result["answer"],
            }

performance_report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "stage3_performance_calibration",
    "catalog_snapshot_count": 131930,
    "all_contracts_passed": True,
    "results": performance_results,
}
PERFORMANCE_REPORT_PATH = PROJECT_DIR / "runs" / "stage3_performance_calibration_report.json"
PERFORMANCE_REPORT_PATH.write_text(
    json.dumps(performance_report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("\nSTAGE 3 PERFORMANCE CALIBRATION CONTRACT: PASSED")
print("Saved:", PERFORMANCE_REPORT_PATH)
'''
    ),
    markdown(
        """
## Popularity and conversation-memory contracts

Balanced discovery combines semantic relevance with a catalog-wide popularity
pool; it does not hardcode perfume names. The session contract then verifies
that a natural follow-up keeps the previous constraints while excluding items
already recommended in the conversation.
"""
    ),
    code(
        r'''
popular_vanilla = retrieval_client.search({
    "query": "versatile vanilla fragrance",
    "top_k": 10,
    "required_terms": ["vanilla"],
    "discovery_mode": "balanced",
})
popular_vanilla_candidates = [candidate_record(item) for item in popular_vanilla["results"]]
assert len(popular_vanilla_candidates) == 10, popular_vanilla_candidates
assert all(candidate_has_term(candidate, "vanilla") for candidate in popular_vanilla_candidates)
assert sum(int(candidate.get("popularity") or 0) >= 10_000 for candidate in popular_vanilla_candidates[:5]) >= 3, popular_vanilla_candidates[:5]
assert any(
    "catalog_popular" in item.get("reasons", {}).get("candidate_sources", [])
    for item in popular_vanilla["results"][:5]
), popular_vanilla["results"][:5]

contract_session = ScentAISession(pipeline)
first_turn = contract_session.run("Recommend exactly 3 popular fragrances that must have vanilla.")
second_turn = contract_session.run("I want three different options under the same requirements.")
assert first_turn["validation"]["pass"], first_turn
assert second_turn["validation"]["pass"], second_turn
assert second_turn["plan"]["conversation_action"] == "more_options", second_turn["plan"]
assert "vanilla" in second_turn["plan"].get("required_terms", []), second_turn["plan"]
assert second_turn["plan"].get("discovery_mode") == "mainstream", second_turn["plan"]
first_ids = set(first_turn["validation"].get("mentioned_candidate_ids", []))
second_ids = set(second_turn["validation"].get("mentioned_candidate_ids", []))
if not first_ids:
    first_labels = set(first_turn["validation"].get("mentioned_candidates", []))
    first_ids = {candidate["perfume_id"] for candidate in first_turn["candidates"] if candidate["label"] in first_labels}
if not second_ids:
    second_labels = set(second_turn["validation"].get("mentioned_candidates", []))
    second_ids = {candidate["perfume_id"] for candidate in second_turn["candidates"] if candidate["label"] in second_labels}
assert first_ids and second_ids and first_ids.isdisjoint(second_ids), {"first": first_ids, "second": second_ids}

conversation_report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "popular_vanilla_labels": [candidate["label"] for candidate in popular_vanilla_candidates],
    "first_turn": first_turn,
    "second_turn": second_turn,
}
CONVERSATION_REPORT_PATH = PROJECT_DIR / "runs" / "stage3_popularity_conversation_contract.json"
CONVERSATION_REPORT_PATH.write_text(json.dumps(conversation_report, indent=2, ensure_ascii=False), encoding="utf-8")
print("STAGE 3 POPULARITY + CONVERSATION CONTRACT: PASSED")
print("Saved:", CONVERSATION_REPORT_PATH)
'''
    ),
    markdown(
        """
## Advisor A/B diagnosis

This one-time diagnostic uses the same planner and retrieval pipeline for three
answer variants: the previous LoRA prompt, the new advisor LoRA prompt, and the
new advisor prompt on base Gemma. Compare decision value, naturalness,
specificity, repetition, and grounding. Retrieval candidate fingerprints are
checked so prose is not compared across different shortlists.
"""
    ),
    code(
        r'''
ADVISOR_AB_CASES = [
    "Bana vanilyalı, sıcak ama boğucu olmayan tam 3 parfüm öner. Seçeneklerin karakter farklarını anlat.",
    "Ofiste kullanabileceğim temiz ve profesyonel, vanilyasız tam 3 parfüm öner.",
    "Date akşamı için baharatlı ve vanilyalı tam 3 parfüm öner; hangisinin nasıl bir izlenim verdiğini anlat.",
]

from datetime import datetime, timezone

ADVISOR_AB_VARIANTS = [
    {"name": "legacy_lora", "model": LORA_NAME, "prompt": LEGACY_ANSWER_PROMPT},
    {"name": "advisor_lora", "model": LORA_NAME, "prompt": ADVISOR_ANSWER_PROMPT},
    {"name": "advisor_base", "model": MODEL_NAME, "prompt": ADVISOR_ANSWER_PROMPT},
]


def run_advisor_ab(cases=ADVISOR_AB_CASES):
    report_rows = []
    for case_index, query in enumerate(cases, 1):
        print("\n" + "=" * 118)
        print(f"A/B CASE {case_index}: {query}")
        variants = []
        candidate_fingerprint = None
        for variant in ADVISOR_AB_VARIANTS:
            result = pipeline.run(
                query,
                answer_prompt_override=variant["prompt"],
                answer_model_override=variant["model"],
            )
            fingerprint = [candidate["perfume_id"] for candidate in result.get("candidates", [])]
            if candidate_fingerprint is None:
                candidate_fingerprint = fingerprint
            assert fingerprint == candidate_fingerprint, {
                "message": "A/B variants received different retrieval candidates",
                "expected": candidate_fingerprint,
                "actual": fingerprint,
                "variant": variant["name"],
            }
            variants.append({"variant": variant["name"], **result})
            print("\n" + "-" * 118)
            print(
                variant["name"],
                "| route=", result["route"],
                "| valid=", result["validation"]["pass"],
                "| seconds=", result["timings"]["total_seconds"],
            )
            print(result["answer"])
        report_rows.append({
            "query": query,
            "candidate_fingerprint": candidate_fingerprint,
            "candidate_labels": [candidate["label"] for candidate in variants[0]["candidates"]],
            "variants": variants,
        })

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "stage3_advisor_ab",
        "rubric": [
            "decision_value",
            "natural_warmth",
            "query_specificity",
            "meaningful_contrast",
            "low_template_repetition",
            "grounding_and_filter_compliance",
        ],
        "cases": report_rows,
    }
    path = PROJECT_DIR / "runs" / "stage3_advisor_ab_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nADVISOR A/B DIAGNOSIS COMPLETE")
    print("Saved:", path)
    return report


advisor_ab_report = run_advisor_ab()
'''
    ),
    markdown(
        """
## Retrieval bias audit

This fast red-team suite checks whether explicit traits are satisfied by actual
accord/note fields rather than merely echoed in perfume names. It also checks
brand diversity and the known Clean/clean collision. No LLM generation is used.
"""
    ),
    code(
        r'''
bias_cases = [
    {"trait": "vanilla", "semantic_query": "romantic date night fragrance"},
    {"trait": "spicy", "semantic_query": "energetic date night fragrance"},
    {"trait": "oud", "semantic_query": "dark resinous evening fragrance"},
    {"trait": "rose", "semantic_query": "elegant floral spring fragrance"},
    {"trait": "musk", "semantic_query": "soft intimate skin scent"},
    {"trait": "amber", "semantic_query": "warm enveloping evening fragrance"},
    {"trait": "leather", "semantic_query": "confident dressed-up evening fragrance"},
    {"trait": "coffee", "semantic_query": "cozy gourmand winter fragrance"},
    {"trait": "coconut", "semantic_query": "tropical summer holiday fragrance"},
    {"trait": "pineapple", "semantic_query": "bright fruity summer fragrance"},
]

bias_rows = []
for case in bias_cases:
    response = retrieval_client.search({
        "query": case["semantic_query"],
        "top_k": 10,
        "required_terms": [case["trait"]],
    })
    candidates = [candidate_record(item) for item in response["results"]]
    compliant = [candidate for candidate in candidates if candidate_has_term(candidate, case["trait"])]
    name_echoes = [
        candidate for candidate in candidates
        if f" {normalize_text(case['trait'])} " in f" {normalize_text(candidate['name'])} "
    ]
    row = {
        **case,
        "result_count": len(candidates),
        "trait_compliance_rate": round(len(compliant) / len(candidates), 4) if candidates else 0.0,
        "name_echo_rate": round(len(name_echoes) / len(candidates), 4) if candidates else 0.0,
        "unique_brand_count": len({candidate["brand"] for candidate in candidates}),
        "labels": [candidate["label"] for candidate in candidates],
    }
    row["pass"] = (
        row["result_count"] >= 5
        and row["trait_compliance_rate"] == 1.0
        and row["name_echo_rate"] <= 0.6
        and row["unique_brand_count"] >= 4
    )
    bias_rows.append(row)
    print(
        case["trait"],
        "count=", row["result_count"],
        "trait=", row["trait_compliance_rate"],
        "name_echo=", row["name_echo_rate"],
        "brands=", row["unique_brand_count"],
        "pass=", row["pass"],
    )

clean_response = retrieval_client.search({
    "query": "professional clean office fragrance",
    "top_k": 10,
    "exclude_terms": ["vanilla"],
})
clean_candidates = [candidate_record(item) for item in clean_response["results"]]
clean_guard = {
    "clean_brand_count": sum(candidate["brand"] == "Clean" for candidate in clean_candidates),
    "vanilla_violation_count": sum(candidate_has_term(candidate, "vanilla") for candidate in clean_candidates),
    "labels": [candidate["label"] for candidate in clean_candidates],
}
clean_guard["pass"] = clean_guard["clean_brand_count"] <= 1 and clean_guard["vanilla_violation_count"] == 0

bias_report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "stage3_retrieval_bias_audit",
    "all_passed": all(row["pass"] for row in bias_rows) and clean_guard["pass"],
    "trait_cases": bias_rows,
    "clean_brand_collision": clean_guard,
}
BIAS_REPORT_PATH = PROJECT_DIR / "runs" / "stage3_retrieval_bias_audit.json"
BIAS_REPORT_PATH.write_text(json.dumps(bias_report, indent=2, ensure_ascii=False), encoding="utf-8")
assert bias_report["all_passed"], bias_report
print("\nSTAGE 3 RETRIEVAL BIAS AUDIT: PASSED")
print("Saved:", BIAS_REPORT_PATH)
'''
    ),
    markdown(
        """
## Final evaluation v1 - 120 frozen cases

This is the release-quality evaluation pass. It covers recommendation, perfume
profiles, comparisons, similarity, hard filters, entity resolution, stateful
conversation, unsupported requests, and noisy Turkish/English input. Every row
is flushed to Drive immediately and the runner resumes completed cases after a
disconnect. It writes an automatic summary and a stratified 40-row human-review
CSV.
"""
    ),
    code(
        f'''
FINAL_EVAL_SET_SOURCE = {FINAL_EVAL_SET_SOURCE!r}
FINAL_EVAL_RUNTIME_SOURCE = {FINAL_EVAL_RUNTIME_SOURCE!r}

compile(FINAL_EVAL_RUNTIME_SOURCE, "<scentai_final_eval_runtime>", "exec")
exec(compile(FINAL_EVAL_RUNTIME_SOURCE, "<scentai_final_eval_runtime>", "exec"), globals())

FINAL_EVAL_DIR = PROJECT_DIR / "runs" / "final_evaluation"
FINAL_EVAL_CASES_PATH = FINAL_EVAL_DIR / "final_eval_v1.jsonl"
FINAL_EVAL_OUTPUTS_PATH = FINAL_EVAL_DIR / "final_eval_outputs.jsonl"
FINAL_EVAL_SUMMARY_PATH = FINAL_EVAL_DIR / "final_eval_summary.json"
FINAL_EVAL_HUMAN_REVIEW_PATH = FINAL_EVAL_DIR / "final_eval_human_review.csv"
FINAL_EVAL_METADATA_PATH = FINAL_EVAL_DIR / "final_eval_metadata.json"
FINAL_EVAL_RESUME = True
FINAL_EVAL_LIMIT = None  # Set to 10 only for a quick runner smoke test.

FINAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
FINAL_EVAL_CASES_PATH.write_text(FINAL_EVAL_SET_SOURCE, encoding="utf-8")
final_eval_cases = [json.loads(line) for line in FINAL_EVAL_SET_SOURCE.splitlines() if line.strip()]
assert len(final_eval_cases) == 120, len(final_eval_cases)

final_eval_summary = run_final_evaluation(
    pipeline,
    final_eval_cases,
    outputs_path=FINAL_EVAL_OUTPUTS_PATH,
    summary_path=FINAL_EVAL_SUMMARY_PATH,
    human_review_path=FINAL_EVAL_HUMAN_REVIEW_PATH,
    metadata_path=FINAL_EVAL_METADATA_PATH,
    resume=FINAL_EVAL_RESUME,
    limit=FINAL_EVAL_LIMIT,
    run_metadata={{
        "model_name": MODEL_NAME,
        "answer_model": MODEL_NAME,
        "repair_model": LORA_NAME,
        "adapter_dir": str(ADAPTER_DIR),
        "retrieval_health": retrieval_client.health(),
    }},
)

print("\\n" + "=" * 110)
print("FINAL EVALUATION COMPLETE")
print(json.dumps(final_eval_summary, indent=2, ensure_ascii=False))
print("Outputs:", FINAL_EVAL_OUTPUTS_PATH)
print("Summary:", FINAL_EVAL_SUMMARY_PATH)
print("Human review:", FINAL_EVAL_HUMAN_REVIEW_PATH)
print("Metadata:", FINAL_EVAL_METADATA_PATH)
'''
    ),
    markdown("## Interactive full-pipeline use"),
    code(
        r'''
RESULTS_PATH = PROJECT_DIR / "runs" / "stage3_interactive_results.jsonl"
scentai_session = ScentAISession(pipeline)


def ask_scentai(query, *, save=True):
    result = scentai_session.run(query)
    print("Route:", result["route"])
    print("Planner:", result["plan"]["intent"], "| confidence=", result["plan"]["confidence"])
    print("Candidates:")
    for index, candidate in enumerate(result["candidates"], 1):
        print(f"  {index:02d}. {candidate['label']}")
    print("Validation:", result["validation"])
    print("Generation attempts:", result["generation_attempts"])
    print("Total time:", result["timings"]["total_seconds"], "seconds")
    print("\nAnswer:\n" + result["answer"])
    if save:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print("\nSaved:", RESULTS_PATH)
    return result


def reset_scentai_session():
    scentai_session.reset()
    print("ScentAI conversation memory cleared.")


my_query = "Bana popüler ve vanilyalı 3 parfüm öner."
my_result = ask_scentai(my_query)

# Natural follow-up example; this keeps the vanilla/mainstream constraints and
# excludes the perfumes that were actually recommended above:
# my_result = ask_scentai("Başka üç seçenek istiyorum.")
# Start an unrelated topic with: reset_scentai_session()
'''
    ),
]

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
