# Paste this cell into Colab after the fine-tuned model/LoRA is loaded.
# It expects these notebook variables/functions to already exist:
# model, tokenizer, FastModel, PROCESSOR, TEXT_TOKENIZER, decode_generated_tokens,
# USE_NATIVE_SYSTEM_ROLE, DISABLE_GEMMA_THINKING, device, OUTPUT_DIR.

import json
import os
from datetime import datetime, timezone
from pathlib import Path

EVAL_SET_PATH = PROJECT_DIR / "train_set" / "eval" / "scentai_eval_v2.jsonl"
EVAL_RUN_DIR = PROJECT_DIR / "train_set" / "eval" / "runs"
EVAL_RUN_NAME = f"{RUN_MODE}_{MODEL_PRESET}_eval_v2_strictprompt_v2"
EVAL_OUTPUTS_PATH = EVAL_RUN_DIR / f"{EVAL_RUN_NAME}_outputs.jsonl"
EVAL_METADATA_PATH = EVAL_RUN_DIR / f"{EVAL_RUN_NAME}_metadata.json"
EVAL_LIMIT = None  # set to 20 for a quick smoke pass
EVAL_START_INDEX = 0  # 0-based inclusive; use 0, 40, 80, 120 for chunks
EVAL_END_INDEX = None  # exclusive; use 40, 80, 120, 160 for chunks
EVAL_MAX_NEW_TOKENS = 300
EVAL_RESUME = False

EVAL_SYSTEM = """You are ScentAI, a careful perfume recommendation assistant.
Your job is to recommend from the provided database context, not from memory.

GROUNDING CONTRACT:
- Recommend ONLY perfumes that appear inside [PERFUMES].
- Treat every perfume card as the complete source of truth for that perfume.
- Use only facts explicitly printed on that exact card.
- Do not use outside knowledge about perfume notes, flankers, brands, popularity, performance, or release details.

FIELD RULES:
- Accords and Notes are different fields.
- If a card has an Accords line, you may mention those listed accords as accords.
- You may mention notes ONLY when that exact card has a Notes, Top Notes, Middle Notes, or Base Notes line.
- If a card has no note field, never write "notes like", "note of", "with notes of", or any specific note names for that perfume.
- If a card has no rating, season, time, gender, longevity, sillage, or value field, do not infer it.

DATABASE LOOKUP MODE:
- If the user asks what the database says, asks for a database record, or asks for exact perfume information, copy fields exactly from the card.
- Do not convert `0.00/5 (0 votes)` to `N/A`.
- Do not simplify accords such as `white floral` into `floral`.
- Do not omit notes from a single-perfume record unless you explicitly say you are summarizing.

STRICT FILTERS:
- Respect [STRICT FILTERS] literally.
- Excluded notes, accords, or perfumes must be omitted, not ranked lower.
- If [STRICT FILTERS] lists forbidden perfumes, do not mention them in recommendations, reasons, or Best pick.
- For "less X" requests, avoid perfumes listing X when valid non-X alternatives exist.
- Do not claim "no listed X note" unless the card actually has a note field. Prefer "the card does not list X as an accord" only when X is absent from the Accords line.
- If no safe perfume remains after strict filters, say that clearly instead of recommending a forbidden perfume.

ANSWER STYLE:
- Recommend 3-5 perfumes unless the user asks for a different number.
- For each perfume, give a short reason using only listed fields.
- Prefer wording like "listed accords include..." instead of inventing note-level detail.
- If the context has no strong match, say so honestly and recommend the closest grounded options.
- Respond in the same language as the user."""


def read_eval_cases(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_output_candidates(primary_path):
    candidates = [primary_path]
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen and candidate.exists():
            unique.append(candidate)
            seen.add(key)
    return unique


def read_completed_outputs(paths):
    if not EVAL_RESUME:
        return {}, {"sources": [], "malformed_lines": 0}
    completed = {}
    malformed_lines = 0
    sources = []
    for path in paths:
        sources.append(str(path))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    print(f"Skipping malformed JSONL line: {path}:{line_number}")
                    continue
                case_id = row.get("id")
                if case_id:
                    completed[case_id] = row
    return completed, {"sources": sources, "malformed_lines": malformed_lines}


def write_completed_snapshot(path, completed, cases):
    if not completed:
        return
    ordered_ids = [case["id"] for case in cases]
    with path.open("w", encoding="utf-8") as handle:
        for case_id in ordered_ids:
            row = completed.get(case_id)
            if row:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate_eval_answer(case):
    user = inject_eval_strict_filters(case["user"].strip(), case.get("checks", {}))
    if USE_NATIVE_SYSTEM_ROLE:
        messages = [{"role": "system", "content": EVAL_SYSTEM}, {"role": "user", "content": user}]
    else:
        content = f"{SYSTEM_OPEN}\n{EVAL_SYSTEM}\n{SYSTEM_CLOSE}\n\n{user}"
        messages = [{"role": "user", "content": content}]

    inputs = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=not DISABLE_GEMMA_THINKING,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=TEXT_TOKENIZER.pad_token_id,
            eos_token_id=TEXT_TOKENIZER.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return decode_generated_tokens(generated)


def inject_eval_strict_filters(user, checks):
    block = build_eval_strict_filter_block(checks)
    if block is None or "[STRICT FILTERS]" in user:
        return user
    marker = "[/PERFUMES]"
    if marker in user:
        return user.replace(marker, marker + "\n\n" + block, 1)
    return block + "\n\n" + user


def build_eval_strict_filter_block(checks):
    excluded_terms = checks.get("excluded_terms", []) or []
    forbidden_perfumes = checks.get("forbidden_perfumes", []) or []
    if not excluded_terms and not forbidden_perfumes:
        return None
    lines = ["[STRICT FILTERS]"]
    if excluded_terms:
        lines.append("Excluded terms: " + ", ".join(excluded_terms))
    if forbidden_perfumes:
        lines.append("Forbidden perfumes: " + "; ".join(forbidden_perfumes))
    lines.append("You must not recommend forbidden perfumes. If no allowed option fits, say no safe match.")
    lines.append("[/STRICT FILTERS]")
    return "\n".join(lines)


FastModel.for_inference(model)
cases = read_eval_cases(EVAL_SET_PATH)
if EVAL_LIMIT is not None:
    cases = cases[:EVAL_LIMIT]
if EVAL_END_INDEX is not None or EVAL_START_INDEX:
    cases = cases[EVAL_START_INDEX:EVAL_END_INDEX]

EVAL_OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
candidate_paths = find_output_candidates(EVAL_OUTPUTS_PATH)
completed, resume_info = read_completed_outputs(candidate_paths)
write_completed_snapshot(EVAL_OUTPUTS_PATH, completed, cases)
mode = "a" if completed and EVAL_RESUME else "w"

metadata = {
    "eval_set_path": str(EVAL_SET_PATH),
    "eval_outputs_path": str(EVAL_OUTPUTS_PATH),
    "eval_limit": EVAL_LIMIT,
    "eval_start_index": EVAL_START_INDEX,
    "eval_end_index": EVAL_END_INDEX,
    "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
    "eval_resume": EVAL_RESUME,
    "case_count": len(cases),
    "already_completed": len(completed),
    "resume_sources": resume_info["sources"],
    "resume_malformed_lines": resume_info["malformed_lines"],
    "model_name": MODEL_NAME,
    "model_preset": MODEL_PRESET,
    "run_mode": RUN_MODE,
    "output_dir": str(OUTPUT_DIR),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
}
EVAL_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
with EVAL_METADATA_PATH.open("a", encoding="utf-8") as meta_handle:
    meta_handle.flush()
    os.fsync(meta_handle.fileno())
print("Eval resume sources:", resume_info["sources"])
print("Eval malformed lines skipped:", resume_info["malformed_lines"])
print("Eval already completed:", len(completed), "/", len(cases))

with EVAL_OUTPUTS_PATH.open(mode, encoding="utf-8") as handle:
    for index, case in enumerate(cases, 1):
        if case["id"] in completed:
            print(f"[{index}/{len(cases)}] skip {case['id']} {case['category']}")
            continue
        answer = generate_eval_answer(case)
        row = {
            "id": case["id"],
            "category": case["category"],
            "mode": case.get("mode"),
            "difficulty": case.get("difficulty"),
            "tags": case.get("tags", []),
            "answer": answer,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        print(f"[{index}/{len(cases)}] {case['id']} {case['category']} chars={len(answer)}")

print("Saved eval outputs to:", EVAL_OUTPUTS_PATH)
print("Saved eval metadata to:", EVAL_METADATA_PATH)
