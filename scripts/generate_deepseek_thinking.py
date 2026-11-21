#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prompt_templates import render_prompt


LABELS = set("ABCDE")


@dataclass(frozen=True)
class GenerationResult:
    source_record_id: str
    row_index: int
    thinking: str
    answer_option: str | None
    final_answer: str
    raw_model_content: str
    raw_reasoning_content: str
    parsed_json: dict[str, Any]
    usage: dict[str, Any] | None
    error: str | None = None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def record_id(row: dict[str, Any], row_index: int, id_key: str) -> str:
    value = row.get(id_key)
    if value is None or str(value).strip() == "":
        value = row.get("pair_id")
    if value is None or str(value).strip() == "":
        value = row_index
    return str(value)


def completed_ids(path: Path, id_key: str, retry_failed: bool) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_failed and row.get("deepseek_error"):
                continue
            rid = row.get("source_record_id") or row.get(id_key)
            if rid is not None:
                done.add(str(rid))
    return done


def option_lines(row: dict[str, Any]) -> list[str]:
    options = []
    for opt in row.get("options") or []:
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label") or "").strip().upper()
        text = str(opt.get("text") or "").strip()
        if label in LABELS and text:
            options.append((label, text))
    return [f"{label}. {text}" for label, text in sorted(options)]


def option_text(row: dict[str, Any], label: str | None) -> str:
    if not label:
        return ""
    target = label.strip().upper()
    for opt in row.get("options") or []:
        if not isinstance(opt, dict):
            continue
        if str(opt.get("label") or "").strip().upper() == target:
            return str(opt.get("text") or "").strip()
    return ""


def reference_reasoning(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for step in row.get("reasoning_steps") or []:
        if not isinstance(step, dict):
            continue
        step_id = step.get("step")
        reasoning = str(step.get("reasoning") or "").strip()
        if reasoning:
            parts.append(f"{step_id}. {reasoning}" if step_id is not None else reasoning)
    return "\n".join(parts)


def sample_payload_for_solve(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": row.get("domain"),
        "subdomain": row.get("subdomain"),
        "question": row.get("question"),
        "options": option_lines(row),
    }


def sample_payload_with_reference(row: dict[str, Any]) -> dict[str, Any]:
    gold = str(row.get("answer_option") or row.get("correct_option") or "").strip().upper()
    gold_text = str(row.get("answer_option_text") or row.get("final_answer") or "").strip()
    ref_answer = str(row.get("answer") or "").strip()
    ref_steps = reference_reasoning(row)
    return {
        "domain": row.get("domain"),
        "subdomain": row.get("subdomain"),
        "question": row.get("question"),
        "options": option_lines(row),
        "correct_option": gold,
        "correct_option_text": gold_text,
        "reference_answer": ref_answer,
        "reference_reasoning_steps": ref_steps,
    }


def build_solve_user_prompt(row: dict[str, Any]) -> str:
    return render_prompt("cot_generation", sample_json=sample_payload_for_solve(row))


def build_reference_user_prompt(row: dict[str, Any]) -> str:
    return render_prompt("cot_reference_generation", sample_json=sample_payload_with_reference(row))


SOLVE_SYSTEM_PROMPT = "You are a careful multiple-choice question assistant."


REFERENCE_SYSTEM_PROMPT = (
    "You write concise, faithful reasoning traces for multiple-choice SFT data. "
    "Check the reference against the evidence and output only valid JSON."
)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S | re.I)
    return match.group(1).strip() if match else stripped


def parse_json_object(text: str) -> dict[str, Any]:
    text = strip_code_fence(text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def normalize_answer_option(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in LABELS:
        return text
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else None


def extract_choice(text: str) -> str | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    patterns = [
        r"(?:^|\n)\s*Answer\s*:\s*\(?([A-E])\)?\b",
        r"(?:^|\n)\s*Final\s+answer\s*:\s*\(?([A-E])\)?\b",
        r"^\s*\(?([A-E])\)?\s*$",
        r"^\s*\(?([A-E])\)?\s*[\.:\)]",
        r"\b(?:answer|final answer|option)\s*(?:is|:)?\s*\(?([A-E])\)?\b",
        r"\\boxed\{\s*([A-E])\s*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).upper()
    letters = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", text.upper())
    return letters[-1] if letters else None


def extract_reasoning(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>|</think>", "", text, flags=re.I).strip()
    match = re.search(
        r"(?:^|\n)\s*Reasoning\s*:\s*(.*?)(?=(?:^|\n)\s*(?:Answer|Final\s+answer)\s*:)",
        text,
        flags=re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    text = re.sub(r"(?:^|\n)\s*(?:Answer|Final\s+answer)\s*:\s*\(?[A-E]\)?\b.*$", "", text, flags=re.I | re.S)
    return text.strip()


def clean_reasoning_for_sft(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"<think>|</think>", "", text, flags=re.I).strip()
    text = re.sub(
        r'^\s*We are given\s*:\s*".*?"\s*(?:Options\s*:\s*.*?)(?=\n\s*(?:We need|The key|Key|This|The patient|Given|First|Now|Option|Let\'s|In this))',
        "",
        text,
        flags=re.I | re.S,
    ).strip()
    text = re.sub(
        r"^\s*We are given\s*:.*?Options\s*:\s*.*?(?=\n\s*(?:Interpretation|We need|The key|Key|This|The patient|Given|First|Now|Option|Let's|In this))",
        "",
        text,
        flags=re.I | re.S,
    ).strip()
    text = re.sub(
        r"^\s*Options\s*:\s*.*?(?=\n\s*(?:Interpretation|We need|The key|Key|This|The patient|Given|First|Now|Option|Let's|In this))",
        "",
        text,
        flags=re.I | re.S,
    ).strip()
    text = re.sub(
        r'^\s*We are given\s*:\s*".*?"\s*(?=\n\s*(?:We need|The key|Key|This|The patient|Given|First|Now|Option|Let\'s|In this))',
        "",
        text,
        flags=re.I | re.S,
    ).strip()
    text = re.sub(r"^\s*We need to reason step by step(?:\s+and\s+choose\s+the\s+best\s+answer)?\.?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^\s*(?:and\s+choose\s+the\s+best\s+answer|We need to select[^.]*\.)\s*", "", text, flags=re.I).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs and re.match(r"^We are given\b", paragraphs[0], flags=re.I):
        paragraphs = paragraphs[1:]
    if paragraphs and re.match(r"^Options\s*:", paragraphs[0], flags=re.I):
        paragraphs = paragraphs[1:]
    text = "\n\n".join(paragraphs).strip()
    return text


def openai_client(api_key: str, api_base: str, timeout: float, disable_http_env: bool) -> Any:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if api_base:
        kwargs["base_url"] = api_base
    if disable_http_env:
        import httpx

        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
    return OpenAI(**kwargs)


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_model(row: dict[str, Any], row_index: int, args: argparse.Namespace) -> GenerationResult:
    rid = record_id(row, row_index, args.id_key)
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or API_KEY.")

    api_base = args.api_base
    client = openai_client(api_key, api_base, args.timeout, args.disable_http_env)
    try:
        messages: list[dict[str, str]] = []
        system_prompt = REFERENCE_SYSTEM_PROMPT if args.prompt_mode == "reference" else SOLVE_SYSTEM_PROMPT
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": build_reference_user_prompt(row)
                if args.prompt_mode == "reference"
                else build_solve_user_prompt(row),
            }
        )
        request: dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.response_format_json:
            request["response_format"] = {"type": "json_object"}
        if not args.omit_extra_body:
            request["extra_body"] = {"enable_thinking": args.enable_thinking}

        last_error: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                completion = client.chat.completions.create(**request)
                message = completion.choices[0].message
                content = getattr(message, "content", None) or ""
                reasoning_content = getattr(message, "reasoning_content", None) or ""
                parsed = parse_json_object(content)

                thinking = str(parsed.get("thinking") or reasoning_content or extract_reasoning(content)).strip()
                answer_option = normalize_answer_option(parsed.get("answer_option")) or extract_choice(content)
                final_answer = str(parsed.get("final_answer") or "").strip()
                if not final_answer:
                    final_answer = option_text(row, answer_option) or content.strip()
                return GenerationResult(
                    source_record_id=rid,
                    row_index=row_index,
                    thinking=thinking,
                    answer_option=answer_option,
                    final_answer=final_answer,
                    raw_model_content=content,
                    raw_reasoning_content=reasoning_content,
                    parsed_json=parsed,
                    usage=usage_to_dict(getattr(completion, "usage", None)),
                )
            except Exception as exc:
                last_error = exc
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * (attempt + 1))
        assert last_error is not None
        raise RuntimeError(str(last_error)) from last_error
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def make_output_row(row: dict[str, Any], result: GenerationResult, args: argparse.Namespace) -> dict[str, Any]:
    gold = normalize_answer_option(row.get("answer_option") or row.get("correct_option"))
    is_correct = (result.answer_option == gold) if result.answer_option else False
    out = dict(row)
    out["source_record_id"] = result.source_record_id
    out["source_row_index"] = result.row_index
    out[args.thinking_field] = result.thinking
    out["deepseek_answer_option"] = result.answer_option
    out["deepseek_final_answer"] = result.final_answer
    out["deepseek_answer_matches_gold"] = is_correct
    out["deepseek_correct"] = is_correct
    out["deepseek_model"] = args.model
    out["deepseek_prompt_mode"] = args.prompt_mode
    out["deepseek_enable_thinking"] = args.enable_thinking
    out["deepseek_raw_model_content"] = result.raw_model_content
    if result.raw_reasoning_content:
        out["deepseek_raw_reasoning_content"] = result.raw_reasoning_content
    out["deepseek_usage"] = result.usage
    if result.error:
        out["deepseek_error"] = result.error
    return out


def make_error_row(row: dict[str, Any], row_index: int, error: Exception, args: argparse.Namespace) -> dict[str, Any]:
    rid = record_id(row, row_index, args.id_key)
    out = dict(row)
    out["source_record_id"] = rid
    out["source_row_index"] = row_index
    out["deepseek_correct"] = False
    out["deepseek_model"] = args.model
    out["deepseek_prompt_mode"] = args.prompt_mode
    out["deepseek_enable_thinking"] = args.enable_thinking
    out["deepseek_error"] = repr(error)
    return out


def sft_instruction(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Answer the following multiple-choice question. Think step by step, then give the final answer.",
            "",
            "Question:",
            str(row.get("question") or "").strip(),
            "",
            "Options:",
            *option_lines(row),
        ]
    )


def make_sft_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    answer = normalize_answer_option(row.get("deepseek_answer_option")) or normalize_answer_option(row.get("answer_option") or row.get("correct_option")) or ""
    final_answer = str(row.get("deepseek_final_answer") or row.get("answer_option_text") or row.get("final_answer") or answer).strip()
    thinking = clean_reasoning_for_sft(str(row.get(args.thinking_field) or ""))
    output = f"<think>\n{thinking}\n</think>\nAnswer: {answer}"
    if final_answer and final_answer != answer:
        output += f"\nFinal answer: {final_answer}"
    return {
        "instruction": sft_instruction(row),
        "input": "",
        "output": output,
    }


def write_sft_file(enriched_path: Path, sft_path: Path, args: argparse.Namespace) -> dict[str, int]:
    counts = {"total": 0, "written": 0, "skipped_empty_thinking": 0, "skipped_mismatch": 0, "skipped_error": 0}
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with enriched_path.open(encoding="utf-8") as fin, sft_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["total"] += 1
            if row.get("deepseek_error"):
                counts["skipped_error"] += 1
                continue
            if not str(row.get(args.thinking_field) or "").strip():
                counts["skipped_empty_thinking"] += 1
                continue
            if args.sft_skip_mismatch and row.get("deepseek_answer_matches_gold") is not True:
                counts["skipped_mismatch"] += 1
                continue
            fout.write(json_dumps(make_sft_row(row, args)) + "\n")
            counts["written"] += 1
    return counts


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.name.removesuffix(".jsonl") + ".deepseek_visible_cot.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--sft-output", default="", help="Optional Alpaca-style JSONL output path.")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument(
        "--prompt-mode",
        choices=("solve", "reference"),
        default=os.environ.get("PROMPT_MODE", "solve"),
        help="solve: no gold answer is shown to the model; reference: old mode with gold/reference answer.",
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("API_BASE") or "https://api.deepseek.com",
    )
    parser.add_argument("--id-key", default="pair_id")
    parser.add_argument("--thinking-field", default="deepseek_thinking")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "4096")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("LLM_RETRIES", "3")))
    parser.add_argument("--retry-sleep", type=float, default=float(os.environ.get("RETRY_SLEEP", "2")))
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ENABLE_THINKING", False),
        help="Control extra_body enable_thinking. Default is false for SFT-friendly visible reasoning.",
    )
    parser.add_argument(
        "--omit-extra-body",
        action="store_true",
        default=env_flag("OMIT_EXTRA_BODY"),
        help="Do not send provider-specific extra_body fields. Use this for OpenAI models such as gpt-4o.",
    )
    parser.add_argument("--response-format-json", action="store_true", default=env_flag("RESPONSE_FORMAT_JSON"))
    parser.add_argument("--disable-http-env", action="store_true", default=env_flag("OPENAI_DISABLE_HTTP_ENV"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", default=True)
    parser.add_argument("--no-retry-failed", action="store_false", dest="retry_failed")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--sft-skip-mismatch", action="store_true", default=True)
    parser.add_argument("--include-mismatch-in-sft", action="store_false", dest="sft_skip_mismatch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    indexed_rows = list(enumerate(rows))
    if args.num_shards > 1:
        indexed_rows = [(idx, row) for idx, row in indexed_rows if idx % args.num_shards == args.shard_index]

    done = completed_ids(output_path, args.id_key, args.retry_failed)
    pending = [
        (idx, row)
        for idx, row in indexed_rows
        if record_id(row, idx, args.id_key) not in done
    ]

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "sft_output": args.sft_output or None,
                "model": args.model,
                "prompt_mode": args.prompt_mode,
                "enable_thinking": args.enable_thinking,
                "omit_extra_body": args.omit_extra_body,
                "api_base": args.api_base,
                "rows": len(rows),
                "shard_rows": len(indexed_rows),
                "already_done": len(done),
                "pending": len(pending),
                "workers": args.workers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    completed = 0
    failed = 0
    with output_path.open("a", encoding="utf-8") as fout:
        for start in range(0, len(pending), args.workers):
            batch = pending[start : start + args.workers]
            results_by_index: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(call_model, row, idx, args): (idx, row)
                    for idx, row in batch
                }
                for future in as_completed(futures):
                    idx, row = futures[future]
                    try:
                        result = future.result()
                        results_by_index[idx] = make_output_row(row, result, args)
                    except Exception as exc:
                        failed += 1
                        if args.stop_on_error:
                            raise
                        results_by_index[idx] = make_error_row(row, idx, exc, args)

            for idx, _ in batch:
                fout.write(json_dumps(results_by_index[idx]) + "\n")
                completed += 1
            fout.flush()
            print(f"processed {completed}/{len(pending)} pending rows; failed={failed}", flush=True)

    summary = {"processed": completed, "failed": failed, "output": str(output_path)}
    if args.sft_output:
        summary["sft_counts"] = write_sft_file(output_path, Path(args.sft_output), args)
        summary["sft_output"] = args.sft_output
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
