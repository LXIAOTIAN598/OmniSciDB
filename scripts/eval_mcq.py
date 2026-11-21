#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = set("ABCDE")


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    backend: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["eval_id"]))
            except Exception:
                continue
    return done


def option_lines(row: dict[str, Any]) -> list[str]:
    opts = []
    for opt in row.get("options") or []:
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label") or "").strip().upper()
        text = str(opt.get("text") or "").strip()
        if label in LABELS and text:
            opts.append((label, text))
    return [f"{label}. {text}" for label, text in sorted(opts)]


def build_prompt(row: dict[str, Any], cot: bool) -> str:
    if cot:
        instructions = [
            "Answer the following multiple-choice question.",
            "Think through the problem step by step.",
            "After your reasoning, put the final answer on a new line in exactly this format:",
            "Answer: <A/B/C/D/E>",
        ]
    else:
        instructions = [
            "Answer the following multiple-choice question.",
            "Choose the single best option.",
            "Return only one letter: A, B, C, D, or E.",
        ]
    return "\n".join(
        [
            *instructions,
            "",
            "Question:",
            str(row.get("question") or "").strip(),
            "",
            "Options:",
            *option_lines(row),
            "",
            "Answer:",
        ]
    )


def chat_prompt(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    for pat in patterns:
        match = re.search(pat, text, flags=re.I)
        if match:
            return match.group(1).upper()
    letters = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", text.upper())
    if letters:
        return letters[-1]
    return None


def load_hf_model(model_path: str) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer, backend="hf")


def load_vllm_model(
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    max_num_seqs: int,
    max_num_batched_tokens: int | None,
    enforce_eager: bool,
) -> ModelBundle:
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": enforce_eager,
    }
    if max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    return ModelBundle(model=LLM(**kwargs), tokenizer=tokenizer, backend="vllm")


@torch.no_grad()
def generate_hf(
    bundle: ModelBundle,
    prompts: list[str],
    max_new_tokens: int,
    max_input_tokens: int,
) -> list[str]:
    old_padding_side = bundle.tokenizer.padding_side
    bundle.tokenizer.padding_side = "left"
    inputs = bundle.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(bundle.model.device) for key, value in inputs.items()}
    outputs = bundle.model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=bundle.tokenizer.pad_token_id,
        eos_token_id=bundle.tokenizer.eos_token_id,
    )
    prompt_width = inputs["input_ids"].shape[1]
    texts = [
        bundle.tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
        for row in outputs
    ]
    bundle.tokenizer.padding_side = old_padding_side
    return texts


def generate_vllm(
    bundle: ModelBundle,
    prompts: list[str],
    max_new_tokens: int,
    seed: int | None,
) -> list[str]:
    from vllm import SamplingParams

    outputs = bundle.model.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            seed=seed,
            skip_special_tokens=True,
        ),
    )
    return [item.outputs[0].text.strip() for item in outputs]


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", choices=("vllm", "hf"), default=os.getenv("BACKEND", "vllm"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--cot", action="store_true", default=os.getenv("COT", "0") == "1")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tensor_parallel_size", type=int, default=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--gpu_memory_utilization", type=float, default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--vllm_dtype", default=os.getenv("VLLM_DTYPE", "auto"))
    parser.add_argument("--vllm_max_model_len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--vllm_max_num_seqs", type=int, default=int(os.getenv("VLLM_MAX_NUM_SEQS", "16")))
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "4096")))
    parser.add_argument("--vllm_enforce_eager", action="store_true", default=os.getenv("VLLM_ENFORCE_EAGER", "0") == "1")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    indexed_rows = list(enumerate(rows))
    if args.num_shards > 1:
        indexed_rows = [
            (idx, row)
            for idx, row in indexed_rows
            if idx % args.num_shards == args.shard_index
        ]

    done = completed_ids(pred_path)
    pending = [
        (idx, row)
        for idx, row in indexed_rows
        if str(row.get("pair_id") or idx) not in done
    ]

    if args.backend == "hf":
        bundle = load_hf_model(args.model_path)
    else:
        bundle = load_vllm_model(
            args.model_path,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            args.vllm_max_model_len,
            args.vllm_dtype,
            args.vllm_max_num_seqs,
            args.vllm_max_num_batched_tokens or None,
            args.vllm_enforce_eager,
        )

    with pred_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [
                chat_prompt(bundle.tokenizer, build_prompt(row, args.cot), args.enable_thinking)
                for _, row in batch
            ]
            if bundle.backend == "hf":
                outputs = generate_hf(bundle, prompts, args.max_new_tokens, args.max_input_tokens)
            else:
                outputs = generate_vllm(bundle, prompts, args.max_new_tokens, args.seed)
            for (idx, row), prompt, output in zip(batch, prompts, outputs):
                gold = row.get("answer_option")
                pred = extract_choice(output)
                scored = gold in LABELS
                record = {
                    "eval_id": str(row.get("pair_id") or idx),
                    "row_index": idx,
                    "pair_id": row.get("pair_id"),
                    "domain": row.get("domain"),
                    "subdomain": row.get("subdomain"),
                    "validation_score": row.get("validation_score"),
                    "question": row.get("question"),
                    "gold": gold,
                    "prediction": pred,
                    "correct": (pred == gold) if scored else None,
                    "scored": scored,
                    "raw_output": output,
                    "prompt": prompt,
                    "answer_option_source": row.get("answer_option_source"),
                    "original_correct_option": row.get("correct_option"),
                    "original_final_answer": row.get("final_answer"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"generated {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)

    predictions = read_jsonl(pred_path)
    total = len(predictions)
    scored_rows = [r for r in predictions if r.get("scored")]
    correct = sum(1 for r in scored_rows if r.get("correct") is True)
    unparsed = sum(1 for r in scored_rows if r.get("prediction") not in LABELS)

    by_domain: dict[str, dict[str, Any]] = {}
    for row in scored_rows:
        domain = row.get("domain") or "unknown"
        item = by_domain.setdefault(domain, {"total": 0, "correct": 0, "accuracy": None})
        item["total"] += 1
        item["correct"] += int(row.get("correct") is True)
    for item in by_domain.values():
        item["accuracy"] = item["correct"] / item["total"] if item["total"] else None

    summary = {
        "input": str(input_path),
        "prediction_path": str(pred_path),
        "model_path": args.model_path,
        "backend": args.backend,
        "total_predictions": total,
        "total_input_rows_before_sharding": len(rows),
        "shard_rows": len(indexed_rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "scored": len(scored_rows),
        "unscored_gold_missing": total - len(scored_rows),
        "correct": correct,
        "accuracy": correct / len(scored_rows) if scored_rows else None,
        "unparsed_model_outputs": unparsed,
        "prediction_distribution": dict(Counter(r.get("prediction") for r in scored_rows)),
        "gold_distribution": dict(Counter(r.get("gold") for r in scored_rows)),
        "by_domain": by_domain,
        "generation": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "cot": args.cot,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
            "vllm_max_model_len": args.vllm_max_model_len if args.backend == "vllm" else None,
            "vllm_max_num_seqs": args.vllm_max_num_seqs if args.backend == "vllm" else None,
            "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens if args.backend == "vllm" else None,
        },
    }
    write_summary(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
