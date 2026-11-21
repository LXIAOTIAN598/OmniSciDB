#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prompt_templates import render_prompt

REQUEST_CONTEXT = threading.local()


class Tee:

    def __init__(self, console, logfile, lock):
        self.console, self.logfile, self.lock = console, logfile, lock

    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.logfile.write(text)
            self.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.console.flush()
            self.logfile.flush()

    def isatty(self):
        return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_id(row_id: int, record: dict[str, Any]) -> str:
    return str(record.get("idx", record.get("source_idx", row_id)))


def truncate_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    return text[:head] + "\n...[TRUNCATED]...\n" + text[-(max_chars - head):]


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    else:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")
    return payload

CONTEXT_TYPES = {
    "Entity": "Object, system, agent, or case being described.",
    "Property": "Qualitative attribute of an entity.",
    "Metric": "Measurement dimension or criterion; distinct from its value.",
    "Event": "Given occurrence or intervention.",
    "Mechanism": "Explicitly supplied process explaining an interaction.",
    "Observation": "Supplied observation or case fact, not an inferred finding.",
    "Condition": "Circumstance or applicability condition.",
    "Quantity": "Supplied numerical value with unit/dimension requirements.",
    "Rule": "Explicitly supplied principle, criterion, or procedural rule.",
    "Formula": "Explicitly supplied mathematical relationship.",
    "Constraint": "Supplied admissibility restriction or boundary.",
    "Candidate": "Alternative or hypothesis explicitly given for evaluation.",
    "Evidence": "Explicitly supplied evidence item or report.",
    "Dataset": "Supplied collection, table, or series of observations.",
    "Method": "Explicitly supplied procedure or analysis method.",
}
CONTEXT_RELATIONS = {
    "has-property": "Entity -> Property: source possesses target attribute.",
    "measured-by": "Entity/Property -> Metric: target measures source.",
    "participates-in": "Entity -> Event/Mechanism: source participates in target.",
    "affects": "Source influences target without asserting a specific cause.",
    "causes": "Source causes target, only when stated in the context.",
    "supported-by": "Source contextual fact -> target supplied evidence.",
    "subject-to": "Source is subject to target Condition/Constraint/Rule.",
    "has-value": "Metric/Property -> Quantity: supplied value of source.",
    "part-of": "Source is a component or member of target.",
    "instance-of": "Source is an instance of the category represented by target.",
    "precedes": "Source occurs before target in the supplied context.",
    "defined-by": "Source Metric/Property -> target Formula/Rule/Method.",
    "applies-to": "Source Rule/Formula/Method/Condition -> target governed item.",
    "contradicts": "Source and target are explicitly conflicting supplied facts.",
}
OPERATIONS = {
    "Extract": "Read and organize facts, conditions, or candidate propositions already supplied in context.",
    "Retrieve": "Obtain indispensable domain knowledge NOT supplied in context; specify the required principle precisely.",
    "Compute": "Calculate a quantity using bound values and a supplied/required method.",
    "Compare": "Compare bound values, states, candidates, or results by a criterion.",
    "Aggregate": "Combine multiple values or evidence items into a summary.",
    "Infer": "Derive a new conclusion, explanation, classification, or elimination judgment from inputs.",
    "Verify": "Check a result against evidence, rules, or constraints.",
    "Select": "Choose the candidate option or option set satisfying the question's selection criterion, using prior judgments.",
}
SCHEMA_VERSION = "cross_domain_ontology_v2"


class GraphValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphValidationError(message)


def indexed_nodes(nodes: Any, prefix: str, types: dict) -> dict:
    require(isinstance(nodes, list) and bool(nodes), f"{prefix}: nodes must be nonempty")
    result = {}
    for node in nodes:
        require(isinstance(node, dict), "Each node must be an object")
        nid = node.get("id")
        require(isinstance(nid, str) and re.fullmatch(prefix + r"[1-9]\d*", nid) is not None,
                f"Invalid node id: {nid!r}")
        require(nid not in result, f"Duplicate node id: {nid}")
        require(isinstance(node.get("type"), str) and node["type"] in types,
                f"Unknown type on {nid}: {node.get('type')}")
        result[nid] = node
    return result


def validate_context(payload: Any) -> dict:
    require(isinstance(payload, dict), "Context output must be an object")
    graph = payload.get("context_graph", payload)
    require(isinstance(graph, dict), "context_graph must be an object")
    nodes = indexed_nodes(graph.get("nodes"), "C", CONTEXT_TYPES)
    counts: Counter = Counter()
    clean_nodes = []
    for nid, node in nodes.items():
        counts[node["type"]] += 1
        clean_nodes.append({"id": nid, "type": node["type"],
                            "abstract_label": f"{node['type']}_{counts[node['type']]}",
                            "visibility": "context"})
    edges = graph.get("edges")
    require(isinstance(edges, list), "Context edges must be a list")
    clean_edges, seen = [], set()
    for edge in edges:
        require(isinstance(edge, dict), "Context edge must be an object")
        s, t, typ = edge.get("source"), edge.get("target"), edge.get("type")
        require(isinstance(s, str) and isinstance(t, str) and s in nodes and t in nodes,
                f"Unknown context endpoint: {s!r} -> {t!r}")
        require(s != t, f"Context self edge: {s}")
        require(isinstance(typ, str) and typ in CONTEXT_RELATIONS, f"Unknown relation: {typ}")
        key = (s, t, typ)
        require(key not in seen, f"Duplicate context edge: {key}")
        seen.add(key)
        clean_edges.append({"id": f"CE{len(clean_edges)+1}", "source": s,
                            "target": t, "type": typ})
    return {"nodes": clean_nodes, "edges": clean_edges}


def validate_reasoning(payload: Any, context: dict) -> dict:
    require(isinstance(payload, dict), "Reasoning output must be an object")
    graph = payload.get("reasoning_graph", payload)
    require(isinstance(graph, dict), "reasoning_graph must be an object")
    nodes = indexed_nodes(graph.get("nodes"), "R", OPERATIONS)
    cids = {n["id"] for n in context["nodes"]}
    edges, bindings, clean = [], [], []
    incoming = {nid: 0 for nid in nodes}
    outgoing: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        inputs = node.get("inputs")
        require(isinstance(inputs, list) and bool(inputs), f"{nid}: inputs must be nonempty")
        require(all(isinstance(x, str) for x in inputs), f"{nid}: inputs must be IDs")
        require(len(set(inputs)) == len(inputs), f"{nid}: duplicate input")
        instruction = node.get("instruction")
        require(isinstance(instruction, str) and bool(instruction.strip()),
                f"{nid}: missing reusable instruction")
        refs = set(re.findall(r"\{([CR][1-9]\d*)\}", instruction))
        require(refs == set(inputs), f"{nid}: instruction placeholders must match inputs exactly")
        for src in inputs:
            require(src in cids or src in nodes, f"{nid}: unknown input {src}")
            require(src != nid, f"{nid}: self dependency")
            if src in cids:
                bindings.append({"source": src, "target": nid, "type": "context-input"})
            else:
                edges.append({"id": f"RE{len(edges)+1}", "source": src,
                              "target": nid, "type": "result-input"})
                outgoing[src].append(nid)
                incoming[nid] += 1
        clean.append({"id": nid, "type": node["type"], "inputs": inputs,
                      "instruction": instruction})
    queue = deque(nid for nid in nodes if incoming[nid] == 0)
    order = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for target in outgoing[nid]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    require(len(order) == len(nodes), "Reasoning dependencies contain a cycle")
    final = graph.get("answer_node_id")
    require(isinstance(final, str) and final in nodes, "answer_node_id must reference an operation")
    require(not outgoing[final], "Answer operation must be a sink")
    ancestors, stack = set(), [final]
    while stack:
        nid = stack.pop()
        if nid not in ancestors:
            ancestors.add(nid)
            stack.extend(src for src in nodes[nid]["inputs"] if src in nodes)
    require(ancestors == set(nodes), "Every operation must contribute to the answer")
    if nodes[final]["type"] == "Select":
        candidates = {n["id"] for n in context["nodes"] if n["type"] == "Candidate"}
        used_context = {src for nid in ancestors for src in nodes[nid]["inputs"] if src in cids}
        require(bool(candidates), "Select requires supplied Candidate slots")
        require(candidates <= used_context,
                f"Select must consider all candidates; unused: {sorted(candidates-used_context)}")
        require(any(src in nodes for src in nodes[final]["inputs"]),
                "Select must consume a preceding judgment or comparison")
    if "edges" in graph:
        supplied = graph["edges"]
        require(isinstance(supplied, list) and all(isinstance(e, dict) for e in supplied),
                "Reasoning edges must be objects")
        expected = [(e["source"], e["target"]) for e in edges]
        actual = [(e.get("source"), e.get("target")) for e in supplied]
        require(Counter(map(repr, actual)) == Counter(map(repr, expected)),
                "Explicit reasoning edges disagree with input dependencies")
    return {"nodes": clean, "edges": edges, "context_bindings": bindings,
            "topological_order": order, "answer_node_id": final}


def mermaid(context: dict, reasoning: dict) -> str:
    lines = ["flowchart TD", '  subgraph GC["Context Graph"]']
    lines += [f'    {n["id"]}["{n["abstract_label"]}"]' for n in context["nodes"]]
    lines += [f'    {e["source"]} -->|{e["type"]}| {e["target"]}' for e in context["edges"]]
    lines += ["  end", '  subgraph GR["Reasoning Graph"]']
    lines += [f'    {n["id"]}["{n["id"]}: {n["type"]}"]' for n in reasoning["nodes"]]
    lines += [f'    {e["source"]} --> {e["target"]}' for e in reasoning["edges"]]
    lines += ["  end"]
    lines += [f'  {e["source"]} -.->|input| {e["target"]}' for e in reasoning["context_bindings"]]
    return "\n".join(lines)


def call_llm(prompt: str, args: argparse.Namespace) -> dict:
    from openai import OpenAI
    is_deepseek = "deepseek" in args.model.lower()
    api_key = (os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
               or (os.environ.get("DEEPSEEK_API_KEY") if is_deepseek else None))
    if not api_key:
        raise RuntimeError("Missing API key: set API_KEY or OPENAI_API_KEY in this terminal")
    base_url = (os.environ.get("API_BASE") or os.environ.get("OPENAI_BASE_URL")
                or (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
                    if is_deepseek else None))
    thinking = args.thinking
    if thinking == "auto":
        thinking = "disabled" if is_deepseek else "provider-default"
    request = {
        "model": args.model,
        "messages": [{"role": "system", "content":
                      "Extract reusable cross-domain ontology graphs. Return strict JSON only."},
                     {"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens, "stream": False,
        "response_format": {"type": args.response_format},
    }
    if thinking in {"enabled", "disabled"}:
        request["extra_body"] = {"thinking": {"type": thinking}}
    if thinking == "enabled":
        request["reasoning_effort"] = args.reasoning_effort
    else:
        request["temperature"] = args.temperature
    address = urlsplit(base_url or "https://api.openai.com/v1")
    safe_url = urlunsplit((address.scheme, address.hostname or "", address.path, "", ""))
    label = (f"source={getattr(REQUEST_CONTEXT, 'source', 'api-check')} "
             f"stage={getattr(REQUEST_CONTEXT, 'stage', 'api-check')}")
    last = None
    for attempt in range(args.llm_retries + 1):
        started = time.monotonic()
        hit_token_limit = False
        print(f"[API {label} attempt={attempt+1}] endpoint={safe_url} model={args.model} "
              f"format={args.response_format} thinking={thinking} max_tokens={args.max_tokens}", flush=True)
        try:
            with OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                        timeout=args.request_timeout) as client:
                response = client.chat.completions.create(**request)
            if not response.choices:
                raise RuntimeError("Provider returned no choices")
            choice = response.choices[0]
            hit_token_limit = choice.finish_reason == "length"
            text = choice.message.content
            reasoning = getattr(choice.message, "reasoning_content", None)
            usage = getattr(response, "usage", None)
            metadata = {
                "response_id": getattr(response, "id", None),
                "served_model": getattr(response, "model", None),
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "content_characters": len(text) if isinstance(text, str) else 0,
                "reasoning_characters": len(reasoning) if isinstance(reasoning, str) else 0,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            print(f"[API response {label}] {json.dumps(metadata, ensure_ascii=False)}", flush=True)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(
                    "Provider returned empty message.content; "
                    f"finish_reason={choice.finish_reason!r}; "
                    f"reasoning_characters={len(reasoning) if isinstance(reasoning, str) else 0}; "
                    f"completion_tokens={getattr(usage, 'completion_tokens', None)}; "
                    f"refusal_present={bool(getattr(choice.message, 'refusal', None))}. "
                    "JSON mode may return whitespace even with finish_reason='stop'. "
                    "If finish_reason='length', increase --max-tokens or disable thinking. "
                    "Reasoning text is not used as the final JSON answer."
                )
            if choice.finish_reason == "length":
                raise RuntimeError(f"Response hit the token limit (thinking={thinking}); "
                                   "generate a more compact graph or increase --max-tokens")
            return parse_json_response(text)
        except Exception as exc:
            last = exc
            print(f"[API {label} {attempt+1}/{args.llm_retries+1}] {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            if attempt < args.llm_retries:
                if hit_token_limit:
                    request["messages"] = [request["messages"][0], {"role": "user", "content":
                        prompt + "\n\nThe previous response exceeded the output token limit. "
                        "Regenerate a complete, more compact JSON object. Keep only the required "
                        "fields and essential distinct slots and direct relations. Do not enumerate "
                        "hypothetical facts, repeat records, or add explanations. "
                        "Return minified JSON without indentation."}]
                    print(f"[API {label}] Retrying with compact-output instruction", flush=True)
                time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(f"Model request failed: {last}") from last


def render(template: str, sample: dict) -> str:
    return template.replace("{{sample_json}}", json.dumps(sample, ensure_ascii=False, indent=2))


def extract_validated(prompt: str, validator, args: argparse.Namespace) -> tuple[dict, list]:
    history = []
    request = prompt
    for attempt in range(args.repair_rounds + 1):
        payload = call_llm(request, args)
        try:
            return validator(payload), history
        except GraphValidationError as exc:
            history.append(str(exc))
            print(f"[validation repair {attempt}/{args.repair_rounds}] {exc}", file=sys.stderr, flush=True)
            if attempt == args.repair_rounds:
                raise GraphValidationError("; ".join(history)) from exc
            request = render_prompt("ontology_extraction_repair", original_prompt=prompt,
                                    candidate_json=payload, validation_error=str(exc))
    raise AssertionError("unreachable")


def context_sample(record: dict, args: argparse.Namespace) -> dict:
    question = record.get("question")
    require(isinstance(question, str) and bool(question.strip()),
            "Expected source QA row with nonempty question; graph-only JSONL is not source QA")
    return {"question": truncate_text(question, args.max_question_chars),
            "node_type_guide": CONTEXT_TYPES, "edge_type_guide": CONTEXT_RELATIONS}


def extract_one(row_id: int, record: dict, templates: tuple, args: argparse.Namespace) -> dict:
    result = {"row_id": row_id, "source_idx": source_id(row_id, record),
              "model": args.model, "schema_version": SCHEMA_VERSION,
              "operation_types": list(OPERATIONS),
              "graph": None, "validation_repairs": {}, "errors": []}
    try:
        REQUEST_CONTEXT.source = result["source_idx"]
        REQUEST_CONTEXT.stage = "context"
        print(f"[source={result['source_idx']}] Extracting Context Graph", flush=True)
        context, repairs = extract_validated(render(templates[0], context_sample(record, args)),
                                             validate_context, args)
        result["validation_repairs"]["context"] = repairs
        sample = {"question": truncate_text(record["question"], args.max_question_chars),
                  "reference_answer": truncate_text(record.get("reference_answer", ""), args.max_answer_chars),
                  "final_answer": truncate_text(record.get("final_answer", ""), args.max_answer_chars),
                  "source_solution": truncate_text(record.get("response", ""), args.max_answer_chars),
                  "context_graph": context, "operation_guide": OPERATIONS}
        REQUEST_CONTEXT.stage = "reasoning"
        print(f"[source={result['source_idx']}] Extracting Reasoning Graph", flush=True)
        reasoning, repairs = extract_validated(render(templates[1], sample),
                                               lambda p: validate_reasoning(p, context), args)
        result["validation_repairs"]["reasoning"] = repairs
        result["graph"] = {"schema_version": SCHEMA_VERSION, "context_graph": context,
                           "reasoning_graph": reasoning, "mermaid": mermaid(context, reasoning)}
    except Exception as exc:
        result["errors"] = [f"{type(exc).__name__}: {exc}"]
        print(f"[source={result['source_idx']}] FAILED: {result['errors'][0]}", file=sys.stderr, flush=True)
    finally:
        REQUEST_CONTEXT.__dict__.clear()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "backup/data/reference_data/sample/sample.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reference_data/sample/cross_domain_ontology.v2.10k.jsonl")
    parser.add_argument("--log-file", type=Path, help="Append console output and errors to this file; defaults to OUTPUT with .log suffix")
    parser.add_argument("--context-prompt", type=Path, default=ROOT / "prompts/cross_domain_context_extraction.md")
    parser.add_argument("--reasoning-prompt", type=Path, default=ROOT / "prompts/cross_domain_reasoning_extraction.md")
    parser.add_argument("--model", default=os.environ.get("SKELETON_MODEL", "deepseek-flash"))
    parser.add_argument("--response-format", choices=("json_object", "text"), default="json_object")
    parser.add_argument("--thinking", choices=("auto", "enabled", "disabled"), default="auto",
                        help="auto explicitly disables thinking for DeepSeek; leaves other providers unchanged")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high",
                        help="Used only with --thinking enabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    parser.add_argument("--check-api", action="store_true", help="Test one small JSON response without reading input or writing output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repair-rounds", type=int, default=2)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-question-chars", type=int, default=0, help="0 preserves the full question")
    parser.add_argument("--max-answer-chars", type=int, default=12000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Skip successful rows; retry failed rows")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print first context prompt, without calling the model")
    args = parser.parse_args()
    if args.start < 0 or (args.limit is not None and args.limit <= 0) or args.workers <= 0:
        parser.error("start must be >= 0; limit and workers must be > 0")
    if min(args.repair_rounds, args.llm_retries, args.retry_sleep) < 0 or args.max_tokens <= 0:
        parser.error("Retry/repair settings must be >= 0; max-tokens must be > 0")
    if args.request_timeout <= 0:
        parser.error("request-timeout must be > 0")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ")
    if args.log_file and args.log_file.resolve() in (args.input.resolve(), args.output.resolve()):
        parser.error("Log file must differ from input and output")
    return args


def main() -> int:
    args = parse_args()
    if args.dry_run and args.log_file is None:
        return run(args)
    logfile = args.log_file or args.output.with_suffix(".api-check.log" if args.check_api else ".log")
    if logfile.resolve() in (args.input.resolve(), args.output.resolve()):
        raise ValueError("Log file must differ from input and output; set --log-file")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as stream:
        lock = threading.RLock()
        with redirect_stdout(Tee(sys.stdout, stream, lock)), redirect_stderr(Tee(sys.stderr, stream, lock)):
            print(f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] Run started", flush=True)
            print(f"Log: {logfile.resolve()}\nModel: {args.model}\nInput: {args.input}\nOutput: {args.output}", flush=True)
            print(f"workers={args.workers} limit={args.limit} resume={args.resume} "
                  f"max_tokens={args.max_tokens} timeout={args.request_timeout}s "
                  f"response_format={args.response_format} thinking={args.thinking} "
                  f"reasoning_effort={args.reasoning_effort}", flush=True)
            started = time.monotonic()
            status = 1
            try:
                status = run(args)
            except KeyboardInterrupt:
                print("Interrupted by user", file=sys.stderr, flush=True)
                status = 130
            except Exception:
                traceback.print_exc()
            print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                  f"Run ended: exit={status}, elapsed={time.monotonic()-started:.1f}s", flush=True)
            return status


def run(args: argparse.Namespace) -> int:
    if args.check_api:
        response = call_llm('Return exactly this JSON object: {"ok": true}', args)
        print("API check returned valid JSON:", json.dumps(response, ensure_ascii=False))
        return 0
    templates = (args.context_prompt.read_text(encoding="utf-8"),
                 args.reasoning_prompt.read_text(encoding="utf-8"))
    rows = load_jsonl(args.input)
    selected = list(enumerate(rows))[args.start:args.start + args.limit if args.limit else None]
    if args.dry_run:
        if not selected:
            raise ValueError("No input rows selected")
        print(render(templates[0], context_sample(selected[0][1], args)))
        return 0
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; choose --resume or --overwrite")
    if args.resume and args.output.exists():
        seen = {str(r["source_idx"]) for r in load_jsonl(args.output)
                if r.get("schema_version") == SCHEMA_VERSION and r.get("graph") and not r.get("errors")}
        selected = [(i, r) for i, r in selected if source_id(i, r) not in seen]
    if args.overwrite and args.output.exists():
        args.output.unlink()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(lambda item: extract_one(*item, templates, args), selected), 1):
            append_jsonl(args.output, [result])
            failed += bool(result["errors"])
            status = "failed" if result["errors"] else "written"
            print(f"{done}/{len(selected)} source={result['source_idx']} {status}", flush=True)
    print(f"Completed: {len(selected)-failed} successful, {failed} failed -> {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
