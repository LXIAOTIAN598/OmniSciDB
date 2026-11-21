import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.prompt_templates import load_prompt, prompt_hashes

from .client import JSONClient
from .graphs import audit, digest, grounding_order, load_graph, read_jsonl
from .pipeline import Pipeline, Settings, require


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def output_ready(args, protected):
    paths = [args.output, args.output.with_suffix(".events.jsonl"), args.output.with_suffix(".manifest.json")]
    require(len({p.resolve() for p in paths}) == len(paths), "output artifact paths collide")
    protected = {p.resolve() for p in protected if p}
    require(not any(p.resolve() in protected for p in paths), "output would overwrite an input")
    require(not any(p.exists() for p in paths) or args.resume or args.overwrite,
            "output exists; choose --resume or --overwrite")
    if args.overwrite:
        for path in paths:
            path.unlink(missing_ok=True)
    return paths[1], paths[2]


def new_client(args):
    return JSONClient(args.model, base_url=args.base_url, timeout=args.timeout,
                      retries=args.llm_retries, max_tokens=args.max_tokens,
                      thinking=args.thinking)


def generate(args):
    profile = json.loads(args.profile.read_text()) if args.profile else {"domain": args.domain}
    require(isinstance(profile, dict) and isinstance(profile.get("domain"), str) and profile["domain"].strip(),
            "domain profile must contain a nonempty domain string")
    settings = Settings(args.scenario_attempts, args.repair_rounds, args.qa_attempts,
                        args.min_score, args.question_type, args.language)
    config = {"model": args.model, "base_url": args.base_url or os.environ.get("API_BASE")
              or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL"),
              "thinking": args.thinking, "max_tokens": args.max_tokens,
              "settings": asdict(settings), "profile": profile, "replicas": args.replicas,
              "prompt_hashes": prompt_hashes()}
    if config["base_url"]:
        from urllib.parse import urlsplit, urlunsplit
        url = urlsplit(config["base_url"])
        config["base_url"] = urlunsplit((url.scheme, url.hostname or "", url.path, "", ""))
    config_hash = digest(config)
    if args.dry_run:
        for line, record in read_jsonl(args.input):
            if line - 1 < args.start:
                continue
            try:
                graph = load_graph(record)
            except (ValueError, TypeError, KeyError) as exc:
                print(f"skip line {line}: {exc}", file=sys.stderr)
                continue
            print(json.dumps({"line": line, "config": config, "graph_hash": digest(graph),
                              "context_nodes": len(graph["context_graph"]["nodes"]),
                              "reasoning_order": graph["reasoning_graph"]["topological_order"],
                              "grounding_order": grounding_order(graph["context_graph"])},
                             ensure_ascii=False, indent=2))
            return 0
        raise ValueError("No valid graphs in requested range")
    client = new_client(args)
    try:
        events, manifest = output_ready(args, [args.input, args.profile])
        if args.resume and manifest.exists():
            previous = json.loads(manifest.read_text())
            require(previous["config_hash"] == config_hash, "resume configuration differs; use a new output")
        done = set()
        if args.resume and args.output.exists():
            for _, row in read_jsonl(args.output):
                require(row.get("config_hash") == config_hash, "output contains incompatible run configuration")
                if row.get("status") == "accepted":
                    done.add(row["job_id"])
        write_json(manifest, {"input": str(args.input.resolve()), "config": config, "config_hash": config_hash})
        counts = {"accepted": 0, "rejected": 0, "invalid_graphs": 0, "resumed": 0}
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        def run_job(graph, replica):
            pipeline = Pipeline(client, settings)
            try:
                return pipeline.run(graph, {**profile, "variant": replica}), pipeline.events, None
            except Exception as exc:
                return None, pipeline.events, f"{type(exc).__name__}: {exc}"

        pending = {}

        def collect():
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                job, source, replica = pending.pop(future)
                result, history, error = future.result()
                if error is None:
                    result.update(source_idx=source, replica=replica, job_id=job, config_hash=config_hash)
                    append(args.output, result)
                    done.add(job)
                    counts["accepted"] += 1
                    status = "accepted"
                else:
                    counts["rejected"] += 1
                    status = "rejected"
                    print(f"source={source}: {error}", file=sys.stderr)
                append(events, {"job_id": job, "status": status, "error": error, "events": history})

        processed = 0
        seen = set()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for line, record in read_jsonl(args.input):
                if line - 1 < args.start:
                    continue
                try:
                    graph = load_graph(record)
                except (ValueError, TypeError, KeyError) as exc:
                    counts["invalid_graphs"] += 1
                    append(events, {"status": "invalid_graph", "line": line, "error": str(exc)})
                    continue
                source = str(record.get("source_idx", record.get("row_id", line - 1)))
                identity = (source, digest(graph))
                if identity in seen:
                    continue
                seen.add(identity)
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1
                for replica in range(args.replicas):
                    job = digest({"source": source, "graph": identity[1], "config": config_hash, "replica": replica})
                    if job in done:
                        counts["resumed"] += 1
                        continue
                    print(f"source={source} replica={replica}: generating", flush=True)
                    pending[executor.submit(run_job, graph, replica)] = (job, source, replica)
                    if len(pending) >= 2 * args.workers:
                        collect()
            while pending:
                collect()
        print(json.dumps(counts))
        require(processed > 0, "No valid graphs in requested range")
        return 1 if counts["rejected"] else 0
    finally:
        client.close()


def select(args):
    sources = {}
    for line, row in read_jsonl(args.source_input):
        sources[str(row.get("source_idx", row.get("idx", line - 1)))] = row
    client = new_client(args)
    try:
        events, _ = output_ready(args, [args.input, args.source_input])
        done = {r["selection_id"] for _, r in read_jsonl(args.output)} if args.resume and args.output.exists() else set()
        processed = 0
        for line, record in read_jsonl(args.input):
            try:
                graph = load_graph(record)
            except (ValueError, TypeError, KeyError) as exc:
                append(events, {"line": line, "error": str(exc)})
                continue
            source = str(record.get("source_idx", record.get("row_id", line - 1)))
            identity = digest({"source": source, "graph": graph, "model": args.model,
                               "threshold": args.threshold, "source_qa": sources.get(source),
                               "prompt_hashes": prompt_hashes()})
            if identity in done:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1
            require(source in sources, f"Missing original QA for source {source}; cannot score fidelity")
            review = client("ontology evaluator",
                load_prompt("ontology_evaluation"),
                {"graph": graph, "source_qa": sources[source]})
            scores = [review.get(k) for k in ("cross_domain_transferability", "structural_fidelity", "reasoning_potential")]
            require(all(type(v) in (int, float) and 0 <= v <= 10 for v in scores), "Invalid selection scores")
            mean = sum(scores) / 3
            append(events, {"source_idx": source, "selection": review, "mean": mean})
            if mean >= args.threshold:
                append(args.output, {**record, "graph": graph, "selection": review,
                                     "selection_mean": mean, "selection_id": identity})
                done.add(identity)
        return 0
    finally:
        client.close()


def export(args):
    require(args.input.resolve() != args.output.resolve(), "input and output must differ")
    require(not args.output.exists() or args.overwrite, "export exists; use --overwrite")
    rows = []
    for _, row in read_jsonl(args.input):
        if row.get("status") != "accepted":
            continue
        qa = row["qa"]
        prompt = qa["context"] + "\n\n" + qa["question"]
        if args.command == "export-qa":
            options = qa.get("options", {})
            require(bool(options) and set(options) <= set("ABCDE"),
                    "legacy MCQ tools require options labeled A–E; use export-sft for other tasks")
            rows.append({"pair_id": row["job_id"], "source_idx": row["source_idx"],
                         "domain": row["domain_profile"]["domain"],
                         "subdomain": row["scenario"]["subdomain"], "question": prompt,
                         "options": [{"label": k, "text": v} for k, v in options.items()],
                         "answer_option": qa["answer"], "answer_option_text": options[qa["answer"]],
                         "answer": qa["explanation"],
                         "reasoning_steps": [{"step": s["node_id"], "reasoning": s["output"]}
                                             for s in qa["reasoning_steps"]]})
            continue
        if qa.get("options"):
            prompt += "\n" + "\n".join(f"{k}. {v}" for k, v in qa["options"].items())
        answer = qa["explanation"] + "\n\nAnswer: " + qa["answer"]
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": answer}]})
    require(bool(rows), "No accepted records to export")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"exported": len(rows), "output": str(args.output)}))
    return 0


def parser():
    root = argparse.ArgumentParser(description="Synthesize scientific QA from context and reasoning graphs")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("audit", help="Offline full-library structural audit")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    for name in ("generate", "select"):
        p = commands.add_parser(name)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--model", default=os.environ.get("OMNISCIDB_MODEL", "deepseek-v4-flash"))
        p.add_argument("--base-url")
        p.add_argument("--timeout", type=float, default=120)
        p.add_argument("--max-tokens", type=int, default=8192)
        p.add_argument("--thinking", choices=("provider-default", "enabled", "disabled"), default="provider-default")
        p.add_argument("--llm-retries", type=int, default=2)
        p.add_argument("--limit", type=int)
        mode = p.add_mutually_exclusive_group()
        mode.add_argument("--resume", action="store_true")
        mode.add_argument("--overwrite", action="store_true")
        if name == "generate":
            domain = p.add_mutually_exclusive_group(required=True)
            domain.add_argument("--domain")
            domain.add_argument("--profile", type=Path)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--replicas", type=int, default=1)
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--scenario-attempts", type=int, default=3)
            p.add_argument("--repair-rounds", type=int, default=2)
            p.add_argument("--qa-attempts", type=int, default=3)
            p.add_argument("--min-score", type=float, default=8.5)
            p.add_argument("--question-type", choices=("multiple-choice", "open-ended"), default="multiple-choice")
            p.add_argument("--language", default="English")
            p.add_argument("--dry-run", action="store_true")
        else:
            p.add_argument("--source-input", type=Path, required=True)
            p.add_argument("--threshold", type=float, default=7.0)
    for name in ("export-sft", "export-qa"):
        p = commands.add_parser(name)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--overwrite", action="store_true")
    return root


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        for name in ("limit", "replicas", "workers", "scenario_attempts", "qa_attempts", "max_tokens", "timeout"):
            value = getattr(args, name, None)
            require(value is None or value > 0, f"{name} must be positive")
        for name in ("start", "repair_rounds", "llm_retries"):
            require(getattr(args, name, 0) >= 0, f"{name} must be nonnegative")
        for name in ("min_score", "threshold"):
            require(0 <= getattr(args, name, 0) <= 10, f"{name} must be in [0,10]")
        if args.command == "audit":
            report = audit(args.input)
            if args.output:
                require(args.input.resolve() != args.output.resolve(), "audit output must differ from input")
                write_json(args.output, report)
            print(json.dumps({k: v for k, v in report.items() if k != "invalid"}, ensure_ascii=False, indent=2))
            return 0
        return {"generate": generate, "select": select, "export-sft": export, "export-qa": export}[args.command](args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
