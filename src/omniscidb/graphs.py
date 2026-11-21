import hashlib
import json
from collections import Counter, deque
from pathlib import Path

from scripts.extract_cross_domain_ontology import validate_context, validate_reasoning


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            yield line_no, row


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_graph(record):
    graph = record.get("graph", record)
    if not isinstance(graph, dict):
        raise ValueError("missing/null graph: " + "; ".join(map(str, record.get("errors", []))))
    context = validate_context(graph.get("context_graph"))
    reasoning = validate_reasoning(graph.get("reasoning_graph"), context)
    supplied = graph["reasoning_graph"].get("context_bindings")
    if supplied is not None:
        expected = Counter((x["source"], x["target"], x["type"])
                           for x in reasoning["context_bindings"])
        actual = Counter((x.get("source"), x.get("target"), x.get("type")) for x in supplied)
        if actual != expected:
            raise ValueError("context_bindings disagree with operation inputs")
    return {"schema_version": "cross_domain_ontology_v2", "context_graph": context,
            "reasoning_graph": reasoning}


def grounding_order(context):
    neighbors = {n["id"]: [] for n in context["nodes"]}
    for edge in context["edges"]:
        neighbors[edge["source"]].append(edge["target"])
        neighbors[edge["target"]].append(edge["source"])
    seen, order = set(), []
    for node in neighbors:
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            queue.extend(neighbors[current])
    return order


def audit(path):
    report = {"rows": 0, "valid_rows": 0, "invalid_rows": 0,
              "duplicate_source_rows": 0, "operation_counts": Counter(), "invalid": []}
    seen = set()
    for line, row in read_jsonl(path):
        report["rows"] += 1
        source = str(row.get("source_idx", row.get("row_id", line - 1)))
        report["duplicate_source_rows"] += source in seen
        seen.add(source)
        try:
            graph = load_graph(row)
            report["valid_rows"] += 1
            report["operation_counts"].update(n["type"] for n in graph["reasoning_graph"]["nodes"])
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            report["invalid_rows"] += 1
            report["invalid"].append({"line": line, "source_idx": source, "error": str(exc)})
    report["unique_sources"] = len(seen)
    return report
