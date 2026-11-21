import hashlib
import json
from importlib.resources import files


ROLE_PROMPTS = {
    "ontology evaluator": "ontology_evaluation",
    "scenario planner": "scenario_planning",
    "progressive grounding": "context_grounding",
    "grounding critic": "context_validation",
    "local grounding repair": "context_local_repair",
    "reasoning task planner": "reasoning_task_planning",
    "conductor": "qa_generation",
    "independent solver": "independent_solving",
    "QA critic": "qa_validation",
}
PROMPT_NAMES = tuple(sorted(set(ROLE_PROMPTS.values()) | {
    "cross_domain_context_extraction", "cross_domain_reasoning_extraction",
    "ontology_extraction_repair", "qa_refinement", "cot_generation", "cot_reference_generation",
}))


def load_prompt(name):
    if name not in PROMPT_NAMES:
        raise ValueError(f"Unknown prompt: {name}")
    result = files("prompts").joinpath(f"{name}.md").read_text(encoding="utf-8").strip()
    if not result:
        raise ValueError(f"Empty prompt: {name}")
    return result


def render_prompt(name, **values):
    result = load_prompt(name)
    import re
    def replace(match):
        key = match.group(1)
        if key not in values:
            raise ValueError(f"Missing {key} for prompt {name}")
        value = values[key]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return re.sub(r"\{\{([a-z_]+)\}\}", replace, result)


def prompt_hashes():
    return {name: hashlib.sha256(load_prompt(name).encode()).hexdigest() for name in PROMPT_NAMES}
