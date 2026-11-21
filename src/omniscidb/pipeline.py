import ast
import copy
import math
import operator
import re
from dataclasses import dataclass

from scripts.prompt_templates import ROLE_PROMPTS, load_prompt, render_prompt

from .graphs import digest, grounding_order


class Rejected(ValueError):
    pass


@dataclass
class Settings:
    scenario_attempts: int = 3
    repair_rounds: int = 2
    qa_attempts: int = 3
    min_score: float = 8.5
    question_type: str = "multiple-choice"
    language: str = "English"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_realization(value):
    require(isinstance(value, dict) and text(value.get("text")), "grounded node needs nonempty text")
    if "quantity" in value:
        quantity = value["quantity"]
        require(isinstance(quantity, dict), "quantity must be an object")
        require(type(quantity.get("value")) in (int, float) and math.isfinite(quantity["value"]),
                "quantity value must be finite")
        require(text(quantity.get("unit")), "quantity requires a unit (use '1' for dimensionless)")


def validate_grounding(graph, grounded):
    expected = {n["id"] for n in graph["context_graph"]["nodes"]}
    require(set(grounded) == expected, "grounding must cover exactly all CG nodes")
    for value in grounded.values():
        validate_realization(value)


def validate_scenario(graph, scenario):
    require(text(scenario.get("scenario")) and text(scenario.get("subdomain")), "incomplete scenario")
    ids = {n["id"] for n in graph["context_graph"]["nodes"]}
    requirements = scenario.get("grounding_requirements")
    require(isinstance(requirements, dict) and set(requirements) == ids
            and all(text(v) for v in requirements.values()), "scenario must specify requirements for every CG node")
    relations = scenario.get("relation_requirements")
    require(isinstance(relations, list) and all(isinstance(e, dict) for e in relations),
            "scenario requires relation_requirements")
    expected = {(e["source"], e["target"], e["type"]) for e in graph["context_graph"]["edges"]}
    actual = [(e.get("source"), e.get("target"), e.get("type")) for e in relations]
    require(len(actual) == len(expected) and set(actual) == expected
            and all(text(e.get("requirement")) for e in relations),
            "scenario relation requirements must preserve every CG edge exactly")


def arithmetic(expression, variables):
    require(isinstance(expression, str) and len(expression) <= 500, "invalid arithmetic expression")
    tree = ast.parse(expression, mode="eval")
    require(sum(1 for _ in ast.walk(tree)) <= 100, "arithmetic expression too complex")
    functions = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.Pow: operator.pow}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.Name) and node.id in variables:
            result = variables[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in functions:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                require(abs(right) <= 12, "exponent exceeds arithmetic limit")
            result = functions[type(node.op)](left, right)
        else:
            raise ValueError("unsupported arithmetic; only numbers, bound quantities and + - * / ** allowed")
        require(type(result) in (int, float) and math.isfinite(result), "nonfinite arithmetic")
        return result

    return visit(tree.body)


def validate_plan(graph, plan):
    require(text(plan.get("objective")), "plan requires objective")
    rg = graph["reasoning_graph"]
    required = {b["source"] for b in rg["context_bindings"]}
    visible = plan.get("visible_context_ids")
    hidden = plan.get("withheld_reasoning_ids")
    require(isinstance(visible, list) and all(isinstance(x, str) for x in visible), "invalid visible_context_ids")
    require(required <= set(visible) <= {n["id"] for n in graph["context_graph"]["nodes"]},
            "plan must present context required by all RG bindings")
    require(isinstance(hidden, list) and set(hidden) == set(rg["topological_order"]),
            "plan must withhold all RG intermediate results and final conclusion")
    require(text(plan.get("question_strategy")), "plan needs question_strategy")
    operations = plan.get("operation_plan")
    require(isinstance(operations, list) and all(isinstance(x, dict) for x in operations),
            "plan requires operation_plan")
    require([x.get("node_id") for x in operations] == rg["topological_order"],
            "operation_plan must follow every RG operation in order")
    nodes = {n["id"]: n for n in rg["nodes"]}
    for item in operations:
        instruction = item.get("target_instruction")
        require(text(instruction) and set(re.findall(r"\{([CR][1-9]\d*)\}", instruction))
                == set(nodes[item["node_id"]]["inputs"]),
                "target instruction placeholders must match original RG inputs")
    background = plan.get("background_knowledge")
    require(isinstance(background, list) and all(isinstance(x, dict) for x in background),
            "plan requires background_knowledge list")
    retrieve_ids = {n["id"] for n in nodes.values() if n["type"] == "Retrieve"}
    require(len(background) == len(retrieve_ids)
            and {x.get("node_id") for x in background} == retrieve_ids
            and all(text(x.get("requirement")) for x in background),
            "background knowledge must specify exactly the Retrieve operations")


def validate_qa(graph, grounded, qa, question_type):
    for key in ("context", "question", "answer", "explanation"):
        require(text(qa.get(key)), f"QA requires {key}")
    rg = graph["reasoning_graph"]
    steps = qa.get("reasoning_steps")
    require(isinstance(steps, list) and all(isinstance(s, dict) for s in steps), "missing reasoning_steps")
    require([s.get("node_id") for s in steps] == rg["topological_order"],
            "trace must cover every RG operation once in topological order")
    by_id = {n["id"]: n for n in rg["nodes"]}
    for step in steps:
        node = by_id[step["node_id"]]
        require(step.get("inputs") == node["inputs"], f"trace inputs changed at {node['id']}")
        require(step.get("operation") == node["type"], f"operation changed at {node['id']}")
        require(text(step.get("output")), f"missing result at {node['id']}")
    options = qa.get("options", {})
    if question_type == "multiple-choice":
        require(isinstance(options, dict) and 2 <= len(options) <= 10, "MCQ requires labeled options")
        require(all(text(k) and text(v) for k, v in options.items()), "invalid MCQ option")
        require(len(set(v.strip().casefold() for v in options.values())) == len(options), "duplicate options")
        require(qa["answer"] in options, "MCQ answer must be exactly one option label")
        candidates = {n["id"] for n in graph["context_graph"]["nodes"] if n["type"] == "Candidate"}
        mapping = qa.get("candidate_option_map", {})
        require(isinstance(mapping, dict) and set(mapping) == candidates, "candidate_option_map must cover all candidates")
        if candidates:
            require(all(isinstance(v, str) for v in mapping.values()) and len(mapping) == len(options)
                    and set(mapping.values()) == set(options), "candidate-to-option mapping must be one-to-one")
    else:
        require(not options, "open-ended QA should not contain options")
    checks = qa.get("numerical_checks", [])
    require(isinstance(checks, list), "numerical_checks must be a list")
    covered = set()
    for check in checks:
        require(isinstance(check, dict) and check.get("node_id") in by_id, "invalid numerical check")
        nid = check["node_id"]
        variables = check.get("variables", {})
        require(isinstance(variables, dict), "variables must be an object")
        for cid, val in variables.items():
            require(cid in grounded and "quantity" in grounded[cid], "numeric variable must bind to CG quantity")
            require(type(val) in (int, float) and val == grounded[cid]["quantity"]["value"],
                    "arithmetic operands disagree with grounded quantities")
        result = arithmetic(check.get("expression"), variables)
        expected = check.get("expected")
        require(type(expected) in (int, float) and math.isfinite(expected), "invalid expected number")
        require(math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-8), f"arithmetic mismatch at {nid}")
        require(text(check.get("unit")), "numeric check must declare output unit")
        covered.add(nid)
    require({n["id"] for n in by_id.values() if n["type"] == "Compute"} <= covered,
            "each Compute operation requires a numerical check")


class Pipeline:
    def __init__(self, client, settings=None):
        self.client = client
        self.settings = settings or Settings()
        self.events = []

    def ask(self, role, instruction, payload):
        response = self.client(role, instruction, copy.deepcopy(payload))
        require(isinstance(response, dict), f"{role}: expected object")
        prompt_name = ROLE_PROMPTS[role]
        if role == "conductor" and payload.get("feedback"):
            prompt_name = "qa_refinement"
        self.events.append({"role": role, "prompt_name": prompt_name,
                            "prompt_hash": digest(instruction), "response": copy.deepcopy(response)})
        return response

    def review_grounding(self, graph, scenario, grounded):
        return self.ask("grounding critic",
            load_prompt('context_validation'),
            {"graph": graph, "scenario": scenario, "grounded_nodes": grounded})

    def repair_grounding(self, graph, scenario, grounded, review):
        issues = review.get("issues", [])
        affected = {cid for issue in issues for cid in issue.get("node_ids", [])}
        require(bool(affected) and affected <= set(grounded), "grounding repair requires localized valid node IDs")
        allowed = set(affected)
        for edge in graph["context_graph"]["edges"]:
            if edge["source"] in affected or edge["target"] in affected:
                allowed.update([edge["source"], edge["target"]])
        patch = self.ask("local grounding repair",
            load_prompt('context_local_repair'),
            {"graph": graph, "scenario": scenario, "grounded_nodes": grounded,
             "review": review, "allowed_nodes": sorted(allowed)})
        nodes = patch.get("nodes")
        require(isinstance(nodes, dict) and bool(nodes) and set(nodes) <= allowed, "repair changed unaffected nodes")
        for value in nodes.values():
            validate_realization(value)
        result = copy.deepcopy(grounded)
        result.update(nodes)
        validate_grounding(graph, result)
        return result

    def instantiate(self, graph, profile, scenario_index):
        scenario = self.ask("scenario planner",
            load_prompt('scenario_planning'),
            {"context_graph": graph["context_graph"], "domain_profile": profile,
             "scenario_index": scenario_index,
             "previous_scenarios": [e["response"] for e in self.events if e["role"] == "scenario planner"]})
        if scenario.get("feasible") is not True:
            raise Rejected("No compatible scenario")
        validate_scenario(graph, scenario)
        grounded = {}
        nodes = {n["id"]: n for n in graph["context_graph"]["nodes"]}
        for cid in grounding_order(graph["context_graph"]):
            realization = self.ask("progressive grounding",
                load_prompt('context_grounding'),
                {"current_node": nodes[cid], "scenario": scenario, "domain_profile": profile,
                 "context_graph": graph["context_graph"],
                 "grounded_nodes": grounded,
                 "relations": [e for e in graph["context_graph"]["edges"] if cid in (e["source"], e["target"])]})
            validate_realization(realization)
            grounded[cid] = realization
        validate_grounding(graph, grounded)
        for repair in range(self.settings.repair_rounds + 1):
            review = self.review_grounding(graph, scenario, grounded)
            if review.get("valid") is True and review.get("issues") == []:
                return scenario, grounded
            if repair < self.settings.repair_rounds:
                grounded = self.repair_grounding(graph, scenario, grounded, review)
        raise Rejected("Grounding rejected after local repair budget")

    def plan(self, graph, grounded, feedback=None):
        plan = self.ask("reasoning task planner",
            load_prompt('reasoning_task_planning'),
            {"graph": graph, "grounded_nodes": grounded, "feedback": feedback,
             "question_type": self.settings.question_type, "language": self.settings.language})
        validate_plan(graph, plan)
        return plan

    def synthesize(self, graph, grounded, plan, feedback, previous_qa=None):
        return self.ask("conductor",
            (render_prompt("qa_refinement", generation_instructions=load_prompt("qa_generation"))
             if feedback else load_prompt("qa_generation")),
            {"graph": graph, "grounded_nodes": grounded, "plan": plan, "feedback": feedback,
             "previous_qa": previous_qa,
             "question_type": self.settings.question_type, "language": self.settings.language})

    def verify(self, graph, grounded, plan, qa):
        solution = self.ask("independent solver",
            load_prompt('independent_solving'),
            {key: qa[key] for key in ("context", "question", "options") if key in qa})
        review = self.ask("QA critic",
            load_prompt('qa_validation'),
            {"graph": graph, "grounded_nodes": grounded, "plan": plan, "qa": qa,
             "independent_solution": solution})
        score = review.get("score")
        checks = review.get("dependency_checks")
        coverage_valid = (isinstance(checks, list) and all(isinstance(x, dict) for x in checks)
                          and [x.get("node_id") for x in checks] == graph["reasoning_graph"]["topological_order"]
                          and all(x.get("covered") is True and text(x.get("evidence")) for x in checks))
        if review.get("valid") is True and (not coverage_valid
                or not text(review.get("shortcut_check")) or not text(review.get("leakage_check"))):
            issues = review.get("issues")
            review["issues"] = (issues if isinstance(issues, list) else []) + [
                "Critic must provide complete dependency evidence and explicit shortcut/leakage assessments."]
            review["valid"] = False
            review["repair_stage"] = "qa"
        passed = (review.get("valid") is True and review.get("issues") == []
                  and type(score) in (int, float) and self.settings.min_score <= score <= 10
                  and solution.get("answerable") is True and text(solution.get("answer"))
                  and review.get("answer_agrees") is True and coverage_valid
                  and text(review.get("shortcut_check")) and text(review.get("leakage_check")))
        if self.settings.question_type == "multiple-choice":
            passed = passed and solution.get("answer") == qa["answer"]
        return passed, review, solution

    def run(self, graph, profile):
        self.events = []
        original = copy.deepcopy(graph)
        fingerprint = digest(graph)
        last = "No attempts"
        for scenario_index in range(self.settings.scenario_attempts):
            try:
                scenario, grounded = self.instantiate(graph, profile, scenario_index)
                feedback, plan, previous_qa = None, None, None
                for attempt in range(self.settings.qa_attempts):
                    try:
                        if plan is None:
                            plan = self.plan(graph, grounded, feedback)
                        qa = self.synthesize(graph, grounded, plan, feedback, previous_qa)
                        previous_qa = copy.deepcopy(qa)
                        validate_qa(graph, grounded, qa, self.settings.question_type)
                        passed, feedback, solution = self.verify(graph, grounded, plan, qa)
                        if passed:
                            require(graph == original, "abstract graph mutated")
                            return {"status": "accepted", "graph_hash": fingerprint,
                                    "domain_profile": profile, "scenario": scenario,
                                    "graph": original, "grounded_nodes": grounded, "plan": plan,
                                    "qa": qa, "review": feedback, "independent_solution": solution}
                        last = str(feedback)
                        if feedback.get("repair_stage") == "grounding":
                            grounded = self.repair_grounding(graph, scenario, grounded,
                                {"issues": [{"node_ids": feedback.get("node_ids", []), "message": last}]})
                            review = self.review_grounding(graph, scenario, grounded)
                            if review.get("valid") is not True or review.get("issues") != []:
                                raise Rejected("Stage 2 repair failed semantic revalidation")
                            plan = None
                        elif feedback.get("repair_stage") == "plan":
                            plan = None
                    except Rejected:
                        raise
                    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                        feedback = {"deterministic_error": str(exc)}
                        last = str(exc)
                        plan = None
                        self.events.append({"role": "validation", "error": last})
            except (ValueError, TypeError, KeyError) as exc:
                last = str(exc)
                self.events.append({"role": "scenario rejection", "error": last})
        raise Rejected(f"Instance discarded after bounded retries: {last}")
