# Prompt reference

These templates serve the dual-graph pipeline: context-graph and reasoning-graph extraction, three-score selection, scenario planning, node-by-node grounding, validation and local repair, task planning, QA generation and refinement, independent solving, and the critic. 
| Task | Prompt file | Caller |
|---|---|---|
| Context-graph extraction | `cross_domain_context_extraction.md` | `extract_cross_domain_ontology.py` |
| Reasoning-graph extraction | `cross_domain_reasoning_extraction.md` | same script |
| Structural repair | `ontology_extraction_repair.md` | repair after a failed extraction check |
| Three-score selection | `ontology_evaluation.md` | `select` |
| Scenario planning | `scenario_planning.md` | `Pipeline.instantiate` |
| Progressive grounding | `context_grounding.md` | one call per context-graph node |
| Semantic validation | `context_validation.md` | `Pipeline.review_grounding` |
| Local repair | `context_local_repair.md` | `Pipeline.repair_grounding` |
| Task planning | `reasoning_task_planning.md` | `Pipeline.plan`, including replanning |
| Conductor | `qa_generation.md` | first QA generation |
| Independent solving | `independent_solving.md` | `Pipeline.verify`; sees only the task and options |
| Critic and fault attribution | `qa_validation.md` | `Pipeline.verify` |
| Iterative repair | `qa_refinement.md` | regeneration with feedback and the previous QA |
| Extra chain-of-thought | `cot_generation.md` | CoT tool, default `solve` mode |
| Reference-assisted explanation | `cot_reference_generation.md` | CoT tool, `reference` mode |
