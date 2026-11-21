# OmniSciDB

OmniSciDB synthesizes scientific reasoning data by transferring structure across domains. Questions from different disciplines rely on different knowledge, but they often share the same organization and reasoning patterns. The method first abstracts a domain-free ontology from a source question: the context graph describes entities, observations, conditions, and their relations, and the reasoning graph describes operations such as Extract, Compute, and Compare, together with their dependencies. In a target domain, the context graph is grounded into a coherent scenario from a domain profile alone, without documents from that domain. The reasoning graph is then reused to plan the item, generate the question and answer, and check that the evidence is sufficient and that the reasoning follows the original dependencies.

## Prompts

`prompts/` contains 15 execution templates. They cover dual-graph extraction and repair, three-score selection, scenario planning, progressive grounding, semantic validation, local repair, task planning, QA generation, independent solving, the critic, QA refinement, and chain-of-thought. See the [prompt reference](prompts/README.md) for call relationships and knowledge conventions.

## Pipeline

| Stage | Code | Behavior |
|---|---|---|
| CG/RG extraction | `scripts/extract_cross_domain_ontology.py` | Two extraction passes, type and dependency checks, and repair when extraction fails |
| Library selection | `select` | Score transferability, structural fidelity to the source QA, and reasoning potential, then filter by the mean score. Fidelity scoring requires the original QA |
| Scenario planning | `Pipeline.instantiate` | Check domain compatibility and replace the scenario when it is incompatible. The abstract structure stays unchanged |
| Progressive grounding | `grounding_order` / `instantiate` | Ground one node at a time along context-graph adjacency. Cyclic and disconnected context graphs are supported. Later nodes can see nodes already grounded |
| Validation and local repair | `repair_grounding` | Check node coverage and numeric format. Assess every relation and scientific consistency. Repair only the faulty node and its direct neighbors |
| Planner / Conductor | `plan` / `synthesize` | Plan which evidence to reveal and which reasoning results to hide, then generate the question, answer, and operation trace from the original reasoning graph |
| Improver | `verify` / `run` | The solver sees only the context, question, and options. The critic checks the answer, leakage, shortcuts, dependencies, and distractors, then falls back to planning or grounding |
| Saving and training input | `generate` / `export-sft` | Bounded concurrency, resume from successful samples, failure audit, and ShareGPT export |

## Run

Install the dependencies, then set an API key. `OPENAI_API_KEY`, `API_KEY`, and `DEEPSEEK_API_KEY` are all accepted. Set `OPENAI_BASE_URL`, `API_BASE`, or `DEEPSEEK_BASE_URL` when the endpoint is not the provider default.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com
```

Raw question-answer JSONL needs a nonempty `question` in each row. Extract both graphs before audit, selection, or generation.

```bash
python scripts/extract_cross_domain_ontology.py \
  --input data/raw.jsonl \
  --output data/graphs.jsonl

python scripts/run_omniscidb.py audit --input data/graphs.jsonl

python scripts/run_omniscidb.py select \
  --input data/graphs.jsonl \
  --source-input data/raw.jsonl \
  --output data/selected.jsonl

python scripts/run_omniscidb.py generate \
  --input data/selected.jsonl \
  --profile configs/medicine.json \
  --output outputs/medicine.jsonl

python scripts/run_omniscidb.py export-sft \
  --input outputs/medicine.jsonl \
  --output data/sft/train.jsonl
```