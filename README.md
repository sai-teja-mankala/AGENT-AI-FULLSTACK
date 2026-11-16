# Question Generation Project

Users upload source material, configure an assessment, and receive multiple-choice questions grounded in that material. Every question passes quality evaluators before it is returned. Offline experiments compare pipeline versions against a golden dataset.

Full build specification: [CLAUDE.md](CLAUDE.md).

> **Status:** The question-generation agent (`Django-Backend`) works end to end. The other services are still in progress.

## Services

| Folder | Role | Status |
|---|---|---|
| `Django-Backend` | Question-generation agent, evaluators, offline experiments, Phoenix tracing | Working |
| `Nest-Backend` | Public API: authentication, PostgreSQL, job creation, RabbitMQ, Socket.IO | In progress |
| `RAG-pipeline` | Document parsing, chunking, embeddings, Qdrant retrieval | In progress |
| `Rabbitmq` | Queue and exchange definitions | In progress |
| `React-front-end` | User interface | In progress |

## Infrastructure

`docker-compose.yml` starts the local infrastructure:

- PostgreSQL on port 5432
- RabbitMQ on ports 5672 and 15672 (management UI)
- Qdrant on port 6333
- Phoenix on ports 6006 (UI) and 4317 (OTLP)

## Question-generation agent (`Django-Backend`)

### How a generation run works

1. The caller sends a payload (see [sample_payload.json](Django-Backend/sample_payload.json)): learning objectives, question types, difficulty levels, number of questions, locale and source text.
2. `run_pipeline()` in `agents_for_question_generation/agents/langgraph_pipeline.py` runs the LangGraph agent. It generates the questions, checks them and regenerates the ones that fail.
3. The agent runs its quality gates on each question and regenerates the ones that fail. Each run also records regeneration attempts and any pipeline errors under `pipeline_stats`.
4. If `EVALUATORS_ENABLED` is set, the post-run evaluators then score the finished questions. Their results go to Phoenix traces. They never change the returned questions.

### Evaluators

| Evaluator | Type | What it checks |
|---|---|---|
| `structure` | Deterministic | Question text, at least two answers, one correct answer, no duplicate answers |
| `all_checks` | LLM judges | The 2 stem checks and 7 distractor checks, on each question |
| `quality_relevancy` | LLM judge | Quality and relevancy against the configured thresholds, per learning objective |

Set `EVALUATORS_ENABLED=structure,all_checks,quality_relevancy`, or `true` to enable all of them.

`GENERATION_EVALUATORS_ENABLED=true` also runs judges after each generation step. It adds LLM calls and is off by default.

### Offline experiments

`run_experiment` runs the agent over a golden dataset stored in Phoenix and scores each output with the same evaluators, plus a `final_status` check (COMPLETED or FAILED, with the pipeline errors as the reason). The run is stored in Phoenix under the version name you give. Experiments write no job records and publish nothing to RabbitMQ.

Each example makes several LLM calls, so check the cost before running a large dataset.

## Setup

The project was built with Python 3.14. The venv lives in `Django-Backend/.venv`.

```bash
cd Django-Backend
cp .env.example .env          # then fill in the values (never commit .env)
python -m venv .venv          # skip if .venv already exists
.venv/bin/pip install -r requirements.lock.txt
docker compose up -d phoenix  # from the repo root
```

### Environment variables

Names only. Values go in `Django-Backend/.env`.

- LLM: `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `DEEPINFRA_API_KEY`, `GENERATOR_MODEL`, `JUDGE_MODEL`, `AGENT_USE_STRUCTURED_OUTPUT`
- Tracing: `PHOENIX_COLLECTOR_ENDPOINT`, `PHOENIX_PROJECT_NAME`, `PHOENIX_TRACING_ENABLED`
- Evaluators: `EVALUATORS_ENABLED`, `GENERATION_EVALUATORS_ENABLED`
- Experiments: `PHOENIX_GOLDEN_DATASET_NAME`

## Running

Run these from `Django-Backend`.

Generate questions from a payload file:

```bash
.venv/bin/python manage.py run_agent sample_payload.json
```

Run the offline experiment for a pipeline version:

```bash
.venv/bin/python manage.py run_experiment --pipeline-version rag-v1
```

Run the tests:

```bash
.venv/bin/python manage.py test agents_for_question_generation
```

## Tests

The agent's test suite covers the evaluators, the tracing setup, the experiment command and its helpers. Evaluator tests use stubbed judges with fixed scores, so they make no model calls.

## Documentation

- [CLAUDE.md](CLAUDE.md): the full specification, build order and rules for every service.
