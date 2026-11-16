# ARIZE-PHOENIX-EXPERIMENT: offline experiment runner (CLAUDE.md section 8.4). Command line only, never in the request path.
# 1) Loads the golden dataset from Phoenix (name from PHOENIX_GOLDEN_DATASET_NAME).
# 2) Runs the agent on each example with this pipeline version.
# 3) Scores each output with the same evaluators used in production, plus a final-status check.
# 4) Stores the run in Phoenix under --version.
# Never writes job records and never publishes to RabbitMQ.
import json
import os

from django.core.management.base import BaseCommand, CommandError
from phoenix.client import Client

from agents_for_question_generation.agents.langgraph_pipeline import _create_pipeline_config, _run_pipeline_impl
from agents_for_question_generation.evaluators import builtin  # noqa: F401  (registers the built-in evaluators)
from agents_for_question_generation.evaluators.registry import REGISTRY
from agents_for_question_generation.tracing import init_tracing

# Golden-dataset cells can arrive as JSON strings (e.g. from a CSV export); these fields must be turned back into their types.
LIST_FIELDS = ("learningObjectives", "questionTypes", "difficultyLevels")
INT_FIELDS = ("numberOfQuestions",)


def coerce_payload(raw):
    """Turn a golden example's input back into the payload shape the pipeline expects."""
    payload = dict(raw or {})
    for key in LIST_FIELDS:
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = json.loads(value) if value.strip() else []
    for key in INT_FIELDS:
        if isinstance(payload.get(key), str):
            payload[key] = int(payload[key])
    return payload


def _context_for(payload):
    """Same context keys run_pipeline() gives production evaluators."""
    research = payload.get("research", {}) or {}
    container_id = payload.get("assessment_container_id")
    return {
        "payload": payload,
        "pipeline_config": _create_pipeline_config(research),
        "container_id": container_id,
        "job_id": payload.get("task_id"),
        "assessment_id": container_id,
    }


def to_experiment_evaluator(evaluator):
    """Wrap a production evaluator (evaluate(output, context) -> EvalResult) as a Phoenix experiment evaluator."""
    def _evaluator(output, input):  # noqa: A002  (Phoenix binds these argument names)
        result = evaluator.evaluate(output, _context_for(coerce_payload(input)))
        values = {"score": result.score, "label": "pass" if result.passed else "fail", "explanation": result.reason}
        return {k: v for k, v in values.items() if v is not None}
    _evaluator.__name__ = evaluator.name
    return _evaluator


def final_status(output, input):  # noqa: A002
    """COMPLETED when questions were delivered, otherwise FAILED with the pipeline's own error messages as the reason."""
    questions = sum(len(lo.get("questions", []) or []) for lo in output or [])
    stats = (output[0] if output else {}).get("pipeline_stats", {}) or {}
    errors = [str(e) for e in stats.get("pipeline_errors", [])]
    completed = questions > 0
    reason = "; ".join(errors) or ("no questions delivered" if not completed else f"{questions} question(s) delivered")
    return {"score": 1.0 if completed else 0.0, "label": "COMPLETED" if completed else "FAILED", "explanation": reason}


def make_task(version):
    """The experiment task: run the agent on one golden example. Production evaluators are not run here."""
    def run_agent(input):  # noqa: A002  (Phoenix binds this argument name to the example input)
        payload = coerce_payload(input)
        payload.setdefault("task_id", f"experiment-{version}")
        return _run_pipeline_impl(payload)
    return run_agent


class Command(BaseCommand):
    help = "Run an offline experiment over the golden dataset in Phoenix. Writes no job records and publishes nothing to RabbitMQ."

    def add_arguments(self, parser):
        # Not "--version": Django's BaseCommand already defines that flag.
        parser.add_argument("--pipeline-version", required=True,
                            help="Pipeline version name, e.g. rag-v1. Used as the experiment name.")

    def handle(self, *args, **opts):
        dataset_name = os.getenv("PHOENIX_GOLDEN_DATASET_NAME", "").strip()
        if not dataset_name:
            raise CommandError("PHOENIX_GOLDEN_DATASET_NAME is not set in the environment")
        version = opts["pipeline_version"].strip()
        if not version:
            raise CommandError("--pipeline-version must not be empty")

        init_tracing()
        client = Client()
        dataset = client.datasets.get_dataset(dataset=dataset_name)
        evaluators = [to_experiment_evaluator(ev) for ev in REGISTRY.values()] + [final_status]
        client.experiments.run_experiment(
            dataset=dataset,
            task=make_task(version),
            evaluators=evaluators,
            experiment_name=version,
            experiment_metadata={"pipeline_version": version, "dataset": dataset_name},
            print_summary=True,
        )
        self.stdout.write(f"Experiment '{version}' finished on dataset '{dataset_name}'.")
