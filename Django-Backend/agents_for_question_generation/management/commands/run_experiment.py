# ARIZE-PHOENIX-EXPERIMENT: offline experiment runner (CLAUDE.md section 8.4). Command line only, never in the request path.
# Never writes job records and never publishes to RabbitMQ.
#
# EXPERIMENT FLOW (search for "exp-step-<n>" to follow it):
#   exp-step-1   read PHOENIX_GOLDEN_DATASET_NAME from the environment
#   exp-step-2   validate --pipeline-version
#   exp-step-3   connect to Phoenix and fetch the golden dataset
#   exp-step-4   build the evaluator list (production judges + old-vs-new + final status)
#   exp-step-5   per example: run the agent with evaluate=False (production judges are not run here)
#   exp-step-6   per example: turn the golden input and reference output back into Python types
#   exp-step-7   per example: build the evaluator context (pipeline config, container and job ids, reference)
#   exp-step-8 to exp-step-11: old-vs-new comparison, see evaluators/comparison.py
#   exp-step-12  per example: final status (COMPLETED or FAILED)
#   exp-step-13  Phoenix stores the run under --pipeline-version and the summary is printed
import json
import os

from django.core.management.base import BaseCommand, CommandError
from phoenix.client import Client

from agents_for_question_generation.agents.langgraph_pipeline import _create_pipeline_config, run_pipeline
from agents_for_question_generation.evaluators import builtin  # noqa: F401  (registers the built-in evaluators)
from agents_for_question_generation.evaluators.comparison import OldVsNewEvaluator
from agents_for_question_generation.evaluators.registry import REGISTRY

# Golden-dataset cells can arrive as JSON strings (e.g. from a CSV export); these fields must be turned back into their types.
LIST_FIELDS = ("learningObjectives", "questionTypes", "difficultyLevels")
INT_FIELDS = ("numberOfQuestions",)


def coerce_payload(raw):
    """exp-step-6: turn a golden example's input back into the payload shape the pipeline expects."""
    payload = dict(raw or {})
    for key in LIST_FIELDS:
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = json.loads(value) if value.strip() else []
    for key in INT_FIELDS:
        if isinstance(payload.get(key), str):
            payload[key] = int(payload[key])
    return payload


def coerce_output(raw):
    """exp-step-6: the reference output may be stored as a JSON string; return the parsed value."""
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else None
    return raw


def _context_for(payload):
    """exp-step-7: the same context keys run_pipeline() gives production evaluators."""
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
    """Wrap an evaluator (evaluate(output, context) -> EvalResult) as a Phoenix experiment evaluator.

    Phoenix binds the golden dataset's reference output to `expected`; it is passed on as context["reference"].
    """
    def _evaluator(output, input, expected=None):  # noqa: A002  (Phoenix binds these argument names)
        context = _context_for(coerce_payload(input))                    # exp-step-7
        context["reference"] = coerce_output(expected)                   # exp-step-6
        result = evaluator.evaluate(output, context)
        values = {
            "score": result.score,
            "label": "pass" if result.passed else "fail",
            "explanation": result.reason,
            "metadata": result.metadata or None,
        }
        return {k: v for k, v in values.items() if v is not None}
    _evaluator.__name__ = evaluator.name
    return _evaluator


def final_status(output, input, expected=None):  # noqa: A002
    """exp-step-12: COMPLETED when questions were delivered, otherwise FAILED with the pipeline's own errors as the reason."""
    questions = sum(len(lo.get("questions", []) or []) for lo in output or [])
    stats = (output[0] if output else {}).get("pipeline_stats", {}) or {}
    errors = [str(e) for e in stats.get("pipeline_errors", [])]
    completed = questions > 0
    reason = "; ".join(errors) or ("no questions delivered" if not completed else f"{questions} question(s) delivered")
    return {"score": 1.0 if completed else 0.0, "label": "COMPLETED" if completed else "FAILED", "explanation": reason}


def make_task(version):
    """exp-step-5: run the agent on one golden example. Production evaluators are skipped (evaluate=False)."""
    def run_agent(input):  # noqa: A002  (Phoenix binds this argument name to the example input)
        payload = coerce_payload(input)
        payload.setdefault("task_id", f"experiment-{version}")
        return run_pipeline(payload, evaluate=False)
    return run_agent


class Command(BaseCommand):
    help = "Run an offline experiment over the golden dataset in Phoenix. Writes no job records and publishes nothing to RabbitMQ."

    def add_arguments(self, parser):
        # Not "--version": Django's BaseCommand already defines that flag.
        parser.add_argument("--pipeline-version", required=True,
                            help="Pipeline version name, e.g. rag-v1. Used as the experiment name.")

    def handle(self, *args, **opts):
        dataset_name = os.getenv("PHOENIX_GOLDEN_DATASET_NAME", "").strip()   # exp-step-1
        if not dataset_name:
            raise CommandError("PHOENIX_GOLDEN_DATASET_NAME is not set in the environment")
        version = opts["pipeline_version"].strip()                            # exp-step-2
        if not version:
            raise CommandError("--pipeline-version must not be empty")

        client = Client()                                                     # exp-step-3
        dataset = client.datasets.get_dataset(dataset=dataset_name)
        evaluators = (                                                        # exp-step-4
            [to_experiment_evaluator(ev) for ev in REGISTRY.values()]
            + [to_experiment_evaluator(OldVsNewEvaluator()), final_status]
        )
        client.experiments.run_experiment(                                    # exp-step-5, 13
            dataset=dataset,
            task=make_task(version),
            evaluators=evaluators,
            experiment_name=version,
            experiment_metadata={"pipeline_version": version, "dataset": dataset_name},
            print_summary=True,
        )
        self.stdout.write(f"Experiment '{version}' finished on dataset '{dataset_name}'.")
