# ARIZE-PHOENIX-GENERATION-EVALUATOR: LLM-as-judge run right after a GENERATION node, reusing the validation prompts.
# Decoupled from node code: graph_builder.py wraps the node; nothing is written to graph state, so it can never
# change routing or output. Opt-in: env GENERATION_EVALUATORS_ENABLED=true (adds extra judge LLM calls).
#
#   build_question / regenerate_stem_*      -> judge the generated stems (stem clarity/bias + LO/cognitive prompts)
#   build_options  / correct_failed_distractors -> judge the generated options (the 7 distractor check prompts)
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict

from ..tracing.node_specs import NODE_SPECS
from . import judges

logger = logging.getLogger(__name__)

STEM_NODES = {"build_question", "regenerate_stem_clarity_bias", "regenerate_stem_lo_cognitive"}
OPTION_NODES = {"build_options": "raw_outputs", "correct_failed_distractors": "distractor_corrections_output"}
GENERATION_EVAL_NODES = STEM_NODES | set(OPTION_NODES)


def generation_evaluators_enabled() -> bool:
    return os.getenv("GENERATION_EVALUATORS_ENABLED", "false").lower() in ("1", "true", "yes")


def _evaluate(node_name: str, state: Dict[str, Any]) -> None:
    cfg, cid = state.get("pipeline_config"), state.get("container_id")
    spec = NODE_SPECS.get(node_name)
    extra = {
        "evaluation_stage": f"generation:{node_name}",
        "node_name": node_name,
        "job_id": state.get("task_id"),
        "assessment_id": (state.get("input") or {}).get("assessment_container_id"),
        "attempt_number": spec.attempt(state, {}) if spec and spec.attempt else None,
    }
    if node_name in STEM_NODES:
        for entry in state.get("questions") or []:
            p = entry.get("payload") or {}
            tasks = judges.stem_check_tasks(p.get("questionType", "MULTIPLE_CHOICE"), p.get("difficultyLevel", ""),
                                            p.get("learningObjective", ""), entry.get("questionText", ""), cfg, cid)
            judges.emit_outcomes(judges.run_parallel(tasks), "generation_eval", extra)
    else:
        for payload, question in judges.iter_questions(state.get(OPTION_NODES[node_name]) or []):
            tasks = judges.distractor_check_tasks(question, payload, cfg, cid)
            judges.emit_outcomes(judges.run_parallel(tasks), "generation_eval", extra)


def with_generation_evaluators(node_name: str, fn: Callable) -> Callable:
    """Wrap a generation node so enabled judges run on its output. Judge failures are swallowed."""
    if node_name not in GENERATION_EVAL_NODES:
        return fn

    @wraps(fn)
    def wrapper(state):
        out = fn(state)
        if generation_evaluators_enabled():
            try:
                _evaluate(node_name, out if isinstance(out, dict) else state)
            except Exception:
                logger.exception("EVAL: generation evaluator failed for %s (ignored)", node_name)
        return out

    return wrapper
