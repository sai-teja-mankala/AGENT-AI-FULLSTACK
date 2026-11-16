# ARIZE-PHOENIX-EVALUATOR-JUDGES: LLM-as-judge helpers that REUSE the pipeline's existing validation prompts
# (stem clarity/bias, stem LO/cognitive, 7 distractor checks). Shared by the generation-step and final-output evaluators.
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..tracing.spans import emit_span


@dataclass
class CheckOutcome:
    name: str
    score: Optional[float]
    threshold: Optional[float]
    passed: bool
    reason: str = ""
    start_ns: int = 0
    end_ns: int = 0


def _timed(name: str, threshold, call: Callable[[], Dict[str, Any]], reason_fn=None) -> CheckOutcome:
    start = time.time_ns()
    try:
        res = call() or {}
        reason = (reason_fn(res) if reason_fn else None) or res.get("feedback") or res.get("reason") or ""
        score = res.get("score")
        return CheckOutcome(name, score if isinstance(score, (int, float)) else None, threshold,
                            bool(res.get("pass")), str(reason)[:500], start, time.time_ns())
    except Exception as exc:  # a failing judge is reported as a failed check, never raised
        return CheckOutcome(name, None, threshold, False, f"judge error: {exc}", start, time.time_ns())


def stem_check_tasks(question_type, difficulty, learning_objective, stem, cfg, container_id) -> List[Callable]:
    from ..agents import langgraph_pipeline as lp
    tasks = []
    if lp._is_eligible_for_stem_clarity_bias(question_type):
        tasks.append(lambda: _timed("stem_clarity_bias", None, lambda: lp._call_stem_clarity_bias_validation(
            question_type, stem, cfg, container_id)))
    if lp._is_eligible_for_stem_lo_cognitive(question_type):
        tasks.append(lambda: _timed("stem_lo_cognitive", None, lambda: lp._call_stem_lo_cognitive_validation(
            question_type, difficulty, learning_objective, stem, cfg, container_id),
            reason_fn=lambda r: "; ".join(lp._collect_stem_lo_cog_failure_reasons(r))))
    return tasks


def distractor_check_tasks(question: Dict[str, Any], payload: Dict[str, Any], cfg, container_id) -> List[Callable]:
    from ..agents.langgraph_pipeline import VALIDATION_AGENTS
    from ..agents.pipeline_constants import CHECK_THRESHOLDS
    tasks = []
    for key, (_num, _label, fn, _ret) in VALIDATION_AGENTS.items():
        tasks.append(lambda key=key, fn=fn: _timed(f"{key}_check", CHECK_THRESHOLDS.get(key),
                                                   lambda: fn(question, payload, cfg, container_id)))
    return tasks


def run_parallel(tasks: List[Callable]) -> List[CheckOutcome]:
    """Run judge calls concurrently, keeping the current OTel span as parent."""
    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 9)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, t) for t in tasks]
        return [f.result() for f in futures]


def emit_outcomes(outcomes: List[CheckOutcome], prefix: str, extra: Dict[str, Any]) -> None:
    for o in outcomes:
        emit_span(f"{prefix}.{o.name}", {
            "operation_type": "evaluation",
            "check_name": o.name,
            "actual_score": o.score,
            "threshold": o.threshold,
            "validation_status": "pass" if o.passed else "fail",
            "eval.label": "pass" if o.passed else "fail",
            "eval.explanation": o.reason,
            **extra,
        }, start_ns=o.start_ns, end_ns=o.end_ns, kind="EVALUATOR")


def iter_questions(items: List[Dict[str, Any]]):
    """Yield (payload, question) from pipeline items shaped {payload, result:{questions:[...]}}."""
    for item in items or []:
        for q in (item.get("result") or {}).get("questions", []) or []:
            if isinstance(q, dict):
                yield item.get("payload") or {}, q
