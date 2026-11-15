import logging
import time
from typing import Any, Dict, List, Optional

from ..tracing.spans import emit_span
from . import EvalResult
from .registry import get_enabled

logger = logging.getLogger(__name__)


# ARIZE-PHOENIX-EVALUATOR-RUNNER: runs enabled evaluators after the pipeline; each result becomes an evaluator.* span
def run_evaluators(output: List[Dict[str, Any]], context: Dict[str, Any],
                   evaluators: Optional[list] = None) -> List[EvalResult]:
    """Run each enabled evaluator independently; one failing never affects the others
    or the pipeline. Each result is emitted as an EVALUATOR span under the current span."""
    results: List[EvalResult] = []
    for ev in (evaluators if evaluators is not None else get_enabled()):
        start = time.time_ns()
        try:
            res = ev.evaluate(output, context)
        except Exception as exc:
            logger.exception("EVAL: %s failed", getattr(ev, "name", ev))
            res = EvalResult(getattr(ev, "name", "unknown"), None, False, f"evaluator error: {exc}")
        results.append(res)
        emit_span(f"evaluator.{res.name}", {
            "operation_type": "evaluation",
            "check_name": res.name,
            "actual_score": res.score,
            "threshold": res.threshold,
            "validation_status": "pass" if res.passed else "fail",
            "eval.label": "pass" if res.passed else "fail",
            "eval.explanation": res.reason,
            "job_id": context.get("job_id"),
            "assessment_id": context.get("assessment_id"),
            "eval.metadata": res.metadata or None,
        }, start_ns=start, end_ns=time.time_ns())
    return results
