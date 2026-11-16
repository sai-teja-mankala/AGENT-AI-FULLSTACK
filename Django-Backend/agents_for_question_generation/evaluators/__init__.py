"""Reusable, node-independent evaluators run over the finished pipeline output.

Enable with env EVALUATORS_ENABLED=structure,all_checks,quality_relevancy, or =true/all for every evaluator (default: none).
Add an evaluator: subclass/implement `Evaluator` and call `register(...)`.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# ARIZE-PHOENIX-EVALUATOR: result of one evaluator (score, pass/fail, reason) - shown as an evaluator.* span in Phoenix
@dataclass
class EvalResult:
    name: str
    score: Optional[float]
    passed: bool
    reason: str = ""
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ARIZE-PHOENIX-EVALUATOR: interface every evaluator implements (name + evaluate(output, context))
class Evaluator(Protocol):
    name: str

    def evaluate(self, output: List[Dict[str, Any]], context: Dict[str, Any]) -> EvalResult: ...


from .registry import register, get_enabled, REGISTRY  # noqa: E402
from .runner import run_evaluators  # noqa: E402

__all__ = ["EvalResult", "Evaluator", "register", "get_enabled", "REGISTRY", "run_evaluators"]
