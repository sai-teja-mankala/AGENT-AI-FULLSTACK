from typing import Any, Dict, List

from . import EvalResult
from .registry import register


def _questions(output: List[Dict[str, Any]]):
    for lo in output or []:
        for q in lo.get("questions", []) or []:
            if isinstance(q, dict):
                yield q


# ARIZE-PHOENIX-EVALUATOR: "structure" (deterministic, no LLM)
class StructureEvaluator:
    """Deterministic: question text present, >=2 answers, >=1 correct, no duplicate answers."""
    name = "structure"

    def evaluate(self, output, context) -> EvalResult:
        errors, total = [], 0
        for i, q in enumerate(_questions(output)):
            if "answer" not in q:  # e.g. MATCHING has a different shape
                continue
            total += 1
            answers = q.get("answer") or []
            texts = [str(a.get("answerText", "")).strip().lower() for a in answers]
            if not str(q.get("questionText", "")).strip():
                errors.append(f"q{i}: empty question")
            if len(answers) < 2:
                errors.append(f"q{i}: fewer than 2 answers")
            if not any(a.get("isCorrect") for a in answers):
                errors.append(f"q{i}: no correct answer")
            if len(set(texts)) != len(texts):
                errors.append(f"q{i}: duplicate answers")
        passed = total > 0 and not errors
        reason = "; ".join(errors) if errors else (f"{total} question(s) structurally valid" if total else "no questions")
        return EvalResult(self.name, 1.0 if passed else 0.0, passed, reason, threshold=1.0,
                          metadata={"questions_checked": total, "error_count": len(errors)})


# ARIZE-PHOENIX-EVALUATOR: "quality_relevancy" (LLM judge, reuses the pipeline's existing evaluate_quality_and_relevancy)
class QualityRelevancyEvaluator:
    """LLM judge; reuses the pipeline's evaluate_quality_and_relevancy (needs context['pipeline_config'])."""
    name = "quality_relevancy"

    def evaluate(self, output, context) -> EvalResult:
        from ..agents.langgraph_pipeline import (
            evaluate_quality_and_relevancy, DEFAULT_QUALITY_RUBRIC, _learning_objective_descriptions)
        payload = context["payload"]
        q_thr = payload.get("qualityThreshold", 85)
        r_thr = payload.get("relevancyThreshold", 85)
        rubric = payload.get("qualityRubric") or DEFAULT_QUALITY_RUBRIC.strip()
        los = _learning_objective_descriptions(payload)
        scores, passes, reasons = [], [], []
        for lo in output or []:
            quality, relevancy = evaluate_quality_and_relevancy(
                lo, rubric, q_thr, los, r_thr, payload.get("locale", "en-US"),
                context.get("pipeline_config"), container_id=context.get("container_id"))
            for label, r in (("quality", quality), ("relevancy", relevancy)):
                if isinstance(r.get("score"), (int, float)):
                    scores.append(r["score"])
                passes.append(bool(r.get("pass")))
                if not r.get("pass"):
                    reasons.append(f"{label}: {r.get('issues')}")
        return EvalResult(self.name, min(scores) if scores else None, bool(passes) and all(passes),
                          "; ".join(reasons) or "quality and relevancy passed", threshold=min(q_thr, r_thr),
                          metadata={"quality_threshold": q_thr, "relevancy_threshold": r_thr})


register(StructureEvaluator())
register(QualityRelevancyEvaluator())
