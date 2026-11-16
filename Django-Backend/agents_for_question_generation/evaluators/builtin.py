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


# ARIZE-PHOENIX-EVALUATOR: "all_checks" - runs EVERY validation prompt (2 stem + 7 distractor checks) against each
# FINAL question/answer set and reports which pass or fail. One span per check + one summary span.
class AllChecksEvaluator:
    name = "all_checks"

    def evaluate(self, output, context) -> EvalResult:
        from . import judges
        cfg, cid = context.get("pipeline_config"), context.get("container_id")
        extra = {"evaluation_stage": "final_output", "job_id": context.get("job_id"),
                 "assessment_id": context.get("assessment_id")}
        outcomes = []
        for lo in output or []:
            for q in _questions([lo]):
                if "answer" not in q:  # MATCHING: these prompts do not apply
                    continue
                tasks = judges.stem_check_tasks(
                    q.get("questionType", "MULTIPLE_CHOICE"), lo.get("LevelOfQuiz", ""),
                    lo.get("LearningObjective", ""), q.get("questionText", ""), cfg, cid)
                tasks += judges.distractor_check_tasks(q, context.get("payload") or {}, cfg, cid)
                results = judges.run_parallel(tasks)
                judges.emit_outcomes(results, "evaluator.all_checks", extra)
                outcomes.extend(results)
        failed = [o for o in outcomes if not o.passed]
        passed = bool(outcomes) and not failed
        ratio = (len(outcomes) - len(failed)) / len(outcomes) if outcomes else None
        reason = ("all %d checks passed" % len(outcomes)) if passed else (
            "; ".join(f"{o.name}: {o.reason or 'failed'}" for o in failed) or "no questions")
        return EvalResult(self.name, ratio, passed, reason, threshold=1.0, metadata={
            "checks": [{"name": o.name, "score": o.score, "threshold": o.threshold, "passed": o.passed}
                       for o in outcomes]})


register(StructureEvaluator())
register(AllChecksEvaluator())
register(QualityRelevancyEvaluator())
