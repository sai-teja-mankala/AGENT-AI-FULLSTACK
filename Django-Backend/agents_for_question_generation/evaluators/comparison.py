# OLD-VS-NEW COMPARISON: offline experiments only. Not registered in REGISTRY, so production never runs it.
# Compares the reference output (OLD) with the new output (NEW) for the same input.
#
# Flow for one experiment example (the numbers continue the exp-step numbering in run_experiment.py):
#   exp-step-8  rubric score of the NEW output       (existing quality/relevancy judge, no new prompt)
#   exp-step-9  rubric score of the OLD output       (same judge)
#   exp-step-10 one LLM call that picks the winner   (new comparison prompt)
#   exp-step-11 winner -> score, label and metadata
import json
from typing import Any, Dict, Optional

from utils.constants import LOCALE_MAPPING

from ..agents.langgraph_pipeline import (
    DEFAULT_QUALITY_RUBRIC,
    _call_llm,
    _learning_objective_descriptions,
    _log_with_container,
    _parse_json,
)
from ..agents.pipeline_prompts import (
    SYSTEM_PROMPT_TEMPLATE_OLD_VS_NEW_COMPARISON,
    USER_PROMPT_TEMPLATE_OLD_VS_NEW_COMPARISON,
)
from . import EvalResult
from .builtin import QualityRelevancyEvaluator

WINNER_SCORES = {"new": 1.0, "tie": 0.5, "old": 0.0}


def parse_verdict(content: str) -> Dict[str, str]:
    """Return {"winner", "reason"}. The winner is 'invalid' when the judge reply is unusable."""
    try:
        parsed = _parse_json(content)
    except Exception:
        return {"winner": "invalid", "reason": "judge reply was not valid JSON"}
    if not isinstance(parsed, dict):
        return {"winner": "invalid", "reason": "judge reply was not a JSON object"}
    winner = str(parsed.get("winner", "")).strip().lower()
    if winner not in WINNER_SCORES:
        return {"winner": "invalid", "reason": f"unexpected winner value: {winner!r}"}
    return {"winner": winner, "reason": str(parsed.get("reason", "")).strip()}


def _ask_judge(reference: Any, output: Any, payload: Dict[str, Any], pipeline_config, container_id: Optional[str]) -> Dict[str, str]:
    """exp-step-10: one LLM call that reads both outputs and picks the winner."""
    prompt = USER_PROMPT_TEMPLATE_OLD_VS_NEW_COMPARISON.format(
        rubric=payload.get("qualityRubric") or DEFAULT_QUALITY_RUBRIC.strip(),
        learning_objectives=json.dumps(_learning_objective_descriptions(payload), ensure_ascii=False),
        locale=LOCALE_MAPPING.get(payload.get("locale", "en-US"), "English_US"),
        original_input=json.dumps(payload, ensure_ascii=False),
        old_output=json.dumps(reference, ensure_ascii=False),
        new_output=json.dumps(output, ensure_ascii=False),
    )
    response = _call_llm(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_OLD_VS_NEW_COMPARISON},
            {"role": "user", "content": prompt},
        ],
        task_type="quality_check",
        pipeline_config=pipeline_config,
        call_context="old_vs_new_comparison",
        container_id=container_id,
    )
    return parse_verdict(response.choices[0].message.content or "")


class OldVsNewEvaluator:
    """Scores the new output against the reference output for the same input.

    Needs context["reference"] (the old output). passed=True means the new output is not worse.
    """
    name = "old_vs_new"

    def evaluate(self, output, context) -> EvalResult:
        reference = context.get("reference")
        if not reference:
            return EvalResult(self.name, None, False, "no reference output for this example")
        try:
            quality = QualityRelevancyEvaluator()
            new_rubric = quality.evaluate(output, context)        # exp-step-8
            old_rubric = quality.evaluate(reference, context)     # exp-step-9
            verdict = _ask_judge(reference, output, context["payload"],   # exp-step-10
                                 context.get("pipeline_config"), context.get("container_id"))
        except Exception as exc:
            _log_with_container("error", f"OLD-VS-NEW comparison failed: {exc}", container_id=context.get("container_id"))
            return EvalResult(self.name, None, False, f"comparison error: {exc}")

        winner = verdict["winner"]                                # exp-step-11
        return EvalResult(
            self.name,
            WINNER_SCORES.get(winner),
            winner in ("new", "tie"),
            verdict["reason"],
            metadata={
                "winner": winner,
                "old_rubric": {"score": old_rubric.score, "passed": old_rubric.passed, "reason": old_rubric.reason},
                "new_rubric": {"score": new_rubric.score, "passed": new_rubric.passed, "reason": new_rubric.reason},
            },
        )
