"""Declarative per-node tracing metadata (kept out of the agent code).

Every value is read from real node results/state; nothing is fabricated.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..agents.pipeline_constants import CHECK_THRESHOLDS

GEN_TASK, VAL_TASK = "generation", "validation"


# ARIZE-PHOENIX-NODE-SPEC: declares what each node's span records (operation type, in/out state keys, attempt, score extractor)
@dataclass
class NodeSpec:
    operation_type: str                       # generation | validation | regeneration | routing
    task_type: str = VAL_TASK                 # selects generator vs judge model name
    check_name: Optional[str] = None
    in_keys: Tuple[str, ...] = ()
    out_keys: Tuple[str, ...] = ()
    attempt: Optional[Callable] = None        # (state, pre) -> int
    pre: Optional[Callable] = None            # state -> dict captured BEFORE the node runs
    extract: Optional[Callable] = None        # (state, out, pre) -> (attrs, children)


def _counter(key: str, plus: int = 0) -> Callable:
    return lambda state, pre: (state.get(key, 0) or 0) + plus


def _status(passed: Optional[bool]) -> Optional[str]:
    return None if passed is None else ("pass" if passed else "fail")


# ARIZE-PHOENIX-VALIDATION-SPAN: 7 parallel distractor checks, one node == one check == one span
# (actual_score = lowest score in batch, threshold = CHECK_THRESHOLDS, validation_status)
def _distractor_check(check: str, key: str) -> NodeSpec:
    def extract(state, out, pre):
        scores, passes, feedback = [], [], []
        for item in (out or {}).get(key, []):
            for c in (item.get("checks") or {}).values():
                if isinstance(c.get("score"), (int, float)):
                    scores.append(c["score"])
                passes.append(bool(c.get("pass")))
                if not c.get("pass") and c.get("feedback"):
                    feedback.append(c["feedback"])
        attrs = {
            "threshold": CHECK_THRESHOLDS.get(check),
            "validation_status": _status(all(passes) if passes else None),
            "question_scores": scores or None,
            "validation_feedback": feedback[0] if feedback else None,
        }
        if scores:
            attrs["actual_score"] = min(scores)  # lowest score across questions in this batch
            attrs["score_aggregation"] = "min"
        return attrs, []

    return NodeSpec("validation", VAL_TASK, f"{check}_check", out_keys=(key,),
                    attempt=_counter("distractor_correction_attempt", 1), extract=extract)


# ARIZE-PHOENIX-VALIDATION-SPAN: stem gates (clarity/bias, LO/cognitive) - verdict based, no numeric score exists
def _stem_gate(result_key: str, failed_key: str, counter: str) -> NodeSpec:
    def extract(state, out, pre):
        failed = state.get(failed_key) or []
        verdicts = [e.get(result_key, {}).get("verdict") for e in (state.get("questions") or []) if e.get(result_key)]
        return {"validation_status": _status(not failed), "failed_count": len(failed),
                "verdicts": verdicts or None}, []

    return NodeSpec("validation", VAL_TASK, result_key.replace("_validation", ""),
                    attempt=_counter(counter, 1), out_keys=(failed_key,), extract=extract)


# ARIZE-PHOENIX-REGENERATION-SPAN: stem regeneration (trigger = which stem gate failed)
def _stem_regen(failed_key: str, result_key: str, trigger: str, counter: str) -> NodeSpec:
    def pre(state):
        verdicts = [e.get(result_key, {}).get("verdict") for e in (state.get(failed_key) or []) if e.get(result_key)]
        return {"verdicts": verdicts}

    def extract(state, out, p):
        return {"regeneration_trigger": trigger, "trigger_verdicts": p.get("verdicts") or None}, []

    return NodeSpec("regeneration", GEN_TASK, trigger, in_keys=(failed_key,), out_keys=("questions",),
                    attempt=_counter(counter), pre=pre, extract=extract)


# ARIZE-PHOENIX-VALIDATION-SPAN: quality + relevancy = one LLM call, emitted as two distinct child check spans
def _quality_extract_factory(passed_key: str, failed_key: str, q_default: int = 85):
    def extract(state, out, pre):
        entries = (state.get(passed_key) or [])[pre["passed_len"]:] + (state.get(failed_key) or [])
        inp = state.get("input") or {}
        children, attrs = [], {}
        for label, thr in (("quality", inp.get("qualityThreshold", q_default)),
                           ("relevancy", inp.get("relevancyThreshold", q_default))):
            results = [e.get(label) for e in entries if isinstance(e.get(label), dict)]
            if not results:
                continue
            scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
            passed = all(bool(r.get("pass")) for r in results)
            issues = [i for r in results if not r.get("pass") for i in (r.get("issues") or [])]
            child = {"check_name": f"{label}_check", "threshold": thr, "validation_status": _status(passed),
                     "operation_type": "validation", "question_scores": scores or None,
                     "validation_feedback": issues[:3] or None}
            if scores:
                child["actual_score"] = min(scores)
                child["score_aggregation"] = "min"
            children.append({"name": f"validation.{label}_check", "attrs": child})
            attrs[f"{label}_status"] = child["validation_status"]
        attrs["validation_status"] = _status(all(c["attrs"]["validation_status"] == "pass" for c in children)
                                             if children else None)
        return attrs, children

    return extract


def _quality_pre(passed_key: str):
    return lambda state: {"passed_len": len(state.get(passed_key) or [])}


# ARIZE-PHOENIX-REGENERATION-SPAN: quality correction - captures failing scores/thresholds BEFORE the node rewrites them
def _correct_quality_pre(state):
    inp = state.get("input") or {}
    rows = []
    for e in state.get("quality_validation_failed") or []:
        rows.append({"quality": (e.get("quality") or {}).get("score"),
                     "relevancy": (e.get("relevancy") or {}).get("score"),
                     "failed": [k for k in ("quality", "relevancy") if not (e.get(k) or {}).get("pass")]})
    return {"rows": rows, "q_thr": inp.get("qualityThreshold", 85), "r_thr": inp.get("relevancyThreshold", 85)}


def _correct_quality_extract(state, out, pre):
    rows = pre.get("rows") or []
    failed = sorted({k for r in rows for k in r["failed"]})
    return {"regeneration_trigger": "+".join(failed) or "quality_relevancy",
            "actual_score": min([r["quality"] for r in rows if isinstance(r["quality"], (int, float))] or [None]) if rows else None,
            "trigger_scores": [{"quality": r["quality"], "relevancy": r["relevancy"]} for r in rows] or None,
            "threshold": pre.get("q_thr"), "relevancy_threshold": pre.get("r_thr")}, []


# ARIZE-PHOENIX-REGENERATION-SPAN: distractor correction - records which checks triggered it and their scores
def _correct_distractors_pre(state):
    failed = {}
    for item in state.get("validated_outputs") or []:
        for q in (item.get("result") or {}).get("questions", []):
            dv = q.get("distractor_validation") if isinstance(q, dict) else None
            if dv and dv.get("correction_action") == "CORRECT":
                for name in dv.get("failed_checks", []):
                    failed.setdefault(name if isinstance(name, str) else str(name), []).append(
                        (dv.get("check_scores") or {}).get(name))
    return {"failed": failed}


def _correct_distractors_extract(state, out, pre):
    failed = pre.get("failed") or {}
    return {"regeneration_trigger": ",".join(failed) or "distractor_validation",
            "trigger_scores": failed or None,
            "trigger_thresholds": {k: CHECK_THRESHOLDS.get(k.replace("_check", "")) for k in failed} or None}, []


# ARIZE-PHOENIX-NODE-REGISTRY: graph node name -> NodeSpec. graph_builder.py wraps every node listed here.
# To trace a new node, add one entry here (no change to the node itself).
NODE_SPECS: Dict[str, NodeSpec] = {
    "build_question": NodeSpec("generation", GEN_TASK, out_keys=("questions",), attempt=lambda s, p: 1),
    "build_options": NodeSpec("generation", GEN_TASK, in_keys=("questions",), out_keys=("raw_outputs",),
                              attempt=lambda s, p: 1),
    "validate_stem_clarity_bias": _stem_gate("stem_clarity_bias_validation", "stem_validation_failed",
                                             "stem_validation_attempt"),
    "validate_stem_lo_cognitive": _stem_gate("stem_lo_cognitive_validation", "stem_lo_cog_validation_failed",
                                             "stem_lo_cog_validation_attempt"),
    "regenerate_stem_clarity_bias": _stem_regen("stem_validation_failed", "stem_clarity_bias_validation",
                                                "stem_clarity_bias", "stem_validation_attempt"),
    "regenerate_stem_lo_cognitive": _stem_regen("stem_lo_cog_validation_failed", "stem_lo_cognitive_validation",
                                                "stem_lo_cognitive", "stem_lo_cog_validation_attempt"),
    "agent_check_plausibility": _distractor_check("plausibility", "validation_check_plausibility"),
    "agent_check_misconception": _distractor_check("misconception", "validation_check_misconception"),
    "agent_check_distinctiveness": _distractor_check("distinctiveness", "validation_check_distinctiveness"),
    "agent_check_incorrectness": _distractor_check("incorrectness", "validation_check_incorrectness"),
    "agent_check_clarity": _distractor_check("clarity", "validation_check_clarity"),
    "agent_check_parallelism": _distractor_check("parallelism", "validation_check_parallelism"),
    "agent_check_locale_compliance": _distractor_check("locale_compliance", "validation_check_locale_compliance"),
    "correct_failed_distractors": NodeSpec("regeneration", GEN_TASK, in_keys=("distractor_corrections_output",),
                                           out_keys=("distractor_corrections_output",),
                                           attempt=_counter("distractor_correction_attempt"),
                                           pre=_correct_distractors_pre, extract=_correct_distractors_extract),
    "validate_quality": NodeSpec("validation", VAL_TASK, "quality_relevancy_check",
                                 attempt=_counter("quality_correction_attempts", 1),
                                 pre=_quality_pre("quality_validation_passed"),
                                 extract=_quality_extract_factory("quality_validation_passed",
                                                                  "quality_validation_failed")),
    "correct_quality": NodeSpec("regeneration", GEN_TASK, "quality_relevancy_check",
                                attempt=_counter("quality_correction_attempts"),
                                pre=_correct_quality_pre, extract=_correct_quality_extract),
    "build_question_matching": NodeSpec("generation", GEN_TASK, out_keys=("matching_questions",),
                                        attempt=lambda s, p: 1),
    "build_options_matching": NodeSpec("generation", GEN_TASK, out_keys=("matching_raw_outputs",),
                                       attempt=lambda s, p: 1),
    "validate_quality_matching": NodeSpec("validation", VAL_TASK, "matching_quality_check",
                                          attempt=_counter("matching_quality_correction_attempts", 1),
                                          pre=_quality_pre("matching_quality_passed"),
                                          extract=_quality_extract_factory("matching_quality_passed",
                                                                           "matching_quality_failed")),
    "correct_quality_matching": NodeSpec("regeneration", GEN_TASK, "matching_quality_check",
                                         out_keys=("matching_quality_failed",),
                                         attempt=_counter("matching_quality_correction_attempts")),
}
