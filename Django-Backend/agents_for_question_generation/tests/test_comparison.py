from unittest import TestCase
from unittest.mock import MagicMock, patch

from agents_for_question_generation.evaluators import EvalResult
from agents_for_question_generation.evaluators import comparison as cmp
from agents_for_question_generation.management.commands import run_experiment as exp


def _rubric(score, passed, reason=""):
    return EvalResult("quality_relevancy", score, passed, reason)


class ParseVerdictTests(TestCase):
    def test_valid_winners(self):
        for winner in ("old", "new", "tie"):
            verdict = cmp.parse_verdict('{"winner": "%s", "reason": "because"}' % winner)
            self.assertEqual(verdict, {"winner": winner, "reason": "because"})

    def test_winner_is_case_and_space_tolerant(self):
        self.assertEqual(cmp.parse_verdict('{"winner": " NEW ", "reason": "x"}')["winner"], "new")

    def test_unknown_winner_is_invalid(self):
        self.assertEqual(cmp.parse_verdict('{"winner": "both", "reason": "x"}')["winner"], "invalid")

    def test_non_json_reply_is_invalid(self):
        self.assertEqual(cmp.parse_verdict("not json at all")["winner"], "invalid")

    def test_json_that_is_not_an_object_is_invalid(self):
        self.assertEqual(cmp.parse_verdict('["new"]')["winner"], "invalid")


class OldVsNewEvaluatorTests(TestCase):
    def _run(self, verdict, reference=None):
        context = {"payload": {}, "pipeline_config": None, "container_id": None, "reference": reference}
        with patch.object(cmp, "QualityRelevancyEvaluator") as rubric_cls, \
             patch.object(cmp, "_ask_judge", return_value=verdict) as ask:
            rubric_cls.return_value.evaluate.side_effect = [_rubric(90, True, "new ok"), _rubric(70, False, "old weak")]
            result = cmp.OldVsNewEvaluator().evaluate([{"questions": []}], context)
        return result, ask

    def test_new_wins_scores_one_and_passes(self):
        result, _ = self._run({"winner": "new", "reason": "clearer stem"}, reference=[{"questions": []}])
        self.assertEqual((result.score, result.passed, result.reason), (1.0, True, "clearer stem"))
        self.assertEqual(result.metadata["winner"], "new")
        self.assertEqual(result.metadata["new_rubric"]["score"], 90)
        self.assertEqual(result.metadata["old_rubric"]["score"], 70)

    def test_old_wins_scores_zero_and_fails(self):
        result, _ = self._run({"winner": "old", "reason": "new lost a distractor"}, reference=[{"questions": []}])
        self.assertEqual((result.score, result.passed), (0.0, False))

    def test_tie_scores_half_and_passes(self):
        result, _ = self._run({"winner": "tie", "reason": "equal"}, reference=[{"questions": []}])
        self.assertEqual((result.score, result.passed), (0.5, True))

    def test_invalid_verdict_has_no_score_and_fails(self):
        result, _ = self._run({"winner": "invalid", "reason": "judge reply was not valid JSON"}, reference=[{"questions": []}])
        self.assertIsNone(result.score)
        self.assertFalse(result.passed)

    def test_missing_reference_skips_the_judge(self):
        result, ask = self._run({"winner": "new", "reason": "x"}, reference=None)
        self.assertIsNone(result.score)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "no reference output for this example")
        ask.assert_not_called()

    def test_judge_exception_becomes_a_failed_result(self):
        context = {"payload": {}, "pipeline_config": None, "container_id": None, "reference": [{"questions": []}]}
        with patch.object(cmp, "QualityRelevancyEvaluator") as rubric_cls:
            rubric_cls.return_value.evaluate.side_effect = RuntimeError("llm down")
            result = cmp.OldVsNewEvaluator().evaluate([{"questions": []}], context)
        self.assertIsNone(result.score)
        self.assertEqual(result.reason, "comparison error: llm down")


class ExperimentAdapterReferenceTests(TestCase):
    def setUp(self):
        self.patch = patch.object(exp, "_context_for", side_effect=lambda payload: {"payload": payload})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_reference_is_passed_as_context_and_metadata_is_reported(self):
        seen = {}

        class Recorder:
            name = "recorder"

            def evaluate(self, output, context):
                seen["reference"] = context.get("reference")
                return EvalResult("recorder", 1.0, True, "ok", metadata={"winner": "new"})

        fn = exp.to_experiment_evaluator(Recorder())
        out = fn([], {"task_id": "t"}, expected='[{"questions": []}]')
        self.assertEqual(seen["reference"], [{"questions": []}])
        self.assertEqual(out["metadata"], {"winner": "new"})

    def test_missing_expected_gives_none_reference(self):
        seen = {}

        class Recorder:
            name = "recorder"

            def evaluate(self, output, context):
                seen["reference"] = context.get("reference")
                return EvalResult("recorder", None, False, "")

        exp.to_experiment_evaluator(Recorder())([], {})
        self.assertIsNone(seen["reference"])


class CoerceOutputTests(TestCase):
    def test_json_string_is_parsed(self):
        self.assertEqual(exp.coerce_output('[{"a": 1}]'), [{"a": 1}])

    def test_blank_string_is_none(self):
        self.assertIsNone(exp.coerce_output("   "))

    def test_already_parsed_value_is_unchanged(self):
        value = [{"a": 1}]
        self.assertIs(exp.coerce_output(value), value)
