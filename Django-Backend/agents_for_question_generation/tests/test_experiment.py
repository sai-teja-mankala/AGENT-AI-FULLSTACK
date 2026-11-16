import os
from unittest import TestCase
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError

from agents_for_question_generation.evaluators import EvalResult
from agents_for_question_generation.management.commands import run_experiment as exp


class _StubEvaluator:
    name = "stub"

    def __init__(self, result):
        self._result = result

    def evaluate(self, output, context):
        return self._result


class CoercePayloadTests(TestCase):
    def test_json_string_cells_become_lists_and_ints(self):
        raw = {
            "learningObjectives": '[{"uuid": "lo-1", "text": "Explain X"}]',
            "questionTypes": '["MULTIPLE_CHOICE"]',
            "difficultyLevels": '["Beginner"]',
            "numberOfQuestions": "2",
            "locale": "en-US",
        }
        payload = exp.coerce_payload(raw)
        self.assertEqual(payload["learningObjectives"], [{"uuid": "lo-1", "text": "Explain X"}])
        self.assertEqual(payload["questionTypes"], ["MULTIPLE_CHOICE"])
        self.assertEqual(payload["difficultyLevels"], ["Beginner"])
        self.assertEqual(payload["numberOfQuestions"], 2)
        self.assertEqual(payload["locale"], "en-US")

    def test_already_typed_values_are_left_alone(self):
        raw = {"learningObjectives": [{"uuid": "lo-1"}], "numberOfQuestions": 3}
        self.assertEqual(exp.coerce_payload(raw), raw)

    def test_empty_list_cell_becomes_empty_list(self):
        self.assertEqual(exp.coerce_payload({"questionTypes": "  "})["questionTypes"], [])


class ExperimentEvaluatorAdapterTests(TestCase):
    def setUp(self):
        self.context = {"payload": {}, "pipeline_config": None}
        self.patch = patch.object(exp, "_context_for", return_value=self.context)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_maps_eval_result_to_phoenix_dict(self):
        ev = _StubEvaluator(EvalResult("stub", 0.9, True, "fine", threshold=0.8))
        fn = exp.to_experiment_evaluator(ev)
        self.assertEqual(fn.__name__, "stub")
        self.assertEqual(fn([], {"task_id": "t"}), {"score": 0.9, "label": "pass", "explanation": "fine"})

    def test_failed_result_and_missing_score(self):
        ev = _StubEvaluator(EvalResult("stub", None, False, "broke"))
        out = exp.to_experiment_evaluator(ev)([], {})
        self.assertEqual(out, {"label": "fail", "explanation": "broke"})  # None score is dropped, not sent as null


class FinalStatusTests(TestCase):
    def test_completed_when_questions_delivered(self):
        output = [{"questions": [{"questionText": "q"}], "pipeline_stats": {"pipeline_errors": []}}]
        self.assertEqual(exp.final_status(output, {})["label"], "COMPLETED")

    def test_failed_with_pipeline_errors_as_reason(self):
        output = [{"questions": [], "pipeline_stats": {"pipeline_errors": ["retry budget exhausted", "no sources"]}}]
        result = exp.final_status(output, {})
        self.assertEqual(result["label"], "FAILED")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["explanation"], "retry budget exhausted; no sources")

    def test_failed_without_errors_gives_a_reason(self):
        result = exp.final_status([], {})
        self.assertEqual(result["label"], "FAILED")
        self.assertEqual(result["explanation"], "no questions delivered")


class RunExperimentCommandTests(TestCase):
    def test_missing_dataset_name_fails_before_contacting_phoenix(self):
        with patch.dict(os.environ, {"PHOENIX_GOLDEN_DATASET_NAME": ""}):
            with self.assertRaises(CommandError):
                call_command("run_experiment", "--pipeline-version", "rag-v1")

    def test_empty_version_is_rejected(self):
        with patch.dict(os.environ, {"PHOENIX_GOLDEN_DATASET_NAME": "first-data-set"}):
            with self.assertRaises(CommandError):
                call_command("run_experiment", "--pipeline-version", "   ")
