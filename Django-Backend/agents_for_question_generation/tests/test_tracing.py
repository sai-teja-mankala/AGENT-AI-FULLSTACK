from unittest import TestCase
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agents_for_question_generation.evaluators import EvalResult, run_evaluators
from agents_for_question_generation.evaluators.builtin import StructureEvaluator
from agents_for_question_generation.tracing import config as tcfg
from agents_for_question_generation.tracing.node_specs import NODE_SPECS
from agents_for_question_generation.tracing.spans import dump, pipeline_span, traced_node

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


def _spans():
    return {s.name: s for s in _exporter.get_finished_spans()}


class TracingTests(TestCase):
    def setUp(self):
        _exporter.clear()

    def test_distractor_check_span_has_score_threshold_status(self):
        out = {"validation_check_plausibility": [{"payload": {}, "checks": {
            0: {"score": 70, "pass": False, "feedback": "weak"}, 1: {"score": 90, "pass": True}}}]}
        fn = traced_node("agent_check_plausibility", NODE_SPECS["agent_check_plausibility"])(lambda s: out)
        self.assertIs(fn({"task_id": "j1", "input": {}}), out)
        a = _spans()["agent_check_plausibility"].attributes
        self.assertEqual(a["operation_type"], "validation")
        self.assertEqual(a["check_name"], "plausibility_check")
        self.assertEqual(a["actual_score"], 70)
        self.assertEqual(a["threshold"], 85)
        self.assertEqual(a["validation_status"], "fail")
        self.assertEqual(a["job_id"], "j1")

    def test_quality_node_emits_separate_quality_and_relevancy_spans(self):
        def node(state):
            state["quality_validation_passed"] = [{"quality": {"score": 90, "pass": True},
                                                   "relevancy": {"score": 60, "pass": False}}]
            state["quality_validation_failed"] = []
            return state
        fn = traced_node("validate_quality", NODE_SPECS["validate_quality"])(node)
        fn({"input": {"qualityThreshold": 85, "relevancyThreshold": 80}})
        s = _spans()
        self.assertEqual(s["validation.quality_check"].attributes["actual_score"], 90)
        self.assertEqual(s["validation.relevancy_check"].attributes["validation_status"], "fail")
        self.assertEqual(s["validation.relevancy_check"].attributes["threshold"], 80)
        self.assertEqual(s["validation.relevancy_check"].parent.span_id, s["validate_quality"].context.span_id)

    def test_regeneration_span_records_trigger_and_attempt(self):
        def node(state):
            state["quality_correction_attempts"] = 1
            return state
        state = {"input": {}, "quality_validation_failed": [
            {"quality": {"score": 50, "pass": False}, "relevancy": {"score": 95, "pass": True}}]}
        traced_node("correct_quality", NODE_SPECS["correct_quality"])(node)(state)
        a = _spans()["correct_quality"].attributes
        self.assertEqual(a["operation_type"], "regeneration")
        self.assertEqual(a["regeneration_trigger"], "quality")
        self.assertEqual(a["attempt_number"], 1)

    def test_node_exception_propagates_and_is_recorded(self):
        def boom(state):
            raise ValueError("x")
        with self.assertRaises(ValueError):
            traced_node("build_question", NODE_SPECS["build_question"])(boom)({})
        self.assertEqual(_spans()["build_question"].attributes["execution_status"], "failure")

    def test_broken_extractor_does_not_break_node(self):
        from agents_for_question_generation.tracing.node_specs import NodeSpec
        spec = NodeSpec("validation", extract=lambda *a: 1 / 0)
        self.assertEqual(traced_node("n", spec)(lambda s: "ok")({}), "ok")

    def test_pipeline_span_captures_input_output_status(self):
        with pipeline_span({"task_id": "t1", "assessment_container_id": "a1"}, {"judge_model": "m"}) as finish:
            finish([{"questions": []}])
        a = _spans()["question_generation_pipeline"].attributes
        self.assertEqual(a["execution_status"], "success")
        self.assertEqual((a["job_id"], a["assessment_id"]), ("t1", "a1"))
        self.assertIn("questions", a["output.value"])
        self.assertIn("t1", a["input.value"])

    def test_pipeline_span_failure_reraises(self):
        with self.assertRaises(RuntimeError):
            with pipeline_span({}):
                raise RuntimeError("bad")
        self.assertEqual(_spans()["question_generation_pipeline"].attributes["execution_status"], "failure")

    def test_dump_scrubs_secrets_and_truncates(self):
        self.assertNotIn("sk-123", dump({"api_key": "sk-123"}))
        self.assertIn("truncated", dump("x" * 100, limit=10))

    def test_init_tracing_never_raises_and_runs_once(self):
        tcfg._initialized, tcfg._tracer_provider = False, None
        with patch.dict("os.environ", {"PHOENIX_TRACING_ENABLED": "true"}), \
                patch("phoenix.otel.register", side_effect=RuntimeError("down")) as reg:
            self.assertIsNone(tcfg.init_tracing())
            self.assertIsNone(tcfg.init_tracing())
            self.assertEqual(reg.call_count, 1)
        tcfg._initialized = True


class EvaluatorTests(TestCase):
    def setUp(self):
        _exporter.clear()

    def test_structure_pass_and_fail(self):
        good = [{"questions": [{"questionText": "Q?", "answer": [
            {"answerText": "a", "isCorrect": True}, {"answerText": "b", "isCorrect": False}]}]}]
        bad = [{"questions": [{"questionText": "Q?", "answer": [
            {"answerText": "a", "isCorrect": False}, {"answerText": "A", "isCorrect": False}]}]}]
        ev = StructureEvaluator()
        self.assertTrue(ev.evaluate(good, {}).passed)
        r = ev.evaluate(bad, {})
        self.assertFalse(r.passed)
        self.assertIn("duplicate", r.reason)

    def test_runner_isolates_failures_and_emits_spans(self):
        class Boom:
            name = "boom"
            def evaluate(self, o, c): raise RuntimeError("nope")
        class Ok:
            name = "ok"
            def evaluate(self, o, c): return EvalResult("ok", 0.9, True, "fine", threshold=0.8)
        res = run_evaluators([], {"job_id": "j"}, evaluators=[Boom(), Ok()])
        self.assertEqual([r.passed for r in res], [False, True])
        s = _spans()
        self.assertEqual(s["evaluator.ok"].attributes["actual_score"], 0.9)
        self.assertEqual(s["evaluator.boom"].attributes["validation_status"], "fail")

    def test_enabled_via_env(self):
        from agents_for_question_generation.evaluators import get_enabled
        with patch.dict("os.environ", {"EVALUATORS_ENABLED": ""}):
            self.assertEqual(get_enabled(), [])
        with patch.dict("os.environ", {"EVALUATORS_ENABLED": "structure"}):
            self.assertEqual([e.name for e in get_enabled()], ["structure"])
        # Regression: "true" must enable every registered evaluator (not silently none)
        with patch.dict("os.environ", {"EVALUATORS_ENABLED": "true"}):
            self.assertEqual({e.name for e in get_enabled()}, {"structure", "all_checks", "quality_relevancy"})
        # Regression: a malformed value (e.g. "EVALUATORS_ENABLED=structure" pasted into the value) must warn, not pass silently
        with patch.dict("os.environ", {"EVALUATORS_ENABLED": "EVALUATORS_ENABLED=structure,all_checks"}):
            with self.assertLogs("agents_for_question_generation.evaluators.registry", level="WARNING") as logs:
                self.assertEqual([e.name for e in get_enabled()], ["all_checks"])
            self.assertIn("evaluators_enabled=structure", logs.output[0])  # value is lowercased before parsing


# ARIZE-PHOENIX-EVALUATOR tests: judge LLM calls are stubbed with fixed outcomes
from agents_for_question_generation.evaluators import judges  # noqa: E402
from agents_for_question_generation.evaluators.builtin import AllChecksEvaluator  # noqa: E402
from agents_for_question_generation.evaluators.generation import with_generation_evaluators  # noqa: E402


def _stub_tasks(*specs):
    return [lambda n=n, p=p: judges.CheckOutcome(n, 70.0, 80, p, "r", 1, 2) for n, p in specs]


class JudgeEvaluatorTests(TestCase):
    def setUp(self):
        _exporter.clear()

    def test_all_checks_reports_each_check_and_aggregate(self):
        out = [{"LearningObjective": "lo", "LevelOfQuiz": "Beginner", "questions": [
            {"questionText": "Q?", "answer": [{"answerText": "a", "isCorrect": True}]}]}]
        with patch.object(judges, "stem_check_tasks", return_value=_stub_tasks(("stem_clarity_bias", True))), \
                patch.object(judges, "distractor_check_tasks", return_value=_stub_tasks(("clarity_check", False))):
            res = AllChecksEvaluator().evaluate(out, {"payload": {}})
        self.assertFalse(res.passed)
        self.assertEqual(res.score, 0.5)
        self.assertIn("clarity_check", res.reason)
        s = _spans()
        self.assertEqual(s["evaluator.all_checks.clarity_check"].attributes["validation_status"], "fail")
        self.assertEqual(s["evaluator.all_checks.stem_clarity_bias"].attributes["evaluation_stage"], "final_output")

    def test_all_checks_skips_matching_shape(self):
        res = AllChecksEvaluator().evaluate([{"questions": [{"pairs": []}]}], {})
        self.assertFalse(res.passed)
        self.assertEqual(res.metadata["checks"], [])

    def test_generation_evaluator_runs_only_when_enabled_and_keeps_output(self):
        state = {"questions": [{"questionText": "Q?", "payload": {}}], "task_id": "j"}
        node = with_generation_evaluators("build_question", lambda s: s)
        with patch.object(judges, "stem_check_tasks", return_value=_stub_tasks(("stem_clarity_bias", True))) as t:
            with patch.dict("os.environ", {"GENERATION_EVALUATORS_ENABLED": "false"}):
                self.assertIs(node(state), state)
            t.assert_not_called()
            with patch.dict("os.environ", {"GENERATION_EVALUATORS_ENABLED": "true"}):
                self.assertIs(node(state), state)
            t.assert_called_once()
        a = _spans()["generation_eval.stem_clarity_bias"].attributes
        self.assertEqual(a["evaluation_stage"], "generation:build_question")
        self.assertEqual(a["job_id"], "j")

    def test_generation_evaluator_failure_never_breaks_node(self):
        node = with_generation_evaluators("build_options", lambda s: s)
        state = {"raw_outputs": [{"payload": {}, "result": {"questions": [{"x": 1}]}}]}
        with patch.object(judges, "distractor_check_tasks", side_effect=RuntimeError("boom")), \
                patch.dict("os.environ", {"GENERATION_EVALUATORS_ENABLED": "true"}):
            self.assertIs(node(state), state)

    def test_non_generation_node_not_wrapped(self):
        fn = lambda s: s
        self.assertIs(with_generation_evaluators("validate_quality", fn), fn)
