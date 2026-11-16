"""Span helpers. Tracing errors are swallowed; business exceptions always propagate."""
import functools
import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

TRACER_NAME = "agent-question-generation"
MAX_PAYLOAD_CHARS = 4000
_SENSITIVE = ("api_key", "apikey", "authorization", "secret", "password", "token_value")


# ARIZE-PHOENIX-SPAN: returns the OTel tracer; spans go to Phoenix via the provider set in tracing/config.py
def _tracer():
    return trace.get_tracer(TRACER_NAME)


# ARIZE-PHOENIX-SENSITIVE-DATA: masks api keys/passwords before anything is put on a span
def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("***" if any(s in str(k).lower() for s in _SENSITIVE) else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def dump(obj: Any, limit: int = MAX_PAYLOAD_CHARS) -> str:
    try:
        text = json.dumps(_scrub(obj), default=str, ensure_ascii=False)
    except Exception:
        text = str(obj)
    return text if len(text) <= limit else text[:limit] + f"...[truncated {len(text) - limit} chars]"


# ARIZE-PHOENIX-ATTRIBUTES: single place where metadata (scores, thresholds, ids...) is written to a span
def _set_attrs(span, attrs: Dict[str, Any]) -> None:
    for k, v in attrs.items():
        if v is None:
            continue
        try:
            span.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else dump(v))
        except Exception:
            pass


# ARIZE-PHOENIX-EVALUATOR-SPAN: emits one finished EVALUATOR child span (per validation check / evaluator result)
def emit_span(name: str, attrs: Dict[str, Any], start_ns: Optional[int] = None, end_ns: Optional[int] = None,
              kind: str = "EVALUATOR", failed: bool = False) -> None:
    """Create a finished child span of the current span (e.g. one per check/evaluator)."""
    try:
        span = _tracer().start_span(name, start_time=start_ns)
        _set_attrs(span, {"openinference.span.kind": kind, **attrs})
        if failed:
            span.set_status(Status(StatusCode.ERROR))
        span.end(end_time=end_ns)
    except Exception:
        logger.debug("TRACING: emit_span failed", exc_info=True)


# ARIZE-PHOENIX-PIPELINE-SPAN: ROOT span for one full agent run (complete input, final output, status, latency)
@contextmanager
def pipeline_span(payload: Dict[str, Any], model_config: Optional[Dict[str, Any]] = None):
    """Root span for one agent execution. Yields a `finish(output)` callable."""
    started = time.monotonic()
    state = {"output": None}
    try:
        cm = _tracer().start_as_current_span("question_generation_pipeline")
        span = cm.__enter__()
        _set_attrs(span, {
            "openinference.span.kind": "CHAIN",
            "operation_type": "pipeline",
            "node_name": "run_pipeline",
            "input.value": dump(payload, 20000),
            "input.mime_type": "application/json",
            "job_id": payload.get("task_id"),
            "assessment_id": payload.get("assessment_container_id"),
            "question_types": payload.get("questionTypes"),
            "model_config": model_config,
        })
    except Exception:
        logger.debug("TRACING: pipeline_span start failed", exc_info=True)
        cm = span = None

    def finish(output: Any) -> None:
        state["output"] = output

    try:
        yield finish
    except BaseException as exc:
        _close_pipeline(cm, span, started, state["output"], exc)
        raise
    else:
        _close_pipeline(cm, span, started, state["output"], None)


def _close_pipeline(cm, span, started, output, exc) -> None:
    if cm is None:
        return
    try:
        _set_attrs(span, {
            "output.value": dump(output, 20000) if output is not None else None,
            "output.mime_type": "application/json",
            "execution_status": "failure" if exc else "success",
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "error.message": str(exc) if exc else None,
        })
        if exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
        else:
            span.set_status(Status(StatusCode.OK))
        cm.__exit__(None, None, None)
    except Exception:
        logger.debug("TRACING: pipeline_span close failed", exc_info=True)


# ARIZE-PHOENIX-NODE-SPAN: wraps each LangGraph node (generation / validation / regeneration) in its own span.
# What gets recorded per node is declared in tracing/node_specs.py
def traced_node(name: str, spec) -> Callable:
    """Decorator factory: wraps a LangGraph node fn(state) in a span using a NodeSpec."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state):
            try:
                cm = _tracer().start_as_current_span(name)
                span = cm.__enter__()
                start_ns = time.time_ns()
                t0 = time.monotonic()
                pre = spec.pre(state) if spec.pre else {}
                _set_attrs(span, {
                    "openinference.span.kind": "CHAIN",
                    "node_name": name,
                    "operation_type": spec.operation_type,
                    "check_name": spec.check_name,
                    "input.value": dump({k: state.get(k) for k in spec.in_keys}) if spec.in_keys else None,
                })
            except Exception:
                logger.debug("TRACING: node span start failed", exc_info=True)
                return fn(state)

            exc = None
            out = None
            try:
                out = fn(state)
                return out
            except BaseException as e:
                exc = e
                raise
            finally:
                try:
                    _finish_node(span, spec, name, state, out, pre, start_ns, t0, exc)
                    cm.__exit__(None, None, None)
                except Exception:
                    logger.debug("TRACING: node span finish failed", exc_info=True)
        return wrapper
    return decorator


# ARIZE-PHOENIX-NODE-SPAN: after the node runs, attach model, attempt, score/threshold/status from real results
def _finish_node(span, spec, name, state, out, pre, start_ns, t0, exc) -> None:
    end_ns = time.time_ns()
    result_state = out if isinstance(out, dict) else state
    attrs: Dict[str, Any] = {"duration_ms": round((time.monotonic() - t0) * 1000, 1)}
    try:
        cfg = state.get("pipeline_config")
        if cfg is not None:
            attrs["model_name"] = cfg.get_client_and_deployment(spec.task_type)[1]
    except Exception:
        pass
    attrs["job_id"] = state.get("task_id")
    attrs["assessment_id"] = (state.get("input") or {}).get("assessment_container_id")
    if spec.attempt:
        attrs["attempt_number"] = spec.attempt(state, pre)
    if spec.out_keys:
        attrs["output.value"] = dump({k: result_state.get(k) for k in spec.out_keys})
    children = []
    if spec.extract and exc is None:
        try:
            extra, children = spec.extract(state, out, pre)
            attrs.update(extra or {})
        except Exception:
            logger.debug("TRACING: extractor failed for %s", name, exc_info=True)
    if exc:
        attrs["execution_status"] = "failure"
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    _set_attrs(span, attrs)
    for child in children:  # per-check child spans, timed with the parent call
        emit_span(child["name"], {**child["attrs"], "node_name": name,
                                  "attempt_number": attrs.get("attempt_number"),
                                  "model_name": attrs.get("model_name"),
                                  "job_id": attrs.get("job_id")},
                  start_ns=start_ns, end_ns=end_ns, kind="EVALUATOR")
