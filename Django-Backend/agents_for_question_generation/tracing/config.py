"""Centralized, env-driven Phoenix tracing setup. Never raises."""
import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_tracer_provider = None
_initialized = False


def tracing_enabled() -> bool:
    return os.getenv("PHOENIX_TRACING_ENABLED", "true").lower() not in ("0", "false", "no")


# ARIZE-PHOENIX-INIT: one-time Phoenix registration + LangChain/OpenAI auto-instrumentation (env-driven, never raises)
def init_tracing():
    """Register Phoenix + OpenInference instrumentors once per process.

    Env: PHOENIX_COLLECTOR_ENDPOINT (default http://localhost:6006),
    PHOENIX_PROJECT_NAME (default agent-question-generation), PHOENIX_TRACING_ENABLED.
    Returns the tracer provider, or None if disabled/failed.
    """
    global _tracer_provider, _initialized
    if _initialized:
        return _tracer_provider
    with _lock:
        if _initialized:
            return _tracer_provider
        _initialized = True  # do not retry on failure
        if not tracing_enabled():
            logger.info("TRACING: disabled via PHOENIX_TRACING_ENABLED")
            return None
        try:
            from phoenix.otel import register
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from openinference.instrumentation.openai import OpenAIInstrumentor

            endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
            _tracer_provider = register(
                project_name=os.getenv("PHOENIX_PROJECT_NAME", "agent-question-generation"),
                endpoint=f"{endpoint}/v1/traces",
                batch=True,
                set_global_tracer_provider=True,
                verbose=False,
            )
            LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
            OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)
            logger.info("TRACING: Phoenix initialized -> %s", endpoint)
        except Exception:
            _tracer_provider = None
            logger.exception("TRACING: initialization failed; continuing without tracing")
        return _tracer_provider
