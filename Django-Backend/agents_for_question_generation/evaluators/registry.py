import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

REGISTRY: Dict[str, object] = {}


# ARIZE-PHOENIX-EVALUATOR-REGISTRY: add new evaluators here via register(...)
def register(evaluator) -> None:
    REGISTRY[evaluator.name] = evaluator


# ARIZE-PHOENIX-EVALUATOR-CONFIG: which evaluators run is controlled by env EVALUATORS_ENABLED (default: none)
# Truthy/"all" values (true, 1, yes, on, all) enable EVERY registered evaluator (structure, all_checks, quality_relevancy)
_ENABLE_ALL_VALUES = {"all", "true", "1", "yes", "on"}


def get_enabled() -> List[object]:
    """Evaluators named in EVALUATORS_ENABLED (comma list, or all/true/1/yes/on for every one); default none."""
    from . import builtin  # noqa: F401  (registers built-ins)
    raw = os.getenv("EVALUATORS_ENABLED", "").strip().lower()
    if not raw:
        return []
    if raw in _ENABLE_ALL_VALUES:
        return list(REGISTRY.values())
    names = [n.strip() for n in raw.split(",") if n.strip()]
    # ARIZE-PHOENIX-EVALUATOR-CONFIG: unknown names used to be dropped silently, so a typo (e.g. "EVALUATORS_ENABLED=structure"
    # pasted into the value) meant no evaluator ran with no error. Warn so the misconfiguration is visible in the logs.
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        logger.warning("EVAL: unknown evaluator name(s) %s in EVALUATORS_ENABLED; valid names: %s",
                       unknown, sorted(REGISTRY))
    return [REGISTRY[n] for n in names if n in REGISTRY]
