import os
from typing import Dict, List

REGISTRY: Dict[str, object] = {}


# ARIZE-PHOENIX-EVALUATOR-REGISTRY: add new evaluators here via register(...)
def register(evaluator) -> None:
    REGISTRY[evaluator.name] = evaluator


# ARIZE-PHOENIX-EVALUATOR-CONFIG: which evaluators run is controlled by env EVALUATORS_ENABLED (default: none)
def get_enabled() -> List[object]:
    """Evaluators named in EVALUATORS_ENABLED (comma list, or 'all'); default none."""
    from . import builtin  # noqa: F401  (registers built-ins)
    raw = os.getenv("EVALUATORS_ENABLED", "").strip().lower()
    if not raw:
        return []
    names = list(REGISTRY) if raw == "all" else [n.strip() for n in raw.split(",") if n.strip()]
    return [REGISTRY[n] for n in names if n in REGISTRY]
