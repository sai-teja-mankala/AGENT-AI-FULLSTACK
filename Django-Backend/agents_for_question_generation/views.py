import json
import logging
import time

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .agents.langgraph_pipeline import run_pipeline

logger = logging.getLogger(__name__)


@require_GET
def health(request):
    return JsonResponse({"status": "ok"})


@csrf_exempt
@require_POST
def generate(request):
    logger.info("GENERATE request received (%d bytes)", len(request.body))
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        logger.warning("GENERATE rejected: body is not valid JSON")
        return JsonResponse({"error": "Body must be valid JSON"}, status=400)
    if not payload.get("learningObjectives"):
        logger.warning("GENERATE rejected: learningObjectives missing")
        return JsonResponse({"error": "learningObjectives is required"}, status=400)

    logger.info(
        "GENERATE start | LOs=%d | types=%s | difficulties=%s | questions per LO=%s",
        len(payload["learningObjectives"]),
        payload.get("questionTypes"),
        payload.get("difficultyLevels"),
        payload.get("numberOfQuestions"),
    )
    started = time.monotonic()
    try:
        result = run_pipeline(payload)
    except Exception as exc:
        logger.exception("GENERATE failed after %.1fs", time.monotonic() - started)
        return JsonResponse({"error": str(exc)}, status=500)

    total = sum(len(lo.get("questions", [])) for lo in result)
    logger.info("GENERATE done in %.1fs | questions returned=%d", time.monotonic() - started, total)
    return JsonResponse({"result": result}, safe=False)
