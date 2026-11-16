"""
Utility functions for validation and JSON parsing with improved error handling.

Provides standardized validation response parsing for distractor validation checks
and quality checks with comprehensive logging and fallback mechanisms.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_and_parse_validation_json(
    content: str, 
    check_name: str, 
    expected_format_type: str = "validation",
    threshold: int = 80,
    default_response: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM validation responses with comprehensive error handling.
    
    Args:
        content: Raw LLM response content
        check_name: Name of the validation check (for logging)
        expected_format_type: "validation" or "quality" to determine expected format
        threshold: Score threshold for pass/fail determination
        default_response: Fallback response on parsing failure
        
    Returns:
        Dict containing validated response with score, pass, and feedback/issues
    """
    try:
        # Extract JSON from potential markdown code blocks
        cleaned_content = _extract_json_from_response(content)
        
        # Parse the cleaned JSON
        parsed = json.loads(cleaned_content)
        
        # Validate and standardize the response format
        return _standardize_validation_response(
            parsed, check_name, expected_format_type, threshold
        )
        
    except json.JSONDecodeError as e:
        logger.warning(
            f"{check_name} JSON parsing failed. "
            f"Original: {content[:200]}... "
            f"Cleaned: {_extract_json_from_response(content)[:200]}... "
            f"Error: {str(e)}"
        )
        return _get_default_response(check_name, expected_format_type, default_response)
        
    except Exception as e:
        logger.error(f"Unexpected error parsing {check_name}: {str(e)}")
        return _get_default_response(check_name, expected_format_type, default_response)


def _extract_json_from_response(content: str) -> str:
    """
    Extract JSON content from LLM response, handling various formats.
    
    Handles:
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - json\n{...}
    - raw JSON
    """
    stripped = content.strip()
    
    # Handle markdown code blocks
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        
        stripped = stripped.strip()
    
    # Handle "json" prefix that some models add
    if stripped.startswith("json"):
        stripped = stripped[4:].strip()
    
    return stripped


def _standardize_validation_response(
    parsed: Any, 
    check_name: str, 
    expected_format_type: str, 
    threshold: int
) -> Dict[str, Any]:
    """
    Standardize validation response format and ensure required fields.
    """
    if not isinstance(parsed, dict):
        logger.warning(f"{check_name} returned non-dict response: {type(parsed)}")
        return _get_default_response(check_name, expected_format_type)
    
    if expected_format_type == "validation":
        # Distractor validation format: {"score": 85, "pass": true, "feedback": "..."}
        score = parsed.get("score", 0)
        feedback = parsed.get("feedback", f"No feedback provided for {check_name}")
        
        # Always calculate pass based on score vs threshold
        passed = score >= threshold
        
        return {
            "score": score,
            "pass": passed,
            "feedback": feedback
        }
        
    elif expected_format_type == "quality":
        # Quality check format: {"score": 85, "pass": true, "issues": [...]}
        score = parsed.get("score", 0)
        issues = parsed.get("issues", [f"No issues provided for {check_name}"])
        
        # Ensure issues is a list
        if not isinstance(issues, list):
            issues = [str(issues)]
        
        passed = score >= threshold
        
        return {
            "score": score,
            "pass": passed,
            "issues": issues
        }

    elif expected_format_type == "boolean_quality":
        # Boolean quality check format: {"pass": true|false, "issues": [...]}
        # No numeric score/threshold - the LLM judges pass/fail directly
        # against the rubric's checklist criteria (used by MATCHING quality
        # validation instead of the scored "quality" format above).
        passed = bool(parsed.get("pass", False))
        issues = parsed.get("issues", [f"No issues provided for {check_name}"])

        if not isinstance(issues, list):
            issues = [str(issues)]

        return {
            "pass": passed,
            "issues": issues
        }

    elif expected_format_type == "stem_validation":
        return _standardize_stem_validation_response(parsed, check_name)

    elif expected_format_type == "stem_dimension_validation":
        return _standardize_stem_dimension_validation_response(parsed, check_name)

    else:
        logger.error(f"Unknown expected_format_type: {expected_format_type}")
        return _get_default_response(check_name, expected_format_type)


def _standardize_stem_validation_response(parsed: Dict[str, Any], check_name: str) -> Dict[str, Any]:
    """
    Standardize stem clarity/bias validation response.

    Expected format:
    {
      "verdict": "PASS"|"FAIL",
      "reason": "<1-2 sentences when FAIL; empty string when PASS>"
    }
    """
    verdict = str(parsed.get("verdict", "")).upper()
    if verdict not in ("PASS", "FAIL"):
        logger.warning(f"{check_name} invalid verdict: {verdict}")
        verdict = "FAIL"

    reason = str(parsed.get("reason", "") or "").strip()
    if verdict == "PASS":
        reason = ""
    elif not reason:
        reason = "Stem clarity/bias check failed without a specific reason"

    return {
        "verdict": verdict,
        "reason": reason,
        "pass": verdict == "PASS",
    }


def _standardize_stem_dimension_validation_response(parsed: Dict[str, Any], check_name: str) -> Dict[str, Any]:
    """
    Standardize per-dimension stem validation response
    (LO alignment + cognitive level — each evaluated independently).

    Expected format:
    {
      "lo_alignment":    {"verdict": "PASS"|"FAIL", "reason": "..."},
      "cognitive_level": {"verdict": "PASS"|"FAIL", "reason": "..."}
    }

    Returns the same per-dimension shape plus a computed overall "pass"
    (True only when BOTH dimensions PASS).
    """
    def _standardize_dimension(dimension_key: str) -> Dict[str, str]:
        raw = parsed.get(dimension_key)
        if not isinstance(raw, dict):
            logger.warning(f"{check_name} missing/invalid dimension: {dimension_key}")
            return {"verdict": "FAIL", "reason": f"{dimension_key} was not evaluated"}

        verdict = str(raw.get("verdict", "")).upper()
        if verdict not in ("PASS", "FAIL"):
            logger.warning(f"{check_name} invalid verdict for {dimension_key}: {verdict}")
            verdict = "FAIL"

        reason = str(raw.get("reason", "") or "").strip()
        if verdict == "PASS":
            reason = ""
        elif not reason:
            reason = f"{dimension_key} check failed without a specific reason"

        return {"verdict": verdict, "reason": reason}

    lo_alignment = _standardize_dimension("lo_alignment")
    cognitive_level = _standardize_dimension("cognitive_level")

    return {
        "lo_alignment": lo_alignment,
        "cognitive_level": cognitive_level,
        "pass": lo_alignment["verdict"] == "PASS" and cognitive_level["verdict"] == "PASS",
    }


def _get_default_response(
    check_name: str, 
    expected_format_type: str, 
    default_response: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get appropriate default response for failed parsing."""
    
    if default_response:
        return default_response
    
    if expected_format_type == "validation":
        return {
            "score": 0,
            "pass": False,
            "feedback": f"Failed to parse {check_name} response"
        }
    elif expected_format_type == "quality":
        return {
            "score": 0,
            "pass": False,
            "issues": [f"Failed to parse {check_name} response"]
        }
    elif expected_format_type == "boolean_quality":
        return {
            "pass": False,
            "issues": [f"Failed to parse {check_name} response"]
        }
    elif expected_format_type == "stem_validation":
        return {
            "verdict": "FAIL",
            "reason": f"Failed to parse {check_name} response",
            "pass": False,
        }
    elif expected_format_type == "stem_dimension_validation":
        return {
            "lo_alignment": {"verdict": "FAIL", "reason": f"Failed to parse {check_name} response"},
            "cognitive_level": {"verdict": "FAIL", "reason": f"Failed to parse {check_name} response"},
            "pass": False,
        }
    else:
        return {
            "score": 0,
            "pass": False,
            "error": f"Unknown format type for {check_name}"
        }


def log_validation_parsing_debug(check_name: str, content: str, parsed_result: Dict[str, Any]):
    """Log detailed debugging information for validation parsing."""
    # Debug logging removed for cleaner output
    pass