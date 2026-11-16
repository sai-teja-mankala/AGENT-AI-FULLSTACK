import json
import logging
import os
import random
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.token_tracking import get_token_tracker, reset_token_tracker
from .prompts import (
    STEM_LO_COGNITIVE_SYSTEM_PROMPT,
    STEM_LO_COGNITIVE_USER_PROMPT,
)

# Constants
MULTIPLE_CHOICE_MULTI_SELECT = "MULTIPLE_CHOICE_MULTI_SELECT"  # Question type that requires 2-3 correct options

from .KS_prioritization_prompts import (
    MATCHING_SYSTEM_CONTRACT_KNOWLEDGE_SOURCE,
    SYSTEM_CONTRACT_KNOWLEDGE_SOURCE,
)
from .pipeline_prompts import (
    SYSTEM_CONTRACT,
    DEFAULT_QUALITY_RUBRIC,
    OPTIONS_GENERATION_SINGLE_SELECT_PROMPT,
    OPTIONS_GENERATION_MULTI_SELECT_PROMPT,
    QUESTION_GENERATION_PROMPT,
    SCENARIO_REQUIREMENTS_TEMPLATE,
    RES_FORMAT,
    RES_FORMAT_MULTI_SELECT,
    SYSTEM_PROMPT_TEMPLATE_QUALITY_CHECK,
    SYSTEM_PROMPT_TEMPLATE_RELEVANCY_CHECK,
    SYSTEM_PROMPT_TEMPLATE_QUALITY_RELEVANCY_CHECK,
    SYSTEM_PROMPT_TEMPLATE_FIX_FORMAT,
    USER_PROMPT_TEMPLATE_FIX_FORMAT,
    USER_PROMPT_QUALITY_CORRECTION,
    USER_PROMPT_TEMPLATE_QUALITY_CHECK,
    USER_PROMPT_TEMPLATE_RELEVANCY_CHECK,
    USER_PROMPT_TEMPLATE_QUALITY_RELEVANCY_CHECK,
    DISTRACTOR_PLAUSIBILITY_CHECK_PROMPT,
    DISTRACTOR_MISCONCEPTION_CHECK_PROMPT,
    DISTRACTOR_DISTINCTIVENESS_CHECK_PROMPT,
    DISTRACTOR_INCORRECTNESS_CHECK_PROMPT,
    DISTRACTOR_CLARITY_CHECK_PROMPT,
    DISTRACTOR_PARALLELISM_LENGTH_CHECK_PROMPT,
    LOCALE_COMPLIANCE_CHECK_PROMPT,
    DISTRACTOR_BATCH_CORRECTION_SYSTEM_PROMPT,
    DISTRACTOR_BATCH_CORRECTION_USER_PROMPT,
    VALIDATION_SYSTEM_MESSAGE,
    STEM_CLARITY_BIAS_SYSTEM_PROMPT,
    STEM_CLARITY_BIAS_USER_PROMPT,
    MATCHING_SYSTEM_CONTRACT,
    MATCHING_STEM_GENERATION_PROMPT,
    MATCHING_OPTIONS_GENERATION_PROMPT,
    RES_FORMAT_MATCHING,
    DEFAULT_MATCHING_QUALITY_RUBRIC,
    SYSTEM_PROMPT_TEMPLATE_MATCHING_QUALITY_CHECK,
    USER_PROMPT_TEMPLATE_MATCHING_QUALITY_CHECK,
    USER_PROMPT_MATCHING_QUALITY_CORRECTION,
)
from .pipeline_constants import (
    MAX_DISTRACTOR_CORRECTION_ATTEMPTS,
    MAX_QUALITY_CORRECTION_ATTEMPTS,
    MAX_FORMAT_CORRECTION_ATTEMPTS,
    MAX_GUARDRAIL_REGENERATION_ATTEMPTS,
    MAX_STEM_VALIDATION_ATTEMPTS,
    MAX_STEM_LO_COG_VALIDATION_ATTEMPTS,
    MAX_QUESTION_RETRY_ATTEMPTS,
    RATE_LIMIT_RETRY_DELAY_SECONDS,
    CHECK_THRESHOLDS,
    QUESTION_GENERATION_BATCH_SIZE,
    OPTIONS_GENERATION_BATCH_SIZE,
    # Memory cleanup key groups
    EARLY_PIPELINE_KEYS,
    STEM_VALIDATION_KEYS,
    STEM_LO_COG_VALIDATION_KEYS,
    VALIDATION_AGENT_KEYS,
    DISTRACTOR_VALIDATION_KEYS,
    FINAL_PIPELINE_KEYS,
    MATCHING_PIPELINE_KEYS,
    MAX_MATCHING_QUALITY_CORRECTION_ATTEMPTS,
    USE_STRUCTURED_OUTPUT
)
from .pipeline_types import GraphState, PipelineInput, PromptPayload, PipelineConfig
from utils.constants import AGENT_SUPPORTED_QUESTION_TYPES, LOCALE_MAPPING
from ..utils.azure_clients import (
      get_generation_client,
      get_validation_client,
      get_generation_model,
      get_validation_model,
)
from utils.guardrails import ContentGuardrails, run_output_guardrails
from ..utils.validation_utils import extract_and_parse_validation_json
from ..utils.structured_output_utils import call_structured_llm, get_schema_for_context
from ..utils.response_schemas import (
    QuestionGenerationResponse,
    OptionsGenerationResponse,
    ValidationResponse,
)

logger = logging.getLogger(__name__)


class PipelineCancelled(Exception):
    """Raised when the owning job is cancelled while the pipeline is running."""
    pass


# Container ID logging utilities - passed from state
def _get_container_prefix(container_id: str = None, level: str = "info") -> str:
    """Get structured log prefix with container ID for ES filtering."""
    active_container_id = container_id or 'unknown'
    return f"{active_container_id}-{level}"

def _log_with_container(level: str, message: str, container_id: str = None, **kwargs) -> None:
    """Log with structured container prefix for ES filtering.
    
    Args:
        level: Log level (info, warning, error)
        message: Log message
        container_id: Optional container_id for structured logging
        **kwargs: Additional logging parameters
    """
    prefix = _get_container_prefix(container_id, level)
    
    formatted_message = f"[{prefix}] {message}"
    
    if level == "error":
        logger.error(formatted_message, **kwargs)
    elif level == "warning":
        logger.warning(formatted_message, **kwargs)
    else:
        logger.info(formatted_message, **kwargs)

def _log_pipeline_error(error: Exception, context: str, step: str, container_id: str = None) -> Dict[str, Any]:
    """Log pipeline error for job tracking with container ID."""
    error_log = {
        'error': str(error),
        'context': context,
        'step': step
    }
    _log_with_container("error", f"PIPELINE ERROR [{context}:{step}]: {error}", container_id=container_id)
    return error_log


def _clear_state_keys(state: GraphState, keys: List[str], context: str = "") -> None:
    """
    Clear specified keys from state to optimize memory usage.
    
    Args:
        state: The GraphState dictionary
        keys: List of state keys to clear
        context: Optional context for logging (e.g., "validation data", "pipeline data")
    """
    cleared_count = 0
    for key in keys:
        if key in state:
            if isinstance(state[key], list):
                state[key].clear()
            del state[key]
            cleared_count += 1
    
    if cleared_count > 0 and context:
        container_id = state.get("container_id")
        _log_with_container("info", f"MEMORY: Cleared {cleared_count} state keys ({context})", container_id=container_id)




def _create_pipeline_config(research_config: dict = None, container_id: str = None) -> PipelineConfig:
    """
    Create pipeline configuration from research config.
    Pure function that creates configuration without side effects.
    """
    research_config = research_config or {}

    generation_client = get_generation_client()
    validation_client = get_validation_client()

    generation_deployment = research_config.get("generationModel") or get_generation_model()
    validation_deployment = research_config.get("validationModel") or get_validation_model()

    _log_with_container("info", f"Pipeline configured - Generation: {generation_deployment}, Validation: {validation_deployment}", container_id=container_id)

    return PipelineConfig(
        generation_client=generation_client,
        generation_deployment=generation_deployment,
        validation_client=validation_client,
        validation_deployment=validation_deployment,
        research_config=research_config,
    )


def _call_llm(messages: list, task_type: str = "generation", pipeline_config: PipelineConfig = None, 
              response_format: Optional[Any] = None, call_context: str = "", container_id: str = None, **kwargs):
    """
    Call LLM with optional structured output support.
    Automatically selects generation or validation based on task_type.
    
    Args:
        messages: List of messages for chat completion
        task_type: Type of task for client selection
        pipeline_config: Pipeline configuration
        response_format: Optional Pydantic model for structured output
        call_context: Context for logging
        container_id: Container ID for logging
        **kwargs: Additional parameters for API call
        
    Returns:
        Either structured response (if response_format provided and feature enabled)
        or traditional response (backward compatibility)
    """
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")

    client, deployment = pipeline_config.get_client_and_deployment(task_type)
    
    # Use structured output if enabled and schema provided
    if USE_STRUCTURED_OUTPUT and response_format is not None:
        return call_structured_llm(
            client=client,
            deployment=deployment,
            messages=messages,
            response_format=response_format,
            task_type=task_type,
            pipeline_config=pipeline_config,
            call_context=call_context,
            container_id=container_id,
            **kwargs
        )
    
    # Fallback to traditional unstructured call
    return _call_openai(
        client=client,
        deployment=deployment,
        messages=messages,
        task_type=task_type,
        call_context=call_context,
        container_id=container_id,
        **kwargs
    )


def _call_openai(client, deployment: str, messages: list, task_type: str = None,
                 call_context: str = "", container_id: str = None, **kwargs):
    """
    Call the chat completions endpoint with rate-limit retry and token tracking.
    """
    max_retries = 3
    effective_context = call_context or f"{deployment}-{task_type}"
    for attempt in range(max_retries):
        started = time.monotonic()
        _log_with_container("info", f"LLM CALL -> {effective_context} | model={deployment} | attempt {attempt + 1}/{max_retries}", container_id=container_id)
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                timeout=120,
                **kwargs
            )
            break
        except Exception as e:
            _log_with_container("error", f"LLM CALL FAILED | {effective_context} | {type(e).__name__}: {str(e)[:300]}", container_id=container_id)
            is_rate_limit = (
                getattr(e, "status_code", None) == 429
                or "429" in str(e)
                or "rate limit" in str(e).lower()
                or "too many requests" in str(e).lower()
            )
            if is_rate_limit and attempt < max_retries - 1:
                _log_with_container("warning", f"Rate limit hit (429) on attempt {attempt + 1}, waiting {RATE_LIMIT_RETRY_DELAY_SECONDS}s before retry...", container_id=container_id)
                time.sleep(RATE_LIMIT_RETRY_DELAY_SECONDS)
                continue
            raise

    _log_with_container("info", f"LLM CALL OK | {effective_context} | {time.monotonic() - started:.1f}s", container_id=container_id)
    if response.usage:
        usage_data = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        get_token_tracker().add_usage(
            task_type=task_type or "unknown",
            usage_data=usage_data,
            call_context=effective_context,
        )
        _log_with_container("info", f"TOKEN: {effective_context} | In: {usage_data['prompt_tokens']:,} | Out: {usage_data['completion_tokens']:,} | Total: {usage_data['total_tokens']:,}", container_id=container_id)
    else:
        _log_with_container("warning", f"TOKEN WARNING: No usage data available for {effective_context} call", container_id=container_id)

    return response


def evaluate_question_with_weighted_scores(validation_checks: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate question using individual check approach - each check evaluated independently."""
    all_scores = []
    failed_checks = []
    passed_checks = []
    
    for check_name, result in validation_checks.items():
        score = result.get("score", 0)
        threshold = CHECK_THRESHOLDS.get(check_name, 80)
        passed = score >= threshold
        
        all_scores.append(score)
        
        if not passed:
            failed_checks.append({
                "check": check_name,
                "score": score,
                "threshold": threshold,
                "gap": threshold - score,
                "feedback": result.get("feedback", "")
            })
        else:
            passed_checks.append({
                "check": check_name,
                "score": score,
                "threshold": threshold,
                "feedback": result.get("feedback", "")
            })
    
    # Simple overall score: average of all individual check scores
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    return {
        "overall_score": round(overall_score, 2),
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "total_checks": len(validation_checks),
        "checks_passed": len(passed_checks),
        "checks_failed": len(failed_checks),
        "check_scores": {check: result.get("score", 0) for check, result in validation_checks.items()}
    }


def determine_correction_action(evaluation_result: Dict[str, Any], current_attempt: int) -> str:
    """
    Determine correction action based on individual check failures.
    
    Returns:
        "ACCEPT": All checks passed - question goes to final output
        "CORRECT": Some checks failed - question needs LLM correction and re-validation
    """
    failed_checks = evaluation_result["failed_checks"]
    
    if len(failed_checks) == 0:
        return "ACCEPT"

    return "CORRECT"




def _extract_json_text(content: str) -> str:
    """
    Extract JSON from markdown code blocks with improved handling.
    
    Handles various LLM response formats:
    - ```json\n{...}\n```
    - ```\n{...}\n```  
    - raw JSON without code blocks
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
    
    # Additional cleaning for common LLM artifacts
    if stripped.startswith("json"):
        stripped = stripped[4:].strip()
    
    return stripped


def _parse_json(content: str, container_id: str = None) -> Any:
    try:
        cleaned = _extract_json_text(content)
        parsed = json.loads(cleaned)
        if (
            isinstance(parsed, list)
            and len(parsed) == 1
            and isinstance(parsed[0], dict)
            and "questions" in parsed[0]
        ):
            return parsed[0]
        return parsed
    except json.JSONDecodeError as e:
        _log_with_container("warning", f"JSON parsing failed. Original content: {content[:200]}...", container_id=container_id)
        _log_with_container("warning", f"Cleaned content: {_extract_json_text(content)[:200]}...", container_id=container_id)
        _log_with_container("warning", f"JSON error: {str(e)}", container_id=container_id)
        raise


def _parse_json_raw(content: str, container_id: str = None) -> Any:
    try:
        cleaned = _extract_json_text(content)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        _log_with_container("warning", f"Raw JSON parsing failed. Content: {content[:200]}...", container_id=container_id)
        _log_with_container("warning", f"Cleaned: {_extract_json_text(content)[:200]}...", container_id=container_id)
        raise


def _call_llm_for_validation(
    system_message: str,
    user_prompt: str,
    check_name: str = "validation",
    threshold: int = 80,
    default_response: Optional[Dict[str, Any]] = None,
    pipeline_config: PipelineConfig = None,
    call_context: str = "",
    container_id: str = None
) -> Dict[str, Any]:
    """
    Reusable method for LLM validation calls with comprehensive JSON parsing.
    
    Uses improved validation utility for robust JSON extraction and parsing
    with fallback mechanisms for all distractor validation checks.
    
    Expected LLM response format: {"score": 85, "pass": true, "feedback": "explanation here"}
    """
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    
    try:
        # Determine schema for structured output
        schema = None
        if USE_STRUCTURED_OUTPUT:
            if any(ctx in call_context for ctx in ["validation", "check"]):
                schema = ValidationResponse
        
        # Use our centralized LLM call function
        response = _call_llm(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            task_type="validation",
            pipeline_config=pipeline_config,
            response_format=schema,
            call_context=call_context or f"validation_{check_name}",
            container_id=container_id
        )
        
        # Handle response based on structured output or traditional parsing
        if USE_STRUCTURED_OUTPUT and schema and hasattr(response, 'choices') and response.choices:
            # Structured response - convert to expected validation format
            message = response.choices[0].message
            
            # Check for refusal first
            if hasattr(message, 'refusal') and message.refusal:
                _log_with_container("warning", f"Model refused {check_name}: {message.refusal}", container_id=container_id)
                return default_response or {"score": 0, "pass": False, "feedback": f"Model refused {check_name}"}
            
            # Get structured data from message.parsed
            structured_data = None
            if hasattr(message, 'parsed') and message.parsed:
                if hasattr(message.parsed, 'model_dump'):
                    structured_data = message.parsed.model_dump()
                elif hasattr(message.parsed, 'dict'):
                    structured_data = message.parsed.dict()
                elif isinstance(message.parsed, dict):
                    # If it's already a dict, use it directly
                    structured_data = message.parsed
                else:
                    # Handle unexpected parsed type - try to convert to dict
                    try:
                        structured_data = dict(message.parsed) if hasattr(message.parsed, '__iter__') else {"raw_parsed": str(message.parsed)}
                        _log_with_container("warning", f"Unexpected parsed type for {check_name}, converted: {type(message.parsed)}", container_id=container_id)
                    except Exception as convert_e:
                        _log_with_container("error", f"Failed to convert parsed data for {check_name}: {convert_e}", container_id=container_id)
                        structured_data = None
            else:
                _log_with_container("warning", f"No parsed content available for {check_name}", container_id=container_id)
            
            # Convert to expected validation format and add 'pass' field
            if structured_data and isinstance(structured_data, dict):
                score = structured_data.get("score", 0)
                feedback = structured_data.get("feedback", f"No feedback provided for {check_name}")
                passed = score >= threshold
                
                return {
                    "score": score,
                    "pass": passed,
                    "feedback": feedback
                }
            elif structured_data is None:
                # No structured data available - return default response instead of empty JSON parsing
                _log_with_container("error", f"Structured data is None for {check_name}, returning default response", container_id=container_id)
                return default_response or {"score": 0, "pass": False, "feedback": f"Structured output parsing failed for {check_name}"}
            else:
                # Structured data exists but is not a dict - fallback to traditional parsing
                _log_with_container("warning", f"Invalid structured data type for {check_name}: {type(structured_data)}, falling back to traditional parsing", container_id=container_id)
        
        # Traditional validation parsing (fallback or when structured output disabled)
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content or "{}"
        else:
            # This might be a structured response without parsed data
            content = "{}"
            
        # Use the improved validation utility for comprehensive JSON parsing
        return extract_and_parse_validation_json(
            content=content,
            check_name=check_name,
            expected_format_type="validation",
            threshold=threshold,
            default_response=default_response
        )
            
    except Exception as e:
        _log_with_container("error", f"Unexpected error in {check_name}: {str(e)}", container_id=container_id)
        return default_response or {
            "score": 0, 
            "pass": False, 
            "feedback": f"Unexpected error in {check_name}: {str(e)}"
        }




def _normalize_learning_objectives(payload: PipelineInput) -> List[Dict[str, Any]]:
    raw = payload.get("learningObjectives") or []
    normalized: List[Dict[str, Any]] = []
    uuid_list = payload.get("learningObjectiveUuid")
    uuid_values: List[str | None] = []
    if isinstance(uuid_list, list):
        uuid_values = [str(val) if val else None for val in uuid_list]
    elif isinstance(uuid_list, str):
        uuid_values = [uuid_list]

    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            description = (
                item.get("description")
                or item.get("learningObjective")
                or item.get("text")
                or ""
            )
            lo_id = item.get("id") or item.get("learningObjectiveUuid")
        else:
            description = str(item)
            lo_id = None
        if not lo_id and idx < len(uuid_values):
            lo_id = uuid_values[idx]
        normalized.append({"id": lo_id, "description": description})
    return normalized


def _learning_objective_descriptions(payload: PipelineInput) -> List[str]:
    return [entry["description"] for entry in _normalize_learning_objectives(payload) if entry.get("description")]


def _group_questions_for_options_batching(questions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group questions by similar context for options generation batching.
    Groups by: Learning Objective + Difficulty Level + Question Type
    
    Args:
        questions: List of question objects with payload and questionText
        
    Returns:
        List of question batches, each batch contains questions with same context
    """
    if not questions:
        return []
    
    # Group questions by context (LO + difficulty + question type)
    context_groups = {}
    
    for question in questions:
        payload = question.get("payload", {})
        context_key = (
            payload.get("learningObjective", ""),
            payload.get("difficultyLevel", ""), 
            payload.get("questionType", "")
        )
        
        if context_key not in context_groups:
            context_groups[context_key] = []
        context_groups[context_key].append(question)
    
    # Create batches within each context group
    all_batches = []
    for context_questions in context_groups.values():
        # Split each context group into optimal batch sizes
        for i in range(0, len(context_questions), OPTIONS_GENERATION_BATCH_SIZE):
            batch = context_questions[i:i + OPTIONS_GENERATION_BATCH_SIZE]
            all_batches.append(batch)
    
    return all_batches


def _get_options_prompt(question_type: str) -> str:
    """Select the appropriate options generation prompt based on question type."""
    return (OPTIONS_GENERATION_MULTI_SELECT_PROMPT 
            if question_type == MULTIPLE_CHOICE_MULTI_SELECT 
            else OPTIONS_GENERATION_SINGLE_SELECT_PROMPT)


def _generate_batch_question_options(question_batch: List[Dict[str, Any]], pipeline_config: PipelineConfig, container_id: str = None) -> List[Dict[str, Any]]:
    """Generate options for questions - handles both single and multiple questions universally."""
    if not question_batch:
        return []
    
    # Use the first question's payload as the template (all questions in batch have same context)
    template_payload = question_batch[0]["payload"]
    question_type = template_payload["questionType"]
    
    # Prepare batch payload with multiple questions
    batch_options_payload = {
        "learning_objective": template_payload["learningObjective"],
        "difficulty": template_payload["difficultyLevel"],
        "question_type": question_type,
        "source_context": template_payload.get("sourceText", ""),
        "locale": template_payload.get("locale", "en-US"),
        "numberOfQuestions": len(question_batch),
        # Keep distractors source-grounded when knowledge-source prioritization is on
        "ksPrioritizeEnabled": template_payload.get("ksPrioritizeEnabled", False),
        "questions": []  # List of questions to generate options for
    }
    
    # Add each question to the batch
    for question_data in question_batch:
        question_info = {
            "questionText": question_data.get("questionText", "")
        }
        
        # No need for separate scenario data - it's embedded in questionText
        batch_options_payload["questions"].append(question_info)
    
    # Choose the appropriate prompt based on question type
    options_prompt = _get_options_prompt(question_type)
    
    # Option counts are handled by hardcoded rules in the prompt template
    # No need to pass separate parameters
    # Make batch API call
    batch_result = _run_decomposed_node(
        options_prompt,
        batch_options_payload,
        pipeline_config,
        call_context="options_generation_batch",
        container_id=container_id
    )
    
    # Parse batch results
    if not isinstance(batch_result, dict):
        _log_with_container("error", f"Options batch result is not a dict, got: {type(batch_result)} - {str(batch_result)[:200]}", container_id=container_id)
        return []
    batch_answers = batch_result.get("question_answers", [])
    if not batch_answers and "answers" in batch_result:
        # Fallback for single question format
        batch_answers = [{"questionText": question_batch[0].get("questionText", ""), "answers": batch_result["answers"]}]
    
    # Create individual outputs from batch results
    outputs = []
    for i, question_data in enumerate(question_batch):
        payload = question_data["payload"]
        
        # Get answers for this specific question
        if i < len(batch_answers):
            answers = batch_answers[i].get("answers", [])
        else:
            _log_with_container("warning", f"Missing answers for question {i+1} in batch", container_id=container_id)
            answers = []
        
        # Build formatted result
        question_result = {
            "questionText": question_data.get("questionText", ""),
            "answer": answers,
        }
        
        # No need for separate scenario data - it's embedded in questionText
        
        formatted_result = {
            "LearningObjective": payload["learningObjective"],
            "questions": [question_result]
        }
        
        outputs.append({
            "payload": payload,
            "result": formatted_result,
        })
    
    return outputs


def _create_question_batches(total_questions: int, batch_size: int = QUESTION_GENERATION_BATCH_SIZE) -> List[Dict[str, Any]]:
    """
    Create optimized batches for question generation grouped by difficulty level.
    
    Args:
        total_questions: Total number of questions to generate for this difficulty level
        batch_size: Maximum questions per batch (default: 5)
        
    Returns:
        List of batch configurations with batch_size and question_indices
        
    Examples:
        10 questions → [{"batch_size": 5, "question_indices": [1,2,3,4,5]}, 
                       {"batch_size": 5, "question_indices": [6,7,8,9,10]}]
        12 questions → [{"batch_size": 5, "question_indices": [1,2,3,4,5]},
                       {"batch_size": 5, "question_indices": [6,7,8,9,10]},
                       {"batch_size": 2, "question_indices": [11,12]}]
    """
    if total_questions <= 0:
        return []
    
    # Always use batching approach - works for any number (1, 2, 5, 10, etc.)
    batches = []
    for start_idx in range(0, total_questions, batch_size):
        end_idx = min(start_idx + batch_size, total_questions)
        batch_count = end_idx - start_idx
        question_indices = list(range(start_idx + 1, end_idx + 1))  # 1-based indices
        
        batches.append({
            "batch_size": batch_count,
            "question_indices": question_indices
        })
    
    return batches


def create_question_configs(state: GraphState) -> GraphState:
    data = state["input"]
    container_id = state.get("container_id")
    _log_with_container("info", "Creating question configurations", container_id=container_id)
    locale = data.get("locale", "en-US")
    source_text = data.get("sourceText", "")
    ks_prioritize = data.get("ksPrioritizeEnabled", False)
    number_of_questions = data.get("numberOfQuestions", 3)
    _log_with_container(
        "info",
        f"KS: create_question_configs ksPrioritizeEnabled={ks_prioritize}, "
        f"sourceText_len={len(source_text or '')}",
        container_id=container_id,
    )

    # DEBUG: Log payload details for calculation verification
    learning_objectives = data.get("learningObjectives", [])
    difficulty_levels = data.get("difficultyLevels", ["Beginner", "Intermediate", "Advanced"])
    raw_question_types = data.get("questionTypes", ["MULTIPLE_CHOICE"])
    # MATCHING is in AGENT_SUPPORTED_QUESTION_TYPES (it does go through the
    # agent pipeline) but is explicitly excluded here because it has its own
    # config builder (create_matching_question_configs) and chain, routed
    # separately by _route_from_start - it never flows through this MCQ
    # config builder.
    question_types = [qt for qt in raw_question_types if qt in AGENT_SUPPORTED_QUESTION_TYPES and qt != "MATCHING"]
    unsupported = [qt for qt in raw_question_types if qt not in AGENT_SUPPORTED_QUESTION_TYPES]
    if unsupported:
        _log_with_container("warning", f"Agent pipeline received unsupported types {unsupported}; filtering to {question_types}", container_id=container_id)
    if not question_types:
        raise ValueError(
            f"Agent pipeline does not support: {unsupported or raw_question_types}. "
            "MATCHING uses a separate generation flow."
        )

    # DEBUG: Check processed learning objectives vs raw
    processed_los = _learning_objective_descriptions(data)
    
    _log_with_container("info", f"PAYLOAD ANALYSIS: numQuestions={number_of_questions} | rawLOs={len(learning_objectives)} | processedLOs={len(processed_los)} | difficulties={difficulty_levels} | questionTypes={question_types}", container_id=container_id)
    
    # Calculate expected total using the same formula as run_pipeline
    expected_by_formula = len(learning_objectives) * len(difficulty_levels) * number_of_questions
    
    _log_with_container("info", f"EXPECTED QUESTION COUNT IN CURRENT FLOW: {len(learning_objectives)} LOs × {len(difficulty_levels)} difficulties × {number_of_questions} questions = {expected_by_formula} total", container_id=container_id)

    # Handle optional numCorrectOptions and numIncorrectOptions with intelligent defaults
    num_correct = data.get("numCorrectOptions")
    num_incorrect = data.get("numIncorrectOptions")

    payloads: List[PromptPayload] = []
    config_counter = 0
    
    for lo_idx, lo in enumerate(_learning_objective_descriptions(data)):
        for qtype_idx, qtype in enumerate(question_types):
            for level_idx, level in enumerate(difficulty_levels):
                _log_with_container("info", f"DEBUG: Processing LO{lo_idx+1} × QType{qtype_idx+1} × Level{level_idx+1}: '{level}'", container_id=container_id)
                
                # Set response format based on question type
                if qtype == MULTIPLE_CHOICE_MULTI_SELECT:
                    response_format = RES_FORMAT_MULTI_SELECT.strip()
                else:
                    response_format = RES_FORMAT.strip()
                
                # Create batched payloads for this configuration to optimize token usage
                # Each payload generates multiple questions in one API call  
                question_batches = _create_question_batches(number_of_questions, QUESTION_GENERATION_BATCH_SIZE)
                _log_with_container("info", f"DEBUG: Created {len(question_batches)} batches for {number_of_questions} questions", container_id=container_id)
                
                for batch_idx, batch in enumerate(question_batches):
                    config_counter += 1
                    _log_with_container("info", f"DEBUG: Config {config_counter}: Batch {batch_idx+1} - {batch['batch_size']} questions (indices: {batch['question_indices']})", container_id=container_id)
                    
                    payloads.append(
                        {
                            "learningObjective": lo,
                            "difficultyLevel": level,
                            "questionType": qtype,
                            "sourceText": source_text,
                            "locale": locale,  # Include locale for validation and generation
                            "numCorrectOptions": num_correct,  # Pass user value or None (let AI decide)
                            "numIncorrectOptions": num_incorrect,  # Pass user value or None (let AI decide)
                            "numberOfQuestions": batch["batch_size"],  # Generate multiple questions per batch
                            "questionIndices": batch["question_indices"],  # Track which questions [1,2,3,4,5]
                            "responseFormat": response_format,
                            "systemPrompt": SYSTEM_CONTRACT.strip(),
                            "userPrompt": "",
                            "ksPrioritizeEnabled": ks_prioritize,
                        }
                    )

    # DEBUG: Verify total expected questions from configs
    total_from_configs = sum(config.get("numberOfQuestions", 1) for config in payloads)
    _log_with_container("info", f"DEBUG: Config verification:", container_id=container_id)
    _log_with_container("info", f"  Total configs created: {len(payloads)}", container_id=container_id)
    _log_with_container("info", f"  Total questions from configs: {total_from_configs}", container_id=container_id)
    _log_with_container("info", f"  Expected by raw LOs: {expected_by_formula}", container_id=container_id)

    state["question_configs"] = payloads
    _log_with_container("info", f"Built {len(payloads)} question configs", container_id=container_id)
    return state


def _run_decomposed_node(node_prompt: str, payload_dict: Dict[str, Any], pipeline_config: PipelineConfig = None, call_context: str = "", container_id: str = None) -> Dict[str, Any]:
    """Helper to run a single decomposed generation node with SYSTEM_CONTRACT."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    
    locale_code = payload_dict.get('locale', 'en-US')
    locale_name = LOCALE_MAPPING.get(locale_code, locale_code)

    # Select source-grounded system contract when knowledge-source prioritization
    # is enabled - forces generation strictly from the provided source_context.
    ks_prioritize = bool(payload_dict.get("ksPrioritizeEnabled"))
    system_contract = SYSTEM_CONTRACT_KNOWLEDGE_SOURCE if ks_prioritize else SYSTEM_CONTRACT
    if ks_prioritize:
        _log_with_container(
            "info",
            f"KS: {call_context or 'generation'} using SOURCE-GROUNDED system contract",
            container_id=container_id,
        )
    formatted_system_contract = system_contract.format(locale=locale_name)

    # Format prompt with parameters from payload
    prompt_params = {"locale": locale_name, "scenario_requirements": payload_dict.get("scenario_requirements", "")}
    if "numberOfQuestions" in payload_dict:
        prompt_params["numberOfQuestions"] = payload_dict["numberOfQuestions"]
    
    formatted_node_prompt = node_prompt.format(**prompt_params)
    
    # Build user message from payload (excluding locale since it's in prompts)
    user_message_parts = []
    for key, value in payload_dict.items():
        if value and key != 'locale':  # Exclude locale from user message
            # Exclude option count parameters for all question types - let static prompts decide
            if key in ["numCorrectOptions", "numIncorrectOptions"]:
                continue  # Static prompts handle option counts reliably for all types
            # Control flag, not question content - handled via system contract selection
            if key == "ksPrioritizeEnabled":
                continue
            user_message_parts.append(f"{key}: {value}")
    user_message = "\n".join(user_message_parts)
    
    try:
        # Determine appropriate schema if structured output is enabled
        schema = None
        if USE_STRUCTURED_OUTPUT:
            if "question_generation" in call_context:
                schema = QuestionGenerationResponse
            elif "options_generation" in call_context:
                schema = OptionsGenerationResponse
        
        # Make the LLM call
        response = _call_llm(
            messages=[
                {"role": "system", "content": formatted_system_contract + "\n\n" + formatted_node_prompt},
                {"role": "user", "content": user_message},
            ],
            task_type="generation",
            pipeline_config=pipeline_config,
            response_format=schema,
            call_context=call_context,
            container_id=container_id
        )
        
        # Handle response based on structured output or traditional parsing
        if USE_STRUCTURED_OUTPUT and schema and hasattr(response, 'choices') and response.choices:
            # Structured response - get from message.parsed
            message = response.choices[0].message
            
            # Check for refusal first
            if hasattr(message, 'refusal') and message.refusal:
                _log_with_container("warning", f"Model refused {call_context}: {message.refusal}", container_id=container_id)
                return {"error": f"Model refused {call_context}: {message.refusal}"}
            
            # Get structured data from message.parsed
            if hasattr(message, 'parsed') and message.parsed:
                if hasattr(message.parsed, 'model_dump'):
                    return message.parsed.model_dump()
                elif hasattr(message.parsed, 'dict'):
                    return message.parsed.dict()
                else:
                    return message.parsed
            else:
                _log_with_container("warning", f"No parsed content for {call_context}, falling back to traditional parsing", container_id=container_id)
                # Fall through to traditional parsing
        
        # Traditional parsing (fallback or when structured output disabled)
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content or ""
        else:
            content = ""
            
        try:
            parsed = _parse_json(content)
            if not isinstance(parsed, dict):
                return {"error": "Non-object JSON from model", "raw": content}
            return parsed
        except json.JSONDecodeError as json_e:
            _log_with_container("error", f"JSON parsing failed for {call_context}: {json_e}", container_id=container_id)
            return {"error": f"JSON parsing failed: {str(json_e)}", "raw_content": content, "context": call_context}
        
    except (ConnectionError, TimeoutError, OSError) as network_e:
        # Network/connection errors should propagate to allow retry at higher level
        _log_with_container("error", f"Network/connection error in {call_context}: {network_e}", container_id=container_id)
        raise
    except Exception as e:
        # Check if this is an OpenAI API error that should propagate
        if hasattr(e, 'status_code') or 'authentication' in str(e).lower() or 'authorization' in str(e).lower():
            _log_with_container("error", f"API error in {call_context}: {e}", container_id=container_id)
            raise
        # For other unexpected errors, log and return error dict (but log as critical)
        _log_with_container("error", f"Unexpected error in {call_context}: {e}", container_id=container_id, exc_info=True)
        return {"error": f"Unexpected error: {str(e)}", "context": call_context}

def _retry_missing_MCQ_questions(config: Dict[str, Any], successful_questions: List[Dict], expected_count: int, pipeline_config: PipelineConfig = None, container_id: str = None) -> List[Dict]:
    """
    Retry question generation until required count is reached or max attempts exhausted.
    
    This function keeps retrying until we have the exact number of questions needed,
    accumulating questions across multiple retry attempts.
    
    Args:
        config: Original batch configuration
        successful_questions: Questions successfully generated in batch
        expected_count: Total number of questions expected
        pipeline_config: Pipeline configuration for LLM calls
    
    Returns:
        Combined list of successful + retry questions (up to expected_count)
    """
    # Start with successful questions
    accumulated_questions = successful_questions.copy()
    initial_count = len(accumulated_questions)
    
    _log_with_container("warning", f"Batch shortfall: {initial_count}/{expected_count} questions. Starting iterative retry process", container_id=container_id)
    
    # Keep retrying until we have enough questions or exhaust attempts
    for attempt in range(MAX_QUESTION_RETRY_ATTEMPTS):
        current_count = len(accumulated_questions)
        
        # Check if we already have enough questions
        if current_count >= expected_count:
            _log_with_container("info", f"Required count reached: {current_count}/{expected_count} questions", container_id=container_id)
            break
            
        # Calculate how many more questions we need
        still_missing = expected_count - current_count
        _log_with_container("info", f"Retry attempt {attempt + 1}: Need {still_missing} more questions (current: {current_count}/{expected_count})", container_id=container_id)
        
        try:
            # Determine scenario requirements for retry (preserve original question type)
            question_type = config.get("questionType", "MULTIPLE_CHOICE")
            if question_type == "SCENARIO_BASED":
                scenario_requirements = SCENARIO_REQUIREMENTS_TEMPLATE.format(numberOfQuestions=still_missing)
            else:
                scenario_requirements = ""  # Empty string for non-scenario questions
            
            # Create retry payload for remaining missing questions
            question_payload = {
                "learning_objective": config["learningObjective"],
                "difficulty": config["difficultyLevel"],
                "source_context": config.get("sourceText", ""),
                "question_type": question_type,
                "numberOfQuestions": still_missing,  # Only ask for what we still need
                "scenario_requirements": scenario_requirements,  # Preserve scenario type
                "locale": config.get("locale", "en-US"),
                "ksPrioritizeEnabled": config.get("ksPrioritizeEnabled", False),
            }
            
            # Generate missing questions using existing method
            retry_result = _run_decomposed_node(
                QUESTION_GENERATION_PROMPT, 
                question_payload, 
                pipeline_config,
                call_context=f"question_retry_batch_attempt_{attempt + 1}",
                container_id=container_id
            )
            
            # Extract questions from response
            retry_batch = retry_result.get("questions", [])
            if not retry_batch and "questionText" in retry_result:
                # Handle single question format
                retry_batch = [{"questionText": retry_result["questionText"]}]
            
            # Filter out empty questions from retry
            valid_retry_questions = [q for q in retry_batch if q.get("questionText", "").strip()]
            
            if valid_retry_questions:
                # Add the new questions to our accumulated collection
                accumulated_questions.extend(valid_retry_questions)
                new_total = len(accumulated_questions)
                
                _log_with_container("info", f"Retry attempt {attempt + 1} success: +{len(valid_retry_questions)} questions (total: {new_total}/{expected_count})", container_id=container_id)
                
                # If we have enough questions now, we can stop
                if new_total >= expected_count:
                    _log_with_container("info", f"Target reached after {attempt + 1} retry attempts: {new_total}/{expected_count}", container_id=container_id)
                    break
            else:
                _log_with_container("warning", f"Retry attempt {attempt + 1}: No valid questions generated", container_id=container_id)
                
        except Exception as e:
            _log_with_container("error", f"Retry attempt {attempt + 1} failed: {e}", container_id=container_id)
    
    # Final results
    final_count = len(accumulated_questions)
    total_retried = final_count - initial_count
    
    if final_count >= expected_count:
        # Trim to exact count if we got more than needed
        final_questions = accumulated_questions[:expected_count]
        _log_with_container("info", f"✅ Retry SUCCESS: {len(final_questions)}/{expected_count} questions (added {total_retried} via retry)", container_id=container_id)
    else:
        final_questions = accumulated_questions
        still_missing = expected_count - final_count
        _log_with_container("warning", f"❌ Retry INCOMPLETE: {final_count}/{expected_count} questions after {MAX_QUESTION_RETRY_ATTEMPTS} attempts (still missing {still_missing})", container_id=container_id)
    
    return final_questions


def _retry_missing_matching_questions(config: Dict[str, Any], successful_questions: List[Dict], expected_count: int, pipeline_config: PipelineConfig = None, container_id: str = None) -> List[Dict]:
    """
    Retry MATCHING stem generation until required count is reached or max attempts exhausted.

    Fully independent of _retry_missing_MCQ_questions() (MCQ) - deliberately not
    sharing a helper function so this can't affect MCQ's already-working
    retry path. Same loop shape and MAX_QUESTION_RETRY_ATTEMPTS budget, but
    its own copy: generates via _run_matching_generation_node +
    MATCHING_STEM_GENERATION_PROMPT, and a "valid" item needs both
    questionText and columnATerms (not just questionText like MCQ).

    Args:
        config: Original batch configuration
        successful_questions: Stems successfully generated in batch
        expected_count: Total number of stems expected
        pipeline_config: Pipeline configuration for LLM calls

    Returns:
        Combined list of successful + retry stems (up to expected_count)
    """
    # Start with successful questions
    accumulated_questions = successful_questions.copy()
    initial_count = len(accumulated_questions)

    _log_with_container("warning", f"MATCHING batch shortfall: {initial_count}/{expected_count} stems. Starting iterative retry process", container_id=container_id)

    # Keep retrying until we have enough stems or exhaust attempts
    for attempt in range(MAX_QUESTION_RETRY_ATTEMPTS):
        current_count = len(accumulated_questions)

        # Check if we already have enough stems
        if current_count >= expected_count:
            _log_with_container("info", f"Required count reached: {current_count}/{expected_count} stems", container_id=container_id)
            break

        # Calculate how many more stems we need
        still_missing = expected_count - current_count
        _log_with_container("info", f"MATCHING retry attempt {attempt + 1}: Need {still_missing} more stems (current: {current_count}/{expected_count})", container_id=container_id)

        try:
            # Create retry payload for remaining missing stems
            gen_payload = {
                "learning_objective": config["learningObjective"],
                "difficulty": config["difficultyLevel"],
                "source_context": config.get("sourceText", ""),
                "numberOfQuestions": still_missing,
                "locale": config.get("locale", "en-US"),
                "ksPrioritizeEnabled": config.get("ksPrioritizeEnabled", False),
            }

            # Generate missing stems using existing method
            retry_result = _run_matching_generation_node(
                MATCHING_STEM_GENERATION_PROMPT,
                gen_payload,
                pipeline_config,
                call_context=f"matching_stem_retry_batch_attempt_{attempt + 1}",
                container_id=container_id
            )

            # Extract stems from response
            retry_batch = retry_result.get("questions", [])

            # Filter out incomplete stems from retry (needs both text and Column A terms)
            valid_retry_questions = [
                q for q in retry_batch
                if q.get("questionText", "").strip() and q.get("columnATerms")
            ]

            if valid_retry_questions:
                # Add the new stems to our accumulated collection
                accumulated_questions.extend(valid_retry_questions)
                new_total = len(accumulated_questions)

                _log_with_container("info", f"MATCHING retry attempt {attempt + 1} success: +{len(valid_retry_questions)} stems (total: {new_total}/{expected_count})", container_id=container_id)

                # If we have enough stems now, we can stop
                if new_total >= expected_count:
                    _log_with_container("info", f"Target reached after {attempt + 1} MATCHING retry attempts: {new_total}/{expected_count}", container_id=container_id)
                    break
            else:
                _log_with_container("warning", f"MATCHING retry attempt {attempt + 1}: No valid stems generated", container_id=container_id)

        except Exception as e:
            _log_with_container("error", f"MATCHING retry attempt {attempt + 1} failed: {e}", container_id=container_id)

    # Final results
    final_count = len(accumulated_questions)
    total_retried = final_count - initial_count

    if final_count >= expected_count:
        # Trim to exact count if we got more than needed
        final_questions = accumulated_questions[:expected_count]
        _log_with_container("info", f"✅ MATCHING retry SUCCESS: {len(final_questions)}/{expected_count} stems (added {total_retried} via retry)", container_id=container_id)
    else:
        final_questions = accumulated_questions
        still_missing = expected_count - final_count
        _log_with_container("warning", f"❌ MATCHING retry INCOMPLETE: {final_count}/{expected_count} stems after {MAX_QUESTION_RETRY_ATTEMPTS} attempts (still missing {still_missing})", container_id=container_id)

    return final_questions


def build_question(state: GraphState) -> GraphState:
    """Node 2: Generate question text. For scenario-based questions, creates scenarios and questions in one step."""
    container_id = state.get("container_id")
    _log_with_container("info", "Building questions", container_id=container_id)
    
    questions: List[Dict[str, Any]] = []
    
    # Process all question configs - handle scenario-based and regular questions uniformly
    question_configs = state.get("question_configs", [])
    
    for config in question_configs:
        try:
            question_count = config.get("numberOfQuestions", 1)
            question_indices = config.get("questionIndices", list(range(1, question_count + 1)))
            question_type = config.get("questionType", "MULTIPLE_CHOICE")
            
            # Set scenario requirements based on question type using simple placeholder system
            if question_type == "SCENARIO_BASED":
                scenario_requirements = SCENARIO_REQUIREMENTS_TEMPLATE.format(numberOfQuestions=question_count)
                call_context = "question_generation_scenario_integrated"
                _log_with_container("info", f"Generating {question_count} scenario-based questions", container_id=container_id)
            else:
                scenario_requirements = ""  # Empty string for non-scenario questions
                call_context = "question_generation_mcq"
                _log_with_container("info", f"Generating {question_count} {question_type} questions", container_id=container_id)
            
            # Unified question generation approach - same payload structure for all types
            question_payload = {
                "learning_objective": config["learningObjective"],
                "difficulty": config["difficultyLevel"],
                "source_context": config.get("sourceText", ""),
                "question_type": question_type,
                "numberOfQuestions": question_count,
                "scenario_requirements": scenario_requirements,
                "locale": config.get("locale", "en-US"),
                "ksPrioritizeEnabled": config.get("ksPrioritizeEnabled", False),
            }
            
            batch_result = _run_decomposed_node(
                QUESTION_GENERATION_PROMPT, 
                question_payload, 
                state.get("pipeline_config"),
                call_context=call_context,
                container_id=container_id
            )
            
            # Handle batch question responses - expect strict format compliance
            batch_questions = batch_result.get("questions", [])
            if not batch_questions and "questionText" in batch_result:
                # Fallback for single question format (backward compatibility)
                batch_questions = [{"questionText": batch_result["questionText"]}]
            
            # Apply retry logic for missing questions if needed
            if len(batch_questions) < question_count:
                _log_with_container("warning", f"Batch incomplete: {len(batch_questions)}/{question_count} questions received", container_id=container_id)
                try:
                    batch_questions = _retry_missing_MCQ_questions(
                        config=config,
                        successful_questions=batch_questions,
                        expected_count=question_count,
                        pipeline_config=state.get("pipeline_config"),
                        container_id=container_id
                    )
                except Exception as e:
                    error_log = _log_pipeline_error(e, "question_generation", "retry_logic", container_id=container_id)
                    if "pipeline_errors" not in state:
                        state["pipeline_errors"] = []
                    state["pipeline_errors"].append(error_log)
            
            # Create individual question objects from batched results
            for question_idx, question_data in enumerate(batch_questions):
                # Create a copy of the config for each individual question
                individual_config = config.copy()
                
                # Assign proper question index (use batch indices or fallback to sequential)
                if question_idx < len(question_indices):
                    individual_config["questionIndex"] = question_indices[question_idx]
                else:
                    individual_config["questionIndex"] = question_idx + 1
                
                # Remove batch-specific fields from individual config
                individual_config.pop("numberOfQuestions", None)
                individual_config.pop("questionIndices", None)
                
                question_text = question_data.get("questionText", "").strip()
                if not question_text:
                    _log_with_container("warning", f"Empty question text detected at index {question_idx+1}", container_id=container_id)
                
                question_entry = {
                    "payload": individual_config,
                    "questionText": question_text,
                }
                questions.append(question_entry)
                
        except Exception as e:
            # Log error and store for job tracking
            error_log = _log_pipeline_error(e, "question_generation", "batch_generation", container_id=container_id)
            if "pipeline_errors" not in state:
                state["pipeline_errors"] = []
            state["pipeline_errors"].append(error_log)
            continue
    
    # Calculate expected total from all configs for verification
    total_expected = sum(config.get("numberOfQuestions", 1) for config in state.get("question_configs", []))
    
    state["questions"] = questions
    _log_with_container("info", f"Built {len(questions)} questions", container_id=container_id)
    
    if len(questions) != total_expected:
        shortfall = total_expected - len(questions)
        _log_with_container("warning", f"Question shortfall: {len(questions)}/{total_expected} generated (missing {shortfall})", container_id=container_id)
    
    return state


def _is_eligible_for_stem_clarity_bias(question_type: str) -> bool:
    """Return True if the question type is in scope for stem clarity/bias validation.

    MATCHING is excluded even though it is agent-supported: a matching
    question's "stem" is just an instruction (e.g. "Match the following
    terms") rather than a single question stem, so clarity/bias validation
    doesn't apply to it.
    """
    return question_type in AGENT_SUPPORTED_QUESTION_TYPES and question_type != "MATCHING"


def _build_stem_clarity_bias_prompt(question_type: str, question_stem: str) -> str:
    """Fill the stem clarity/bias user prompt template."""
    return STEM_CLARITY_BIAS_USER_PROMPT.format(
        question_type=question_type,
        question_stem=question_stem or "",
    )


def _is_stem_clarity_bias_failure(validation_result: Dict[str, Any]) -> bool:
    """Return True when the overall stem clarity/bias verdict is FAIL."""
    if not isinstance(validation_result, dict):
        return True
    if validation_result.get("verdict") == "FAIL":
        return True
    return not validation_result.get("pass", False)


def _collect_stem_failure_reasons(validation_result: Dict[str, Any]) -> List[str]:
    """Collect the failure reason from a failed stem validation result."""
    if validation_result.get("verdict") != "FAIL":
        return []
    reason = str(validation_result.get("reason", "") or "").strip()
    return [reason] if reason else []


def _call_stem_clarity_bias_validation(
    question_type: str,
    question_stem: str,
    pipeline_config: PipelineConfig,
    container_id: str = None,
) -> Dict[str, Any]:
    """Call the LLM to validate stem clarity and bias/fairness in a single request."""
    user_prompt = _build_stem_clarity_bias_prompt(question_type, question_stem)
    default_response = {
        "verdict": "FAIL",
        "reason": "Stem validation call failed",
        "pass": False,
    }
    
    try:
        response = _call_llm(
            messages=[
                {"role": "system", "content": STEM_CLARITY_BIAS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            task_type="validation",
            pipeline_config=pipeline_config,
            call_context="stem_clarity_bias_validation",
            container_id=container_id,
        )
        content = response.choices[0].message.content or "{}"
        return extract_and_parse_validation_json(
            content=content,
            check_name="stem_clarity_bias",
            expected_format_type="stem_validation",
            default_response=default_response,
        )
    except Exception as e:
        _log_with_container(
            "error",
            f"Unexpected error in stem clarity/bias validation: {str(e)}",
            container_id=container_id,
        )
        return default_response


def validate_stem_clarity_bias(state: GraphState) -> GraphState:
    """Validate question stems for clarity and bias/fairness before options generation."""
    container_id = state.get("container_id")
    attempt_num = state.get("stem_validation_attempt", 0) + 1
    _log_with_container(
        "info",
        f"Validating question stems for clarity and bias/fairness (attempt {attempt_num})",
        container_id=container_id,
    )

    # Re-validation loops read previously failed stems; first pass uses all questions.
    questions = state.get("stem_validation_failed") or state.get("questions", [])
    passed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    pipeline_config = state.get("pipeline_config")

    for question_entry in questions:
        payload = question_entry.get("payload", {})
        question_type = payload.get("questionType", "MULTIPLE_CHOICE")
        question_stem = question_entry.get("questionText", "")

        if not _is_eligible_for_stem_clarity_bias(question_type):
            _log_with_container(
                "info",
                f"Stem clarity/bias check skipped for question type {question_type}",
                container_id=container_id,
            )
            passed.append(question_entry)
            continue

        validation_result = _call_stem_clarity_bias_validation(
            question_type=question_type,
            question_stem=question_stem,
            pipeline_config=pipeline_config,
            container_id=container_id,
        )
        question_entry["stem_clarity_bias_validation"] = validation_result

        _log_with_container(
            "info",
            f"Stem validation — verdict: {validation_result.get('verdict', 'UNKNOWN')}",
            container_id=container_id,
        )

        if _is_stem_clarity_bias_failure(validation_result):
            failed.append(question_entry)
        else:
            passed.append(question_entry)

    state["questions"] = questions
    state["stem_validation_passed"] = state.get("stem_validation_passed", []) + passed
    state["stem_validation_failed"] = failed
    _log_with_container(
        "info",
        f"Stem validation complete: {len(passed)} passed, {len(failed)} failed",
        container_id=container_id,
    )
    return state


def _regenerate_single_stem(
    question_entry: Dict[str, Any],
    failure_reasons: List[str],
    pipeline_config: PipelineConfig,
    container_id: str = None,
) -> str:
    """Regenerate a single question stem using failure reasons from validation."""
    payload = question_entry.get("payload", {})
    question_type = payload.get("questionType", "MULTIPLE_CHOICE")

    if question_type == "SCENARIO_BASED":
        scenario_requirements = SCENARIO_REQUIREMENTS_TEMPLATE.format(numberOfQuestions=1)
        call_context = "stem_regeneration_scenario_integrated"
    else:
        scenario_requirements = ""
        call_context = "stem_regeneration_mcq"

    question_payload = {
        "learning_objective": payload.get("learningObjective", ""),
        "difficulty": payload.get("difficultyLevel", ""),
        "source_context": payload.get("sourceText", ""),
        "question_type": question_type,
        "numberOfQuestions": 1,
        "scenario_requirements": scenario_requirements,
        "locale": payload.get("locale", "en-US"),
        "failure_reasons": "; ".join(failure_reasons),
        "ksPrioritizeEnabled": payload.get("ksPrioritizeEnabled", False),
    }

    batch_result = _run_decomposed_node(
        QUESTION_GENERATION_PROMPT,
        question_payload,
        pipeline_config,
        call_context=call_context,
        container_id=container_id,
    )

    batch_questions = batch_result.get("questions", [])
    if not batch_questions and "questionText" in batch_result:
        batch_questions = [{"questionText": batch_result["questionText"]}]

    if batch_questions:
        new_stem = batch_questions[0].get("questionText", "").strip()
        if new_stem:
            original_stem = question_entry.get("questionText", "")
            _log_with_container(
                "info",
                f"Stem regeneration — type: {question_type} | "
                f"original: {original_stem} | regenerated: {new_stem}",
                container_id=container_id,
            )
            return new_stem

    _log_with_container(
        "warning",
        "Stem regeneration returned empty question text; keeping original stem",
        container_id=container_id,
    )
    return question_entry.get("questionText", "")


def regenerate_stem_clarity_bias(state: GraphState) -> GraphState:
    """Regenerate question stems that failed clarity/bias validation."""
    container_id = state.get("container_id")
    pipeline_config = state.get("pipeline_config")
    failed_questions = state.get("stem_validation_failed", [])

    _log_with_container(
        "info",
        f"Regenerating {len(failed_questions)} question stem(s) after clarity/bias failure",
        container_id=container_id,
    )

    regenerated: List[Dict[str, Any]] = []
    for question_entry in failed_questions:
        validation_result = question_entry.get("stem_clarity_bias_validation", {})
        failure_reasons = _collect_stem_failure_reasons(validation_result)
        if not failure_reasons:
            failure_reasons = ["Stem failed clarity and/or bias/fairness validation."]

        new_stem = _regenerate_single_stem(
            question_entry,
            failure_reasons,
            pipeline_config,
            container_id=container_id,
        )
        updated_entry = question_entry.copy()
        updated_entry["questionText"] = new_stem
        updated_entry.pop("stem_clarity_bias_validation", None)
        regenerated.append(updated_entry)

    state["questions"] = regenerated
    state["stem_validation_attempt"] = state.get("stem_validation_attempt", 0) + 1
    state["stem_validation_failed"] = regenerated

    _log_with_container(
        "info",
        f"Stem regeneration complete; {len(regenerated)} stem(s) rewritten",
        container_id=container_id,
    )
    return state


def _route_after_stem_validation(state: GraphState) -> str:
    """Route to stem regeneration or continue to options generation."""
    container_id = state.get("container_id")
    failed = state.get("stem_validation_failed", [])
    attempts = state.get("stem_validation_attempt", 0)

    if failed and attempts < MAX_STEM_VALIDATION_ATTEMPTS:
        _log_with_container(
            "info",
            f"Routing to stem regeneration (attempt {attempts + 1}/{MAX_STEM_VALIDATION_ATTEMPTS})",
            container_id=container_id,
        )
        return "regenerate"

    if failed:
        _log_with_container(
            "warning",
            f"Max stem validation attempts ({MAX_STEM_VALIDATION_ATTEMPTS}) reached. "
            f"{len(failed)} question(s) still flagged — proceeding with current stems",
            container_id=container_id,
        )

    return "continue"


def _merge_stem_validated_questions(state: GraphState) -> None:
    """Rebuild the full question list from accumulated stem validation results."""
    passed = state.get("stem_validation_passed")
    if passed is None:
        return
    failed = state.get("stem_validation_failed", [])
    state["questions"] = passed + failed


# ==========================================================================
# STEM LO-ALIGNMENT / COGNITIVE-LEVEL VALIDATION
# ==========================================================================
# Second stem-level quality gate, running after clarity/bias and before
# options generation. A single LLM call evaluates two dimensions
# independently: (1) whether the stem assesses the stated learning
# objective, and (2) whether its cognitive demand matches the assigned
# difficulty tier. Bloom's Taxonomy mapping is resolved inside the prompt;
# the raw difficulty (easy/medium/difficult) is passed through unchanged.


def _is_eligible_for_stem_lo_cognitive(question_type: str) -> bool:
    """Return True if the question type is in scope for LO/cognitive validation.

    Same scope as stem clarity/bias: all agent-supported MCQ-family types
    except MATCHING, whose "stem" is an instruction ("Match the following
    terms") rather than an assessable question.
    """
    return question_type in AGENT_SUPPORTED_QUESTION_TYPES and question_type != "MATCHING"


def _build_stem_lo_cognitive_prompt(
    question_type: str,
    difficulty_level: str,
    learning_objective: str,
    question_stem: str,
) -> str:
    """Fill the stem LO/cognitive user prompt template."""
    return STEM_LO_COGNITIVE_USER_PROMPT.format(
        question_type=question_type,
        difficulty_level=difficulty_level or "",
        learning_objective=learning_objective or "",
        question_stem=question_stem or "",
    )


def _is_stem_lo_cognitive_failure(validation_result: Dict[str, Any]) -> bool:
    """Return True when either dimension failed (overall stem verdict FAIL)."""
    if not isinstance(validation_result, dict):
        return True
    return not validation_result.get("pass", False)


def _collect_stem_lo_cog_failure_reasons(validation_result: Dict[str, Any]) -> List[str]:
    """Collect per-dimension failure reasons from a failed LO/cognitive result."""
    reasons: List[str] = []
    if not isinstance(validation_result, dict):
        return reasons
    for dimension_key, label in (
        ("lo_alignment", "LO alignment"),
        ("cognitive_level", "Cognitive level"),
    ):
        dimension = validation_result.get(dimension_key, {})
        if isinstance(dimension, dict) and dimension.get("verdict") == "FAIL":
            reason = str(dimension.get("reason", "") or "").strip()
            reasons.append(f"{label}: {reason}" if reason else f"{label}: failed")
    return reasons


def _call_stem_lo_cognitive_validation(
    question_type: str,
    difficulty_level: str,
    learning_objective: str,
    question_stem: str,
    pipeline_config: PipelineConfig,
    container_id: str = None,
) -> Dict[str, Any]:
    """Call the LLM to validate LO alignment and cognitive level in a single request."""
    user_prompt = _build_stem_lo_cognitive_prompt(
        question_type, difficulty_level, learning_objective, question_stem
    )
    default_response = {
        "lo_alignment": {"verdict": "FAIL", "reason": "Stem validation call failed"},
        "cognitive_level": {"verdict": "FAIL", "reason": "Stem validation call failed"},
        "pass": False,
    }

    try:
        response = _call_llm(
            messages=[
                {"role": "system", "content": STEM_LO_COGNITIVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            task_type="validation",
            pipeline_config=pipeline_config,
            call_context="stem_lo_cognitive_validation",
            container_id=container_id,
        )
        content = response.choices[0].message.content or "{}"
        return extract_and_parse_validation_json(
            content=content,
            check_name="stem_lo_cognitive",
            expected_format_type="stem_dimension_validation",
            default_response=default_response,
        )
    except Exception as e:
        _log_with_container(
            "error",
            f"Unexpected error in stem LO/cognitive validation: {str(e)}",
            container_id=container_id,
        )
        return default_response


def validate_stem_lo_cognitive(state: GraphState) -> GraphState:
    """Validate stems for LO alignment and cognitive level before options generation."""
    container_id = state.get("container_id")

    # On the first pass, fold the clarity/bias results into the working
    # question list. Regeneration loops (attempt > 0) operate on their own
    # failed list, so we must NOT re-merge and discard regenerated stems.
    if state.get("stem_lo_cog_validation_attempt", 0) == 0:
        _merge_stem_validated_questions(state)

    attempt_num = state.get("stem_lo_cog_validation_attempt", 0) + 1
    _log_with_container(
        "info",
        f"Validating question stems for LO alignment and cognitive level (attempt {attempt_num})",
        container_id=container_id,
    )

    # Re-validation loops read previously failed stems; first pass uses all questions.
    questions = state.get("stem_lo_cog_validation_failed") or state.get("questions", [])
    passed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    pipeline_config = state.get("pipeline_config")

    for question_entry in questions:
        payload = question_entry.get("payload", {})
        question_type = payload.get("questionType", "MULTIPLE_CHOICE")
        question_stem = question_entry.get("questionText", "")

        if not _is_eligible_for_stem_lo_cognitive(question_type):
            _log_with_container(
                "info",
                f"Stem LO/cognitive check skipped for question type {question_type}",
                container_id=container_id,
            )
            passed.append(question_entry)
            continue

        validation_result = _call_stem_lo_cognitive_validation(
            question_type=question_type,
            difficulty_level=payload.get("difficultyLevel", ""),
            learning_objective=payload.get("learningObjective", ""),
            question_stem=question_stem,
            pipeline_config=pipeline_config,
            container_id=container_id,
        )
        question_entry["stem_lo_cognitive_validation"] = validation_result

        _log_with_container(
            "info",
            "Stem LO/cognitive validation — "
            f"LO: {validation_result.get('lo_alignment', {}).get('verdict', 'UNKNOWN')}, "
            f"cognitive: {validation_result.get('cognitive_level', {}).get('verdict', 'UNKNOWN')}",
            container_id=container_id,
        )

        if _is_stem_lo_cognitive_failure(validation_result):
            failed.append(question_entry)
        else:
            passed.append(question_entry)

    state["questions"] = questions
    state["stem_lo_cog_validation_passed"] = state.get("stem_lo_cog_validation_passed", []) + passed
    state["stem_lo_cog_validation_failed"] = failed
    _log_with_container(
        "info",
        f"Stem LO/cognitive validation complete: {len(passed)} passed, {len(failed)} failed",
        container_id=container_id,
    )
    return state


def regenerate_stem_lo_cognitive(state: GraphState) -> GraphState:
    """Regenerate question stems that failed LO/cognitive validation."""
    container_id = state.get("container_id")
    pipeline_config = state.get("pipeline_config")
    failed_questions = state.get("stem_lo_cog_validation_failed", [])

    _log_with_container(
        "info",
        f"Regenerating {len(failed_questions)} question stem(s) after LO/cognitive failure",
        container_id=container_id,
    )

    regenerated: List[Dict[str, Any]] = []
    for question_entry in failed_questions:
        validation_result = question_entry.get("stem_lo_cognitive_validation", {})
        failure_reasons = _collect_stem_lo_cog_failure_reasons(validation_result)
        if not failure_reasons:
            failure_reasons = ["Stem failed LO alignment and/or cognitive level validation."]

        new_stem = _regenerate_single_stem(
            question_entry,
            failure_reasons,
            pipeline_config,
            container_id=container_id,
        )
        updated_entry = question_entry.copy()
        updated_entry["questionText"] = new_stem
        updated_entry.pop("stem_lo_cognitive_validation", None)
        regenerated.append(updated_entry)

    state["questions"] = regenerated
    state["stem_lo_cog_validation_attempt"] = state.get("stem_lo_cog_validation_attempt", 0) + 1
    state["stem_lo_cog_validation_failed"] = regenerated

    _log_with_container(
        "info",
        f"Stem LO/cognitive regeneration complete; {len(regenerated)} stem(s) rewritten",
        container_id=container_id,
    )
    return state


def _route_after_stem_lo_cognitive(state: GraphState) -> str:
    """Route to LO/cognitive stem regeneration or continue to options generation."""
    container_id = state.get("container_id")
    failed = state.get("stem_lo_cog_validation_failed", [])
    attempts = state.get("stem_lo_cog_validation_attempt", 0)

    if failed and attempts < MAX_STEM_LO_COG_VALIDATION_ATTEMPTS:
        _log_with_container(
            "info",
            f"Routing to stem LO/cognitive regeneration "
            f"(attempt {attempts + 1}/{MAX_STEM_LO_COG_VALIDATION_ATTEMPTS})",
            container_id=container_id,
        )
        return "regenerate"

    if failed:
        _log_with_container(
            "warning",
            f"Max stem LO/cognitive validation attempts ({MAX_STEM_LO_COG_VALIDATION_ATTEMPTS}) "
            f"reached. {len(failed)} question(s) still flagged — proceeding with current stems",
            container_id=container_id,
        )

    return "continue"


def _merge_stem_lo_cog_validated_questions(state: GraphState) -> None:
    """Rebuild the full question list from accumulated LO/cognitive results."""
    passed = state.get("stem_lo_cog_validation_passed")
    if passed is None:
        return
    failed = state.get("stem_lo_cog_validation_failed", [])
    state["questions"] = passed + failed


def build_options(state: GraphState) -> GraphState:
    """Node 3: Generate answer options for each question using optimized batching."""
    container_id = state.get("container_id")
    _log_with_container("info", "Building options", container_id=container_id)

    # LO/cognitive is the last stem gate and already folds clarity/bias results
    # into its passed/failed lists on first pass. Merge only those results so
    # stale clarity stems cannot overwrite LO-regenerated question text.
    _merge_stem_lo_cog_validated_questions(state)
    questions = state.get("questions", [])
    if not questions:
        _log_with_container("warning", "No questions available for options generation", container_id=container_id)
        state["raw_outputs"] = []
        return state
    
    # Group questions into batches for optimized API calls
    question_batches = _group_questions_for_options_batching(questions)
    
    outputs: List[Dict[str, Any]] = []
    total_api_calls = 0
    
    for batch_idx, question_batch in enumerate(question_batches):
        try:
            batch_size = len(question_batch)
            total_api_calls += 1
            
            # Always use batch method - it handles both single and multiple questions
            batch_outputs = _generate_batch_question_options(question_batch, state.get("pipeline_config"), container_id=container_id)
            outputs.extend(batch_outputs)
        except Exception as e:
            error_log = _log_pipeline_error(e, "option_generation", "batch_processing", container_id=container_id)
            if "pipeline_errors" not in state:
                state["pipeline_errors"] = []
            state["pipeline_errors"].append(error_log)
            continue
    
    state["raw_outputs"] = outputs
    _log_with_container("info", f"Built {len(questions)} complete questions with options", container_id=container_id)
    
    # Verify 1:1 mapping between questions and outputs
    if len(outputs) != len(questions):
        _log_with_container("warning", f"Options count mismatch: {len(outputs)}/{len(questions)} outputs generated", container_id=container_id)
        
    _clear_state_keys(state, EARLY_PIPELINE_KEYS + STEM_VALIDATION_KEYS + STEM_LO_COG_VALIDATION_KEYS, "early pipeline data")

    return state


def create_matching_question_configs(state: GraphState) -> GraphState:
    """Build one config per (learning objective, difficulty, batch) for MATCHING.
    Mirrors create_question_configs()'s batching approach but is independent of
    it - MATCHING has no question-type sub-branching (single/multi-select/
    scenario) and no numCorrectOptions/numIncorrectOptions concept.
    """
    data = state["input"]
    container_id = state.get("container_id")

    raw_question_types = data.get("questionTypes", [])
    if "MATCHING" not in raw_question_types:
        state["matching_question_configs"] = []
        return state

    _log_with_container("info", "Creating MATCHING question configurations", container_id=container_id)

    locale = data.get("locale", "en-US")
    source_text = data.get("sourceText", "")
    ks_prioritize = data.get("ksPrioritizeEnabled", False)
    number_of_questions = data.get("numberOfQuestions", 3)
    difficulty_levels = data.get("difficultyLevels", ["Beginner", "Intermediate", "Advanced"])
    _log_with_container(
        "info",
        f"KS: create_matching_question_configs ksPrioritizeEnabled={ks_prioritize}, "
        f"sourceText_len={len(source_text or '')}",
        container_id=container_id,
    )

    payloads: List[PromptPayload] = []
    for lo in _learning_objective_descriptions(data):
        for level in difficulty_levels:
            question_batches = _create_question_batches(number_of_questions, QUESTION_GENERATION_BATCH_SIZE)
            for batch in question_batches:
                payloads.append(
                    {
                        "learningObjective": lo,
                        "difficultyLevel": level,
                        "questionType": "MATCHING",
                        "sourceText": source_text,
                        "locale": locale,
                        "numberOfQuestions": batch["batch_size"],
                        "questionIndices": batch["question_indices"],
                        "responseFormat": RES_FORMAT_MATCHING.strip(),
                        "systemPrompt": MATCHING_SYSTEM_CONTRACT.strip(),
                        "userPrompt": "",
                        "ksPrioritizeEnabled": ks_prioritize,
                    }
                )

    state["matching_question_configs"] = payloads
    _log_with_container("info", f"Built {len(payloads)} MATCHING question configs", container_id=container_id)
    return state


def _extract_required_format_snippet(node_prompt: str) -> str:
    """Pull just the target JSON template out of a MATCHING generation
    prompt (the text after '**Format STRICTLY:**'), rather than passing the
    whole prompt as the "required format" for a correction call. The full
    prompt also contains generation instructions (e.g. "write N diverse
    question stems..."), which could bias a format-correction call into
    generating fresh content instead of just reformatting what the model
    already produced. Falls back to the full prompt if the marker isn't
    found, so correction still gets attempted either way.
    """
    marker = "**Format STRICTLY:**"
    idx = node_prompt.find(marker)
    if idx == -1:
        return node_prompt
    return node_prompt[idx + len(marker):].strip()


def _attempt_matching_format_correction(
    raw_output: str,
    required_format: str,
    locale_name: str,
    pipeline_config: PipelineConfig,
    call_context: str = "",
    container_id: str = None,
    max_attempts: int = MAX_FORMAT_CORRECTION_ATTEMPTS,
) -> Optional[Dict[str, Any]]:
    """Correction attempt(s) for a MATCHING generation response that failed
    JSON parsing. Reuses the same format-fix prompt pair already used for MCQ
    (SYSTEM_PROMPT_TEMPLATE_FIX_FORMAT/USER_PROMPT_TEMPLATE_FIX_FORMAT, see
    validate_and_fix_format/format_conversion) instead of dropping the item
    straight to an error result.

    Retries up to max_attempts times (default MAX_FORMAT_CORRECTION_ATTEMPTS,
    the same budget MCQ's own format-fix loop uses in validate_and_fix_format)
    if a correction attempt itself comes back invalid - each retry feeds the
    latest bad output back in as the thing to fix, rather than giving up after
    a single try. Returns the corrected dict on success, or None if every
    attempt fails - callers should fall back to their own error handling in
    that case.
    """
    current_raw_output = raw_output
    for attempt in range(1, max_attempts + 1):
        attempt_context = f"{call_context}_format_fix_attempt_{attempt}"
        content = ""
        try:
            response = _call_llm(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_FIX_FORMAT.format(locale=locale_name)},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE_FIX_FORMAT.format(
                        res_format=required_format,
                        raw_output=current_raw_output,
                        locale=locale_name,
                    )},
                ],
                task_type="format_fix",
                pipeline_config=pipeline_config,
                call_context=attempt_context,
                container_id=container_id,
            )
            content = response.choices[0].message.content or ""
            corrected = _parse_json(content)
            if not isinstance(corrected, dict):
                _log_with_container("warning", f"Format correction attempt {attempt}/{max_attempts} for {call_context} returned non-object JSON", container_id=container_id)
                current_raw_output = content
                continue
            _log_with_container("info", f"Format correction succeeded for {call_context} on attempt {attempt}/{max_attempts}", container_id=container_id)
            return corrected
        except json.JSONDecodeError as e:
            _log_with_container("warning", f"Format correction attempt {attempt}/{max_attempts} for {call_context} failed (still invalid JSON): {e}", container_id=container_id)
            current_raw_output = content or current_raw_output
            continue
        except Exception as e:
            # Not a JSON problem (e.g. network/API error) - still respects
            # max_attempts rather than giving up immediately on the first try.
            _log_with_container("error", f"Format correction attempt {attempt}/{max_attempts} for {call_context} failed unexpectedly: {e}", container_id=container_id)
            continue

    _log_with_container("error", f"Format correction for {call_context} exhausted {max_attempts} attempts - giving up", container_id=container_id)
    return None


def _run_matching_generation_node(node_prompt: str, payload_dict: Dict[str, Any], pipeline_config: PipelineConfig = None, call_context: str = "", container_id: str = None) -> Dict[str, Any]:
    """MATCHING's generation helper. Uses MATCHING_SYSTEM_CONTRACT and
    MATCHING-specific structured-output schemas when USE_STRUCTURED_OUTPUT is
    enabled, falling back to JSON parsing otherwise.
    """
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")

    locale_code = payload_dict.get('locale', 'en-US')
    locale_name = LOCALE_MAPPING.get(locale_code, locale_code)

    # Source-grounded contract when knowledge-source prioritization is enabled.
    ks_prioritize = bool(payload_dict.get("ksPrioritizeEnabled"))
    matching_system_contract = (
        MATCHING_SYSTEM_CONTRACT_KNOWLEDGE_SOURCE if ks_prioritize else MATCHING_SYSTEM_CONTRACT
    )
    if ks_prioritize:
        _log_with_container(
            "info",
            f"KS: {call_context or 'matching_generation'} using SOURCE-GROUNDED system contract",
            container_id=container_id,
        )
    formatted_system_contract = matching_system_contract.format(locale=locale_name)

    prompt_params = {"locale": locale_name}
    if "numberOfQuestions" in payload_dict:
        prompt_params["numberOfQuestions"] = payload_dict["numberOfQuestions"]

    formatted_node_prompt = node_prompt.format(**prompt_params)

    user_message_parts = []
    for key, value in payload_dict.items():
        # `if value` would silently drop falsy-but-meaningful values like
        # numberOfQuestions=0; only skip truly absent/empty values.
        if value not in (None, '', [], {}) and key != 'locale':
            # Control flag, not question content - drives contract selection only
            if key == "ksPrioritizeEnabled":
                continue
            user_message_parts.append(f"{key}: {value}")
    user_message = "\n".join(user_message_parts)

    messages = [
        {"role": "system", "content": formatted_system_contract + "\n\n" + formatted_node_prompt},
        {"role": "user", "content": user_message},
    ]

    # Use structured output when enabled — MATCHING schemas guarantee column counts.
    if USE_STRUCTURED_OUTPUT:
        schema = get_schema_for_context(call_context)
        if schema is not None:
            try:
                response = call_structured_llm(
                    client=pipeline_config.generation_client,
                    deployment=pipeline_config.generation_deployment,
                    messages=messages,
                    response_format=schema,
                    task_type="generation",
                    pipeline_config=pipeline_config,
                    call_context=call_context,
                    container_id=container_id,
                )
                message = response.choices[0].message
                if hasattr(message, 'parsed') and message.parsed is not None:
                    return message.parsed.model_dump()
            except Exception as e:
                _log_with_container(
                    "warning",
                    f"Structured output failed for {call_context}, falling back to JSON parsing: {e}",
                    container_id=container_id,
                )

    try:
        response = _call_llm(
            messages=messages,
            task_type="generation",
            pipeline_config=pipeline_config,
            call_context=call_context,
            container_id=container_id
        )

        content = ""
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content or ""

        try:
            parsed = _parse_json(content)
            if not isinstance(parsed, dict):
                return {"error": "Non-object JSON from model", "raw": content}
            return parsed
        except json.JSONDecodeError as json_e:
            _log_with_container("warning", f"JSON parsing failed for {call_context}: {json_e}. Attempting format correction before giving up.", container_id=container_id)
            corrected = _attempt_matching_format_correction(
                raw_output=content,
                required_format=_extract_required_format_snippet(formatted_node_prompt),
                locale_name=locale_name,
                pipeline_config=pipeline_config,
                call_context=call_context,
                container_id=container_id,
            )
            if corrected is not None:
                return corrected
            return {"error": f"JSON parsing failed: {str(json_e)}", "raw_content": content, "context": call_context}

    except (ConnectionError, TimeoutError, OSError) as network_e:
        _log_with_container("error", f"Network/connection error in {call_context}: {network_e}", container_id=container_id)
        raise
    except Exception as e:
        if hasattr(e, 'status_code') or 'authentication' in str(e).lower() or 'authorization' in str(e).lower():
            _log_with_container("error", f"API error in {call_context}: {e}", container_id=container_id)
            raise
        _log_with_container("error", f"Unexpected error in {call_context}: {e}", container_id=container_id, exc_info=True)
        return {"error": f"Unexpected error: {str(e)}", "context": call_context}


def build_question_matching(state: GraphState) -> GraphState:
    """MATCHING Node 1: generate the question stem + Column A terms.
    Fully independent of build_question() (MCQ) - own prompt, own state key,
    no scenario/multi-select branching since none of that applies to MATCHING.
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Building MATCHING questions (stem + Column A)", container_id=container_id)

    questions: List[Dict[str, Any]] = []
    configs = state.get("matching_question_configs", [])

    for config in configs:
        try:
            question_count = config.get("numberOfQuestions", 1)
            question_indices = config.get("questionIndices", list(range(1, question_count + 1)))

            gen_payload = {
                "learning_objective": config["learningObjective"],
                "difficulty": config["difficultyLevel"],
                "source_context": config.get("sourceText", ""),
                "numberOfQuestions": question_count,
                "locale": config.get("locale", "en-US"),
                "ksPrioritizeEnabled": config.get("ksPrioritizeEnabled", False),
            }

            batch_result = _run_matching_generation_node(
                MATCHING_STEM_GENERATION_PROMPT,
                gen_payload,
                state.get("pipeline_config"),
                call_context="matching_stem_generation",
                container_id=container_id
            )

            batch_questions = batch_result.get("questions", [])

            valid_batch_questions = [
                q for q in batch_questions
                if q.get("questionText", "").strip() and q.get("columnATerms")
            ]

            if len(valid_batch_questions) < question_count:
                _log_with_container("warning", f"MATCHING stem batch incomplete: {len(valid_batch_questions)}/{question_count} stems received", container_id=container_id)
                try:
                    valid_batch_questions = _retry_missing_matching_questions(
                        config=config,
                        successful_questions=valid_batch_questions,
                        expected_count=question_count,
                        pipeline_config=state.get("pipeline_config"),
                        container_id=container_id
                    )
                except Exception as e:
                    error_log = _log_pipeline_error(e, "matching_stem_generation", "retry_logic", container_id=container_id)
                    if "pipeline_errors" not in state:
                        state["pipeline_errors"] = []
                    state["pipeline_errors"].append(error_log)

            for question_idx, question_data in enumerate(valid_batch_questions):
                individual_config = config.copy()
                if question_idx < len(question_indices):
                    individual_config["questionIndex"] = question_indices[question_idx]
                else:
                    individual_config["questionIndex"] = question_idx + 1
                individual_config.pop("numberOfQuestions", None)
                individual_config.pop("questionIndices", None)

                question_text = question_data.get("questionText", "").strip()
                column_a_terms = question_data.get("columnATerms", [])
                if not question_text or not column_a_terms:
                    _log_with_container("warning", f"MATCHING stem/Column A incomplete at index {question_idx + 1} - skipping", container_id=container_id)
                    continue

                questions.append({
                    "payload": individual_config,
                    "questionText": question_text,
                    "columnATerms": column_a_terms,
                })

        except Exception as e:
            error_log = _log_pipeline_error(e, "matching_stem_generation", "batch_generation", container_id=container_id)
            if "pipeline_errors" not in state:
                state["pipeline_errors"] = []
            state["pipeline_errors"].append(error_log)
            continue

    # Mirrors build_question()'s (MCQ) final shortfall summary log.
    total_expected = sum(config.get("numberOfQuestions", 1) for config in state.get("matching_question_configs", []))

    state["matching_questions"] = questions
    _log_with_container("info", f"Built {len(questions)} MATCHING stems", container_id=container_id)

    if len(questions) != total_expected:
        shortfall = total_expected - len(questions)
        _log_with_container("warning", f"MATCHING stem shortfall: {len(questions)}/{total_expected} generated (missing {shortfall})", container_id=container_id)

    return state


def _normalize_matching_options_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert {label, text} column entries and term/definition pairing keys
    (per MATCHING_OPTIONS_GENERATION_PROMPT) into the legacy single-key
    {label: text} shape the rest of the pipeline expects. Applied once here so
    both the structured-output and JSON-parsing-fallback paths get the same
    shape - entries already in the legacy shape pass through unchanged.
    """
    for q in result.get("questions", []):
        for col_key in ("column_a_answers", "column_b_answers"):
            q[col_key] = [
                {e["label"]: e["text"]} if isinstance(e, dict) and "label" in e and "text" in e else e
                for e in q.get(col_key, [])
            ]
        for p in q.get("answers", []):
            if "term" in p:
                p["column_a_answers"] = p.pop("term")
            if "definition" in p:
                p["column_b_answers"] = p.pop("definition")
    return result


def build_options_matching(state: GraphState) -> GraphState:
    """MATCHING Node 2: generate Column B, the correct pairings, and
    explanations given the stem + Column A from build_question_matching.
    Fully independent of build_options() (MCQ) - no options-batching-by-context
    grouping since MATCHING questions are generated and completed one at a time.
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Building MATCHING options (Column B + pairings)", container_id=container_id)

    questions = state.get("matching_questions", [])
    outputs: List[Dict[str, Any]] = []

    for question_entry in questions:
        payload = question_entry["payload"]
        try:
            gen_payload = {
                "questionText": question_entry.get("questionText", ""),
                "columnATerms": json.dumps(question_entry.get("columnATerms", [])),
                "locale": payload.get("locale", "en-US"),
            }

            question_result = None
            # Retry once on column-count schema violations (column_a/b mismatch).
            for attempt in range(2):
                result = _run_matching_generation_node(
                    MATCHING_OPTIONS_GENERATION_PROMPT,
                    gen_payload,
                    state.get("pipeline_config"),
                    call_context="matching_options_generation",
                    container_id=container_id
                )
                result = _normalize_matching_options_result(result)

                if not result.get("questions") and "error" in result:
                    break

                candidate = (result.get("questions") or [result])[0]
                is_valid, errors = _is_valid_response_format(
                    {"questions": [candidate]}, "MATCHING"
                )
                if is_valid:
                    question_result = candidate
                    break
                _log_with_container(
                    "warning",
                    f"MATCHING schema invalid (attempt {attempt + 1}/2): {'; '.join(errors[:2])}",
                    container_id=container_id,
                )

            if question_result is None:
                if not result.get("questions") and "error" in result:
                    error_log = _log_pipeline_error(
                        RuntimeError(result.get("error", "Unknown error")),
                        "matching_options_generation",
                        "batch_generation",
                        container_id=container_id,
                    )
                    if "pipeline_errors" not in state:
                        state["pipeline_errors"] = []
                    state["pipeline_errors"].append(error_log)
                else:
                    # Use last result even if schema invalid (best-effort).
                    question_result = (result.get("questions") or [result])[0]

            if question_result is None:
                continue

            formatted_result = {
                "LearningObjective": payload["learningObjective"],
                "questions": [question_result],
            }
            outputs.append({"payload": payload, "result": formatted_result})

        except Exception as e:
            error_log = _log_pipeline_error(e, "matching_options_generation", "batch_generation", container_id=container_id)
            if "pipeline_errors" not in state:
                state["pipeline_errors"] = []
            state["pipeline_errors"].append(error_log)
            continue

    state["matching_raw_outputs"] = outputs
    _log_with_container("info", f"Built {len(outputs)} complete MATCHING questions", container_id=container_id)
    _clear_state_keys(state, ["matching_question_configs", "matching_questions"], "matching early pipeline data")
    return state


def evaluate_matching_quality(result: Any, rubric: str, locale: str = 'en-US', pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """MATCHING-specific quality evaluator. Independent of evaluate_quality()
    (MCQ) - own system/user prompts so the rubric and failure criteria can be
    tuned for pairing-based items without touching MCQ's.

    Boolean pass/fail rather than a numeric score + threshold: the LLM judges
    directly against the checklist rubric, mirroring the same boolean pattern
    already used for stem clarity/bias validation elsewhere in this file.
    """
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    if "error" in result:
        return {"issues": ["Invalid JSON"], "pass": False}

    locale_name = LOCALE_MAPPING.get(locale, 'English_US')

    try:
        prompt = USER_PROMPT_TEMPLATE_MATCHING_QUALITY_CHECK.format(
            rubric=rubric,
            questions=json.dumps(result),
            locale=locale_name
        )

        response = _call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_MATCHING_QUALITY_CHECK},
                {"role": "user", "content": prompt},
            ],
            task_type="quality_check",
            pipeline_config=pipeline_config,
            call_context="matching_quality_check",
            container_id=container_id
        )

        content = response.choices[0].message.content or ""

        return extract_and_parse_validation_json(
            content=content,
            check_name="matching_quality_check",
            expected_format_type="boolean_quality"
        )

    except Exception as e:
        _log_with_container("error", f"Error in MATCHING quality evaluation: {str(e)}", container_id=container_id)
        return {
            "pass": False,
            "issues": [f"Quality evaluation error: {str(e)}"]
        }


def validate_quality_matching(state: GraphState) -> GraphState:
    """MATCHING quality validation. Fully independent of validate_quality()
    (MCQ) - own rubric, own threshold input keys, own state keys. Relevancy
    checking is shared (evaluate_relevancy) since judging "is this relevant to
    the LO" has no type-specific structure to isolate.
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Validating MATCHING quality and relevancy", container_id=container_id)

    already_passed = state.get("matching_quality_passed", [])
    items_to_validate = state.get("matching_quality_failed")
    if items_to_validate is None or len(items_to_validate) == 0:
        items_to_validate = state.get("matching_raw_outputs", [])

    rubric = state["input"].get("matchingQualityRubric") or DEFAULT_MATCHING_QUALITY_RUBRIC.strip()
    relevancy_threshold = state["input"].get("relevancyThreshold", 85)
    learning_objectives = _learning_objective_descriptions(state["input"])

    passed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for item in items_to_validate:
        payload = item["payload"]
        result = item["result"]
        locale = payload.get('locale', 'en-US')

        # Boolean pass/fail - no numeric score/threshold. See evaluate_matching_quality.
        quality = evaluate_matching_quality(result, rubric, locale, state.get("pipeline_config"), container_id=container_id)
        relevancy = evaluate_relevancy(result, learning_objectives, relevancy_threshold, locale, state.get("pipeline_config"), container_id=container_id)

        entry = {
            "payload": payload,
            "result": result,
            "quality": quality,
            "relevancy": relevancy,
        }

        if quality.get("pass") and relevancy.get("pass"):
            passed.append(entry)
        else:
            failed.append(entry)

    _clear_state_keys(state, ["matching_raw_outputs"], "matching raw outputs consumed")

    total_passed_accumulated = already_passed + passed

    state["matching_quality_passed"] = total_passed_accumulated
    state["matching_quality_failed"] = failed
    # Failed items are still passed through to improved_outputs even if they
    # fail again after the correction loop exhausts its max attempts, so the
    # pipeline doesn't stall waiting on a perfect result.
    state["improved_outputs"] = [
        {"payload": entry["payload"], "result": entry["result"]}
        for entry in total_passed_accumulated + failed
    ]

    _log_with_container("info", f"MATCHING quality validation: {len(total_passed_accumulated)} passed out of total, {len(failed)} failed on this turn", container_id=container_id)
    return state


def correct_quality_matching(state: GraphState) -> GraphState:
    """MATCHING quality correction. Fully independent of correct_quality()
    (MCQ) - own correction prompt (USER_PROMPT_MATCHING_QUALITY_CORRECTION) and
    own system contract (MATCHING_SYSTEM_CONTRACT).
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Correcting failed MATCHING quality or relevancy", container_id=container_id)
    corrected_failed: List[Dict[str, Any]] = []

    for item in state.get("matching_quality_failed", []):
        payload = item["payload"]
        result = item["result"]
        quality = item.get("quality", {})
        relevancy = item.get("relevancy", {})
        failure_reasons: List[str] = []

        if not quality.get("pass"):
            issues = quality.get("issues") or []
            issue_text = "; ".join([issue if isinstance(issue, str) else json.dumps(issue) for issue in issues])
            failure_reasons.append("Quality issues: " + issue_text)

        if not relevancy.get("pass"):
            issues = relevancy.get("issues") or []
            issue_text = "; ".join([issue if isinstance(issue, str) else json.dumps(issue) for issue in issues])
            failure_reasons.append("Relevancy issues: " + issue_text)

        failure_reason_text = "\n".join([r for r in failure_reasons if r]) or "Failed evaluation."

        locale_code = payload.get('locale', 'en-US')
        locale_name = LOCALE_MAPPING.get(locale_code, 'English_US')

        correction_prompt = USER_PROMPT_MATCHING_QUALITY_CORRECTION.format(
            learning_obj=payload["learningObjective"],
            difficulty_level=payload["difficultyLevel"],
            failure_reasons=failure_reason_text,
            question_json=json.dumps(result),
            response_format=payload["responseFormat"],
            locale=locale_name
        )

        try:
            response = _call_llm(
                messages=[
                    {"role": "system", "content": MATCHING_SYSTEM_CONTRACT.format(locale=locale_name)},
                    {"role": "user", "content": correction_prompt},
                ],
                task_type="quality_correction",
                pipeline_config=state.get("pipeline_config"),
                call_context="matching_quality_correction",
                container_id=container_id
            )
            content = response.choices[0].message.content or ""
            corrected = _parse_json(content)
            corrected_failed.append({"payload": payload, "result": corrected})
        except json.JSONDecodeError as e:
            _log_with_container("warning", f"MATCHING quality correction failed - fallback to original output to prevent lockup: {e}", container_id=container_id)
            corrected_failed.append({"payload": payload, "result": result})
        except Exception as e:
            _log_with_container("error", f"MATCHING quality correction LLM call failed - fallback to original output to prevent lockup: {e}", container_id=container_id)
            corrected_failed.append({"payload": payload, "result": result})

    state["matching_quality_failed"] = corrected_failed

    state["matching_quality_correction_attempts"] = state.get("matching_quality_correction_attempts", 0) + 1
    _log_with_container("info", f"Corrected {len(corrected_failed)} failed MATCHING quality outputs", container_id=container_id)
    return state


def _route_from_validate_quality_matching(state: GraphState) -> str:
    container_id = state.get("container_id")
    failed = state.get("matching_quality_failed", [])
    max_attempts = state["input"].get("maxMatchingQualityFixAttempts", MAX_MATCHING_QUALITY_CORRECTION_ATTEMPTS)
    attempts = state.get("matching_quality_correction_attempts", 0)

    if failed and attempts < max_attempts:
        _log_with_container("info", f"Routing back to MATCHING quality correction loop (Attempt {attempts + 1}/{max_attempts})", container_id=container_id)
        return "correct_quality_matching"

    if failed:
        _log_with_container("info", f"Max MATCHING quality correction attempts ({max_attempts}) reached. Passing both passed and failed items to final trail.", container_id=container_id)

    return "end"


def finalize_matching_output(state: GraphState) -> GraphState:
    """Cleanup node. validate_quality_matching already writes MATCHING's
    final passed results directly into the shared 'improved_outputs' key
    (mirroring MCQ's validate_quality), which is what run_pipeline()'s outer
    format-correction loop and _build_output() read from regardless of which
    chain produced them - so there's nothing left to bridge here, just
    MATCHING's remaining transient state to clear.
    """
    _clear_state_keys(state, MATCHING_PIPELINE_KEYS, "matching pipeline cleanup")
    return state


def _route_from_start(state: GraphState) -> str:
    """Routes the graph's entry point to either the MCQ chain or the fully
    separate MATCHING chain, based on which config builder populated state
    before the graph was invoked (create_question_configs vs.
    create_matching_question_configs in run_pipeline()). Simplifying
    assumption: each run_pipeline() call is scoped to exactly one question
    type today, so this is a simple either/or, never a merge.
    """
    container_id = state.get("container_id")
    branch = "matching" if state.get("matching_question_configs") else "mcq"
    _log_with_container("info", f"Routing from start to '{branch}' chain", container_id=container_id)
    return branch


def _check_plausibility(question: Dict[str, Any], payload: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """Check if distractors are plausible to learners with partial knowledge."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    distractors = [a.get('answerText', '') for a in question.get('answer', []) if not a.get('isCorrect')]
    
    prompt = DISTRACTOR_PLAUSIBILITY_CHECK_PROMPT.format(
        question_text=question.get('questionText', ''),
        distractors=json.dumps(distractors, indent=2),
        plausibility_threshold=CHECK_THRESHOLDS["plausibility"]
    )
    
    return _call_llm_for_validation(
        system_message=VALIDATION_SYSTEM_MESSAGE.format(validation_check_role="distractor plausibility"),
        user_prompt=prompt,
        check_name="plausibility_check",
        threshold=CHECK_THRESHOLDS["plausibility"],
        pipeline_config=pipeline_config,
        call_context="plausibility_check",
        container_id=container_id
    )


def _check_misconception_representation(question: Dict[str, Any], payload: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """Check if each distractor represents a different misconception category."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    distractors = [
        {"text": a.get('answerText', ''), "explanation": a.get('explanation', '')}
        for a in question.get('answer', []) if not a.get('isCorrect')
    ]
    
    prompt = DISTRACTOR_MISCONCEPTION_CHECK_PROMPT.format(
        question_text=question.get('questionText', ''),
        distractors=json.dumps(distractors, indent=2),
        misconception_threshold=CHECK_THRESHOLDS["misconception"]
    )
    
    return _call_llm_for_validation(
        system_message=VALIDATION_SYSTEM_MESSAGE.format(validation_check_role="misconception diversity"),
        user_prompt=prompt,
        check_name="misconception_check",
        threshold=CHECK_THRESHOLDS["misconception"],
        pipeline_config=pipeline_config,
        call_context="misconception_check",
        container_id=container_id
    )


def _check_distinctiveness(question: Dict[str, Any], payload: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """Check if all options are sufficiently distinct from each other."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    all_options = [a.get('answerText', '') for a in question.get('answer', [])]
    
    prompt = DISTRACTOR_DISTINCTIVENESS_CHECK_PROMPT.format(
        question_text=question.get('questionText', ''),
        all_options=json.dumps(all_options, indent=2),
        distinctiveness_threshold=CHECK_THRESHOLDS["distinctiveness"]
    )
    
    return _call_llm_for_validation(
        system_message=VALIDATION_SYSTEM_MESSAGE.format(validation_check_role="option distinctiveness"),
        user_prompt=prompt,
        check_name="distinctiveness_check",
        threshold=CHECK_THRESHOLDS["distinctiveness"],
        pipeline_config=pipeline_config,
        call_context="distinctiveness_check",
        container_id=container_id
    )


def _check_incorrectness(question: Dict[str, Any], payload: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """Check if incorrect options are genuinely incorrect in the given context."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    all_options = [
        {"text": a.get('answerText', ''), "isCorrect": a.get('isCorrect', False)}
        for a in question.get('answer', [])
    ]
    
    prompt = DISTRACTOR_INCORRECTNESS_CHECK_PROMPT.format(
        question_text=question.get('questionText', ''),
        learning_objective=payload.get('learningObjective', ''),
        all_options=json.dumps(all_options, indent=2),
        incorrectness_threshold=CHECK_THRESHOLDS["incorrectness"]
    )
    
    return _call_llm_for_validation(
        system_message=VALIDATION_SYSTEM_MESSAGE.format(validation_check_role="answer correctness"),
        user_prompt=prompt,
        check_name="incorrectness_check",
        threshold=CHECK_THRESHOLDS["incorrectness"],
        pipeline_config=pipeline_config,
        call_context="incorrectness_check",
        container_id=container_id
    )


def _check_clarity(question: Dict[str, Any], payload: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """Check if all options are clearly worded and unambiguous."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    all_options = [a.get('answerText', '') for a in question.get('answer', [])]
    
    prompt = DISTRACTOR_CLARITY_CHECK_PROMPT.format(
        all_options=json.dumps(all_options, indent=2),
        clarity_threshold=CHECK_THRESHOLDS["clarity"]
    )
    
    return _call_llm_for_validation(
        system_message=VALIDATION_SYSTEM_MESSAGE.format(validation_check_role="clarity and precision"),
        user_prompt=prompt,
        check_name="clarity_check",
        threshold=CHECK_THRESHOLDS["clarity"],
        pipeline_config=pipeline_config,
        call_context="clarity_check",
        container_id=container_id
    )

def _check_parallelism_length(question: Dict[str, Any], payload: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """Check if options have parallel structure and similar length."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    all_options = [a.get('answerText', '') for a in question.get('answer', [])]
    difficulty = payload.get('difficultyLevel', 'Intermediate')
    
    prompt = DISTRACTOR_PARALLELISM_LENGTH_CHECK_PROMPT.format(
        all_options=json.dumps(all_options, indent=2),
        difficulty=difficulty,
        parallelism_threshold=CHECK_THRESHOLDS["parallelism"]
    )
    
    return _call_llm_for_validation(
        system_message=VALIDATION_SYSTEM_MESSAGE.format(validation_check_role="formatting consistency"),
        user_prompt=prompt,
        check_name="parallelism_check",
        threshold=CHECK_THRESHOLDS["parallelism"],
        pipeline_config=pipeline_config,
        call_context="parallelism_check",
        container_id=container_id
    )




def _check_locale_compliance(question: Dict[str, Any], payload: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """Check if question content strictly follows the specified locale."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    
    # Get target locale from payload or default
    target_locale = payload.get('locale', 'en-US')
    target_language = LOCALE_MAPPING.get(target_locale, 'English_US')
    
    # Extract all text content for validation
    question_text = question.get('questionText', '')
    all_options = [a.get('answerText', '') for a in question.get('answer', [])]
    
    prompt = LOCALE_COMPLIANCE_CHECK_PROMPT.format(
        target_locale=target_locale,
        target_language=target_language,
        question_text=question_text,
        options=json.dumps(all_options, indent=2),
        locale_threshold=CHECK_THRESHOLDS["locale_compliance"]
    )
    
    return _call_llm_for_validation(
        system_message=VALIDATION_SYSTEM_MESSAGE.format(validation_check_role="locale and cultural compliance"),
        user_prompt=prompt,
        check_name="locale_compliance_check",
        threshold=CHECK_THRESHOLDS["locale_compliance"],
        pipeline_config=pipeline_config,
        call_context="locale_compliance_check",
        container_id=container_id
    )

# Configuration for all validation agents: (agent_number, check_name, check_function, return_key)
VALIDATION_AGENTS: Dict[str, Tuple[int, str, Callable, str]] = {
    "plausibility": (1, "plausibility", _check_plausibility, "validation_check_plausibility"),
    "misconception": (2, "misconception", _check_misconception_representation, "validation_check_misconception"),
    "distinctiveness": (3, "distinctiveness", _check_distinctiveness, "validation_check_distinctiveness"),
    "incorrectness": (4, "incorrectness", _check_incorrectness, "validation_check_incorrectness"),
    "clarity": (5, "clarity", _check_clarity, "validation_check_clarity"),
    "parallelism": (6, "parallelism", _check_parallelism_length, "validation_check_parallelism"),
    "locale_compliance": (7, "locale compliance", _check_locale_compliance, "validation_check_locale_compliance"),
}


def _run_validation_agent(
    state: GraphState,
    agent_number: int,
    check_name: str,
    check_function: Callable,
    return_key: str
) -> Dict[str, Any]:
    """
    Generic validation agent that runs any check function.
    
    All validation functions must have signature: (question, payload)
    
    Args:
        state: Current pipeline state
        agent_number: Agent number for logging (1-7)
        check_name: Name of the check for logging
        check_function: Specific validation function to run
        return_key: Key to use in returned dictionary
        
    Returns:
        Dict with validation results under the specified return_key
    """
    items_to_validate = state.get("distractor_corrections_output", state["raw_outputs"])
    container_id = state.get("container_id")
    _log_with_container("info", f"Running validation agent : {check_name}", container_id=container_id)
    # Reduce individual agent verbosity - results shown in summary
    results = []
    
    for item in items_to_validate:
        item_result = {"payload": item["payload"], "checks": {}}
        if "error" in item.get("result", {}):
            results.append(item_result)
            continue
            
        questions = item.get("result", {}).get("questions", [])
        for idx, question in enumerate(questions):
            if not isinstance(question, dict):
                continue

            check_result = check_function(question, item["payload"], state.get("pipeline_config"), container_id)
            item_result["checks"][idx] = check_result

        results.append(item_result)

    return {return_key: results}


def agent_check_plausibility(state: GraphState) -> Dict[str, Any]:
    """Agent 1: Check distractor plausibility."""
    agent_num, check_name, check_func, return_key = VALIDATION_AGENTS["plausibility"]
    return _run_validation_agent(state, agent_num, check_name, check_func, return_key)


def agent_check_misconception(state: GraphState) -> Dict[str, Any]:
    """Agent 2: Check misconception representation."""
    agent_num, check_name, check_func, return_key = VALIDATION_AGENTS["misconception"]
    return _run_validation_agent(state, agent_num, check_name, check_func, return_key)


def agent_check_distinctiveness(state: GraphState) -> Dict[str, Any]:
    """Agent 3: Check option distinctiveness."""
    agent_num, check_name, check_func, return_key = VALIDATION_AGENTS["distinctiveness"]
    return _run_validation_agent(state, agent_num, check_name, check_func, return_key)


def agent_check_incorrectness(state: GraphState) -> Dict[str, Any]:
    """Agent 4: Check answer incorrectness."""
    agent_num, check_name, check_func, return_key = VALIDATION_AGENTS["incorrectness"]
    return _run_validation_agent(state, agent_num, check_name, check_func, return_key)


def agent_check_clarity(state: GraphState) -> Dict[str, Any]:
    """Agent 5: Check option clarity."""
    agent_num, check_name, check_func, return_key = VALIDATION_AGENTS["clarity"]
    return _run_validation_agent(state, agent_num, check_name, check_func, return_key)


def agent_check_parallelism(state: GraphState) -> Dict[str, Any]:
    """Agent 6: Check parallelism and length parity."""
    agent_num, check_name, check_func, return_key = VALIDATION_AGENTS["parallelism"]
    return _run_validation_agent(state, agent_num, check_name, check_func, return_key)




def agent_check_locale_compliance(state: GraphState) -> Dict[str, Any]:
    """Agent 7: Check locale compliance - content follows target locale standards."""
    agent_num, check_name, check_func, return_key = VALIDATION_AGENTS["locale_compliance"]
    return _run_validation_agent(state, agent_num, check_name, check_func, return_key)


def aggregate_validation_results(state: GraphState) -> GraphState:
    """Collect results from all 8 parallel validation agents and apply quality evaluation."""
    items_to_validate = state.get("distractor_corrections_output", state["raw_outputs"])

    # Check if we're processing retry (only failed questions)
    is_retry = "distractor_corrections_output" in state
    retry_info = " (RETRY: failed questions only)" if is_retry else ""
    container_id = state.get("container_id")
    
    _log_with_container("info", f"Aggregating validation results{retry_info}", container_id=container_id)
    
    # OPTIMIZATION: Pre-fetch all validation results once with structured access
    validation_types = [
        ("plausibility", state.get("validation_check_plausibility", [])),
        ("misconception", state.get("validation_check_misconception", [])),
        ("distinctiveness", state.get("validation_check_distinctiveness", [])),
        ("incorrectness", state.get("validation_check_incorrectness", [])),
        ("clarity", state.get("validation_check_clarity", [])),
        ("parallelism", state.get("validation_check_parallelism", [])),
        ("locale_compliance", state.get("validation_check_locale_compliance", []))
    ]

    # OPTIMIZATION: Cache thresholds to avoid repeated CHECK_THRESHOLDS lookups
    threshold_cache = {check_name: CHECK_THRESHOLDS.get(check_name, 80) 
                      for check_name, _ in validation_types}

    # OPTIMIZATION: Pre-compute agent summary more efficiently
    agent_summary = []
    for check_name, results in validation_types:
        if results:
            # More efficient counting using cached thresholds
            threshold = threshold_cache[check_name]
            passed_count = sum(1 for result in results 
                             if any(q.get("score", 0) >= threshold 
                                   for q in result.get("checks", {}).values()))
            total_count = len(results)
            status_icon = "PASS" if passed_count == total_count else "FAIL"
            agent_summary.append(f"{check_name[:4].upper()}:{passed_count}/{total_count}{status_icon}")


    validated_items: List[Dict[str, Any]] = []
    global_question_counter = 1
    
    # OPTIMIZATION: Track action counts during processing (single pass)
    action_counts = {"ACCEPT": 0, "CORRECT": 0}
    
    for item_idx, item in enumerate(items_to_validate):
        payload = item["payload"]
        result = item["result"]
        
        if "error" in result:
            validated_items.append(item)
            continue
            
        questions = result.get("questions", [])
        if not isinstance(questions, list):
            validated_items.append(item)
            continue
        
        validated_questions: List[Dict[str, Any]] = []
        
        # OPTIMIZATION: Pre-fetch validation checks for this item once
        item_validation_data = {}
        default_check = {"score": 0, "pass": False, "feedback": "Missing"}
        
        for check_name, results in validation_types:
            if item_idx < len(results):
                item_checks = results[item_idx].get("checks", {})
                item_validation_data[check_name] = item_checks
            else:
                item_validation_data[check_name] = {}
        
        for question_idx, question in enumerate(questions):
            display_question_num = global_question_counter
            if not isinstance(question, dict):
                validated_questions.append(question)
                continue
            
            # OPTIMIZATION: Build validation_checks using pre-fetched data
            validation_checks = {}
            for check_name, item_checks in item_validation_data.items():
                validation_checks[check_name] = item_checks.get(question_idx, default_check)
            
            # Get attempt number for logging
            attempt_num = state.get("distractor_correction_attempt", 0) + 1

            _log_with_container("info", f"Question {display_question_num} (Attempt {attempt_num}) validation:", container_id=container_id)
            _log_with_container("info", f"{'Check':<18} {'Score':<5} {'Status':<6} {'Threshold':<9} {'Gap':<4}", container_id=container_id)
            _log_with_container("info", "-" * 50, container_id=container_id)

            failed_checks = {}
            passed_checks = {}

            for metric, check_result in validation_checks.items():
                llm_score = check_result.get("score", 0)
                threshold = threshold_cache[metric]  # Use cached threshold
                feedback = check_result.get("feedback", "No feedback provided")

                if llm_score >= threshold:
                    passed_checks[metric] = {
                        "score": llm_score,
                        "threshold": threshold,
                        "feedback": feedback
                    }
                    status = "PASS"
                    gap = f"+{llm_score - threshold}"
                    status_icon = "PASS"
                else:
                    failed_checks[metric] = feedback
                    status = "FAIL"
                    gap = f"-{threshold - llm_score}"
                    status_icon = "FAIL"

                check_name = metric.upper()[:15]  # Truncate long names
                _log_with_container("info", f"{status_icon} {check_name:<15} {llm_score:<5} {status:<6} {threshold:<9} {gap:<4}", container_id=container_id)

            _log_with_container("info", "", container_id=container_id)  # Empty line for readability
            
            isAllChecksPassed = len(failed_checks) == 0
            
            evaluation_result = evaluate_question_with_weighted_scores(validation_checks)
            correction_action = determine_correction_action(
                evaluation_result,
                state.get("distractor_correction_attempt", 0)
            )
            
            # OPTIMIZATION: Count actions during processing (no second pass needed)
            action_counts[correction_action] = action_counts.get(correction_action, 0) + 1
            
            # Final summary
            num_failed = len(failed_checks)
            num_passed = len(passed_checks)
            total_checks = num_failed + num_passed

            if isAllChecksPassed:
                _log_with_container("info", f"SUCCESS: RESULT: {num_passed}/{total_checks} PASSED | Overall: {evaluation_result['overall_score']:.1f}/100 → {correction_action}", container_id=container_id)
            else:
                _log_with_container("warning", f"WARNING: RESULT: {num_failed}/{total_checks} FAILED | Overall: {evaluation_result['overall_score']:.1f}/100 → {correction_action}", container_id=container_id)
                # Show which specific checks failed
                failed_names = [name.upper()[:12] for name in failed_checks.keys()]
                _log_with_container("info", f"   Failed checks: {', '.join(failed_names)}", container_id=container_id)

            question["distractor_validation"] = {
                "passed": isAllChecksPassed,
                "checks": validation_checks,
                "failed_checks": failed_checks,
                "question_index": question_idx,
                "evaluation_result": evaluation_result,
                "correction_action": correction_action
            }
            
            validated_questions.append(question)
            
            # Legacy logging - now handled in summary above
            if not isAllChecksPassed:
                _log_with_container("info", f"Question {display_question_num}: Failed {len(failed_checks)}/7 checks, action: {correction_action}", container_id=container_id)
            
            # Increment global counter after processing each question
            global_question_counter += 1
        
        result["questions"] = validated_questions
        validated_items.append({"payload": payload, "result": result})
    
    state["validated_outputs"] = validated_items

    # OPTIMIZATION: Use pre-computed action counts (no additional iteration)
    total_questions = global_question_counter - 1  # Subtract 1 since we started at 1
    questions_needing_correction = action_counts.get("CORRECT", 0)

    if total_questions > 0:
        action_summary = [f"{action}:{count}" for action, count in action_counts.items() if count > 0]
        _log_with_container("info", f"VALIDATION COMPLETE: {total_questions} questions | {' | '.join(action_summary)}", container_id=container_id)

        # Only show next step if correction is needed
        if questions_needing_correction > 0:
            _log_with_container("info", f"CORRECTION NEEDED: {questions_needing_correction} questions need fixing", container_id=container_id)
        else:
            _log_with_container("info", f"SUCCESS: All questions acceptable - proceed to finalization", container_id=container_id)
    else:
        _log_with_container("info", "No questions to validate", container_id=container_id)

    # These collections are never accessed again after this point and consume significant memory
    _clear_state_keys(state, VALIDATION_AGENT_KEYS, "validation agent outputs")

    return state




def _correct_failed_question_with_llm(question: Dict[str, Any], payload: Dict[str, Any], failed_checks: Dict[str, str], all_checks: Dict[str, Any], pipeline_config: PipelineConfig = None, container_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Apply LLM-based correction to a question with failed distractor validation checks.

    Args:
        question: The question dictionary with failed distractors
        payload: The payload containing question configuration
        failed_checks: Dictionary of failed checks with feedback
        all_checks: Dictionary of all validation checks (passed and failed)

    Returns:
        Corrected question dictionary or None if correction failed
    """
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    try:
        # Prepare answer count information for prompts
        answers = question.get("answer", [])
        total_answers = len(answers)
        correct_answers = sum(1 for a in answers if a.get("isCorrect", False))
        incorrect_answers = total_answers - correct_answers

        # OPTIMIZATION: Cache thresholds for correction feedback
        correction_thresholds = {check_name: CHECK_THRESHOLDS.get(check_name, 80) 
                               for check_name in set(list(failed_checks.keys()) + list(all_checks.keys()))}
        
        # Format failed check feedback
        failed_feedback_lines = []
        for check_name, feedback in failed_checks.items():
            threshold = correction_thresholds[check_name]
            score = all_checks.get(check_name, {}).get("score", 0)
            gap = threshold - score
            failed_feedback_lines.append(f"- {check_name.upper()} (Score: {score}/{threshold}, Gap: -{gap}): {feedback}")

        # Format passed check feedback
        passed_feedback_lines = []
        for check_name, check_result in all_checks.items():
            if check_name not in failed_checks:
                score = check_result.get("score", 0)
                threshold = correction_thresholds[check_name]
                if score >= threshold:
                    feedback = check_result.get("feedback", "Passed validation")
                    passed_feedback_lines.append(f"- {check_name.upper()} (Score: {score}/{threshold}): {feedback}")

        all_feedback = "\n".join(failed_feedback_lines) if failed_feedback_lines else "No failed checks"
        passed_feedback = "\n".join(passed_feedback_lines) if passed_feedback_lines else "No passed checks"
        
        # Get locale for correction prompts
        locale_code = payload.get('locale', 'en-US')
        locale_name = LOCALE_MAPPING.get(locale_code, 'English_US')
        
        # Prepare correction prompts with dynamic answer counts and locale
        system_prompt = DISTRACTOR_BATCH_CORRECTION_SYSTEM_PROMPT.format(
            expected_incorrect=incorrect_answers,
            expected_correct=correct_answers,
            expected_total=total_answers,
            questions_locale=locale_name
        )

        user_prompt = DISTRACTOR_BATCH_CORRECTION_USER_PROMPT.format(
            original_question=json.dumps(question, indent=2),
            all_feedback=all_feedback,
            passed_feedback=passed_feedback,
            questions_locale=locale_name
        )

        # Call LLM for correction
        response = _call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            task_type="correction",
            pipeline_config=pipeline_config,
            call_context="distractor_correction",
            container_id=container_id
        )

        content = response.choices[0].message.content or ""

        # Parse the corrected question
        corrected = _parse_json(content)

        if isinstance(corrected, dict) and "answer" in corrected:
            # Validate corrected answer counts
            corrected_answers = corrected.get("answer", [])
            corrected_total = len(corrected_answers)
            corrected_correct = sum(1 for a in corrected_answers if a.get("isCorrect", False))
            corrected_incorrect = corrected_total - corrected_correct

            if (corrected_total == total_answers and
                corrected_correct == correct_answers and
                corrected_incorrect == incorrect_answers):
                
                # Clean up large feedback variables to optimize memory
                del failed_feedback_lines, passed_feedback_lines, all_feedback, passed_feedback
                return corrected
            else:
                _log_with_container("warning",
                    f"LLM correction failed: Answer count mismatch. Expected {correct_answers} correct + {incorrect_answers} incorrect = {total_answers} total, got {corrected_correct} correct + {corrected_incorrect} incorrect = {corrected_total} total",
                    container_id=container_id
                )
                # Clean up large feedback variables before return
                del failed_feedback_lines, passed_feedback_lines, all_feedback, passed_feedback
                return None
        else:
            _log_with_container("warning", "LLM correction failed: Invalid response format", container_id=container_id)
            # Clean up large feedback variables before return  
            del failed_feedback_lines, passed_feedback_lines, all_feedback, passed_feedback
            return None

    except json.JSONDecodeError as e:
        _log_with_container("error", f"LLM correction failed: Invalid JSON response - {str(e)}", container_id=container_id)
        # Clean up large feedback variables before return
        try:
            del failed_feedback_lines, passed_feedback_lines, all_feedback, passed_feedback
        except:
            pass  # Variables may not exist if error occurred early
        return None
    except Exception as e:
        _log_with_container("error", f"LLM correction failed: Unexpected error - {str(e)}", container_id=container_id)
        # Clean up large feedback variables before return
        try:
            del failed_feedback_lines, passed_feedback_lines, all_feedback, passed_feedback
        except:
            pass  # Variables may not exist if error occurred early
        return None


def correct_failed_distractors(state: GraphState) -> GraphState:
    """Correct distractors that failed validation using appropriate strategy."""
    current_attempt = state.get("distractor_correction_attempt", 0) + 1
    max_attempts = MAX_DISTRACTOR_CORRECTION_ATTEMPTS
    
    state["distractor_correction_attempt"] = current_attempt
    container_id = state.get("container_id")
    _log_with_container("info", f"Starting distractor corrections (attempt {current_attempt}/{max_attempts})", container_id=container_id)
    
    validated_items = state["validated_outputs"]
    _log_with_container("info", f"DEBUG: Processing {len(validated_items)} validated items for segregation", container_id=container_id)

    # ===== OPTIMIZATION: SEPARATE PASSED FROM FAILED =====
    passed_items: List[Dict[str, Any]] = []  # Questions that passed - no re-validation needed
    failed_items_for_correction: List[Dict[str, Any]] = []  # Only failed questions for correction
    
    # Debug counters
    total_questions_processed = 0
    accept_questions_found = 0
    correct_questions_found = 0
    
    for item_idx, item in enumerate(validated_items):
        payload = item["payload"]
        result = item["result"]
        questions = result.get("questions", [])
        
        _log_with_container("info", f"DEBUG: Item {item_idx+1}: Processing {len(questions)} questions", container_id=container_id)
        
        passed_questions: List[Dict[str, Any]] = []
        failed_questions: List[Dict[str, Any]] = []
        
        for question_idx, question in enumerate(questions):
            total_questions_processed += 1
            validation_data = question.get("distractor_validation", {})
            correction_action = validation_data.get("correction_action", "ACCEPT")
            display_idx = validation_data.get("question_index", question_idx)
            
            _log_with_container("info", f"  DEBUG: Q{display_idx}: correction_action='{correction_action}'", container_id=container_id)

            # PASSED QUESTIONS: Add directly to passed list (no re-validation needed)
            if correction_action == "ACCEPT":
                clean_q = dict(question)
                if "distractor_validation" in clean_q:
                    del clean_q["distractor_validation"]
                passed_questions.append(clean_q)
                accept_questions_found += 1
                _log_with_container("info", f"  SUCCESS: Q{display_idx}: {correction_action} → Skip re-validation", container_id=container_id)
                continue
            
            if validation_data.get("passed", False):
                clean_q = dict(question)
                if "distractor_validation" in clean_q:
                    del clean_q["distractor_validation"]
                passed_questions.append(clean_q)
                accept_questions_found += 1
                _log_with_container("info", f"  SUCCESS: Q{display_idx}: Already passed → Skip re-validation", container_id=container_id)
                continue
            
            # FAILED QUESTIONS: Need correction and re-validation
            failed_checks = validation_data.get("failed_checks", {})
            all_checks = validation_data.get("checks", {})
            
            if not failed_checks:
                clean_q = dict(question)
                if "distractor_validation" in clean_q:
                    del clean_q["distractor_validation"]
                passed_questions.append(clean_q)
                accept_questions_found += 1
                _log_with_container("info", f"  SUCCESS: Q{display_idx}: No failed checks → Skip re-validation", container_id=container_id)
                continue
            
            # This question needs correction - perform LLM-based correction
            correct_questions_found += 1
            _log_with_container("info", f"  Q{display_idx}: Correcting failed distractors", container_id=container_id)
            
            # Prepare correction prompts with validation feedback
            corrected_question = _correct_failed_question_with_llm(question, payload, failed_checks, all_checks, state.get("pipeline_config"), container_id=container_id)
            
            if corrected_question:
                failed_questions.append(corrected_question)
                _log_with_container("info", f"  SUCCESS: Q{display_idx}: Applied LLM-based distractor corrections", container_id=container_id)
            else:
                # If correction failed, keep original for re-validation
                failed_questions.append(question)
                _log_with_container("warning", f"  WARNING: Q{display_idx}: LLM correction failed, keeping original", container_id=container_id)

        # ===== STORE RESULTS SEPARATELY =====
        if passed_questions:
            passed_result = {"questions": passed_questions}
            passed_items.append({"payload": payload, "result": passed_result})
            
        if failed_questions:
            failed_result = {"questions": failed_questions}
            failed_items_for_correction.append({"payload": payload, "result": failed_result})

    # ===== DEBUG: SEGREGATION SUMMARY BEFORE STORAGE =====
    _log_with_container("info", f"DEBUG: Segregation Summary:", container_id=container_id)
    _log_with_container("info", f"  Total questions processed: {total_questions_processed}", container_id=container_id)
    _log_with_container("info", f"  ACCEPT questions found: {accept_questions_found}", container_id=container_id)
    _log_with_container("info", f"  CORRECT questions found: {correct_questions_found}", container_id=container_id)    

    # ===== STORE SEPARATED RESULTS IN STATE =====
    existing_passed = state.get("passed_questions_final", [])
    
    existing_passed.extend(passed_items)
    state["passed_questions_final"] = existing_passed  # Preserve all accepted questions from all attempts
    
    total_passed_count = sum(len(item['result']['questions']) for item in existing_passed)
    _log_with_container("info", f"DEBUG: After extending - total passed questions: {total_passed_count}", container_id=container_id)
    
    state["distractor_corrections_output"] = failed_items_for_correction  # Only these get re-validated

    passed_count = sum(len(item['result']['questions']) for item in passed_items)
    failed_count = sum(len(item['result']['questions']) for item in failed_items_for_correction)

    _log_with_container("info", f"SUMMARY: Segregation: {passed_count} PASSED (final) | {failed_count} FAILED (retry needed)", container_id=container_id)
    if failed_count > 0:
        _log_with_container("info", f"Next: {failed_count} questions → correction + re-validation", container_id=container_id)
    _log_with_container("info", f"Corrected {len(failed_items_for_correction)} items with failed questions (attempt {current_attempt}/{max_attempts})", container_id=container_id)
    
    return state


def finalize_distractor_validation(state: GraphState) -> GraphState:
    container_id = state.get("container_id")
    _log_with_container("info", "SUCCESS: Finalizing distractor validation", container_id=container_id)
    
    # ===== COMBINE PASSED QUESTIONS + RE-VALIDATED CORRECTED QUESTIONS =====
    passed_items = state.get("passed_questions_final", [])  # Questions that were never re-validated
    validated_corrected_items = state.get("validated_outputs", [])  # Questions that were corrected and re-validated
    current_attempt = state.get("distractor_correction_attempt", 0)
    max_attempts = MAX_DISTRACTOR_CORRECTION_ATTEMPTS
    
    _log_with_container("info", "DEBUG: Final state before combining:", container_id=container_id)
    _log_with_container("info", f"  passed_questions_final items: {len(passed_items)}", container_id=container_id)
    _log_with_container("info", f"  validated_outputs items: {len(validated_corrected_items)}", container_id=container_id)
    _log_with_container("info", f"  Current attempt: {current_attempt}/{max_attempts}", container_id=container_id)
    
    # Count questions in each source
    passed_questions_count = sum(len(item['result']['questions']) for item in passed_items)
    validated_questions_count = sum(len(item['result']['questions']) for item in validated_corrected_items)
    
    _log_with_container("info", f"  passed_questions_final questions: {passed_questions_count}", container_id=container_id)
    _log_with_container("info", f"  validated_outputs questions: {validated_questions_count}", container_id=container_id)
    
    
    # SIMPLIFIED APPROACH: validated_corrected_items contains ALL retry results (passed + failed)
    # No need to check distractor_corrections_output as it's just the input to the retry
    # The retry process outputs ALL results to validated_corrected_items
    
    _log_with_container("info", "Combining final results...", container_id=container_id)
    _log_with_container("info", f"  Passed originally: {sum(len(item['result']['questions']) for item in passed_items)}", container_id=container_id)
    _log_with_container("info", f"  Re-validated after correction: {sum(len(item['result']['questions']) for item in validated_corrected_items)}", container_id=container_id)
    
    # OPTIMIZATION: Count validation results and clean metadata in single pass
    cleaned_corrected_items: List[Dict[str, Any]] = []
    final_corrected_questions_passed = 0
    final_corrected_questions_failed = 0
    
    for item in validated_corrected_items:
        result = item["result"]
        questions = result.get("questions", [])
        cleaned_questions = []
        
        for question in questions:
            validation = question.get("distractor_validation", {})
            passed = validation.get("passed", False)
            question_idx = validation.get("question_index", "?")
            
            # Count validation results in single pass
            if passed:
                final_corrected_questions_passed += 1
                _log_with_container("info", f"  Final Q{question_idx}: PASSED after correction → Including in output", container_id=container_id)
            else:
                final_corrected_questions_failed += 1
                _log_with_container("info", f"  Final Q{question_idx}: FAILED after correction → Including in output anyway (per requirement)", container_id=container_id)
            
            # Remove validation metadata and include ALL questions (passed AND failed)
            clean_question = dict(question)
            if "distractor_validation" in clean_question:
                del clean_question["distractor_validation"]
            cleaned_questions.append(clean_question)
        
        # Create new item with cleaned questions
        cleaned_item = {
            "payload": item["payload"],
            "result": {"questions": cleaned_questions}
        }
        cleaned_corrected_items.append(cleaned_item)

    _log_with_container("info", f"DEBUG: Corrected questions after final retry:", container_id=container_id)
    _log_with_container("info", f"  Passed after correction: {final_corrected_questions_passed}", container_id=container_id)
    _log_with_container("info", f"  Failed after correction: {final_corrected_questions_failed}", container_id=container_id)
    _log_with_container("info", f"  Total corrected questions: {final_corrected_questions_passed + final_corrected_questions_failed}", container_id=container_id)
    _log_with_container("info", f"  ALL corrected questions will be included in final output", container_id=container_id)

    # Combine ALL final questions (passed + corrected_passed + corrected_failed)
    improved_items = passed_items + cleaned_corrected_items

    # Count total questions in final output for verification
    total_output_questions = sum(len(item['result']['questions']) for item in improved_items)

    _log_with_container("info", f"DEBUG: Final combination:", container_id=container_id)
    _log_with_container("info", f"  passed_items: {len(passed_items)} items", container_id=container_id)
    _log_with_container("info", f"  cleaned_corrected_items: {len(cleaned_corrected_items)} items", container_id=container_id)
    _log_with_container("info", f"  Total final items: {len(improved_items)} items", container_id=container_id)

    # Use passed + corrected as the finalization baseline
    input_to_finalization_count = passed_questions_count + validated_questions_count
    if input_to_finalization_count > 0:
        _log_with_container("info", f"DEBUG: Verification against finalization inputs:", container_id=container_id)
        _log_with_container("info", f"  Questions entering finalization: {input_to_finalization_count}", container_id=container_id)
        _log_with_container("info", f"  Questions in final output: {total_output_questions}", container_id=container_id)

        if total_output_questions != input_to_finalization_count:
            _log_with_container("error", f"CRITICAL: Question count mismatch during finalization!", container_id=container_id)
            _log_with_container("error", f"  Entered finalization: {input_to_finalization_count}", container_id=container_id)
            _log_with_container("error", f"  Final output: {total_output_questions}", container_id=container_id)
            _log_with_container("error", f"  Lost: {input_to_finalization_count - total_output_questions}", container_id=container_id)
        else:
            _log_with_container("info", f"✓ Question count verified: {total_output_questions}/{input_to_finalization_count} questions preserved", container_id=container_id)
    
    # Use consistent clearing method
    _clear_state_keys(state, ["improved_outputs"], "improved outputs")
    state["improved_outputs"] = improved_items
    
    # ===== FINAL VALIDATION STATISTICS =====
    total_passed_originally = sum(len(item['result']['questions']) for item in passed_items)
    total_corrected_questions = final_corrected_questions_passed + final_corrected_questions_failed
    total_questions = total_passed_originally + total_corrected_questions
    final_passed = total_passed_originally + final_corrected_questions_passed
    final_failed = final_corrected_questions_failed

    # Final summary
    if total_questions > 0:
        final_pass_rate = (final_passed / total_questions) * 100
        _log_with_container("info", f"FINAL SUMMARY:", container_id=container_id)
        _log_with_container("info", f"  Originally passed: {total_passed_originally}", container_id=container_id)
        _log_with_container("info", f"  Corrected and passed: {final_corrected_questions_passed}", container_id=container_id)
        _log_with_container("info", f"  Corrected but failed: {final_corrected_questions_failed}", container_id=container_id)
        _log_with_container("info", f"  Total processed: {total_questions}", container_id=container_id)
        _log_with_container("info", f"  Total in final output: {total_output_questions}", container_id=container_id)
        _log_with_container("info", f"  Quality pass rate: {final_pass_rate:.1f}%", container_id=container_id)
        
        if final_failed > 0:
            _log_with_container("info", f"NOTE: {final_failed} questions failed quality checks but are included in output for downstream processing", container_id=container_id)
        
        if total_output_questions != total_questions:
            _log_with_container("error", f"CRITICAL: Output count mismatch! Expected {total_questions}, got {total_output_questions}", container_id=container_id)
        else:
            _log_with_container("info", f"✓ All {total_questions} questions successfully included in final output", container_id=container_id)
    else:
        _log_with_container("info", "No questions to validate", container_id=container_id)
    
    _log_with_container("info", "SUCCESS: Distractor validation complete", container_id=container_id)
    
    # Log validation phase token usage
    token_tracker = get_token_tracker()
    totals = token_tracker.pipeline_totals
    _log_with_container("info", f"TOKEN UPDATE: Validation Complete | Total: {totals['input'] + totals['output']:,} tokens | Calls: {totals['calls']}", container_id=container_id)
    
    _clear_state_keys(state, DISTRACTOR_VALIDATION_KEYS, "distractor validation data")
    
    return state


def _route_after_distractor_validation(state: GraphState) -> str:
    """
    Route after parallel distractor validation using quality-based correction actions.
    """
    container_id = state.get("container_id")
    _log_with_container("info", f"DEBUG: _route_after_distractor_validation called", container_id=container_id)
    
    validated_items = state.get("validated_outputs", [])
    current_attempt = state.get("distractor_correction_attempt", 0)
    max_attempts = MAX_DISTRACTOR_CORRECTION_ATTEMPTS
    
    _log_with_container("info", f"DEBUG: State keys available: {list(state.keys())}", container_id=container_id)
    
    actions = []
    for item in validated_items:
        questions = item.get("result", {}).get("questions", [])
        for question in questions:
            validation = question.get("distractor_validation", {})
            action = validation.get("correction_action", "ACCEPT")
            actions.append(action)
    
    needs_correction = actions.count("CORRECT")
    acceptable = actions.count("ACCEPT")
    
    # Parallel retry approach - check failed questions from segregation
    failed_items = state.get("distractor_corrections_output", [])
    total_failed_questions = sum(len(item["result"]["questions"]) for item in failed_items)
    
    # DEBUG: Show routing decision variables
    _log_with_container("info", f"DEBUG: Routing variables:", container_id=container_id)
    _log_with_container("info", f"  needs_correction (from actions): {needs_correction}", container_id=container_id)
    _log_with_container("info", f"  acceptable (from actions): {acceptable}", container_id=container_id)
    _log_with_container("info", f"  distractor_corrections_output items: {len(failed_items)}", container_id=container_id)
    _log_with_container("info", f"  total_failed_questions: {total_failed_questions}", container_id=container_id)
    
    # Routing decision for parallel retry approach
    _log_with_container("info", f"Routing: Attempt {current_attempt}/{max_attempts} | Need correction: {needs_correction} questions", container_id=container_id)

    # First check if current validation shows any corrections needed
    if needs_correction == 0:
        _log_with_container("info", "SUCCESS: All questions passed → FINALIZE", container_id=container_id)
        return "finalize"
    
    if current_attempt >= max_attempts:
        _log_with_container("warning", f"WARNING: Max retry attempts reached ({max_attempts}) → FINALIZE", container_id=container_id)
        _log_with_container("info", f"   Final result: {acceptable} passed, {needs_correction} still need correction", container_id=container_id)
        _log_with_container("info", f"   All questions (passed + failed) will be included in final output", container_id=container_id)
        return "finalize"
    
    # Start/continue retry loop
    _log_with_container("info", f"Starting retry attempt {current_attempt + 1}/{max_attempts} for {total_failed_questions} questions", container_id=container_id)
    return "correct"


def evaluate_quality(result: Any, rubric: str, threshold: int, locale: str = 'en-US', pipeline_config: PipelineConfig = None, container_id: str = None) -> Dict[str, Any]:
    """
    Evaluate quality of generated content using LLM with improved JSON parsing.
    
    Expected LLM response format: {"score": 85, "pass": true, "issues": ["list of issues"]}
    """
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    if "error" in result:
        return {"score": 0, "issues": ["Invalid JSON"], "pass": False}
    
    # Get locale name for prompt
    locale_name = LOCALE_MAPPING.get(locale, 'English_US')
    
    try:
        prompt = USER_PROMPT_TEMPLATE_QUALITY_CHECK.format(
            rubric=rubric,
            questions=json.dumps(result),
            threshold=str(threshold),
            locale=locale_name
        )
        
        response = _call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_QUALITY_CHECK},
                {"role": "user", "content": prompt},
            ],
            task_type="quality_check",
            pipeline_config=pipeline_config,
            call_context="quality_check",
            container_id=container_id
        )
        
        content = response.choices[0].message.content or ""
        
        # Use the improved validation utility for quality checks
        return extract_and_parse_validation_json(
            content=content,
            check_name="quality_check",
            expected_format_type="quality",
            threshold=threshold
        )
        
    except Exception as e:
        _log_with_container("error", f"Error in quality evaluation: {str(e)}", container_id=container_id)
        return {
            "score": 0,
            "pass": False,
            "issues": [f"Quality evaluation error: {str(e)}"]
        }


def evaluate_relevancy(
    result: Any,
    learning_objectives: List[str],
    threshold: int,
    locale: str = 'en-US',
    pipeline_config: PipelineConfig = None,
    container_id: str = None,
) -> Dict[str, Any]:
    """
    Evaluate relevancy of generated content to learning objectives with improved JSON parsing.
    
    Expected LLM response format: {"score": 85, "pass": true, "issues": ["list of issues"]}
    """
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    if "error" in result:
        return {"score": 0, "issues": ["Result contains error"], "pass": False}
    
    # Get locale name for prompt
    locale_name = LOCALE_MAPPING.get(locale, 'English_US')
    
    try:
        prompt = USER_PROMPT_TEMPLATE_RELEVANCY_CHECK.format(
            learning_objectives=json.dumps(learning_objectives),
            questions=json.dumps(result),
            threshold=str(threshold),
            locale=locale_name
        )
        
        response = _call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_RELEVANCY_CHECK},
                {"role": "user", "content": prompt},
            ],
            task_type="quality_check",
            pipeline_config=pipeline_config,
            call_context="relevancy_check",
            container_id=container_id
        )
        
        content = response.choices[0].message.content or ""
        
        # Use the improved validation utility for relevancy checks
        return extract_and_parse_validation_json(
            content=content,
            check_name="relevancy_check",
            expected_format_type="quality",
            threshold=threshold
        )
        
    except Exception as e:
        _log_with_container("error", f"Error in relevancy evaluation: {str(e)}", container_id=container_id)
        return {
            "score": 0,
            "pass": False,
            "issues": [f"Relevancy evaluation error: {str(e)}"]
        }


def evaluate_quality_and_relevancy(
    result: Any,
    rubric: str,
    quality_threshold: int,
    learning_objectives: List[str],
    relevancy_threshold: int,
    locale: str = 'en-US',
    pipeline_config: PipelineConfig = None,
    container_id: str = None,
) -> tuple:
    """Combined quality + relevancy check — 1 LLM call instead of 2."""
    if pipeline_config is None:
        raise ValueError("pipeline_config is required")
    if "error" in result:
        fail = {"score": 0, "pass": False, "issues": ["Result contains error"]}
        return fail, fail

    locale_name = LOCALE_MAPPING.get(locale, 'English_US')
    try:
        prompt = USER_PROMPT_TEMPLATE_QUALITY_RELEVANCY_CHECK.format(
            rubric=rubric,
            learning_objectives=json.dumps(learning_objectives),
            questions=json.dumps(result),
            quality_threshold=str(quality_threshold),
            relevancy_threshold=str(relevancy_threshold),
            locale=locale_name
        )
        response = _call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_QUALITY_RELEVANCY_CHECK},
                {"role": "user", "content": prompt},
            ],
            task_type="quality_check",
            pipeline_config=pipeline_config,
            call_context="quality_relevancy_check",
            container_id=container_id
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_json(content)
        # Extract sub-objects; fall back to zero-score dicts if missing.
        def _extract(key, threshold):
            sub = parsed.get(key, {})
            score = float(sub.get("score", 0))
            return {
                "score": score,
                "pass": sub.get("pass", score >= threshold),
                "issues": sub.get("issues", [])
            }
        return _extract("quality", quality_threshold), _extract("relevancy", relevancy_threshold)
    except Exception as e:
        _log_with_container("error", f"Error in combined quality/relevancy evaluation: {e}", container_id=container_id)
        fail = {"score": 0, "pass": False, "issues": [str(e)]}
        return fail, fail


def _matching_column_values(column: List[Any]) -> set:
    """Extract the actual term/definition strings out of a MATCHING
    column_a_answers/column_b_answers list, whose entries are single-key
    dicts like {"Option-A": "Term 1"} - the key name is just a label, the
    value is what pairings in 'answers' actually reference. Values are
    stripped so trivial whitespace differences don't break the referential
    lookup in _is_valid_response_format.
    """
    values = set()
    for entry in column:
        if isinstance(entry, dict):
            for v in entry.values():
                if isinstance(v, str):
                    values.add(v.strip())
        elif isinstance(entry, str):
            values.add(entry.strip())
    return values


def _is_valid_response_format(result: Dict[str, Any], question_type: str = "MULTIPLE_CHOICE") -> Tuple[bool, List[str]]:
    """
    Validate response format against expected structure.
    Returns: (is_valid, list_of_errors)
    """
    errors = []

    # Check if result is dict
    if not isinstance(result, dict):
        errors.append(f"Result must be dict, got {type(result).__name__}")
        return False, errors

    # Check if "questions" key exists
    if "questions" not in result:
        errors.append("Missing 'questions' key")
        return False, errors

    questions = result.get("questions")

    # Check if questions is a list
    if not isinstance(questions, list):
        errors.append(f"'questions' must be list, got {type(questions).__name__}")
        return False, errors

    if len(questions) == 0:
        errors.append("'questions' list is empty")
        return False, errors

    if question_type == "MATCHING":
        # MATCHING has no 'answer'/'answerText'/'isCorrect' structure - it
        # pairs column_a_answers with column_b_answers via an 'answers' map.
        # Kept as its own branch rather than reusing the MCQ checks below.
        for idx, question in enumerate(questions):
            if not isinstance(question, dict):
                errors.append(f"Question {idx}: must be dict, got {type(question).__name__}")
                continue

            question_text = question.get("questionText") or question.get("question")
            if not question_text or not isinstance(question_text, str) or not question_text.strip():
                errors.append(f"Question {idx}: missing or invalid 'questionText'/'question'")

            column_a = question.get("column_a_answers")
            column_b = question.get("column_b_answers")
            pairings = question.get("answers")

            if not isinstance(column_a, list) or len(column_a) == 0:
                errors.append(f"Question {idx}: missing or empty 'column_a_answers'")
            if not isinstance(column_b, list) or len(column_b) == 0:
                errors.append(f"Question {idx}: missing or empty 'column_b_answers'")
            if not pairings:
                errors.append(f"Question {idx}: missing 'answers' pairings")
            elif not isinstance(pairings, list):
                errors.append(f"Question {idx}: 'answers' must be list, got {type(pairings).__name__}")
            elif (
                isinstance(column_a, list) and isinstance(column_b, list)
            ):
                # column_b may have more entries than column_a (unused
                # distractors are harmless) but never fewer - that would mean
                # some column_a term has no possible match at all. Every
                # column_a term must have exactly one pairing.
                if len(column_b) < len(column_a) or len(pairings) != len(column_a):
                    errors.append(
                        f"Question {idx}: mismatched lengths - column_a={len(column_a)}, "
                        f"column_b={len(column_b)} (must be >= column_a), "
                        f"answers={len(pairings)} (must equal column_a)"
                    )
                column_a_values = _matching_column_values(column_a)
                column_b_values = _matching_column_values(column_b)
                for pair_idx, pairing in enumerate(pairings):
                    if not isinstance(pairing, dict):
                        errors.append(f"Question {idx}, pairing {pair_idx}: must be dict, got {type(pairing).__name__}")
                        continue
                    a_val = pairing.get("column_a_answers")
                    b_val = pairing.get("column_b_answers")
                    a_val_norm = a_val.strip() if isinstance(a_val, str) else a_val
                    b_val_norm = b_val.strip() if isinstance(b_val, str) else b_val
                    if a_val_norm not in column_a_values:
                        errors.append(f"Question {idx}, pairing {pair_idx}: '{a_val}' not found in column_a_answers")
                    if b_val_norm not in column_b_values:
                        errors.append(f"Question {idx}, pairing {pair_idx}: '{b_val}' not found in column_b_answers")

        is_valid = len(errors) == 0
        return is_valid, errors

    # Validate each question
    for idx, question in enumerate(questions):
        if not isinstance(question, dict):
            errors.append(f"Question {idx}: must be dict, got {type(question).__name__}")
            continue
        
        # Check required fields in question (accept both old and new formats)
        question_text = question.get("questionText") or question.get("question")
        if not question_text:
            errors.append(f"Question {idx}: missing 'questionText' or 'question'")
        elif not isinstance(question_text, str) or not question_text.strip():
            errors.append(f"Question {idx}: question text must be non-empty string")
        
        # Check answer field (accept both "answer" and "answers")
        answers = question.get("answer") or question.get("answers")
        if not answers:
            errors.append(f"Question {idx}: missing 'answer' or 'answers' key")
            continue
        if not isinstance(answers, list):
            errors.append(f"Question {idx}: 'answer' must be list, got {type(answers).__name__}")
            continue
        
        if len(answers) == 0:
            errors.append(f"Question {idx}: 'answer' list is empty")
            continue
        
        # Validate each answer
        for ans_idx, answer in enumerate(answers):
            if not isinstance(answer, dict):
                errors.append(f"Question {idx}, Answer {ans_idx}: must be dict")
                continue
            
            # Check required answer fields (accept both old and new formats)
            answer_text = answer.get("answerText") or answer.get("answer")
            if not answer_text:
                errors.append(f"Question {idx}, Answer {ans_idx}: missing 'answerText' or 'answer'")
            elif not isinstance(answer_text, str) or not answer_text.strip():
                errors.append(f"Question {idx}, Answer {ans_idx}: answer text must be non-empty string")
            
            if "explanation" not in answer:
                errors.append(f"Question {idx}, Answer {ans_idx}: missing 'explanation'")
            elif not isinstance(answer.get("explanation"), str):
                errors.append(f"Question {idx}, Answer {ans_idx}: 'explanation' must be string")
            
            # Check correct flag (accept both "isCorrect" and "correct")
            is_correct = answer.get("isCorrect") if "isCorrect" in answer else answer.get("correct")
            if is_correct is None:
                errors.append(f"Question {idx}, Answer {ans_idx}: missing 'isCorrect' or 'correct'")
            elif not isinstance(is_correct, bool):
                errors.append(f"Question {idx}, Answer {ans_idx}: correct flag must be boolean")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def validate_and_fix_format(state: GraphState) -> GraphState:
    container_id = state.get("container_id")

    # Lightweight validation when using structured output - check business rules
    if USE_STRUCTURED_OUTPUT:
        _log_with_container("info", "Performing lightweight validation for structured output", container_id=container_id)
        questions = state.get("questions", [])
        
        # Basic business rule validation that Pydantic schemas don't cover
        for i, question in enumerate(questions):
            if not isinstance(question, dict):
                continue

            # Check for error keys that shouldn't reach database
            if "error" in question or "context" in question:
                _log_with_container("error", f"Question {i} contains error keys: {question}", container_id=container_id)
                state["error"] = f"Structured output validation failed: error keys found in question {i}"
                return state

            # Validate required fields
            if "questionText" not in question or not question.get("questionText"):
                _log_with_container("error", f"Question {i} missing questionText", container_id=container_id)
                state["error"] = f"Structured output validation failed: missing questionText in question {i}"
                return state

            if question.get("payload", {}).get("questionType") == "MATCHING":
                continue

            # Validate answer options count and structure
            answers = question.get("answer", [])
            if not isinstance(answers, list) or len(answers) < 3 or len(answers) > 6:
                _log_with_container("error", f"Question {i} has invalid answer count: {len(answers) if isinstance(answers, list) else 'not a list'}", container_id=container_id)
                state["error"] = f"Structured output validation failed: invalid answer count in question {i}"
                return state
                
            # Check that at least one answer is correct
            correct_answers = [ans for ans in answers if isinstance(ans, dict) and ans.get("isCorrect")]
            if not correct_answers:
                _log_with_container("error", f"Question {i} has no correct answers", container_id=container_id)
                state["error"] = f"Structured output validation failed: no correct answers in question {i}"
                return state
        
        _log_with_container("info", "Structured output passed lightweight validation", container_id=container_id)
        return state
        
    _log_with_container("info", "Validating output format", container_id=container_id)
    
    fixed: List[Dict[str, Any]] = []
    llm_fixed_count = 0
    isAllFormatsValid = True
    
    items_to_validate = state.get("formatted") or state.get("improved_outputs", [])
    
    for item in items_to_validate:
        payload = item["payload"]
        result = item["result"]
        question_type = payload.get("questionType", "MULTIPLE_CHOICE")
        
        is_valid, errors = _is_valid_response_format(result, question_type)
        
        if not is_valid:
            isAllFormatsValid = False
            error_msg = "; ".join(errors)
            _log_with_container("warning", f"Format issues detected: {error_msg}. Requesting LLM fix", container_id=container_id)
            
            locale_code = payload.get('locale', 'en-US')
            locale_name = LOCALE_MAPPING.get(locale_code, 'English_US')
            
            system_prompt = SYSTEM_PROMPT_TEMPLATE_FIX_FORMAT.format(locale=locale_name)
            user_prompt = USER_PROMPT_TEMPLATE_FIX_FORMAT.format(
                res_format=payload["responseFormat"],
                raw_output=json.dumps(result),
                locale=locale_name
            )
            
            try:
                response = _call_llm(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    task_type="format_fix",
                    pipeline_config=state.get("pipeline_config"),
                    call_context="format_validation_fix",
                    container_id=container_id
                )
                
                content = response.choices[0].message.content or ""
                result = _parse_json(content)
                
                # Validate the LLM-fixed output
                is_valid_after_fix, errors_after = _is_valid_response_format(result, question_type)
                if is_valid_after_fix:
                    llm_fixed_count += 1
                    _log_with_container("info", "LLM format fix successful", container_id=container_id)
                else:
                    _log_with_container("error", f"LLM fix incomplete: {'; '.join(errors_after)}. Using best effort", container_id=container_id)
            except json.JSONDecodeError:
                _log_with_container("error", "LLM failed to fix format (JSON parse error), using original", container_id=container_id)
            except Exception as e:
                _log_with_container("error", f"LLM format fix failed unexpectedly: {e}, using original", container_id=container_id)
        
        fixed.append({"payload": payload, "result": result})

    _clear_state_keys(state, ["improved_outputs"], "validation cleanup")
    state["improved_outputs"] = fixed
    
    state["format_fix_attempts"] = state.get("format_fix_attempts", 0) + 1
    max_attempts = state["input"].get("maxFormatFixAttempts", MAX_FORMAT_CORRECTION_ATTEMPTS)
    state["format_loop"] = (not isAllFormatsValid) and state["format_fix_attempts"] < max_attempts
    
    if llm_fixed_count > 0:
        _log_with_container("info", f"Format validation: {llm_fixed_count} outputs corrected by LLM, {len(fixed) - llm_fixed_count} already valid", container_id=container_id)
    else:
        _log_with_container("info", f"STEP 6 COMPLETE: Format validation | All {len(fixed)} outputs valid", container_id=container_id)
        
    return state


def validate_quality(state: GraphState) -> GraphState:
    """
    Validate items for quality and relevancy using LLM evaluation.
    
    DATA FLOW:
    - Reads from: state["quality_validation_failed"] (corrected items from correct_quality) OR 
                  state["improved_outputs"] (fresh items on initial pass)
    - Writes to: state["quality_validation_failed"] (newly failed items for correct_quality to fix)
    - Next step: Either correct_quality (if items failed) or continue pipeline (if all passed)
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Validating quality and relevancy", container_id=container_id)

    # Clean historical extraction across loop iterations
    already_passed = state.get("quality_validation_passed", [])
    
    # Fallback to improved_outputs on initial pass
    items_to_validate = state.get("quality_validation_failed")
    if items_to_validate is None or len(items_to_validate) == 0:
        items_to_validate = state.get("improved_outputs", [])
        
    rubric = state["input"].get("qualityRubric") or DEFAULT_QUALITY_RUBRIC.strip()
    threshold = state["input"].get("qualityThreshold", 85)
    relevancy_threshold = state["input"].get("relevancyThreshold", 85)
    learning_objectives = _learning_objective_descriptions(state["input"])

    passed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    
    for item in items_to_validate:
        payload = item["payload"]
        result = item["result"]
        locale = payload.get('locale', 'en-US')
        
        # Single combined LLM call instead of 2 sequential calls.
        quality, relevancy = evaluate_quality_and_relevancy(
            result, rubric, threshold, learning_objectives,
            relevancy_threshold, locale, state.get("pipeline_config"), container_id=container_id
        )
        
        entry = {
            "payload": payload,
            "result": result,
            "quality": quality,
            "relevancy": relevancy,
        }
        
        if quality.get("pass") and relevancy.get("pass"):
            passed.append(entry)
        else:
            failed.append(entry)
            
    # Accumulate all items that have successfully passed validation across all runs
    total_passed_accumulated = already_passed + passed
    
    state["quality_validation_passed"] = total_passed_accumulated
    state["quality_validation_failed"] = failed  # Overwrite with only what failed on THIS specific turn

    # On the final attempt, include failed items so they still reach output guardrail.
    max_attempts = state["input"].get("maxQualityFixAttempts", state["input"].get("maxAttempts", MAX_QUALITY_CORRECTION_ATTEMPTS))
    current_attempts = state.get("quality_correction_attempts", 0)
    is_final_pass = current_attempts >= max_attempts

    if is_final_pass and failed:
        _log_with_container("info", f"Final quality pass: forwarding {len(failed)} failed items to output trail.", container_id=container_id)
        state["improved_outputs"] = [
            {"payload": entry["payload"], "result": entry["result"]}
            for entry in total_passed_accumulated + failed
        ]
    else:
        state["improved_outputs"] = [
            {"payload": entry["payload"], "result": entry["result"]}
            for entry in total_passed_accumulated
        ]

    _log_with_container("info", f"Quality validation: {len(total_passed_accumulated)} passed out of total ecosystem, {len(failed)} failed on this turn", container_id=container_id)
    return state

def correct_quality(state: GraphState) -> GraphState:
    """
    Correct items that failed quality validation using LLM.
    
    DATA FLOW: 
    - Reads from: state["quality_validation_failed"] (items that failed validate_quality)
    - Writes to: state["quality_validation_failed"] (corrected items needing re-validation)
    - Next step: validate_quality re-validates the corrected items
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Correcting failed quality or relevancy", container_id=container_id)
    corrected_failed: List[Dict[str, Any]] = []
    
    for item in state.get("quality_validation_failed", []):
        payload = item["payload"]
        result = item["result"]
        quality = item.get("quality", {})
        relevancy = item.get("relevancy", {})
        failure_reasons: List[str] = []
        
        if not quality.get("pass"):
            issues = quality.get("issues") or []
            issue_text = "; ".join([issue if isinstance(issue, str) else json.dumps(issue) for issue in issues])
            failure_reasons.append("Quality issues: " + issue_text)
            
        if not relevancy.get("pass"):
            issues = relevancy.get("issues") or []
            issue_text = "; ".join([issue if isinstance(issue, str) else json.dumps(issue) for issue in issues])
            failure_reasons.append("Relevancy issues: " + issue_text)
            
        failure_reason_text = "\n".join([r for r in failure_reasons if r]) or "Failed evaluation."

        locale_code = payload.get('locale', 'en-US')
        locale_name = LOCALE_MAPPING.get(locale_code, 'English_US')

        correction_prompt = USER_PROMPT_QUALITY_CORRECTION.format(
            learning_obj=payload["learningObjective"],
            difficulty_level=payload["difficultyLevel"],
            question_type=payload["questionType"],
            failure_reasons=failure_reason_text,
            question_json=json.dumps(result),
            response_format=payload["responseFormat"],
            locale=locale_name
        )
        
        response = _call_llm(
            messages=[
                {"role": "system", "content": payload["systemPrompt"]},
                {"role": "user", "content": correction_prompt},
            ],
            task_type="quality_correction",
            pipeline_config=state.get("pipeline_config"),
            call_context="quality_correction",
            container_id=container_id
        )
        content = response.choices[0].message.content or ""
        try:
            corrected = _parse_json(content)
            corrected_failed.append({"payload": payload, "result": corrected})
        except json.JSONDecodeError as e:
            _log_with_container("warning", f"Quality correction failed - fallback to original output to prevent lockup: {e}", container_id=container_id)
            corrected_failed.append({"payload": payload, "result": result})
            
    state["quality_validation_failed"] = corrected_failed

    state["quality_correction_attempts"] = state.get("quality_correction_attempts", 0) + 1
    _log_with_container("info", f"Corrected {len(corrected_failed)} failed quality outputs", container_id=container_id)
    return state

def format_conversion(state: GraphState) -> GraphState:
    container_id = state.get("container_id")
    _log_with_container("info", "Formatting results", container_id=container_id)
    
    passed_items = state.get("quality_validation_passed", [])
    failed_items = state.get("quality_validation_failed", [])
    
    if not passed_items and not failed_items:
        items_to_format = state.get("improved_outputs", [])
    else:
        items_to_format = [
            {"payload": entry["payload"], "result": entry["result"]} for entry in passed_items
        ] + [
            {"payload": entry["payload"], "result": entry["result"]} for entry in failed_items
        ]
        
    _log_with_container("info", f"Formatting results for {len(items_to_format)} total outputs", container_id=container_id)
    formatted: List[Dict[str, Any]] = []
    
    for item in items_to_format:
        payload = item["payload"]
        result = item["result"]
        question_type = payload.get("questionType", "MULTIPLE_CHOICE")
        
        # FIX: Check if the incoming format is already completely valid
        is_valid, _ = _is_valid_response_format(result, question_type)
        
        if is_valid:
            # Safe Pass-Through: If the structure is already healthy, do not call the LLM!
            _log_with_container("info", "Result structure already clean. Applying safe pass-through.", container_id=container_id)
            parsed = result
        else:
            # Active Repair: Only call the LLM if the format is structurally broken
            _log_with_container("warning", "Result structure malformed. Requesting structural conversion.", container_id=container_id)
            locale_name = LOCALE_MAPPING.get(payload.get('locale', 'en-US'), 'English_US')
            
            system_prompt = SYSTEM_PROMPT_TEMPLATE_FIX_FORMAT.format(locale=locale_name)
            user_prompt = USER_PROMPT_TEMPLATE_FIX_FORMAT.format(
                res_format=payload["responseFormat"],
                raw_output=json.dumps(result),
                locale=locale_name
            )
            response = _call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                task_type="format_fix",
                pipeline_config=state.get("pipeline_config"),
                call_context="format_fix",
                container_id=container_id
            )
            content = response.choices[0].message.content or ""
            try:
                parsed = _parse_json_raw(content)
            except json.JSONDecodeError:
                parsed = {"error": "Invalid JSON from model", "raw": content}
            
        formatted.append(
            {
                "payload": payload,
                "learningObjective": payload["learningObjective"],
                "difficultyLevel": payload["difficultyLevel"],
                "questionType": payload["questionType"],
                "result": parsed,
            }
        )
    state["formatted"] = formatted
    _log_with_container("info", f"Formatted {len(formatted)} outputs", container_id=container_id)
    return state


def _route_from_validate_quality(state: GraphState) -> str:
    container_id = state.get("container_id")
    failed = state.get("quality_validation_failed", [])
    max_attempts = state["input"].get("maxQualityFixAttempts", state["input"].get("maxAttempts", MAX_QUALITY_CORRECTION_ATTEMPTS))
    attempts = state.get("quality_correction_attempts", 0)
    
    if failed and attempts < max_attempts:
        _log_with_container("info", f"Routing back to quality correction loop (Attempt {attempts + 1}/{max_attempts})", container_id=container_id)
        return "correct_quality"
        
    if failed:
        # State already updated in validate_quality node (routing functions can't mutate state).
        _log_with_container("info", f"Max quality correction attempts ({max_attempts}) reached. Failed items already forwarded in validate_quality.", container_id=container_id)

    return "end"

def _build_output(payload: PipelineInput, items: List[Dict[str, Any]], container_id: str = None) -> List[Dict[str, Any]]:
    """
    Build output in database format directly - no need for additional formatting.

    Returns format compatible with save_ai_generated_questions():
    [
        {
            "LearningObjective": "LO description",
            "LevelOfQuiz": "Intermediate",
            "questions": [
                {
                    "question": "What is...?",
                    "answers": [
                        {
                            "answer": "Option A",
                            "correct": true,
                            "explanation": "Because..."
                        }
                    ]
                }
            ]
        }
    ]
    """
    normalized_los = _normalize_learning_objectives(payload)

    # Create database format structure
    database_format: List[Dict[str, Any]] = []

    # Build LO structure with database field names
    for entry in normalized_los:
        lo_data = {
            "LearningObjective": entry.get("description", ""),
            "LevelOfQuiz": "Intermediate",  # Default level, can be overridden per question
            "questions": []
        }
        database_format.append(lo_data)

    # Process questions and add to appropriate LO
    for item in items:
        payload_item = item.get("payload", {})
        lo_text = payload_item.get("learningObjective") or item.get("learningObjective")

        # Get question type and difficulty from payload
        question_type = payload_item.get("questionType", "MULTIPLE_CHOICE")
        payload_difficulty = payload_item.get("difficultyLevel")  # This is the difficulty for this batch

        # Find matching LO in database format
        target_lo = next(
            (obj for obj in database_format if obj.get("LearningObjective") == lo_text), None
        )
        if not target_lo:
            continue


        result = item.get("result")
        questions = []
        if isinstance(result, dict):
            questions = result.get("questions") or []
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            questions = result[0].get("questions") or []

        for question in questions:
            if not isinstance(question, dict):
                continue

            # Use the payload difficulty for all questions in this batch - they were generated with this difficulty
            if not payload_difficulty:
                _log_with_container("error", f"ERROR: No difficulty in payload! Payload: {payload_item}", container_id=container_id)
                continue  # Skip items without payload difficulty


            # Generate UUID for the question
            question_id = str(uuid.uuid4())

            if question_type == "MATCHING":
                # Shuffle Column B display order so learners can't pattern-match
                shuffled_column_b = list(question.get("column_b_answers", []))
                random.shuffle(shuffled_column_b)
                question_db_format = {
                    "id": question_id,
                    "questionType": question_type,
                    "aiGeneratedDifficulty": payload_difficulty,
                    "questionText": question.get("questionText") or question.get("question", ""),
                    "column_a_answers": question.get("column_a_answers", []),
                    "column_b_answers": shuffled_column_b,
                    "answers": question.get("answers", []),
                    "isUploaded": False
                }
            else:
                # Build answers in database format with proper IDs
                answers_db_format: List[Dict[str, Any]] = []
                answer_items = question.get("answers") or question.get("answer") or []

                for answer in answer_items:
                    # Generate UUID for each answer
                    answer_id = str(uuid.uuid4())
                    answers_db_format.append({
                        "answerId": answer_id,
                        "answerText": answer.get("answer") or answer.get("answerText", ""),
                        "isCorrect": (
                            answer.get("correct")
                            if answer.get("correct") is not None
                            else answer.get("isCorrect", False)
                        ),
                        "explanation": answer.get("explanation", "")
                    })

                # Build question in database format with required fields
                question_db_format = {
                    "id": question_id,
                    "questionType": question_type,
                    "aiGeneratedDifficulty": payload_difficulty,  # Use the payload difficulty that generated this question
                    "questionText": question.get("questionText") or question.get("question", ""),
                    "answer": answers_db_format,  # Use 'answer' field as expected by API
                    "isUploaded": False
                }

            target_lo["questions"].append(question_db_format)

        # Set LO-level difficulty to the payload difficulty
        if payload_difficulty:
            target_lo["LevelOfQuiz"] = payload_difficulty

    # Filter out LOs with no questions
    return [lo for lo in database_format if lo["questions"]]


def guardrails_check(state: GraphState) -> GraphState:
    """
    Check generated questions against content safety guardrails (parallel execution).

    Runs 3 checks in parallel:
    - Sensitive content detection
    - Inappropriate content detection
    - Off-topic detection

    Questions that fail are marked for regeneration.
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Running content safety guardrails (3 checks in parallel)", container_id=container_id)
    guardrails = ContentGuardrails()
    source_text = state["input"].get("sourceText", "")

    passed_outputs = []
    needs_regeneration = []

    for item in state["raw_outputs"]:
        content = json.dumps(item["result"])
        lo = item["payload"]["learningObjective"]

        results = guardrails.check_all_parallel(content, lo, source_text)

        all_passed = all(r.passed for r in results.values())
        if all_passed:
            passed_outputs.append(item)
        else:
            failed_checks = [name for name, r in results.items() if not r.passed]
            guardrail_reasons = {
                name: r.flagged_content for name, r in results.items()
                if not r.passed and r.flagged_content
            }
            _log_with_container("warning", f"Guardrail failed for question: {failed_checks} - reasons: {guardrail_reasons}", container_id=container_id)
            needs_regeneration.append({
                **item,
                "failed_guardrails": failed_checks,
                "guardrail_reasons": guardrail_reasons,
                "guardrail_results": {k: {"passed": v.passed} for k, v in results.items()}
            })

    state["guardrails_passed"] = passed_outputs
    state["guardrails_failed"] = needs_regeneration
    state["guardrails_regeneration_attempts"] = state.get("guardrails_regeneration_attempts", 0)

    _log_with_container("info", f"Guardrails: {len(passed_outputs)} passed, {len(needs_regeneration)} failed", container_id=container_id)
    return state


def regenerate_for_guardrails(state: GraphState) -> GraphState:
    """
    Regenerate questions that failed guardrails (single retry).

    Adds guardrail feedback to the prompt to guide regeneration
    toward safer content.
    """
    container_id = state.get("container_id")
    _log_with_container("info", "Regenerating questions that failed guardrails", container_id=container_id)
    client = get_generation_client()
    deployment = get_generation_model()

    regenerated = []
    for item in state.get("guardrails_failed", []):
        payload = item["payload"]
        failed_checks = item.get("failed_guardrails", [])
        guardrail_reasons = item.get("guardrail_reasons", {})

        feedback = f"IMPORTANT: Previous generation failed content safety checks: {', '.join(failed_checks)}. "
        if guardrail_reasons:
            reason_details = "; ".join(f"{check}: {reason}" for check, reason in guardrail_reasons.items())
            feedback += f"Specific issues: {reason_details}. "
        feedback += "Please regenerate ensuring the content is appropriate for an educational setting, "
        feedback += "avoids sensitive topics, and stays on-topic with the learning objective."

        response = _call_openai(
            client=client,
            deployment=deployment,
            messages=[
                {"role": "system", "content": payload["systemPrompt"]},
                {"role": "user", "content": payload["userPrompt"] + f"\n\n{feedback}"},
            ],
            task_type="generation",
            container_id=container_id
        )
        content = response.choices[0].message.content or ""
        try:
            result = _parse_json(content)
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON", "raw": content}

        regenerated.append({"payload": payload, "result": result})

    state["raw_outputs"] = regenerated
    state["guardrails_failed"] = []
    state["guardrails_regeneration_attempts"] = state.get("guardrails_regeneration_attempts", 0) + 1

    _log_with_container("info", f"Regenerated {len(regenerated)} questions for guardrails retry", container_id=container_id)
    return state


def _route_after_guardrails(state: GraphState) -> str:
    """
    Route based on guardrails results.

    Returns:
        "regenerate" if there are failed items and retries remaining
        "finalize" to proceed to validation
    """
    failed = state.get("guardrails_failed", [])
    attempts = state.get("guardrails_regeneration_attempts", 0)

    if failed and attempts < MAX_GUARDRAIL_REGENERATION_ATTEMPTS:
        container_id = state.get("container_id")
        _log_with_container("info", f"Routing to guardrails regeneration (attempt {attempts + 1})", container_id=container_id)
        return "regenerate"

    return "finalize"


def finalize_guardrails(state: GraphState) -> GraphState:
    """
    Finalize guardrails check and prepare for validation.

    Combines passed and failed (after max retries) items into raw_outputs
    for the validation phase. This node serves as the fan-out point for
    the 7 parallel validation agents.
    """
    container_id = state.get("container_id")
    failed = state.get("guardrails_failed", [])
    passed = state.get("guardrails_passed", [])

    if failed:
        _log_with_container("warning",
            f"Max guardrail retries ({MAX_GUARDRAIL_REGENERATION_ATTEMPTS}) reached. "
            f"{len(failed)} items still flagged - logging and continuing",
            container_id=container_id
        )
        for item in failed:
            _log_with_container("warning", f"Guardrail-flagged question included in output: {item.get('failed_guardrails')}", container_id=container_id)
        state["raw_outputs"] = passed + [{"payload": f["payload"], "result": f["result"]} for f in failed]
    else:
        state["raw_outputs"] = passed

    _log_with_container("info", f"Guardrails finalized: {len(state['raw_outputs'])} questions proceeding to validation", container_id=container_id)
    return state


def output_guardrail(state: GraphState) -> GraphState:
    """
    Terminal output guardrail: content-safety check on the final questions.

    Runs after quality validation passes, on the fully-validated questions in
    state["improved_outputs"]. Reuses run_output_guardrails (warn-only, no
    regeneration). Any safety warning is appended to state["pipeline_errors"],
    which run_pipeline surfaces on the output and tasks.py converts into job
    warnings. Questions are always returned regardless of the check result.
    """
    container_id = state.get("container_id")
    items = state.get("improved_outputs", [])
    source_text = state.get("input", {}).get("sourceText", "")

    # Fast-path: nothing was generated (e.g. every LO was filtered out by the
    # input guardrails). Make the no-op explicit and skip the work.
    if not items:
        _log_with_container("info", "Output guardrail: no questions to check", container_id=container_id)
        return state

    _log_with_container("info", f"Output guardrail: checking {len(items)} questions", container_id=container_id)

    # Accumulate warnings locally and apply a single immutable state update at
    # the end. Mutating state["pipeline_errors"] in place is unsafe under
    # LangGraph's reducer/checkpointing model.
    new_errors = []

    # Each item corresponds to a single learning objective and carries its own
    # generated questions. We intentionally check one item at a time instead of
    # batching all items into a single call: the off-topic guardrail is
    # LO-specific, and items may span different LOs, so a shared lo_description
    # would produce semantically incorrect relevance checks.
    for item in items:
        # Guard against malformed items (e.g. a partial failure earlier in the
        # pipeline). A missing "result" must not abort the node and drop the
        # questions that were generated successfully.
        result = item.get("result")
        if not result:
            _log_with_container(
                "warning",
                "Output guardrail: skipping item with missing 'result'",
                container_id=container_id,
            )
            continue

        lo = item.get("payload", {}).get("learningObjective", "")

        result_for_check = {
            k: v for k, v in result.items()
            if k not in ("LearningObjective", "locale")
        }

        # The guardrail is warn-only: per the node contract "Questions are always
        # returned regardless of the check result". A guardrail service error
        # (timeout, malformed response, etc.) must therefore degrade gracefully
        # rather than propagate up and drop already-generated questions.
        try:
            _, warning = run_output_guardrails(
                questions=[result_for_check],
                lo_description=lo,
                source_text=source_text,
                regenerate_fn=None,
                logger_instance=logger,
            )
        except Exception as exc:
            # A guardrail service failure (timeout, malformed response, etc.)
            # must not drop already-generated questions, but it also must not
            # vanish into the logs: surface it as a pipeline_errors entry so
            # production can distinguish "guardrails ran and passed" from
            # "guardrails died silently".
            _log_with_container(
                "error",
                f"Output guardrail call failed, continuing without warning: {exc}",
                container_id=container_id,
            )
            new_errors.append(
                {"error": f"Output guardrail service failed: {exc}", "context": "output_guardrail", "step": "content_safety"}
            )
            warning = None

        if warning:
            new_errors.append(
                {"error": warning, "context": "output_guardrail", "step": "content_safety"}
            )

    if new_errors:
        existing = list(state.get("pipeline_errors", []))
        existing.extend(new_errors)
        state["pipeline_errors"] = existing

    return state


def run_pipeline(payload: PipelineInput, evaluate: bool = True) -> List[Dict[str, Any]]:
    """Traced entry point: root Phoenix span + optional post-run evaluators.

    evaluate=False skips the post-run evaluators (used by offline experiments).
    Tracing/evaluator failures never affect the generation result.

    NORMAL GENERATION FLOW (search for "normal-generation-<n>" to follow it):
      normal-generation-1  register Phoenix tracing (once per process)
      normal-generation-2  read the model settings for the trace
      normal-generation-3  open the root span
      normal-generation-4  run the core pipeline (_run_pipeline_impl, steps 8-15 below)
      normal-generation-5  record the final output on the root span
      normal-generation-6  optional post-run evaluators (EVALUATORS_ENABLED)
      normal-generation-7  return the output (evaluators never change it)
    """
    # normal-generation-1: register Phoenix tracing once per process (idempotent)
    from ..tracing import init_tracing, pipeline_span
    init_tracing()
    research = payload.get("research", {}) or {}
    # normal-generation-2: model settings recorded on the root span
    model_cfg = {
        "generator_model": research.get("generationModel") or os.getenv("GENERATOR_MODEL"),
        "judge_model": research.get("validationModel") or os.getenv("JUDGE_MODEL"),
        "llm_provider": os.getenv("LLM_PROVIDER"),
        "structured_output": USE_STRUCTURED_OUTPUT,
    }
    # normal-generation-3: root span - everything below (graph nodes, LLM calls, evaluators) nests under it
    with pipeline_span(payload, model_cfg) as finish:
        output = _run_pipeline_impl(payload)                              # normal-generation-4
        finish(output)                                                    # normal-generation-5
        # normal-generation-6: optional post-run evaluators (env EVALUATORS_ENABLED); never affects the result.
        # Offline experiments pass evaluate=False because they score the output themselves.
        if evaluate:
            try:
                from ..evaluators import run_evaluators, get_enabled
                if get_enabled():
                    run_evaluators(output, {
                        "payload": payload,
                        "pipeline_config": _create_pipeline_config(research),
                        "container_id": payload.get("assessment_container_id"),
                        "job_id": payload.get("task_id"),
                        "assessment_id": payload.get("assessment_container_id"),
                    })
            except Exception:
                logger.exception("EVAL: evaluator run failed (ignored)")
        return output


def _run_pipeline_impl(payload: PipelineInput) -> List[Dict[str, Any]]:
    """
    Execute the complete question generation pipeline using state-based configuration.

    Returns database format directly - ready for store_agent_questions().
    """
    # normal-generation-8: reset the token tracker for this run
    reset_token_tracker()
    
    # Set up container ID and task ID for structured logging
    container_id = payload.get("assessment_container_id", "container_id_not_found")
    task_id = payload.get("task_id", "task_id_not_found")
    
    _log_with_container("info", f"TOKEN TRACKING: Initialized for new pipeline run | Container: {container_id} | Task: {task_id}", container_id=container_id)
    _log_with_container("info", f"STRUCTURED OUTPUT: {'Enabled' if USE_STRUCTURED_OUTPUT else 'Disabled'} for this pipeline run", container_id=container_id)
    
    # Get LO and question type information for context logging
    learning_objectives = payload.get("learningObjectives", [])
    question_types = payload.get("questionTypes", ["MULTIPLE_CHOICE"])
    question_type = question_types[0] if question_types else "MULTIPLE_CHOICE"
    
    if learning_objectives:
        first_lo = learning_objectives[0]
        
        # Handle both string and dictionary formats for learning objectives
        if isinstance(first_lo, dict):
            lo_uuid = first_lo.get("uuid", "unknown")
            lo_text = first_lo.get("text", "No description available")
        else:
            # Legacy format: learning objective is just a string
            lo_uuid = "legacy"
            lo_text = str(first_lo)
        
        _log_with_container("info", f"Question generation started for LO {lo_uuid}: {lo_text} | Question Type: {question_type}", container_id=container_id)
    else:
        _log_with_container("info", f"Starting Question Generation Pipeline | Question Type: {question_type}", container_id=container_id)
    
    research_config = payload.get("research", {})
    # configuring LLM models for generation and validation
    pipeline_config = _create_pipeline_config(research_config, container_id=container_id)

    total_questions_to_generate = len(learning_objectives)
    _log_with_container("info", f"Generating {total_questions_to_generate} questions for {total_questions_to_generate} learning objectives", container_id=container_id)

    difficulty_levels = payload.get("difficultyLevels", ["Intermediate"])
    _log_with_container("info", f"Configuration: {question_types} | {difficulty_levels} | {payload.get('numberOfQuestions', 1)} questions", container_id=container_id)

    if not difficulty_levels or len(difficulty_levels) == 0:
        _log_with_container("error", f"No difficulty levels provided! Payload: {payload}", container_id=container_id)
        raise ValueError("No difficulty levels provided to agent")

    # normal-generation-9: build the initial graph state from the payload
    base_state = {
        "input": payload,
        "container_id": container_id,
        "question_configs": [],
        "questions": [],
        "raw_outputs": [],
        "improved_outputs": [],
        "formatted": [],
        "research": research_config,
        "pipeline_config": pipeline_config,
        "task_id": task_id,
        "matching_question_configs": [],
        "pipeline_errors": [],
    }

    # normal-generation-10: choose the config builder (MATCHING or multiple choice)
    # MATCHING is built by its own config builder and routed by graph_builder
    # to the fully separate MATCHING chain (see _route_from_start). Per the
    # existing calling convention, each run_pipeline() call is scoped to a
    # single question type, so this is a simple either/or, not a merge.
    if "MATCHING" in question_types:
        config_state = create_matching_question_configs(base_state)
    else:
        config_state = create_question_configs(base_state)

    # Import here to avoid circular dependency with graph_builder
    from .graph_builder import build_question_generation_graph
    from .cancellation import is_job_cancelled

    # normal-generation-11: run the graph (generate, validate, regenerate, check options) and stop if the job is cancelled
    app = build_question_generation_graph()
    final_state = config_state
    for step_state in app.stream(config_state, stream_mode="values"):
        final_state = step_state
        if is_job_cancelled(task_id):
            _log_with_container("info", "Job cancelled - aborting pipeline mid-run", container_id=container_id)
            raise PipelineCancelled(f"Job {task_id} cancelled")
    
    # normal-generation-12: format correction (skipped when structured output is on)
    # Skip format correction when using structured output - guaranteed compliance
    if not USE_STRUCTURED_OUTPUT:
        retry = 0
        while retry < MAX_FORMAT_CORRECTION_ATTEMPTS:
            final_state = format_conversion(final_state)
            final_state = validate_and_fix_format(final_state)
            retry += 1
            if not final_state.get("format_loop"):
                break
        _log_with_container("info", "Format validation completed", container_id=container_id)
    else:
        _log_with_container("info", "Skipping format correction - structured output enabled", container_id=container_id)

    # normal-generation-13: build the database-shaped output from the improved outputs
    output = _build_output(payload, final_state.get("improved_outputs", []), container_id=container_id)
    
    total_questions = sum(len(lo.get("questions", [])) for lo in output)
    total_los = len(output)
    
    # normal-generation-14: compare the delivered question count with what was expected
    # Calculate original expectations using the same logic as config creation
    learning_objectives_raw = payload.get("learningObjectives", [])
    learning_objectives_processed = _learning_objective_descriptions(payload)
    difficulty_levels = payload.get("difficultyLevels", ["Intermediate"])  
    questions_per_config = payload.get("numberOfQuestions", 1)
    
    # Use processed LOs count (same as config creation) for accurate calculation
    expected_total_questions = len(learning_objectives_processed) * len(difficulty_levels) * questions_per_config
    expected_by_raw = len(learning_objectives_raw) * len(difficulty_levels) * questions_per_config
    
    _log_with_container("info", f"SUCCESS: Pipeline completed: {total_questions} questions across {total_los} LOs", container_id=container_id)
    _log_with_container("info", f"EXPECTATION CHECK: Expected (processed LOs): {expected_total_questions} | Expected (raw LOs): {expected_by_raw} | Actual delivered: {total_questions}", container_id=container_id)
    
    if total_questions != expected_total_questions:
        shortfall = expected_total_questions - total_questions
        _log_with_container("warning", f"Final shortfall: {total_questions}/{expected_total_questions} questions delivered (missing {shortfall})", container_id=container_id)
        
        # Show both calculations for debugging
        if expected_by_raw != expected_total_questions:
            raw_shortfall = expected_by_raw - total_questions
            _log_with_container("warning", f"If using raw LO count: {total_questions}/{expected_by_raw} delivered (missing {raw_shortfall})", container_id=container_id)
    else:
        _log_with_container("info", f"✓ Expected question count achieved: {total_questions}/{expected_total_questions}", container_id=container_id)
    
    # normal-generation-15: attach pipeline errors and stats to the output, then clean up state and return
    # Attach pipeline errors to output for job logging
    pipeline_errors = final_state.get("pipeline_errors", [])
    if pipeline_errors and output:
        output[0]["pipeline_errors"] = pipeline_errors
    # ARIZE-PHOENIX-EXPERIMENT-STATS: regeneration attempt counts for offline experiments (read by run_experiment only)
    if output:
        output[0]["pipeline_stats"] = {
            "stem_validation_attempts": final_state.get("stem_validation_attempt", 0),
            "stem_lo_cog_validation_attempts": final_state.get("stem_lo_cog_validation_attempt", 0),
            "pipeline_errors": list(pipeline_errors),
        }

    # Log completion with LO and question type context
    if learning_objectives:
        first_lo = learning_objectives[0]
        
        # Handle both string and dictionary formats for learning objectives
        if isinstance(first_lo, dict):
            lo_uuid = first_lo.get("uuid", "unknown")
            lo_text = first_lo.get("text", "No description available")
        else:
            # Legacy format: learning objective is just a string
            lo_uuid = "legacy"
            lo_text = str(first_lo)
        
        _log_with_container("info", f"Question generation ended for LO {lo_uuid}: {lo_text} | Question Type: {question_type}", container_id=container_id)
    else:
        _log_with_container("info", f"Question generation completed | Question Type: {question_type}", container_id=container_id)

    # Display token usage summary
    token_tracker = get_token_tracker()
    token_tracker.log_summary()
    
    _clear_state_keys(final_state, FINAL_PIPELINE_KEYS, "final pipeline cleanup")

    return output
