"""
Structured output utilities for guaranteed JSON compliance in the agent pipeline.

This module combines:
- Structured LLM calling with OpenAI's .with_structured_output()
- Response schema mapping and extraction utilities
- Fallback handling for backward compatibility
"""

import time
import logging
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel

from .token_tracking import get_token_tracker
from ..agents.pipeline_constants import RATE_LIMIT_RETRY_DELAY_SECONDS, USE_STRUCTURED_OUTPUT
from ..agents.pipeline_types import PipelineConfig
from .response_schemas import (
    QuestionGenerationResponse,
    OptionsGenerationResponse,
    QualityCheckResponse,
    ValidationResponse,
    DistractorValidationResponse,
    CorrectionResponse,
    MatchingStemGenerationResponse,
    MatchingOptionsGenerationResponse,
    get_response_schema
)

logger = logging.getLogger(__name__)


def _log_with_container(level: str, message: str, container_id: str = None, **kwargs) -> None:
    """Log with structured container prefix for ES filtering."""
    prefix = f"{container_id or 'unknown'}-{level}"
    formatted_message = f"[{prefix}] {message}"
    
    if level == "error":
        logger.error(formatted_message, **kwargs)
    elif level == "warning":
        logger.warning(formatted_message, **kwargs)
    else:
        logger.info(formatted_message, **kwargs)


def get_schema_for_context(call_context: str, task_type: str = None) -> Optional[Any]:
    """
    Determine the appropriate response schema based on call context and task type.
    
    Args:
        call_context: The context/operation being performed
        task_type: The type of task (generation, validation, etc.)
        
    Returns:
        Appropriate Pydantic schema class or None
    """
    # MATCHING-specific schemas — must come before generic options_generation check
    # because "options_generation" is a substring of "matching_options_generation".
    if "matching_stem_generation" in call_context:
        return MatchingStemGenerationResponse

    if "matching_options_generation" in call_context:
        return MatchingOptionsGenerationResponse

    # Question and options generation (MCQ)
    if any(ctx in call_context for ctx in ["question_generation", "scenario_integrated", "question_retry"]):
        return QuestionGenerationResponse

    if "options_generation" in call_context:
        return OptionsGenerationResponse

    # Quality and validation checks
    if "quality_check" in call_context:
        return QualityCheckResponse
        
    if any(ctx in call_context for ctx in ["validation", "relevancy_check", "guardrails"]):
        return ValidationResponse
        
    # Distractor validation
    if any(ctx in call_context for ctx in [
        "plausibility", "incorrectness", "misconception", 
        "distinctiveness", "clarity", "parallelism", "locale_compliance"
    ]):
        return DistractorValidationResponse
        
    # Correction operations
    if any(ctx in call_context for ctx in ["correction", "fix", "improvement"]):
        return CorrectionResponse
    
    return None


def extract_structured_result(response: Any, fallback_parser=None) -> Any:
    """
    Extract result from structured response with fallback to manual parsing.
    
    Args:
        response: Response object (structured or unstructured)
        fallback_parser: Function to parse unstructured response
        
    Returns:
        Parsed result data
    """
    # Check if this is a structured response
    if USE_STRUCTURED_OUTPUT and hasattr(response, 'choices') and response.choices:
        message = response.choices[0].message
        if hasattr(message, 'parsed') and message.parsed:
            # Convert Pydantic model to dict for compatibility
            if hasattr(message.parsed, 'model_dump'):
                return message.parsed.model_dump()
            elif hasattr(message.parsed, 'dict'):
                return message.parsed.dict()
            else:
                return message.parsed
    
    # Fallback to traditional parsing
    if fallback_parser and hasattr(response, 'choices'):
        content = response.choices[0].message.content or ""
        return fallback_parser(content)
    
    return response


def call_structured_llm(
    client: Any,
    deployment: str,
    messages: List[Dict[str, str]],
    response_format: BaseModel,
    task_type: str = "generation",
    pipeline_config: Optional[PipelineConfig] = None,
    call_context: str = "",
    container_id: str = None,
    **kwargs
) -> Any:
    """
    Call LLM with structured output for guaranteed JSON compliance.
    
    Args:
        client: Azure OpenAI client
        deployment: Model deployment name
        messages: List of messages for the chat completion
        response_format: Pydantic model for response structure
        task_type: Type of task for reasoning configuration
        pipeline_config: Pipeline configuration
        call_context: Context for logging
        container_id: Container ID for structured logging
        **kwargs: Additional parameters for the API call
        
    Returns:
        Structured response object with .parsed attribute
        
    Raises:
        Exception: If API call fails after retries
    """
    if not USE_STRUCTURED_OUTPUT:
        raise ValueError("Structured output is not enabled")
    
    def _track_usage(response: Any, task_type: str, call_context: str) -> None:
        """Helper to track token usage - extracted to avoid duplication."""
        if response.usage:
            tracker = get_token_tracker()
            tracker.add_usage(
                task_type=task_type,
                usage_data={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                call_context=call_context
            )
    
    max_retries = 3
    
    def _make_structured_api_call():
        """Make structured API call with rate limit retry handling."""
        for attempt in range(max_retries):
            try:
                # Structured call via the OpenAI-compatible chat completions API
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    response_format=response_format,
                    timeout=120,
                    stream=False
                )
                
                # Track token usage and return successful response
                _track_usage(response, task_type, call_context)
                return response
                
            except Exception as e:
                # Handle rate limiting
                is_rate_limit = (
                    hasattr(e, 'status_code') and e.status_code == 429
                    or "429" in str(e)
                    or "rate limit" in str(e).lower()
                )
                
                if is_rate_limit and attempt < max_retries - 1:
                    _log_with_container("warning", f"Rate limit hit on attempt {attempt + 1}, retrying in {RATE_LIMIT_RETRY_DELAY_SECONDS}s...", container_id=container_id)
                    time.sleep(RATE_LIMIT_RETRY_DELAY_SECONDS)
                    continue
                
                # Re-raise non-rate-limit errors or final attempt
                _log_with_container("error", f"Structured LLM call failed on attempt {attempt + 1}: {e}", container_id=container_id)
                if attempt == max_retries - 1:
                    raise
    
    try:
        _log_with_container("info", f"STRUCTURED: Calling {deployment} with {response_format.__name__} schema | Context: {call_context}", container_id=container_id)
        
        response = _make_structured_api_call()
        
        # Production logging for structured output success
        _log_with_container("debug", f"STRUCTURED: Received {response_format.__name__} response", container_id=container_id)
        
        # Validate the structured response using correct OpenAI pattern
        if not hasattr(response, 'choices') or not response.choices:
            # Maybe it's a different structure - let's check
            _log_with_container("error", f"STRUCTURED: No choices in response. Available attrs: {[attr for attr in dir(response) if not attr.startswith('_')]}", container_id=container_id)
            raise ValueError("Structured response parsing failed - no choices in response")
            
        message = response.choices[0].message
        
        # Check for refusal first (safety mechanism) - return response for caller to handle
        if hasattr(message, 'refusal') and message.refusal:
            _log_with_container("warning", f"STRUCTURED: Model refused request: {message.refusal}", container_id=container_id)
            return response  # Let caller handle refusal gracefully
        
        # Check for parsed content
        if not hasattr(message, 'parsed') or message.parsed is None:
            _log_with_container("error", f"STRUCTURED: No parsed content in {response_format.__name__} response", container_id=container_id)
            raise ValueError("Structured response parsing failed - no parsed content in message")
        
        _log_with_container("info", f"STRUCTURED: Success - received valid {response_format.__name__} response", container_id=container_id)
        return response
        
    except Exception as e:
        _log_with_container("error", f"STRUCTURED: Failed to get structured response for {call_context}: {e}", container_id=container_id, exc_info=True)
        raise

