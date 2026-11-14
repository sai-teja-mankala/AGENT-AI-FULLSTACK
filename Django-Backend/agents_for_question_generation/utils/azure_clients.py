"""
LLM client utilities for Claude (Anthropic), via Anthropic's OpenAI-compatible
chat completions endpoint.

Requires ANTHROPIC_API_KEY. Model names come from GENERATOR_MODEL / JUDGE_MODEL.
"""

import os
import threading

from openai import OpenAI

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"

_client = None
_client_lock = threading.Lock()


def get_client():
    """Get reusable client (shared for all tasks)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise RuntimeError("ANTHROPIC_API_KEY is not set")
                _client = OpenAI(api_key=api_key, base_url=ANTHROPIC_BASE_URL)
    return _client


def get_generation_client():
    """Get client for generation tasks."""
    return get_client()


def get_validation_client():
    """Get client for validation tasks."""
    return get_client()


def get_generation_model():
    """Get the generator model deployment name."""
    model = os.getenv("GENERATOR_MODEL")
    if not model:
        raise RuntimeError("GENERATOR_MODEL is not set.")
    return model


def get_validation_model():
    """Get the validation/judge model deployment name."""
    model = os.getenv("JUDGE_MODEL")
    if not model:
        raise RuntimeError("JUDGE_MODEL is not set.")
    return model
