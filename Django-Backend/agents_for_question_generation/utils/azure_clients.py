"""
LLM client utilities. Both providers use the OpenAI-compatible chat completions API.

LLM_PROVIDER selects the provider:
- "anthropic" (default): needs ANTHROPIC_API_KEY
- "deepinfra":           needs DEEPINFRA_API_KEY (serves Qwen models)

Model names come from GENERATOR_MODEL / JUDGE_MODEL.
"""

import logging
import os
import threading

from openai import OpenAI

logger = logging.getLogger(__name__)

# provider -> (API key env var, OpenAI-compatible base URL)
PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/"),
    "deepinfra": ("DEEPINFRA_API_KEY", "https://api.deepinfra.com/v1/openai"),
}

_client = None
_client_lock = threading.Lock()


def get_provider() -> str:
    """Return the configured provider name, validated against PROVIDERS."""
    name = os.getenv("LLM_PROVIDER", "anthropic").strip().lower() or os.getenv("LLM_PROVIDER", "deepinfra").strip().lower()
    if name not in PROVIDERS:
        raise RuntimeError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {name!r}")
    return name


def get_client():
    """Get reusable client (shared for all tasks)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                provider = get_provider()
                key_var, base_url = PROVIDERS[provider]
                api_key = os.getenv(key_var)
                if not api_key:
                    raise RuntimeError(f"{key_var} is not set (LLM_PROVIDER={provider})")
                logger.info("LLM client ready | provider=%s | base_url=%s", provider, base_url)
                _client = OpenAI(api_key=api_key, base_url=base_url)
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
