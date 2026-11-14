"""
Token tracking utilities for monitoring API usage across the question generation pipeline.

This module provides a simple token tracking system to monitor input/output tokens
and API calls for analysis and monitoring purposes.
"""

import logging
import threading
from contextvars import ContextVar
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class TokenTracker:
    """Track token usage across the pipeline.

    Thread-safe: add_usage() may be called concurrently from LangGraph's
    parallel validation nodes (which run in worker threads within a single
    pipeline run), so mutations are guarded by a lock.
    """

    def __init__(self):
        self.pipeline_totals = {"input": 0, "output": 0, "calls": 0}
        self.call_details = []
        self._lock = threading.Lock()

    def add_usage(self, task_type: str, usage_data: dict, call_context: str = ""):
        """Add token usage for a specific call."""
        if not usage_data:
            return

        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", input_tokens + output_tokens)

        with self._lock:
            # Update pipeline totals
            self.pipeline_totals["input"] += input_tokens
            self.pipeline_totals["output"] += output_tokens
            self.pipeline_totals["calls"] += 1

            # Store call details for debugging
            self.call_details.append({
                "task_type": task_type,
                "context": call_context,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            })
    
    def get_summary(self) -> dict:
        """Get comprehensive token usage summary.

        Returns a snapshot copy (taken under the lock) so callers can't
        mutate the tracker's internal accumulator, and so the totals are
        consistent even while parallel validation nodes are still writing.
        """
        with self._lock:
            return {
                "pipeline_totals": self.pipeline_totals.copy(),
                "call_count": len(self.call_details)
            }
    
    def log_summary(self):
        """Log token usage summary."""
        logger.info("=" * 50)
        logger.info("TOKEN USAGE SUMMARY")
        logger.info("=" * 50)
        
        # Pipeline totals
        totals = self.pipeline_totals
        total_tokens = totals['input'] + totals['output']
        logger.info(f"Total: {totals['input']:,} input | {totals['output']:,} output | {total_tokens:,} total | {totals['calls']} calls")
        
        logger.info("=" * 50)


# Context-local token tracker instance.
#
# A ContextVar (rather than a module-level global) is used so that multiple
# pipeline runs executing concurrently in separate threads (e.g. one per
# question type) each get their own isolated tracker. LangGraph's sync
# executor submits node work via contextvars.copy_context(), so the tracker
# set by run_pipeline() also propagates correctly into the parallel
# validation-node threads spawned within a single pipeline run.
_token_tracker_var: ContextVar[Optional[TokenTracker]] = ContextVar(
    "question_generation_token_tracker", default=None
)


def get_token_tracker() -> TokenTracker:
    """Get the token tracker for the current context (created on first use)."""
    tracker = _token_tracker_var.get()
    if tracker is None:
        tracker = TokenTracker()
        _token_tracker_var.set(tracker)
    return tracker


def reset_token_tracker():
    """Reset token tracking for new pipeline run (current context only)."""
    _token_tracker_var.set(TokenTracker())