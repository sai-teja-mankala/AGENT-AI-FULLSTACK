"""
LangGraph Graph Builder - Question Generation Pipeline

This module defines the graph structure for the question generation pipeline,
connecting all agent nodes and defining the flow between them.
"""
import logging
import threading
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

from .langgraph_pipeline import (
    # Core generation nodes
    build_question,
    build_options,
    # Stem clarity/bias validation nodes
    validate_stem_clarity_bias,
    regenerate_stem_clarity_bias,
    _route_after_stem_validation,
    # Stem LO-alignment / cognitive-level validation nodes
    validate_stem_lo_cognitive,
    regenerate_stem_lo_cognitive,
    _route_after_stem_lo_cognitive,

    # Guardrails nodes (DISABLED for performance testing)
    # guardrails_check,
    # regenerate_for_guardrails,
    # finalize_guardrails,
    # _route_after_guardrails,
    
    # 7 Parallel validation agent nodes
    agent_check_plausibility,
    agent_check_misconception,
    agent_check_distinctiveness,
    agent_check_incorrectness,
    agent_check_clarity,
    agent_check_parallelism,
    agent_check_locale_compliance,
    
    aggregate_validation_results,
    correct_failed_distractors,
    finalize_distractor_validation,
    validate_quality,
    correct_quality,
    output_guardrail,
    _route_after_distractor_validation,
    _route_from_validate_quality,

    # MATCHING ("match the following") nodes - fully separate chain, no
    # distractor validation (see langgraph_pipeline.py for details)
    build_question_matching,
    build_options_matching,
    validate_quality_matching,
    correct_quality_matching,
    finalize_matching_output,
    _route_from_validate_quality_matching,
    _route_from_start,

    GraphState,
)


# Global cache for compiled graph - compile once, reuse forever
_compiled_graph = None
# Guards first-time compilation when multiple pipeline threads start
# concurrently (e.g. parallel question-type generation).
_compile_lock = threading.Lock()

def build_question_generation_graph():
    """
    Build the complete question generation pipeline graph.
    Uses module-level caching to compile once and reuse across all requests.
    
    Returns:
        Compiled LangGraph with all nodes and edges configured.
    
    Graph Structure:
        - Direct routing to question generation (scenarios handled internally)
        - 2 Generation nodes (question + options, scenarios integrated)
        - 4 Stem validation nodes (clarity/bias + LO/cognitive checks, each with a regeneration loop)
        - [DISABLED] 2 Guardrails nodes (content safety checks) - bypassed for performance
        - 7 Parallel validation agent nodes (comprehensive quality checks with inline format fixing)
        - 1 Aggregator agent node
        - 2 Correction nodes (distractor correction loop)
        - 2 Quality validation nodes
        
    Total: 15 agent nodes (guardrails disabled, scenario coherence removed - scenarios embedded in questions)
    """
    global _compiled_graph

    # Return cached graph if already compiled
    if _compiled_graph is not None:
        return _compiled_graph

    # Serialize first-time compilation across concurrent pipeline threads
    # (parallel question-type generation). The lock is only contended on the
    # very first call(s); afterwards the fast path above returns the cached
    # graph without locking.
    with _compile_lock:
        if _compiled_graph is not None:
            return _compiled_graph
        return _compile_question_generation_graph()


def _compile_question_generation_graph():
    """Build and compile the pipeline graph. Caller must hold _compile_lock."""
    global _compiled_graph

    logger.info("GRAPH: Compiling LangGraph pipeline (first time only)")
    graph = StateGraph[GraphState, None, GraphState, GraphState](GraphState)

    # ARIZE-PHOENIX-NODE-SPAN: wrap each node in a Phoenix span (no-op passthrough if tracing fails).
    # Per-node metadata lives in tracing/node_specs.py
    from ..tracing.node_specs import NODE_SPECS
    from ..tracing.spans import traced_node
    _add_node = graph.add_node

    def add_node(name, fn):
        spec = NODE_SPECS.get(name)
        _add_node(name, traced_node(name, spec)(fn) if spec else fn)

    graph.add_node = add_node

    # Add a dummy start node for conditional routing
    def start_routing(state: GraphState) -> GraphState:
        """Entry point that simply passes through the state for routing."""
        return state
    
    graph.add_node("start", start_routing)
    # TODO Add Input Guardrail
    # TODO Add Document Fetching node here
    # TODO Add RAG Context Validation node here
    graph.add_node("build_question", build_question)
    graph.add_node("validate_stem_clarity_bias", validate_stem_clarity_bias)
    graph.add_node("regenerate_stem_clarity_bias", regenerate_stem_clarity_bias)
    graph.add_node("validate_stem_lo_cognitive", validate_stem_lo_cognitive)
    graph.add_node("regenerate_stem_lo_cognitive", regenerate_stem_lo_cognitive)
    graph.add_node("build_options", build_options)
    
    # Guardrails nodes (DISABLED for performance testing)
    # graph.add_node("guardrails_check", guardrails_check)
    # graph.add_node("regenerate_for_guardrails", regenerate_for_guardrails)
    # graph.add_node("finalize_guardrails", finalize_guardrails)
    
    # 7 Parallel validation agent nodes
    graph.add_node("agent_check_plausibility", agent_check_plausibility)
    graph.add_node("agent_check_misconception", agent_check_misconception)
    graph.add_node("agent_check_distinctiveness", agent_check_distinctiveness)
    graph.add_node("agent_check_incorrectness", agent_check_incorrectness)
    graph.add_node("agent_check_clarity", agent_check_clarity)
    graph.add_node("agent_check_parallelism", agent_check_parallelism)
    graph.add_node("agent_check_locale_compliance", agent_check_locale_compliance)
    graph.add_node("aggregate_validation_results", aggregate_validation_results)
    graph.add_node("correct_failed_distractors", correct_failed_distractors)
    graph.add_node("finalize_distractor_validation", finalize_distractor_validation)

     # TODO: Add document-fetching node
    graph.add_node("validate_quality", validate_quality)
    graph.add_node("correct_quality", correct_quality)

    graph.add_node("build_question_matching", build_question_matching)
    graph.add_node("build_options_matching", build_options_matching)
    graph.add_node("validate_quality_matching", validate_quality_matching)
    graph.add_node("correct_quality_matching", correct_quality_matching)
    graph.add_node("finalize_matching_output", finalize_matching_output)
    graph.add_node("output_guardrail", output_guardrail)

    graph.set_entry_point("start")

    # Route to the MCQ chain or the fully separate MATCHING chain based on
    # which config builder populated state (see _route_from_start). Each
    # run_pipeline() call is scoped to a single question type today, so this
    # is a simple either/or.
    graph.add_conditional_edges(
        "start",
        _route_from_start,
        {
            "mcq": "build_question",
            "matching": "build_question_matching",
        },
    )

    graph.add_edge("build_question", "validate_stem_clarity_bias")

    graph.add_conditional_edges(
        "validate_stem_clarity_bias",
        _route_after_stem_validation,
        {
            "regenerate": "regenerate_stem_clarity_bias",
            "continue": "validate_stem_lo_cognitive",
        },
    )
    graph.add_edge("regenerate_stem_clarity_bias", "validate_stem_clarity_bias")

    # Second stem gate: LO alignment + cognitive level. Runs after clarity/bias
    # resolves and before options are built, with its own regeneration loop.
    graph.add_conditional_edges(
        "validate_stem_lo_cognitive",
        _route_after_stem_lo_cognitive,
        {
            "regenerate": "regenerate_stem_lo_cognitive",
            "continue": "build_options",
        },
    )
    graph.add_edge("regenerate_stem_lo_cognitive", "validate_stem_lo_cognitive")
    
    # Guardrails check after generation, before validation (DISABLED for performance testing)
    # graph.add_edge("build_options", "guardrails_check")
    
    # Conditional routing based on guardrails results (DISABLED)
    # graph.add_conditional_edges(
    #     "guardrails_check",
    #     _route_after_guardrails,
    #     {
    #         "regenerate": "regenerate_for_guardrails",
    #         "finalize": "finalize_guardrails",
    #     },
    # )
    
    # Regeneration loops back to guardrails check (DISABLED)
    # graph.add_edge("regenerate_for_guardrails", "guardrails_check")
    
    # Fan out directly from build_options to all 7 validators (bypassing guardrails)
    graph.add_edge("build_options", "agent_check_plausibility")
    graph.add_edge("build_options", "agent_check_misconception")
    graph.add_edge("build_options", "agent_check_distinctiveness")
    graph.add_edge("build_options", "agent_check_incorrectness")
    graph.add_edge("build_options", "agent_check_clarity")
    graph.add_edge("build_options", "agent_check_parallelism")
    graph.add_edge("build_options", "agent_check_locale_compliance")

    graph.add_edge("agent_check_plausibility", "aggregate_validation_results")
    graph.add_edge("agent_check_misconception", "aggregate_validation_results")
    graph.add_edge("agent_check_distinctiveness", "aggregate_validation_results")
    graph.add_edge("agent_check_incorrectness", "aggregate_validation_results")
    graph.add_edge("agent_check_clarity", "aggregate_validation_results")
    graph.add_edge("agent_check_parallelism", "aggregate_validation_results")
    graph.add_edge("agent_check_locale_compliance", "aggregate_validation_results")
    
    graph.add_conditional_edges(
        "aggregate_validation_results",
        _route_after_distractor_validation,
        {
            "correct": "correct_failed_distractors",
            "finalize": "finalize_distractor_validation",
        },
    )
    
    graph.add_edge("correct_failed_distractors", "agent_check_plausibility")
    graph.add_edge("correct_failed_distractors", "agent_check_misconception")
    graph.add_edge("correct_failed_distractors", "agent_check_distinctiveness")
    graph.add_edge("correct_failed_distractors", "agent_check_incorrectness")
    graph.add_edge("correct_failed_distractors", "agent_check_clarity")
    graph.add_edge("correct_failed_distractors", "agent_check_parallelism")
    graph.add_edge("correct_failed_distractors", "agent_check_locale_compliance")
    
    graph.add_edge("finalize_distractor_validation", "validate_quality")
    
    graph.add_conditional_edges(
        "validate_quality",
        _route_from_validate_quality,
        {"correct_quality": "correct_quality", "end": "output_guardrail"},
    )
    
    graph.add_edge("correct_quality", "validate_quality")

    graph.add_edge("build_question_matching", "build_options_matching")
    graph.add_edge("build_options_matching", "validate_quality_matching")

    graph.add_conditional_edges(
        "validate_quality_matching",
        _route_from_validate_quality_matching,
        {"correct_quality_matching": "correct_quality_matching", "end": "finalize_matching_output"},
    )

    graph.add_edge("correct_quality_matching", "validate_quality_matching")
    graph.add_edge("finalize_matching_output", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    # Compile and cache the graph for reuse across all requests
    _compiled_graph = graph.compile()
    logger.info("GRAPH: Compilation complete - cached for all future requests")
    return _compiled_graph


def reset_compiled_graph():
    """Reset the cached compiled graph. Useful for development/testing."""
    global _compiled_graph
    _compiled_graph = None


__all__ = ["build_question_generation_graph", "reset_compiled_graph"]
