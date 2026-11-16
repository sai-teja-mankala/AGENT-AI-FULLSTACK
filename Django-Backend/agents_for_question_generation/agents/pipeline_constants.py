"""
Configuration constants for the question generation pipeline.
"""

import os

# ==========================================
# FEATURE FLAGS
# ==========================================

# Enable structured output for guaranteed JSON compliance
_structured_output_env = os.getenv("AGENT_USE_STRUCTURED_OUTPUT", "false")
USE_STRUCTURED_OUTPUT = _structured_output_env.lower() == "true"


# Task type constants for better maintainability and type safety
GENERATION_TASK_TYPES = {"generation", "correction", "quality_correction"}
VALIDATION_TASK_TYPES = {"validation", "format_fix", "quality_check"}

# ==========================================
# BATCH CONFIGURATION
# ==========================================
# Optimal batch size for question generation
QUESTION_GENERATION_BATCH_SIZE = 5

OPTIONS_GENERATION_BATCH_SIZE = 3
MIN_QUESTIONS_FOR_BATCHING = 2

# ==========================================
# RETRY LIMITS
# ==========================================

MAX_DISTRACTOR_CORRECTION_ATTEMPTS = 1
MAX_QUALITY_CORRECTION_ATTEMPTS = 1
# Separate from MAX_QUALITY_CORRECTION_ATTEMPTS so MATCHING's correction loop
# can be tuned independently of MCQ's without affecting MCQ's retry budget.
MAX_MATCHING_QUALITY_CORRECTION_ATTEMPTS = 1
MAX_FORMAT_CORRECTION_ATTEMPTS = 2
MAX_GUARDRAIL_REGENERATION_ATTEMPTS = 2
MAX_STEM_VALIDATION_ATTEMPTS = 1
MAX_STEM_LO_COG_VALIDATION_ATTEMPTS = 1
# Individual question retry attempts for batch shortfalls
MAX_QUESTION_RETRY_ATTEMPTS = 2

# Rate limit handling
RATE_LIMIT_RETRY_DELAY_SECONDS = 30

# ==========================================
# DISTRACTOR VALIDATION - WEIGHTED SCORING
# ==========================================

CHECK_THRESHOLDS = {
    "plausibility": 85,
    "incorrectness": 85,
    "misconception": 80,
    "distinctiveness": 80,
    "clarity": 80,
    "scenario_coherence": 80,
    "parallelism": 70,
    "locale_compliance": 90
}

QUALITY_PASS_THRESHOLD = 85


# ================================
# MEMORY CLEANUP KEY GROUPS
# ================================

# Early pipeline data that gets replaced with processed versions
EARLY_PIPELINE_KEYS = ["question_configs", "scenarios", "questions"]

# Stem clarity/bias validation state (cleared after build_options)
STEM_VALIDATION_KEYS = [
    "stem_validation_passed",
    "stem_validation_failed",
    "stem_validation_attempt",
]

# Stem LO-alignment / cognitive-level validation state (cleared after build_options)
STEM_LO_COG_VALIDATION_KEYS = [
    "stem_lo_cog_validation_passed",
    "stem_lo_cog_validation_failed",
    "stem_lo_cog_validation_attempt",
]

# Individual validation agent results (cleared after aggregation)
VALIDATION_AGENT_KEYS = [
    "validation_check_plausibility", "validation_check_misconception", 
    "validation_check_distinctiveness", "validation_check_incorrectness",
    "validation_check_clarity", "validation_check_parallelism",
    "validation_check_scenario_coherence", "validation_check_locale_compliance"
]

# Distractor validation state (cleared after finalization)
DISTRACTOR_VALIDATION_KEYS = ["validated_outputs", "distractor_corrections_output", "passed_questions_final"]

# MATCHING ("match the following") pipeline state - fully separate from the MCQ
# keys above so the two flows can never read/write each other's state.
# Note: final passed results are written directly to the shared
# 'improved_outputs' key (mirroring MCQ's validate_quality) rather than a
# separate matching_improved_outputs key, so there's no matching-specific
# "final output" key to list here - just the transient working state.
MATCHING_PIPELINE_KEYS = [
    "matching_quality_passed",
    "matching_quality_failed",
    "matching_quality_correction_attempts",
]

# Final cleanup - pipeline_config and research only cleared at the very end
FINAL_PIPELINE_KEYS = ["pipeline_config", "research"]
