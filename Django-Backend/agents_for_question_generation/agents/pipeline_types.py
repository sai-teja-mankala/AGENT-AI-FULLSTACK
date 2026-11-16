from typing import Any, Dict, List, Literal, Optional, TypedDict
from dataclasses import dataclass

from .pipeline_constants import GENERATION_TASK_TYPES

QuestionType = Literal["MULTIPLE_CHOICE", "MULTIPLE_CHOICE_MULTI_SELECT", "SCENARIO_BASED", "MATCHING"]


@dataclass
class PipelineConfig:
    """Centralized pipeline configuration with type safety."""
    generation_client: Any  # OpenAI-compatible client for Claude
    generation_deployment: str
    validation_client: Any  # OpenAI-compatible client for Claude
    validation_deployment: str
    research_config: Dict[str, Any]
    
    def get_client_and_deployment(self, task_type: str) -> tuple[Any, str]:
        """Get appropriate client and deployment for task type."""
        if task_type in GENERATION_TASK_TYPES:
            return self.generation_client, self.generation_deployment
        else:
            return self.validation_client, self.validation_deployment


class ResearchConfig(TypedDict, total=False):
    """Research configuration for custom model selection and reasoning levels."""
    generationModel: str
    validationModel: str


class PipelineInput(TypedDict, total=False):
    sourceText: str
    learningObjectives: List[Any]
    numberOfQuestions: int
    questionTypes: List[QuestionType]
    difficultyLevels: List[str]
    numCorrectOptions: Optional[int]
    numIncorrectOptions: Optional[int]
    qualityRubric: str
    qualityThreshold: int
    relevancyThreshold: int
    maxAttempts: int
    maxDistractorFixAttempts: int
    maxQualityFixAttempts: int
    maxFormatFixAttempts: int
    learningObjectiveUuid: str
    research: ResearchConfig  # Optional Research configuration
    ksPrioritizeEnabled: bool  # Source-grounded mode (prioritize_knowledge_source)


class PromptPayload(TypedDict):
    systemPrompt: str
    userPrompt: str
    responseFormat: str
    learningObjective: str
    difficultyLevel: str
    questionType: QuestionType
    questionIndex: int  # Tracks which question this is (1, 2, 3...)
    ksPrioritizeEnabled: bool  # Forwarded per-config; selects source-grounded system contract


class GraphState(TypedDict):
    input: PipelineInput
    question_configs: List[PromptPayload]
    questions: List[Dict[str, Any]]
    raw_outputs: List[Dict[str, Any]]
    improved_outputs: List[Dict[str, Any]]
    formatted: List[Dict[str, Any]]
    
    # Research configuration (optional custom models and reasoning levels)
    research: ResearchConfig
    
    # Pipeline configuration (shared across all nodes)  
    pipeline_config: Optional[Any]
    
    # Tracking identifiers (available throughout the pipeline)
    container_id: str
    task_id: str
    
    # Parallel validation agent outputs (each agent writes to its own key)
    validation_check_plausibility: List[Dict[str, Any]]
    validation_check_misconception: List[Dict[str, Any]]
    validation_check_distinctiveness: List[Dict[str, Any]]
    validation_check_incorrectness: List[Dict[str, Any]]
    validation_check_clarity: List[Dict[str, Any]]
    validation_check_parallelism: List[Dict[str, Any]]
    validation_check_locale_compliance: List[Dict[str, Any]]
    
    # Aggregated validation outputs
    validated_outputs: List[Dict[str, Any]]
    distractor_corrections_output: List[Dict[str, Any]]
    distractor_correction_attempt: int
    passed_questions_final: List[Dict[str, Any]]
    # Quality validation fields
    quality_validation_passed: List[Dict[str, Any]]
    quality_validation_failed: List[Dict[str, Any]]
    quality_correction_attempts: int
    
    # Format validation fields
    format_fix_attempts: int
    format_loop: bool
    
    # Guardrails fields
    guardrails_passed: List[Dict[str, Any]]
    guardrails_failed: List[Dict[str, Any]]
    guardrails_regeneration_attempts: int

    # Pipeline-level warnings/errors (output_guardrail, build failures, etc.).
    # Must be declared here: LangGraph drops undeclared keys from node returns,
    # which previously silently swallowed output-guardrail warnings.
    pipeline_errors: List[Dict[str, Any]]

    # Stem clarity/bias validation fields
    stem_validation_passed: List[Dict[str, Any]]
    stem_validation_failed: List[Dict[str, Any]]
    stem_validation_attempt: int

    # Stem LO-alignment / cognitive-level validation fields
    stem_lo_cog_validation_passed: List[Dict[str, Any]]
    stem_lo_cog_validation_failed: List[Dict[str, Any]]
    stem_lo_cog_validation_attempt: int

    matching_question_configs: List[PromptPayload]
    matching_questions: List[Dict[str, Any]]
    matching_raw_outputs: List[Dict[str, Any]]
    matching_quality_passed: List[Dict[str, Any]]
    matching_quality_failed: List[Dict[str, Any]]
    matching_quality_correction_attempts: int
