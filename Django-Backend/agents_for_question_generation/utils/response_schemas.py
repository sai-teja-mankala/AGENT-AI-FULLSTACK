"""
Pydantic schemas for structured LLM responses in the agent pipeline.
These schemas ensure guaranteed JSON compliance and eliminate parsing errors.
"""

from typing import List, Dict, Optional, Any, Annotated, Type
from pydantic import BaseModel, Field, model_validator


class Answer(BaseModel):
    """Individual answer option schema."""
    answerText: str = Field(..., description="The answer option text")
    explanation: str = Field(..., description="Explanation for why this option is correct/incorrect")
    isCorrect: bool = Field(..., description="Whether this is a correct answer")


class Question(BaseModel):
    """Individual question schema."""
    questionText: str = Field(..., description="The question text")
    answer: Annotated[List[Answer], Field(min_length=3, max_length=6, description="List of answer options")]


class QuestionGenerationResponse(BaseModel):
    """Response schema for question generation (questions + options)."""
    LearningObjective: str = Field(..., description="The learning objective being tested")
    locale: str = Field(..., description="Language locale for the content")
    questions: Annotated[List[Question], Field(min_length=1, description="Generated questions")]


class OptionsGenerationResponse(BaseModel):
    """Response schema for batch options generation."""
    answers: List[Answer] = Field(default_factory=list, description="Generated answer options")
    success: bool = Field(default=True, description="Whether generation was successful")


class ValidationResponse(BaseModel):
    """Response schema for validation checks."""
    score: float = Field(..., ge=0, le=100, description="Validation score (0-100)")
    feedback: str = Field(..., description="Detailed feedback on the validation")
    
    # Note: 'pass' is calculated automatically based on score vs threshold
    # so we don't include it in the schema to avoid conflicts


class DistractorValidationResponse(BaseModel):
    """Response schema for distractor-specific validation checks."""
    plausibility_score: float = Field(..., ge=0, le=100, description="Plausibility score (0-100)")
    incorrectness_confirmed: bool = Field(..., description="Whether distractor is definitively incorrect")
    misconception_type: str = Field(..., description="Type of misconception represented")
    distinctiveness_score: float = Field(..., ge=0, le=100, description="How distinct from other options")
    clarity_score: float = Field(..., ge=0, le=100, description="Clarity and readability score")
    overall_score: float = Field(..., ge=0, le=100, description="Overall distractor quality score")
    feedback: str = Field(..., description="Specific feedback for this distractor")
    needs_improvement: bool = Field(..., description="Whether this distractor needs correction")


class QualityCheckResponse(BaseModel):
    """Response schema for overall quality assessment."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    dimension_scores: Dict[str, float] = Field(
        ..., 
        description="Scores for specific quality dimensions"
    )
    feedback: str = Field(..., description="Comprehensive quality feedback")
    needs_improvement: bool = Field(..., description="Whether quality correction is needed")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")


class CorrectionResponse(BaseModel):
    """Response schema for correction operations."""
    corrected_content: Dict[str, Any] = Field(..., description="The corrected content")
    changes_made: List[str] = Field(..., description="List of changes that were applied")
    improvement_summary: str = Field(..., description="Summary of improvements made")


# --- MATCHING-specific schemas ---

class MatchingStem(BaseModel):
    """Single MATCHING question stem with Column A terms."""
    questionText: str = Field(..., description="The 'Match the following...' stem text")
    columnATerms: List[str] = Field(..., min_length=2, max_length=4, description="Column A terms to be matched")


class MatchingStemGenerationResponse(BaseModel):
    """Structured output schema for MATCHING stem generation."""
    questions: List[MatchingStem] = Field(..., min_length=1, description="Generated MATCHING stems")


class MatchingPairing(BaseModel):
    """Single column-A → column-B pairing with explanation."""
    term: str = Field(..., description="Column A term text")
    definition: str = Field(..., description="Matching Column B definition text")
    explanation: str = Field(..., description="Why this pairing is correct")


class ColumnEntry(BaseModel):
    """Single column entry — fixed keys so OpenAI strict schema accepts it."""
    label: str = Field(..., description="Option label e.g. Option-A")
    text: str = Field(..., description="Term or definition text")


class MatchingQuestion(BaseModel):
    """Fully assembled MATCHING question with both columns and pairings."""
    question: str = Field(..., description="The question stem text")
    column_a_answers: List[ColumnEntry] = Field(..., min_length=2, description="Column A entries")
    column_b_answers: List[ColumnEntry] = Field(..., min_length=2, description="Column B definitions")
    answers: List[MatchingPairing] = Field(..., min_length=2, description="Correct pairings")

    @model_validator(mode="after")
    def _validate_matching_column_counts(self) -> "MatchingQuestion":
        counts = {len(self.column_a_answers), len(self.column_b_answers), len(self.answers)}
        if len(counts) > 1:
            raise ValueError(
                "column_a_answers, column_b_answers, and answers must all have the same length, "
                f"got {len(self.column_a_answers)}, {len(self.column_b_answers)}, {len(self.answers)}"
            )
        return self


class MatchingOptionsGenerationResponse(BaseModel):
    """Structured output schema for MATCHING options (Column B + pairings) generation."""
    questions: List[MatchingQuestion] = Field(..., min_length=1, description="Assembled MATCHING questions")
    
    
# Schema mapping for different operation types
RESPONSE_SCHEMAS = {
    "question_generation": QuestionGenerationResponse,
    "options_generation": OptionsGenerationResponse,
    "quality_check": QualityCheckResponse,
    "validation": ValidationResponse,
    "distractor_validation": DistractorValidationResponse,
    "correction": CorrectionResponse,
    "matching_stem_generation": MatchingStemGenerationResponse,
    "matching_options_generation": MatchingOptionsGenerationResponse,
}


def get_response_schema(operation_type: str) -> Optional[Type[BaseModel]]:
    """
    Get the appropriate response schema for an operation type.
    
    Args:
        operation_type: The type of operation being performed
        
    Returns:
        The corresponding Pydantic model class, or None if not found
    """
    return RESPONSE_SCHEMAS.get(operation_type)
