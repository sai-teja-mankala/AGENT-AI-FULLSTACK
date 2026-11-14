"""
OpenAI-based question generation agent using LangGraph pipeline.
"""

import logging
from typing import Any, Dict, List

from billiard.exceptions import SoftTimeLimitExceeded, TimeLimitExceeded
from .langgraph_pipeline import run_pipeline, PipelineCancelled

logger = logging.getLogger(__name__)


class QuestionGenerationError(Exception):
    """Exception raised when question generation fails."""
    pass


class OpenAIQuestionAgent:
    """
    Question generation agent using LangGraph pipeline with OpenAI models.

    This agent uses a sophisticated multi-stage generation pipeline that includes:
    - Scenario generation (for Intermediate/Advanced questions)
    - Question text generation with Bloom's Taxonomy alignment
    - Answer options generation
    - Distractor quality validation and improvement
    - Quality validation with automatic feedback loops
    """

    def generate_questions(self, pipeline_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate questions using LangGraph pipeline.

        Args:
            pipeline_payload: Complete pipeline payload (passed directly to pipeline)

        Returns:
            List of generated questions with full metadata

        Raises:
            QuestionGenerationError: If generation fails
        """
        try:
            logger.info(
                "Running LangGraph pipeline with payload: LOs=%s, Questions=%s",
                len(pipeline_payload.get("learningObjectives", [])),
                pipeline_payload.get("numberOfQuestions", 5),
            )
            
            # Run the pipeline with the payload directly
            result = run_pipeline(pipeline_payload)
            return result
            
        except SoftTimeLimitExceeded:
            raise  # Let it propagate as-is to the task layer
        except TimeLimitExceeded:
            raise
        except PipelineCancelled:
            raise
        except Exception as e:
            logger.error("Question generation failed: %s", str(e), exc_info=True)
            raise QuestionGenerationError(f"Failed to generate questions: {str(e)}")
