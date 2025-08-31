"""LLM service for evaluation and processing tasks using Groq."""

from typing import Dict, Any, List, Optional
from groq import Groq
from pydantic import BaseModel, Field
import json
import logging
from app.config import settings


class EvaluationRequest(BaseModel):
    """Request model for LLM evaluation."""

    user_input: str = Field(description="Raw user input to evaluate")
    current_field: Optional[str] = Field(
        None, description="Field currently being collected"
    )
    job_data: Dict[str, Any] = Field(
        default_factory=dict, description="Existing job data"
    )
    conversation_context: List[Dict[str, Any]] = Field(
        default_factory=list, description="Recent conversation messages for context"
    )


class EvaluationResult(BaseModel):
    """Result model for LLM evaluation."""

    extracted_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Fields extracted from user input"
    )
    corrected_input: str = Field(description="Spell-checked and normalized input")
    intent_analysis: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis of user intent and response quality"
    )
    validation_issues: List[str] = Field(
        default_factory=list, description="Issues found during validation"
    )
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the evaluation (0-1)"
    )
    needs_clarification: bool = Field(
        default=False, description="Whether response needs clarification"
    )


class LLMService:
    """Service for making LLM calls for evaluation and processing."""

    def __init__(self):
        """Initialize LLM service with Groq client."""
        self.client = Groq(api_key=settings.get_llm_api_key())
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    def evaluate_user_response(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate user response and extract information.

        Args:
            request: Evaluation request with user input and context

        Returns:
            Evaluation result with extracted fields and corrections
        """
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(request)

            # Make LLM call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for more consistent evaluation
                max_tokens=self.max_tokens,
            )

            # Parse response
            return self._parse_evaluation_response(
                response.choices[0].message.content, request.user_input
            )

        except Exception as e:
            logging.error(f"LLM evaluation error: {e}")
            return EvaluationResult(
                corrected_input=request.user_input,
                confidence_score=0.0,
                validation_issues=[f"Evaluation service error: {str(e)}"],
            )

    def _create_evaluation_prompt(self, request: EvaluationRequest) -> str:
        """Create prompt for LLM evaluation."""

        # Get list of all possible fields
        all_fields = [
            "job_title",
            "department",
            "employment_type",
            "location",
            "experience",
            "responsibilities",
            "skills",
            "education",
            "salary",
            "benefits",
            "additional_requirements",
        ]

        # Build context about what's already collected
        collected_fields = list(request.job_data.keys())
        missing_fields = [f for f in all_fields if f not in collected_fields]

        prompt = f"""You are an AI assistant helping to extract and validate job requirement information from user responses.

Current Context:
- Current field being asked about: {request.current_field or "None"}
- Already collected fields: {collected_fields}
- Still needed fields: {missing_fields}

User's Response: "{request.user_input}"

Your task is to:
1. Extract ONLY explicitly mentioned job-related information from this response
2. Correct spelling/grammar mistakes 
3. Validate the extracted information
4. Assess the quality and completeness of the response

CRITICAL: BE EXTREMELY CONSERVATIVE - DO NOT EXTRACT ANYTHING NOT EXPLICITLY STATED.

IMPORTANT: Respond with ONLY a JSON object, no explanations or additional text. The JSON should contain:

{{
  "extracted_fields": {{
    // ONLY include fields that are EXPLICITLY mentioned in the user's response
    // Example: If user says "I am a Software Engineer", include: "job_title": "Software Engineer"
    // Example: If user says "I have 3 years experience", include: "experience": {{"years_min": 3}}
    // DO NOT include any field unless the user specifically mentions it
  }},
  "corrected_input": "spell-checked and grammatically correct version of input",
  "intent_analysis": {{
    "response_type": "direct_answer|partial_answer|clarification_needed|off_topic",
    "completeness": "complete|partial|incomplete", 
    "confidence_level": "high|medium|low",
    "mentions_multiple_fields": boolean
  }},
  "validation_issues": ["list of any issues found"],
  "confidence_score": 0.8,
  "needs_clarification": false
}}

ULTRA-STRICT RULES - EXTRACT ONLY WHAT IS DIRECTLY STATED:

✅ DO EXTRACT IF USER EXPLICITLY SAYS:
- "I am a Software Engineer" → job_title: "Software Engineer"  
- "I have 3 years experience" → experience: {{"years_min": 3}}
- "I know Python and React" → skills: {{"programming_languages": ["Python"], "frameworks_tools": ["React"]}}
- "I want to work remotely" → location: {{"location_type": "remote"}}
- "I want full-time work" → employment_type: "full_time"

❌ DO NOT EXTRACT IF USER DOES NOT MENTION:
- User says "I am an AI Engineer" but doesn't mention location → DO NOT extract location
- User mentions experience but not employment type → DO NOT extract employment_type  
- User doesn't mention skills → DO NOT extract skills
- User doesn't mention salary → DO NOT extract salary

CRITICAL WARNINGS:
- If user only mentions job title and experience, ONLY extract those 2 fields
- Do NOT assume location, employment type, education, skills unless explicitly stated
- Do NOT create empty objects or default values
- When in doubt, DO NOT extract - be extremely conservative
"""

        return prompt

    def _parse_evaluation_response(
        self, llm_response: str, original_input: str
    ) -> EvaluationResult:
        """Parse LLM response into EvaluationResult."""
        try:
            # Extract JSON from markdown code blocks if present
            json_text = self._extract_json_from_response(llm_response)

            # Parse JSON response
            response_data = json.loads(json_text)

            # Create EvaluationResult
            return EvaluationResult(
                extracted_fields=response_data.get("extracted_fields", {}),
                corrected_input=response_data.get("corrected_input", original_input),
                intent_analysis=response_data.get("intent_analysis", {}),
                validation_issues=response_data.get("validation_issues", []),
                confidence_score=response_data.get("confidence_score", 0.5),
                needs_clarification=response_data.get("needs_clarification", False),
            )

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response: {e}")
            logging.error(f"Raw response: {llm_response}")
            return EvaluationResult(
                corrected_input=original_input,
                confidence_score=0.0,
                validation_issues=["Failed to parse evaluation response"],
            )
        except Exception as e:
            logging.error(f"Unexpected error parsing LLM response: {e}")
            return EvaluationResult(
                corrected_input=original_input,
                confidence_score=0.0,
                validation_issues=[f"Evaluation parsing error: {str(e)}"],
            )

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from markdown-formatted response.

        Args:
            response: Raw LLM response that may contain markdown

        Returns:
            Clean JSON string
        """
        import re

        # Remove markdown code blocks
        json_pattern = r"```(?:json)?\s*(.*?)\s*```"
        json_match = re.search(json_pattern, response, re.DOTALL)

        if json_match:
            return json_match.group(1).strip()

        # If no code blocks, try to find JSON-like content
        # Look for content between first { and last }
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx : end_idx + 1]

        # Return as-is if no pattern found
        return response.strip()

    def spell_check_and_normalize(self, text: str) -> str:
        """Simple spell check and normalization using LLM.

        Args:
            text: Text to spell check and normalize

        Returns:
            Corrected and normalized text
        """
        try:
            prompt = f"""Please correct any spelling mistakes and improve grammar in this text, but keep the meaning exactly the same:

"{text}"

Respond with ONLY the corrected text, no explanations or additional content."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            corrected = response.choices[0].message.content.strip()
            return corrected if corrected else text

        except Exception as e:
            logging.error(f"Spell check error: {e}")
            return text


# Create singleton instance
llm_service = LLMService()


def get_llm_service() -> LLMService:
    """Get the singleton LLM service instance."""
    return llm_service


# Convenience functions for common operations
def evaluate_response(
    user_input: str,
    current_field: Optional[str] = None,
    job_data: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
) -> EvaluationResult:
    """Convenience function for evaluating user responses.

    Args:
        user_input: User's raw input
        current_field: Field currently being collected
        job_data: Existing job data
        conversation_context: Recent conversation messages

    Returns:
        Evaluation result with extracted information
    """
    request = EvaluationRequest(
        user_input=user_input,
        current_field=current_field,
        job_data=job_data or {},
        conversation_context=conversation_context or [],
    )

    return llm_service.evaluate_user_response(request)


def quick_spell_check(text: str) -> str:
    """Quick spell check and normalization.

    Args:
        text: Text to correct

    Returns:
        Corrected text
    """
    return llm_service.spell_check_and_normalize(text)
