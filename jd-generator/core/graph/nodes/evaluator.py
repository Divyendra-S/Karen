"""Response evaluator node for intelligent processing of user input."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from ...models.graph_state import GraphState
from ...services.llm_service import evaluate_response, EvaluationResult


def response_evaluator_node(state: GraphState) -> Dict[str, Any]:
    """Evaluate and process user response using LLM analysis.

    This node intelligently extracts information from user responses,
    potentially filling multiple fields at once and correcting issues.

    Args:
        state: Current graph state

    Returns:
        State updates with evaluation results and extracted data
    """
    # Get user input and context
    user_input = _get_last_user_message_content(state)
    current_field = state.get("current_field")
    job_data = state.get("job_data", {})

    if not user_input:
        return _add_evaluation_error(state, "No user input to evaluate")

    # Get recent conversation context for LLM
    conversation_context = _get_conversation_context(state)

    # Evaluate the response using LLM
    evaluation_result = evaluate_response(
        user_input=user_input,
        current_field=current_field,
        job_data=job_data,
        conversation_context=conversation_context,
    )

    # Process evaluation results
    if evaluation_result.needs_clarification:
        return _handle_clarification_needed(state, evaluation_result)

    if evaluation_result.validation_issues:
        return _handle_validation_issues(state, evaluation_result)

    # Successfully evaluated - prepare extracted data for state update
    return _prepare_successful_evaluation(state, evaluation_result)


def _get_last_user_message_content(state: GraphState) -> Optional[str]:
    """Get the content of the last user message."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def _get_conversation_context(
    state: GraphState, limit: int = 5
) -> List[Dict[str, Any]]:
    """Get recent conversation messages for context.

    Args:
        state: Current graph state
        limit: Maximum number of recent messages to include

    Returns:
        List of recent message dictionaries
    """
    messages = state.get("messages", [])
    recent_messages = messages[-limit:] if len(messages) > limit else messages

    # Return simplified message format for LLM context
    return [
        {
            "role": msg.get("role"),
            "content": msg.get("content", "")[:200],  # Truncate for efficiency
            "related_field": msg.get("related_field"),
        }
        for msg in recent_messages
        if msg.get("role") in ["user", "assistant"]
    ]


def _add_evaluation_error(state: GraphState, error_message: str) -> Dict[str, Any]:
    """Add evaluation error to state."""
    new_errors = state.get("validation_errors", []) + [error_message]
    return {**state, "validation_errors": new_errors, "evaluation_failed": True}


def _handle_clarification_needed(
    state: GraphState, evaluation_result: EvaluationResult
) -> Dict[str, Any]:
    """Handle case where user response needs clarification."""

    # Create clarification message
    clarification_message = _create_clarification_message(
        evaluation_result, state.get("current_field")
    )

    new_message = {
        "role": "assistant",
        "content": clarification_message,
        "message_type": "clarification",
        "timestamp": datetime.utcnow().isoformat(),
        "related_field": state.get("current_field"),
    }

    new_messages = state.get("messages", []) + [new_message]

    return {
        **state,
        "messages": new_messages,
        "evaluation_result": evaluation_result.model_dump(),
        "needs_clarification": True,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _handle_validation_issues(
    state: GraphState, evaluation_result: EvaluationResult
) -> Dict[str, Any]:
    """Handle validation issues found during evaluation."""

    # If confidence is still reasonable, proceed with corrections
    if evaluation_result.confidence_score >= 0.7:
        return _prepare_successful_evaluation(state, evaluation_result)

    # Low confidence - ask for clarification
    validation_message = _create_validation_message(evaluation_result.validation_issues)

    new_message = {
        "role": "assistant",
        "content": validation_message,
        "message_type": "clarification",
        "timestamp": datetime.utcnow().isoformat(),
        "related_field": state.get("current_field"),
    }

    new_messages = state.get("messages", []) + [new_message]

    return {
        **state,
        "messages": new_messages,
        "evaluation_result": evaluation_result.model_dump(),
        "validation_errors": evaluation_result.validation_issues,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _prepare_successful_evaluation(
    state: GraphState, evaluation_result: EvaluationResult
) -> Dict[str, Any]:
    """Prepare state updates for successful evaluation."""

    # Store evaluation result and extracted fields for state updater
    return {
        **state,
        "evaluation_result": evaluation_result.model_dump(),
        "extracted_fields": evaluation_result.extracted_fields,
        "corrected_input": evaluation_result.corrected_input,
        "validation_errors": [],
        "needs_clarification": False,
    }


def _create_clarification_message(
    evaluation_result: EvaluationResult, current_field: Optional[str]
) -> str:
    """Create message requesting clarification."""

    intent = evaluation_result.intent_analysis
    response_type = intent.get("response_type", "unclear")

    if response_type == "off_topic":
        return (
            "I notice your response doesn't seem to address the current question. "
            f"Could you please provide information about {current_field or 'the current field'}?"
        )

    if response_type == "partial_answer":
        return (
            "I understand part of what you've shared, but I'd like some clarification. "
            "Could you provide a bit more detail or rephrase your response?"
        )

    # Default clarification
    return (
        "I want to make sure I understand your response correctly. "
        "Could you please clarify or provide a bit more information?"
    )


def _create_validation_message(validation_issues: List[str]) -> str:
    """Create message for validation issues."""
    if len(validation_issues) == 1:
        return (
            f"I noticed an issue: {validation_issues[0]} "
            "Could you please clarify or provide the information differently?"
        )

    issues_text = "\n".join(f"• {issue}" for issue in validation_issues[:3])
    return (
        f"I found a few things that need clarification:\n{issues_text}\n\n"
        "Could you help me understand these points better?"
    )


def merge_extracted_fields_with_job_data(
    existing_job_data: Dict[str, Any], extracted_fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge extracted fields with existing job data intelligently.

    Args:
        existing_job_data: Current job data
        extracted_fields: Fields extracted from evaluation

    Returns:
        Merged job data with extracted fields
    """
    merged_data = existing_job_data.copy()

    for field_name, field_value in extracted_fields.items():
        if field_value is not None and field_value != "":
            # Special handling for complex fields
            if field_name == "skills" and isinstance(field_value, dict):
                merged_data[field_name] = _merge_skills_data(
                    existing_job_data.get(field_name, {}), field_value
                )
            elif field_name == "responsibilities" and isinstance(field_value, list):
                existing_resp = existing_job_data.get(field_name, [])
                merged_data[field_name] = existing_resp + field_value
            elif field_name == "benefits" and isinstance(field_value, list):
                existing_benefits = existing_job_data.get(field_name, [])
                merged_data[field_name] = existing_benefits + field_value
            else:
                # Simple field replacement
                merged_data[field_name] = field_value

    return merged_data


def _merge_skills_data(
    existing_skills: Dict[str, Any], new_skills: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge skills data intelligently."""
    merged = existing_skills.copy()

    # Merge technical skills
    existing_tech = merged.get("technical_skills", [])
    new_tech = new_skills.get("technical_skills", [])
    merged["technical_skills"] = existing_tech + new_tech

    # Merge programming languages
    existing_langs = merged.get("programming_languages", [])
    new_langs = new_skills.get("programming_languages", [])
    merged["programming_languages"] = list(set(existing_langs + new_langs))

    # Merge frameworks and tools
    existing_tools = merged.get("frameworks_tools", [])
    new_tools = new_skills.get("frameworks_tools", [])
    merged["frameworks_tools"] = list(set(existing_tools + new_tools))

    return merged


def get_fields_answered_by_evaluation(evaluation_result: EvaluationResult) -> List[str]:
    """Get list of fields that were answered by the evaluation.

    Args:
        evaluation_result: Result from LLM evaluation

    Returns:
        List of field names that have been answered
    """
    return [
        field_name
        for field_name, field_value in evaluation_result.extracted_fields.items()
        if field_value is not None and field_value != "" and field_value != []
    ]


def should_skip_field_based_on_evaluation(
    field_name: str, evaluation_result: EvaluationResult
) -> bool:
    """Determine if a field should be skipped based on evaluation results.

    Args:
        field_name: Field to check
        evaluation_result: Evaluation result

    Returns:
        True if field should be skipped (already answered)
    """
    answered_fields = get_fields_answered_by_evaluation(evaluation_result)
    return field_name in answered_fields


def calculate_evaluation_quality_score(evaluation_result: EvaluationResult) -> float:
    """Calculate overall quality score for the evaluation.

    Args:
        evaluation_result: Result from evaluation

    Returns:
        Quality score from 0.0 to 1.0
    """
    base_score = evaluation_result.confidence_score

    # Adjust based on extracted fields count
    field_count = len(evaluation_result.extracted_fields)
    field_bonus = min(field_count * 0.1, 0.3)  # Max 0.3 bonus

    # Adjust based on validation issues
    issue_penalty = len(evaluation_result.validation_issues) * 0.1

    # Adjust based on intent analysis
    intent = evaluation_result.intent_analysis
    if intent.get("completeness") == "complete":
        completeness_bonus = 0.1
    elif intent.get("completeness") == "partial":
        completeness_bonus = 0.05
    else:
        completeness_bonus = 0.0

    final_score = base_score + field_bonus + completeness_bonus - issue_penalty
    return max(0.0, min(1.0, final_score))


def create_evaluation_summary(evaluation_result: EvaluationResult) -> str:
    """Create human-readable summary of evaluation results.

    Args:
        evaluation_result: Result from evaluation

    Returns:
        Formatted summary string
    """
    summary_parts = []

    # Fields extracted
    field_count = len(evaluation_result.extracted_fields)
    if field_count > 0:
        summary_parts.append(f"✓ Extracted information for {field_count} field(s)")

    # Quality indicators
    confidence = evaluation_result.confidence_score
    if confidence >= 0.8:
        summary_parts.append("✓ High confidence in understanding")
    elif confidence >= 0.6:
        summary_parts.append("⚠ Medium confidence in understanding")
    else:
        summary_parts.append("⚠ Low confidence - may need clarification")

    # Issues
    if evaluation_result.validation_issues:
        summary_parts.append(
            f"⚠ {len(evaluation_result.validation_issues)} issue(s) found"
        )

    return " | ".join(summary_parts)
