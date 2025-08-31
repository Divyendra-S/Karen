"""Conditional edge logic for graph transitions with pure function approach."""

from typing import Dict, Any, List, Optional
from ...models.graph_state import GraphState


def get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been collected."""
    return [
        field
        for field, value in state.get("job_data", {}).items()
        if value is not None and value != "" and value != []
    ]


def get_missing_required_fields(state: GraphState) -> List[str]:
    """Get list of required fields that are still missing."""
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(get_collected_fields(state))
    return list(required_fields - collected_fields)


def is_ready_for_generation(state: GraphState) -> bool:
    """Check if state has enough data for JD generation."""
    missing_required = get_missing_required_fields(state)
    return len(missing_required) == 0


def should_continue_collecting(state: GraphState) -> str:
    """Determine if we should continue collecting data or move to completion check.

    Args:
        state: Current graph state

    Returns:
        "continue" if more data needed, "complete" if ready for review
    """
    # Check if we have validation errors that need retry
    if state.get("validation_errors") and state.get("retry_count", 0) < 3:
        return "continue"

    # Check if current field was successfully collected
    current_field = state.get("current_field")
    if current_field and current_field not in get_collected_fields(state):
        return "continue"

    # Check if we have minimum required data
    missing_required = get_missing_required_fields(state)
    if missing_required:
        return "continue"

    return "complete"


def should_generate_jd(state: GraphState) -> str:
    """Determine if we should generate JD or continue collecting.

    Args:
        state: Current graph state

    Returns:
        "generate" if ready for JD generation, "continue" if more data needed
    """
    # Check if explicitly marked as complete
    if state.get("is_complete", False):
        return "generate"

    # Check if we have all required fields
    if is_ready_for_generation(state):
        return "generate"

    # Check if user has indicated they're done
    last_messages = state.get("messages", [])[-3:]  # Check last few messages
    user_done_indicators = [
        "that's all",
        "i'm done",
        "finished",
        "complete",
        "generate",
        "create jd",
        "no more",
    ]

    for msg in last_messages:
        if msg.get("role") == "user" and any(
            indicator in msg.get("content", "").lower()
            for indicator in user_done_indicators
        ):
            return "generate"

    return "continue"


def route_to_next_question(state: GraphState) -> str:
    """Route to the appropriate next step based on current state.

    Args:
        state: Current graph state

    Returns:
        Next node name to route to
    """
    # If conversation just started, go to question generation
    if not state.get("messages") or len(state["messages"]) <= 1:
        return "question_generator"

    # If we have validation errors and retries left, retry current question
    if (
        state.get("validation_errors")
        and state.get("retry_count", 0) < 3
        and state.get("current_field")
    ):
        return "question_generator"

    # Check if ready for completion
    if is_ready_for_generation(state):
        return "completeness_checker"

    # Check if user wants to generate with current data
    last_message = get_last_user_message_content(state)
    if last_message and any(
        keyword in last_message.lower()
        for keyword in ["generate", "create jd", "done", "finished"]
    ):
        return "completeness_checker"

    # Default to question generation for more data
    return "question_generator"


def should_retry_field(state: GraphState) -> bool:
    """Check if we should retry collecting current field.

    Args:
        state: Current graph state

    Returns:
        True if field should be retried
    """
    retry_count = state.get("retry_count", 0)
    has_errors = bool(state.get("validation_errors"))
    current_field = state.get("current_field")

    return has_errors and retry_count < 3 and current_field is not None


def determine_next_field(state: GraphState) -> Optional[str]:
    """Determine the next field to collect based on current state.

    Args:
        state: Current graph state

    Returns:
        Next field name to collect or None if all done
    """
    collected_fields = set(get_collected_fields(state))

    # Define collection priority order
    field_priority = [
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

    # Find next uncollected field
    for field in field_priority:
        if field not in collected_fields:
            return field

    return None


def should_ask_optional_fields(state: GraphState) -> bool:
    """Check if we should ask about optional fields.

    Args:
        state: Current graph state

    Returns:
        True if should collect optional fields
    """
    # Check if all required fields are collected
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(get_collected_fields(state))

    if not required_fields.issubset(collected_fields):
        return False

    # Check conversation length - ask optional if conversation is not too long
    message_count = len(state.get("messages", []))
    return message_count < 20  # Arbitrary limit to avoid overly long conversations


def get_conversation_context(state: GraphState) -> Dict[str, Any]:
    """Extract conversation context for decision making.

    Args:
        state: Current graph state

    Returns:
        Dictionary of context information
    """
    messages = state.get("messages", [])
    job_data = state.get("job_data", {})

    return {
        "message_count": len(messages),
        "user_message_count": len([m for m in messages if m.get("role") == "user"]),
        "collected_field_count": len(get_collected_fields(state)),
        "has_validation_errors": bool(state.get("validation_errors")),
        "retry_count": state.get("retry_count", 0),
        "current_phase": state.get("conversation_phase", "greeting"),
        "job_title": job_data.get("job_title", ""),
        "employment_type": job_data.get("employment_type", ""),
    }


def route_based_on_job_type(state: GraphState) -> str:
    """Route to different flows based on job type.

    Args:
        state: Current graph state

    Returns:
        Next node based on job type context
    """
    job_data = state.get("job_data", {})
    job_title = job_data.get("job_title", "").lower()

    # Technical roles need more detailed skill collection
    technical_indicators = [
        "engineer",
        "developer",
        "programmer",
        "architect",
        "devops",
        "data scientist",
        "analyst",
    ]

    if any(indicator in job_title for indicator in technical_indicators):
        # Focus on technical skills
        collected_fields = set(get_collected_fields(state))
        if "skills" not in collected_fields:
            return "question_generator"  # Prioritize skills collection

    # Management roles need leadership questions
    management_indicators = ["manager", "director", "lead", "head", "chief"]

    if any(indicator in job_title for indicator in management_indicators):
        # Focus on leadership and team management
        collected_fields = set(get_collected_fields(state))
        if "experience" not in collected_fields:
            return "question_generator"  # Prioritize experience collection

    # Default routing
    return "question_generator"


def handle_error_recovery(state: GraphState) -> str:
    """Handle error recovery routing.

    Args:
        state: Current graph state with errors

    Returns:
        Next node for error recovery
    """
    retry_count = state.get("retry_count", 0)

    if retry_count >= 3:
        # Too many retries, skip to next field or complete
        if get_missing_required_fields(state):
            return "question_generator"  # Try next required field
        else:
            return "completeness_checker"  # Move to completion

    # Retry current field
    return "question_generator"


def get_last_user_message_content(state: GraphState) -> Optional[str]:
    """Helper to get last user message content.

    Args:
        state: Current graph state

    Returns:
        Last user message content or None
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


# Edge decision functions for specific scenarios
def needs_clarification(state: GraphState) -> bool:
    """Check if current response needs clarification.

    Args:
        state: Current graph state

    Returns:
        True if clarification is needed
    """
    last_response = get_last_user_message_content(state)
    if not last_response:
        return False

    # Check for ambiguous responses
    ambiguous_responses = [
        "yes",
        "no",
        "maybe",
        "idk",
        "i don't know",
        "not sure",
        "depends",
        "?",
    ]

    return (
        len(last_response.strip()) < 3
        or last_response.lower().strip() in ambiguous_responses
    )


def is_response_complete(state: GraphState) -> bool:
    """Check if user response is complete for current field.

    Args:
        state: Current graph state

    Returns:
        True if response appears complete
    """
    current_field = state.get("current_field")
    last_response = get_last_user_message_content(state)

    if not current_field or not last_response:
        return False

    # Field-specific completeness checks
    if current_field == "responsibilities":
        # Responsibilities should be substantial
        return len(last_response.strip()) > 20

    if current_field == "skills":
        # Skills should have multiple items or detailed description
        return len(last_response.strip()) > 10

    if current_field in ["job_title", "department"]:
        # Basic fields need minimum length
        return len(last_response.strip()) > 2

    return len(last_response.strip()) > 5
