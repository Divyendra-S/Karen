"""Simplified conditional edge logic for graph transitions."""

from typing import List, Optional, Dict, Any
from ...models.graph_state import GraphState


def get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been successfully processed."""
    return list(state.get("processed_data", {}).keys())


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
    current_field = state.get("current_field")
    retry_count = state.get("retry_count", 0)
    
    # Safety: If no current field or too many retries, we're done
    if not current_field or retry_count >= 3:
        return "complete"
    
    # If we have a field to collect, continue
    if current_field:
        return "continue"
    
    # Check if we have minimum required data
    required_fields = {"job_title", "skills"}  # Minimum for basic JD
    collected_fields = set(state.get("processed_data", {}).keys())
    
    if not required_fields.issubset(collected_fields):
        return "continue"
    
    return "complete"


def should_generate_jd(state: GraphState) -> str:
    """Determine if we should generate JD or continue collecting.
    
    Args:
        state: Current graph state
        
    Returns:
        "generate" if ready for JD generation, "continue" if more data needed
    """
    # Check minimum required fields
    required_fields = {"job_title", "skills"}
    collected_fields = set(state.get("processed_data", {}).keys())
    
    if required_fields.issubset(collected_fields):
        return "generate"
    
    # Check if user explicitly wants to generate
    if _user_indicated_completion(state):
        return "generate"
    
    return "continue"


def route_to_next_question(state: GraphState) -> str:
    """Simple routing - always generate questions when called.
    
    Args:
        state: Current graph state
        
    Returns:
        Always returns "question_generator"
    """
    return "question_generator"


def _user_indicated_completion(state: GraphState) -> bool:
    """Check if user has indicated they want to complete the conversation."""
    last_message = _get_last_user_message_content(state)
    if not last_message:
        return False
    
    completion_indicators = [
        "generate", "create jd", "done", "finished", "that's all", 
        "complete", "ready", "let's generate", "i'm done"
    ]
    
    last_message_lower = last_message.lower()
    return any(indicator in last_message_lower for indicator in completion_indicators)


def _get_last_user_message_content(state: GraphState) -> Optional[str]:
    """Helper to get last user message content."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def should_retry_field(state: GraphState) -> bool:
    """Check if we should retry collecting current field."""
    retry_count = state.get("retry_count", 0)
    current_field = state.get("current_field")
    clarification_needed = state.get("clarification_needed", [])
    
    return (
        retry_count < 3 and 
        current_field is not None and 
        current_field in clarification_needed
    )


def get_conversation_context(state: GraphState) -> Dict[str, Any]:
    """Extract conversation context for decision making."""
    messages = state.get("messages", [])
    processed_data = state.get("processed_data", {})
    
    return {
        "message_count": len(messages),
        "user_message_count": len([m for m in messages if m.get("role") == "user"]),
        "collected_field_count": len(get_collected_fields(state)),
        "needs_clarification": bool(state.get("clarification_needed")),
        "retry_count": state.get("retry_count", 0),
        "current_phase": state.get("conversation_phase", "greeting"),
        "job_title": processed_data.get("job_title", ""),
        "employment_type": processed_data.get("employment_type", "")
    }


def should_ask_optional_fields(state: GraphState) -> bool:
    """Check if we should ask about optional fields."""
    # Check if all required fields are collected
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(get_collected_fields(state))
    
    if not required_fields.issubset(collected_fields):
        return False
    
    # Check conversation length - ask optional if conversation is not too long
    message_count = len(state.get("messages", []))
    return message_count < 20


def is_field_collection_complete(state: GraphState) -> bool:
    """Check if field collection is logically complete."""
    # Must have required fields
    if not is_ready_for_generation(state):
        return False
    
    # Check if user explicitly wants to finish
    if _user_indicated_completion(state):
        return True
    
    # Check if no current field (natural end)
    current_field = state.get("current_field")
    return current_field is None


def needs_clarification(state: GraphState) -> bool:
    """Check if current response needs clarification."""
    current_field = state.get("current_field")
    clarification_needed = state.get("clarification_needed", [])
    
    return current_field and current_field in clarification_needed


def calculate_conversation_progress(state: GraphState) -> float:
    """Calculate overall conversation progress."""
    total_possible_fields = 11
    completed_fields = len(get_collected_fields(state))
    
    return completed_fields / total_possible_fields