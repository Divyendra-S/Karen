"""State management utilities for LangGraph with functional patterns."""

from typing import Dict, Any, List, Optional, Callable, Tuple
from copy import deepcopy
from ..models.graph_state import GraphState, create_initial_graph_state
from ..models.job_requirements import JobRequirements


def create_state_updater(field_name: str) -> Callable[[GraphState, Any], GraphState]:
    """Higher-order function to create field-specific state updaters.
    
    Args:
        field_name: Name of the field to update
        
    Returns:
        Function that updates the specified field in state
    """
    def update_field(state: GraphState, value: Any) -> GraphState:
        """Update specific field in job data."""
        new_job_data = {**state["job_data"], field_name: value}
        return {**state, "job_data": new_job_data}
    
    return update_field


def add_system_message(state: GraphState, content: str) -> GraphState:
    """Add a system message to the state.
    
    Args:
        state: Current graph state
        content: System message content
        
    Returns:
        Updated state with new system message
    """
    system_message = {
        "role": "system",
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    new_messages = state["messages"] + [system_message]
    return {**state, "messages": new_messages}


def add_assistant_message(state: GraphState, content: str, message_type: str = "response") -> GraphState:
    """Add an assistant message to the state.
    
    Args:
        state: Current graph state
        content: Assistant message content
        message_type: Type of message (question, response, info, etc.)
        
    Returns:
        Updated state with new assistant message
    """
    assistant_message = {
        "role": "assistant",
        "content": content,
        "message_type": message_type,
        "timestamp": datetime.utcnow().isoformat(),
        "related_field": state.get("current_field")
    }
    
    new_messages = state["messages"] + [assistant_message]
    return {**state, "messages": new_messages}


def add_user_message(state: GraphState, content: str) -> GraphState:
    """Add a user message to the state.
    
    Args:
        state: Current graph state
        content: User message content
        
    Returns:
        Updated state with new user message
    """
    user_message = {
        "role": "user",
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
        "related_field": state.get("current_field")
    }
    
    new_messages = state["messages"] + [user_message]
    return {**state, "messages": new_messages}


def update_current_field(state: GraphState, field_name: Optional[str]) -> GraphState:
    """Update the current field being collected.
    
    Args:
        state: Current graph state
        field_name: Name of field to set as current (None to clear)
        
    Returns:
        Updated state with new current field
    """
    return {**state, "current_field": field_name, "retry_count": 0}


def add_validation_error(state: GraphState, error: str) -> GraphState:
    """Add a validation error to the state.
    
    Args:
        state: Current graph state
        error: Error message to add
        
    Returns:
        Updated state with new validation error
    """
    new_errors = state["validation_errors"] + [error]
    new_retry_count = state.get("retry_count", 0) + 1
    
    return {
        **state, 
        "validation_errors": new_errors,
        "retry_count": new_retry_count
    }


def clear_validation_errors(state: GraphState) -> GraphState:
    """Clear all validation errors from state.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with cleared validation errors
    """
    return {**state, "validation_errors": [], "retry_count": 0}


def mark_conversation_complete(state: GraphState) -> GraphState:
    """Mark the conversation as complete.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state marked as complete
    """
    return {
        **state, 
        "is_complete": True,
        "current_field": None,
        "conversation_phase": "completed"
    }


def update_conversation_phase(state: GraphState, phase: str) -> GraphState:
    """Update the conversation phase.
    
    Args:
        state: Current graph state
        phase: New conversation phase
        
    Returns:
        Updated state with new phase
    """
    return {**state, "conversation_phase": phase}


def get_last_user_message(state: GraphState) -> Optional[str]:
    """Get the content of the last user message.
    
    Args:
        state: Current graph state
        
    Returns:
        Last user message content or None if no user messages
    """
    user_messages = [
        msg["content"] for msg in state["messages"] 
        if msg.get("role") == "user"
    ]
    return user_messages[-1] if user_messages else None


def get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been collected.
    
    Args:
        state: Current graph state
        
    Returns:
        List of field names that have values
    """
    return [
        field for field, value in state["job_data"].items()
        if value is not None and value != "" and value != []
    ]


def get_missing_required_fields(state: GraphState) -> List[str]:
    """Get list of required fields that are still missing.
    
    Args:
        state: Current graph state
        
    Returns:
        List of required field names that are missing
    """
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(get_collected_fields(state))
    return list(required_fields - collected_fields)


def calculate_progress_percentage(state: GraphState) -> float:
    """Calculate conversation progress as percentage.
    
    Args:
        state: Current graph state
        
    Returns:
        Progress percentage (0.0 to 100.0)
    """
    total_fields = {
        "job_title", "department", "employment_type", "location",
        "experience", "responsibilities", "skills", "education",
        "salary", "benefits", "additional_requirements"
    }
    
    collected_fields = set(get_collected_fields(state))
    return (len(collected_fields) / len(total_fields)) * 100


def is_ready_for_generation(state: GraphState) -> bool:
    """Check if state has enough data for JD generation.
    
    Args:
        state: Current graph state
        
    Returns:
        True if ready for JD generation
    """
    missing_required = get_missing_required_fields(state)
    return len(missing_required) == 0


def validate_state_transition(
    current_state: GraphState, 
    next_node: str
) -> Tuple[bool, Optional[str]]:
    """Validate that a state transition is allowed.
    
    Args:
        current_state: Current graph state
        next_node: Name of the next node to transition to
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    current_phase = current_state.get("conversation_phase", "greeting")
    
    # Define valid transitions
    valid_transitions = {
        "greeting": ["question_router"],
        "collecting_basic_info": ["question_router", "question_generator", "user_input_collector"],
        "collecting_experience": ["question_router", "question_generator", "user_input_collector"],
        "collecting_skills": ["question_router", "question_generator", "user_input_collector"],
        "collecting_responsibilities": ["question_router", "question_generator", "user_input_collector"],
        "collecting_requirements": ["question_router", "question_generator", "user_input_collector"],
        "reviewing_data": ["jd_generator", "question_router"],
        "generating_jd": ["output"],
        "completed": []
    }
    
    allowed_nodes = valid_transitions.get(current_phase, [])
    
    if next_node not in allowed_nodes:
        return False, f"Invalid transition from {current_phase} to {next_node}"
    
    return True, None


def safe_state_update(
    state: GraphState, 
    update_function: Callable[[GraphState], GraphState]
) -> GraphState:
    """Safely apply state update with error handling.
    
    Args:
        state: Current graph state
        update_function: Function to apply state update
        
    Returns:
        Updated state or original state if update fails
    """
    try:
        return update_function(state)
    except Exception as e:
        # Log error and return original state
        error_state = add_validation_error(state, f"State update error: {str(e)}")
        return error_state


def merge_state_updates(state: GraphState, **updates) -> GraphState:
    """Merge multiple state updates functionally.
    
    Args:
        state: Current graph state
        **updates: Keyword arguments of fields to update
        
    Returns:
        Updated state with merged changes
    """
    return {**state, **updates}


def reset_conversation_state(preserve_session: bool = True) -> GraphState:
    """Reset conversation state to initial values.
    
    Args:
        preserve_session: Whether to preserve session metadata
        
    Returns:
        Fresh graph state
    """
    initial_state = create_initial_graph_state()
    
    if preserve_session:
        # Keep session metadata if it exists
        return initial_state
    
    return initial_state


# State validation functions
def validate_job_data_completeness(job_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that job data has minimum required fields.
    
    Args:
        job_data: Job data dictionary
        
    Returns:
        Tuple of (is_complete, missing_fields)
    """
    required_fields = {"job_title", "responsibilities"}
    missing_fields = []
    
    for field in required_fields:
        value = job_data.get(field)
        if not value or (isinstance(value, list) and len(value) == 0):
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def validate_state_consistency(state: GraphState) -> Tuple[bool, List[str]]:
    """Validate overall state consistency.
    
    Args:
        state: Graph state to validate
        
    Returns:
        Tuple of (is_consistent, error_messages)
    """
    errors = []
    
    # Check message history consistency
    if state["messages"]:
        if state["messages"][0].get("role") != "assistant":
            errors.append("First message should be from assistant")
    
    # Check phase consistency with current field
    phase = state.get("conversation_phase", "greeting")
    current_field = state.get("current_field")
    
    if phase == "completed" and current_field is not None:
        errors.append("Completed conversation should not have current field")
    
    # Check retry count limits
    retry_count = state.get("retry_count", 0)
    if retry_count > 3:
        errors.append("Retry count exceeds maximum allowed")
    
    return len(errors) == 0, errors


from datetime import datetime