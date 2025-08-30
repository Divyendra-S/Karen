"""State updater node for managing job data updates with immutable patterns."""

from typing import Dict, Any, Optional, List
from ...models.graph_state import GraphState


def state_updater_node(state: GraphState) -> Dict[str, Any]:
    """Update state after data collection with proper field management.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with next field routing
    """
    current_field = state.get("current_field")
    clarification_needed = state.get("clarification_needed", [])
    processed_data = state.get("processed_data", {})
    
    if not current_field:
        return state
    
    # If current field needs clarification, stay on it (but cap retry_count)
    if current_field in clarification_needed:
        current_retry = state.get("retry_count", 0)
        new_retry = min(current_retry + 1, 3)
        
        # If we've retried too many times, skip the field
        if new_retry >= 3:
            next_field = determine_next_field(state)
            return {
                **state,
                "current_field": next_field,
                "retry_count": 0,
                "clarification_needed": [item for item in clarification_needed if item != current_field]
            }
        
        return {
            **state,
            "retry_count": new_retry
        }
    
    # If current field was successfully processed, move to next
    if current_field in processed_data:
        next_field = determine_next_field(state)
        
        # Check if we're done
        if next_field is None:
            required_fields = {"job_title", "responsibilities", "skills"}
            collected_fields = set(processed_data.keys())
            is_complete = required_fields.issubset(collected_fields)
            
            return {
                **state,
                "current_field": None,
                "is_complete": is_complete,
                "retry_count": 0,
                "clarification_needed": []
            }
        
        return {
            **state,
            "current_field": next_field,
            "retry_count": 0,
            "clarification_needed": []
        }
    
    # If no data was processed (user skipped), move to next field
    next_field = determine_next_field(state)
    
    return {
        **state,
        "current_field": next_field,
        "retry_count": 0,
        "clarification_needed": []
    }


def determine_next_field(state: GraphState) -> Optional[str]:
    """Determine the next field to collect based on smart prioritization."""
    collected_fields = set(state.get("processed_data", {}).keys())
    
    # Core essential fields in logical order
    core_fields = ["job_title", "experience", "skills", "responsibilities"]
    
    # Context fields that help with better questions
    context_fields = ["location", "department"]
    
    # Optional fields (limit to avoid fatigue)
    optional_fields = ["employment_type", "education"]
    
    # Check core fields first
    for field in core_fields:
        if field not in collected_fields:
            return field
    
    # Then context fields
    for field in context_fields:
        if field not in collected_fields:
            return field
    
    # Only ask 1-2 optional fields maximum
    asked_optional = len([f for f in optional_fields if f in collected_fields])
    if asked_optional < 2:
        for field in optional_fields:
            if field not in collected_fields:
                return field
    
    # Stop here - don't ask about salary/benefits in interview
    return None


def get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been successfully processed."""
    return list(state.get("processed_data", {}).keys())


def calculate_completion_percentage(state: GraphState) -> float:
    """Calculate completion percentage based on collected fields."""
    total_fields = 11  # Total number of fields we collect
    completed_fields = len(get_collected_fields(state))
    return (completed_fields / total_fields) * 100


def get_field_summary(field_name: str, state: GraphState) -> str:
    """Get human-readable summary of a field's value."""
    processed_data = state.get("processed_data", {})
    field_value = processed_data.get(field_name)
    
    if field_value is None:
        return "Not specified"
    
    if field_name == "responsibilities" and isinstance(field_value, list):
        return f"{len(field_value)} responsibilities listed"
    
    if field_name == "skills" and isinstance(field_value, dict):
        tech_count = len(field_value.get("technical_skills", []))
        return f"{tech_count} technical skills specified"
    
    if field_name == "location" and isinstance(field_value, dict):
        location_type = field_value.get("location_type", "").replace("_", " ").title()
        city = field_value.get("city", "")
        return f"{location_type}" + (f" in {city}" if city else "")
    
    if field_name == "experience" and isinstance(field_value, dict):
        level = field_value.get("level", "").title()
        years = field_value.get("years_min", 0)
        return f"{level} level" + (f" ({years}+ years)" if years > 0 else "")
    
    if field_name == "salary" and isinstance(field_value, dict):
        min_sal = field_value.get("min_salary")
        max_sal = field_value.get("max_salary")
        if min_sal and max_sal:
            return f"${min_sal:,.0f} - ${max_sal:,.0f}"
        elif min_sal:
            return f"${min_sal:,.0f}+"
        return "Competitive"
    
    # Default string representation
    if isinstance(field_value, str):
        return field_value[:50] + "..." if len(field_value) > 50 else field_value
    
    return str(field_value)


def create_progress_summary(state: GraphState) -> Dict[str, str]:
    """Create a summary of all collected fields."""
    field_names = [
        "job_title", "department", "employment_type", "location",
        "experience", "responsibilities", "skills", "education",
        "salary", "benefits", "additional_requirements"
    ]
    
    return {
        field: get_field_summary(field, state)
        for field in field_names
        if field in state.get("processed_data", {})
    }


def validate_data_integrity(state: GraphState) -> tuple[bool, List[str]]:
    """Validate that raw responses and processed data are consistent."""
    errors = []
    
    raw_responses = state.get("raw_responses", {})
    processed_data = state.get("processed_data", {})
    
    # Check that we have raw response for every processed field
    for field in processed_data.keys():
        if field not in raw_responses:
            errors.append(f"Missing raw response for processed field: {field}")
    
    # Check for orphaned raw responses (should be rare)
    for field in raw_responses.keys():
        if field not in processed_data and field not in state.get("clarification_needed", []):
            errors.append(f"Raw response exists but no processed data: {field}")
    
    return len(errors) == 0, errors


def needs_completion_check(state: GraphState) -> bool:
    """Check if we have enough data for completion."""
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(state.get("processed_data", {}).keys())
    
    return required_fields.issubset(collected_fields)


def is_field_optional(field_name: str) -> bool:
    """Check if a field is optional for job description generation."""
    optional_fields = {"salary", "benefits", "additional_requirements"}
    return field_name in optional_fields


def should_ask_optional_fields(state: GraphState) -> bool:
    """Determine if we should continue with optional fields."""
    # Check if all required fields are done
    if not needs_completion_check(state):
        return False
    
    # Check conversation length to avoid fatigue
    message_count = len(state.get("messages", []))
    return message_count < 20  # Arbitrary limit


def get_missing_required_fields(state: GraphState) -> List[str]:
    """Get list of required fields that are still missing."""
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(state.get("processed_data", {}).keys())
    return list(required_fields - collected_fields)


def format_field_for_display(field_name: str, field_value: Any) -> str:
    """Format field value for user-friendly display."""
    if field_name == "employment_type":
        return field_value.replace("_", " ").title()
    
    if field_name == "location" and isinstance(field_value, dict):
        location_type = field_value.get("location_type", "").replace("_", " ").title()
        city = field_value.get("city", "")
        return f"{location_type}" + (f" in {city}" if city else "")
    
    if isinstance(field_value, list):
        return f"{len(field_value)} items"
    
    if isinstance(field_value, dict):
        return "Complex data"
    
    return str(field_value)