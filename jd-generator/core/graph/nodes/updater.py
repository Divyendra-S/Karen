"""State updater node for managing job data updates with immutable patterns."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from ...models.graph_state import GraphState


def determine_next_field(state: GraphState) -> Optional[str]:
    """Determine the next field to collect based on current state."""
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


def _handle_evaluation_based_update(state: GraphState) -> Dict[str, Any]:
    """Handle state update based on evaluation results.

    Args:
        state: Current graph state with evaluation results

    Returns:
        Updated state with merged job data
    """
    extracted_fields = state.get("extracted_fields", {})
    current_job_data = state.get("job_data", {})

    # Merge extracted fields with existing job data
    updated_job_data = _merge_extracted_fields_with_job_data(
        current_job_data, extracted_fields
    )

    # Determine next field to collect
    next_field = determine_next_field({**state, "job_data": updated_job_data})

    # Create acknowledgment message if multiple fields were extracted
    acknowledgment_message = None
    if len(extracted_fields) > 1:
        acknowledgment_message = _create_multi_field_acknowledgment(extracted_fields)

    # Prepare state update
    updated_state = {
        **state,
        "job_data": updated_job_data,
        "current_field": next_field,
        "validation_errors": [],
        "extracted_fields": {},  # Clear after processing
        "evaluation_result": None,  # Clear after processing
        "needs_clarification": False,
        "evaluation_failed": False,
    }

    # Add acknowledgment message if needed
    if acknowledgment_message:
        new_message = {
            "role": "assistant",
            "content": acknowledgment_message,
            "message_type": "info",
            "timestamp": datetime.utcnow().isoformat(),
        }
        updated_state["messages"] = state.get("messages", []) + [new_message]

    return updated_state


def _merge_extracted_fields_with_job_data(
    existing_job_data: Dict[str, Any], extracted_fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge extracted fields with existing job data intelligently."""
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
                # Remove duplicates while preserving order
                all_resp = existing_resp + field_value
                merged_data[field_name] = list(dict.fromkeys(all_resp))
            elif field_name == "benefits" and isinstance(field_value, list):
                existing_benefits = existing_job_data.get(field_name, [])
                all_benefits = existing_benefits + field_value
                merged_data[field_name] = list(dict.fromkeys(all_benefits))
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
    merged["technical_skills"] = list(dict.fromkeys(existing_tech + new_tech))

    # Merge programming languages
    existing_langs = merged.get("programming_languages", [])
    new_langs = new_skills.get("programming_languages", [])
    merged["programming_languages"] = list(dict.fromkeys(existing_langs + new_langs))

    # Merge frameworks and tools
    existing_tools = merged.get("frameworks_tools", [])
    new_tools = new_skills.get("frameworks_tools", [])
    merged["frameworks_tools"] = list(dict.fromkeys(existing_tools + new_tools))

    return merged


def _create_multi_field_acknowledgment(extracted_fields: Dict[str, Any]) -> str:
    """Create acknowledgment message for multi-field extraction.

    Args:
        extracted_fields: Fields that were extracted

    Returns:
        Acknowledgment message
    """
    field_names = list(extracted_fields.keys())

    if len(field_names) == 2:
        return f"Great! I've captured information about your {field_names[0]} and {field_names[1]}."
    elif len(field_names) > 2:
        field_list = ", ".join(field_names[:-1]) + f", and {field_names[-1]}"
        return f"Excellent! I've captured information about your {field_list}."

    return "Thanks! I've captured that information."


def get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been collected."""
    return [
        field
        for field, value in state.get("job_data", {}).items()
        if value is not None and value != "" and value != []
    ]


def update_job_field(
    job_data: Dict[str, Any], field_name: str, field_value: Any
) -> Dict[str, Any]:
    """Update specific field in job data immutably.

    Args:
        job_data: Current job data dictionary
        field_name: Name of field to update
        field_value: New value for the field

    Returns:
        New job data dictionary with updated field
    """
    return {**job_data, field_name: field_value}


def merge_nested_field(
    job_data: Dict[str, Any], field_name: str, nested_updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge updates into a nested field immutably.

    Args:
        job_data: Current job data dictionary
        field_name: Name of nested field to update
        nested_updates: Dictionary of updates to merge

    Returns:
        New job data with merged nested field
    """
    current_nested = job_data.get(field_name, {})
    if isinstance(current_nested, dict):
        updated_nested = {**current_nested, **nested_updates}
    else:
        updated_nested = nested_updates

    return {**job_data, field_name: updated_nested}


def validate_field_update(
    field_name: str, field_value: Any, existing_job_data: Dict[str, Any]
) -> tuple[bool, List[str]]:
    """Validate that a field update is consistent with existing data.

    Args:
        field_name: Name of field being updated
        field_value: New value for the field
        existing_job_data: Current job data

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Cross-field validation rules
    if field_name == "employment_type" and field_value == "internship":
        # Internships should have entry-level experience
        existing_experience = existing_job_data.get("experience", {})
        if isinstance(existing_experience, dict):
            exp_level = existing_experience.get("level")
            if exp_level and exp_level not in ["entry", "junior"]:
                errors.append(
                    "Internships typically don't require senior-level experience. "
                    "Consider adjusting the experience requirements."
                )

    if field_name == "location":
        # Validate location consistency
        if isinstance(field_value, dict):
            location_type = field_value.get("location_type")
            city = field_value.get("city")

            if location_type == "on_site" and not city:
                errors.append("On-site positions require a city location.")

    if field_name == "experience":
        # Validate experience consistency with job title
        job_title = existing_job_data.get("job_title", "").lower()
        if isinstance(field_value, dict):
            years_min = field_value.get("years_min", 0)

            if "senior" in job_title and years_min < 3:
                errors.append(
                    "Senior positions typically require at least 3 years of experience."
                )
            elif "entry" in job_title and years_min > 2:
                errors.append(
                    "Entry-level positions typically require 0-2 years of experience."
                )

    return len(errors) == 0, errors


def calculate_field_completion_score(job_data: Dict[str, Any]) -> float:
    """Calculate how complete the job data is.

    Args:
        job_data: Current job data dictionary

    Returns:
        Completion score from 0.0 to 1.0
    """
    all_fields = {
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
    }

    completed_fields = {
        field
        for field in all_fields
        if job_data.get(field) is not None and job_data.get(field) != []
    }

    return len(completed_fields) / len(all_fields)


def get_field_summary(job_data: Dict[str, Any], field_name: str) -> str:
    """Get a summary of a specific field for display.

    Args:
        job_data: Current job data dictionary
        field_name: Field to summarize

    Returns:
        Human-readable summary of the field
    """
    field_value = job_data.get(field_name)

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


def create_progress_summary(job_data: Dict[str, Any]) -> Dict[str, str]:
    """Create a summary of all collected fields.

    Args:
        job_data: Current job data dictionary

    Returns:
        Dictionary mapping field names to human-readable summaries
    """
    field_names = [
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

    return {
        field: get_field_summary(job_data, field)
        for field in field_names
        if job_data.get(field) is not None
    }


def needs_confirmation(field_name: str, field_value: Any) -> bool:
    """Check if a field value needs user confirmation.

    Args:
        field_name: Name of the field
        field_value: Value that was processed

    Returns:
        True if confirmation is recommended
    """
    # Always confirm complex processed data
    complex_fields = {"location", "experience", "skills", "salary"}
    return field_name in complex_fields


def format_field_confirmation(field_name: str, field_value: Any) -> str:
    """Format field value for confirmation display.

    Args:
        field_name: Name of the field
        field_value: Processed field value

    Returns:
        Formatted confirmation text
    """
    if field_name == "location" and isinstance(field_value, dict):
        location_type = field_value.get("location_type", "").replace("_", " ").title()
        city = field_value.get("city", "")
        return f"Location: {location_type}" + (f" in {city}" if city else "")

    if field_name == "experience" and isinstance(field_value, dict):
        level = field_value.get("level", "").title()
        years = field_value.get("years_min", 0)
        return f"Experience: {level} level" + (
            f" with {years}+ years" if years > 0 else ""
        )

    if field_name == "skills" and isinstance(field_value, dict):
        tech_skills = field_value.get("technical_skills", [])
        tech_names = [skill["name"] for skill in tech_skills if isinstance(skill, dict)]
        return f"Skills: {', '.join(tech_names[:5])}" + (
            "..." if len(tech_names) > 5 else ""
        )

    return f"{field_name.replace('_', ' ').title()}: {field_value}"


def state_updater_node(state: GraphState) -> Dict[str, Any]:
    """Update job data state with evaluation results and extracted fields.

    Pure function that takes evaluation results and updates the job data
    in the state, then determines the next field to collect.

    Args:
        state: Current graph state with evaluation results

    Returns:
        Updated state with new job data and next field
    """
    # Handle evaluation results if available
    if state.get("extracted_fields"):
        return _handle_evaluation_based_update(state)

    # Fallback to legacy processing (for backward compatibility)
    current_field = state.get("current_field")
    processed_value = state.get("temp_processed_value")

    if not current_field:
        return state

    # If no processed value (user skipped), move to next field
    if processed_value is None:
        next_field = determine_next_field(state)
        return {**state, "current_field": next_field, "temp_processed_value": None}

    # Update job data with processed value
    new_job_data = update_job_field(
        state.get("job_data", {}), current_field, processed_value
    )

    # Determine next field to collect
    next_field = determine_next_field(state)

    return {
        **state,
        "job_data": new_job_data,
        "current_field": next_field,
        "temp_processed_value": None,
        "validation_errors": [],
    }
