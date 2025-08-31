"""Completeness checker node for validation and readiness assessment."""

from typing import Dict, Any, List
from datetime import datetime
from ...models.graph_state import GraphState
from ...models.job_requirements import JobRequirements, validate_job_requirements


def get_missing_required_fields(state: GraphState) -> List[str]:
    """Get list of required fields that are still missing."""
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(get_collected_fields(state))
    return list(required_fields - collected_fields)


def get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been collected."""
    return [
        field
        for field, value in state.get("job_data", {}).items()
        if value is not None and value != "" and value != []
    ]


def completeness_checker_node(state: GraphState) -> Dict[str, Any]:
    """Check if job data is complete and ready for JD generation.

    Pure function that validates collected job data, checks completeness,
    and either proceeds to generation or requests missing information.

    Args:
        state: Current graph state

    Returns:
        State updates with completion status and any required messages
    """
    job_data = state.get("job_data", {})

    # Validate job data using Pydantic model
    job_requirements, validation_errors = validate_job_requirements(job_data)

    if validation_errors:
        # Has validation errors - need to fix
        error_message = create_validation_error_message(validation_errors)
        new_message = {
            "role": "assistant",
            "content": error_message,
            "message_type": "clarification",
            "timestamp": datetime.utcnow().isoformat(),
        }
        new_messages = state.get("messages", []) + [new_message]
        return {
            **state,
            "messages": new_messages,
            "validation_errors": validation_errors,
        }

    # Check required field completeness
    missing_required = get_missing_required_fields(state)

    if missing_required:
        # Missing required fields
        missing_message = create_missing_fields_message(missing_required, job_data)
        new_message = {
            "role": "assistant",
            "content": missing_message,
            "message_type": "question",
            "timestamp": datetime.utcnow().isoformat(),
        }
        new_messages = state.get("messages", []) + [new_message]
        return {
            **state,
            "messages": new_messages,
            "current_field": missing_required[0],  # Set to first missing field
        }

    # Data is complete - provide summary and confirm
    if job_requirements:
        summary_message = create_completion_summary(job_requirements, job_data)
        new_message = {
            "role": "assistant",
            "content": summary_message,
            "message_type": "confirmation",
            "timestamp": datetime.utcnow().isoformat(),
        }
        new_messages = state.get("messages", []) + [new_message]

        return {
            **state,
            "messages": new_messages,
            "is_complete": True,
            "current_field": None,
            "conversation_phase": "reviewing_data",
        }

    # Fallback - shouldn't reach here
    return state


def create_validation_error_message(validation_errors: List[str]) -> str:
    """Create user-friendly message for validation errors.

    Args:
        validation_errors: List of validation error messages

    Returns:
        Formatted error message for user
    """
    if len(validation_errors) == 1:
        return (
            f"I noticed an issue with the information provided: {validation_errors[0]}\n\n"
            "Could you please clarify or provide the information in a different format?"
        )

    error_list = "\n".join(f"• {error}" for error in validation_errors[:3])
    return (
        f"I found a few issues with the information provided:\n{error_list}\n\n"
        "Could you please help me clarify these points?"
    )


def create_missing_fields_message(
    missing_fields: List[str], job_data: Dict[str, Any]
) -> str:
    """Create message requesting missing required fields.

    Args:
        missing_fields: List of missing required field names
        job_data: Current job data for context

    Returns:
        Message requesting missing information
    """
    if len(missing_fields) == 1:
        field_name = missing_fields[0]
        return get_field_specific_request(field_name, job_data)

    field_list = ", ".join(missing_fields[:-1]) + f", and {missing_fields[-1]}"
    return (
        f"We still need to collect information about: {field_list}.\n\n"
        f"Let's continue with {missing_fields[0]}. "
        f"{get_field_specific_request(missing_fields[0], job_data)}"
    )


def create_completion_summary(
    job_requirements: JobRequirements, job_data: Dict[str, Any]
) -> str:
    """Create summary of collected information for user confirmation.

    Args:
        job_requirements: Validated job requirements model
        job_data: Raw job data dictionary

    Returns:
        Formatted summary message
    """
    summary_parts = [
        "Perfect! I have all the information needed to create your job description. "
        "Here's a summary of what we've collected:\n"
    ]

    # Job basics
    summary_parts.append(f"**Position**: {job_requirements.job_title}")

    if job_requirements.department:
        summary_parts.append(f"**Department**: {job_requirements.department}")

    summary_parts.append(
        f"**Employment Type**: {job_requirements.employment_type.value.replace('_', ' ').title()}"
    )

    # Location
    location_summary = format_location_summary(job_requirements.location)
    summary_parts.append(f"**Location**: {location_summary}")

    # Experience
    experience_summary = format_experience_summary(job_requirements.experience)
    summary_parts.append(f"**Experience**: {experience_summary}")

    # Responsibilities
    resp_count = len(job_requirements.responsibilities)
    summary_parts.append(f"**Responsibilities**: {resp_count} key duties defined")

    # Skills
    skills_summary = format_skills_summary(job_requirements.skills)
    summary_parts.append(f"**Skills**: {skills_summary}")

    # Optional fields
    if job_requirements.salary:
        summary_parts.append(f"**Salary**: {job_requirements.salary.formatted_range}")

    if job_requirements.benefits:
        summary_parts.append(
            f"**Benefits**: {len(job_requirements.benefits)} benefits listed"
        )

    summary_parts.append(
        "\nShall I generate the complete job description now? "
        "Type 'yes' to proceed or let me know if you'd like to change anything."
    )

    return "\n".join(summary_parts)


def get_field_specific_request(field_name: str, job_data: Dict[str, Any]) -> str:
    """Get field-specific request message.

    Args:
        field_name: Name of field to request
        job_data: Current job data for context

    Returns:
        Field-specific request message
    """
    job_title = job_data.get("job_title", "")

    requests = {
        "job_title": "What is the job title for this position?",
        "responsibilities": (
            f"What are the key responsibilities for the {job_title} role? "
            "Please describe the main duties and day-to-day tasks."
        ),
        "skills": (
            f"What skills and competencies are required for the {job_title} position? "
            "Include both technical skills and soft skills."
        ),
        "experience": (
            f"What experience requirements do you have for the {job_title} role? "
            "Please specify years of experience and any specific background needed."
        ),
    }

    return requests.get(field_name, f"Please provide information about {field_name}.")


def format_location_summary(location) -> str:
    """Format location for summary display.

    Args:
        location: LocationRequirement object

    Returns:
        Formatted location string
    """
    location_type = location.location_type.value.replace("_", " ").title()

    if location.city and location.state:
        return f"{location_type} in {location.city}, {location.state}"
    elif location.city:
        return f"{location_type} in {location.city}"
    else:
        return location_type


def format_experience_summary(experience) -> str:
    """Format experience for summary display.

    Args:
        experience: ExperienceRequirement object

    Returns:
        Formatted experience string
    """
    level = experience.level.value.title()
    years = experience.years_min

    parts = [level]
    if years > 0:
        parts.append(f"({years}+ years)")

    if experience.leadership_required:
        parts.append("with leadership experience")

    return " ".join(parts)


def format_skills_summary(skills) -> str:
    """Format skills for summary display.

    Args:
        skills: SkillsRequirement object

    Returns:
        Formatted skills string
    """
    parts = []

    tech_count = len(skills.technical_skills)
    if tech_count > 0:
        parts.append(f"{tech_count} technical skills")

    soft_count = len(skills.soft_skills)
    if soft_count > 0:
        parts.append(f"{soft_count} soft skills")

    lang_count = len(skills.programming_languages)
    if lang_count > 0:
        parts.append(f"{lang_count} programming languages")

    if not parts:
        return "Skills to be defined"

    return ", ".join(parts)


def assess_data_quality(job_data: Dict[str, Any]) -> tuple[float, List[str]]:
    """Assess the quality and completeness of job data.

    Args:
        job_data: Current job data dictionary

    Returns:
        Tuple of (quality_score, improvement_suggestions)
    """
    score = 0.0
    suggestions = []

    # Check basic fields (40% of score)
    basic_fields = ["job_title", "department", "employment_type", "location"]
    basic_complete = sum(1 for field in basic_fields if job_data.get(field))
    score += (basic_complete / len(basic_fields)) * 0.4

    # Check core content (40% of score)
    core_fields = ["responsibilities", "skills", "experience"]
    core_complete = sum(1 for field in core_fields if job_data.get(field))
    score += (core_complete / len(core_fields)) * 0.4

    # Check optional enrichment (20% of score)
    optional_fields = ["education", "salary", "benefits"]
    optional_complete = sum(1 for field in optional_fields if job_data.get(field))
    score += (optional_complete / len(optional_fields)) * 0.2

    # Generate suggestions
    if not job_data.get("salary"):
        suggestions.append(
            "Consider adding salary range to attract more qualified candidates"
        )

    if not job_data.get("benefits"):
        suggestions.append(
            "Adding benefits information can make the position more appealing"
        )

    responsibilities = job_data.get("responsibilities", [])
    if isinstance(responsibilities, list) and len(responsibilities) < 3:
        suggestions.append("Consider adding more detailed responsibilities for clarity")

    return score, suggestions


def should_request_optional_fields(
    job_data: Dict[str, Any], conversation_length: int
) -> tuple[bool, List[str]]:
    """Determine if optional fields should be requested.

    Args:
        job_data: Current job data
        conversation_length: Number of messages in conversation

    Returns:
        Tuple of (should_request, field_priority_list)
    """
    # Don't request if conversation is already long
    if conversation_length > 20:
        return False, []

    # Check which optional fields are missing
    optional_fields = ["salary", "benefits", "additional_requirements"]
    missing_optional = [field for field in optional_fields if not job_data.get(field)]

    # Prioritize by importance
    priority_order = ["salary", "benefits", "additional_requirements"]
    prioritized_missing = [
        field for field in priority_order if field in missing_optional
    ]

    return (
        len(prioritized_missing) > 0,
        prioritized_missing[:2],
    )  # Max 2 optional fields
