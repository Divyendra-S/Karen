"""JD generator and output nodes with functional composition patterns."""

from typing import Dict, Any, List
from datetime import datetime
from ...models.graph_state import GraphState
from ...models.job_requirements import JobRequirements


def jd_generator_node(state: GraphState) -> Dict[str, Any]:
    """Generate professional job description from collected data.

    Pure function that takes complete job data and generates a
    professionally formatted job description.

    Args:
        state: Current graph state with complete job data

    Returns:
        State updates with generated job description
    """
    job_data = state.get("job_data", {})

    try:
        # Convert to JobRequirements model for validation
        job_requirements = JobRequirements(**job_data)

        # Generate the job description
        generated_jd = generate_job_description(job_requirements)

        # Update state with generated JD
        return {
            **state,
            "generated_jd": generated_jd,
            "conversation_phase": "generating_jd",
        }

    except Exception as e:
        # Handle generation errors
        error_message = (
            f"I encountered an issue generating the job description: {str(e)}\n\n"
            "Let me try a different approach. Could you review the information "
            "we've collected and let me know if anything needs to be adjusted?"
        )

        new_message = {
            "role": "assistant",
            "content": error_message,
            "message_type": "error",
            "timestamp": datetime.utcnow().isoformat(),
        }
        new_messages = state.get("messages", []) + [new_message]
        return {**state, "messages": new_messages, "validation_errors": [str(e)]}


def output_node(state: GraphState) -> Dict[str, Any]:
    """Present final job description to user with export options.

    Pure function that formats and presents the generated job description
    with options for export and feedback.

    Args:
        state: Current graph state with generated JD

    Returns:
        Final state with output message and completion status
    """
    generated_jd = state.get("generated_jd")

    if not generated_jd:
        error_message = (
            "I wasn't able to generate the job description. "
            "Please try again or contact support if the issue persists."
        )
        new_message = {
            "role": "assistant",
            "content": error_message,
            "message_type": "error",
            "timestamp": datetime.utcnow().isoformat(),
        }
        new_messages = state.get("messages", []) + [new_message]
        return {**state, "messages": new_messages}

    # Create output message with JD and options
    output_message = create_final_output_message(generated_jd)

    # Create final message
    final_message = {
        "role": "assistant",
        "content": output_message,
        "message_type": "info",
        "timestamp": datetime.utcnow().isoformat(),
    }

    new_messages = state.get("messages", []) + [final_message]

    return {
        **state,
        "messages": new_messages,
        "is_complete": True,
        "current_field": None,
        "conversation_phase": "completed",
    }


def generate_job_description(job_requirements: JobRequirements) -> str:
    """Generate professional job description from requirements.

    Args:
        job_requirements: Validated job requirements model

    Returns:
        Formatted job description text
    """
    sections = []

    # Header
    sections.append(create_jd_header(job_requirements))

    # Job Overview
    sections.append(create_job_overview(job_requirements))

    # Key Responsibilities
    sections.append(create_responsibilities_section(job_requirements))

    # Requirements
    sections.append(create_requirements_section(job_requirements))

    # Qualifications
    sections.append(create_qualifications_section(job_requirements))

    # Compensation (if provided)
    if job_requirements.salary:
        sections.append(create_compensation_section(job_requirements))

    # Benefits (if provided)
    if job_requirements.benefits:
        sections.append(create_benefits_section(job_requirements))

    # Additional Requirements (if provided)
    if job_requirements.additional_requirements:
        sections.append(create_additional_section(job_requirements))

    # Footer
    sections.append(create_jd_footer(job_requirements))

    return "\n\n".join(sections)


def create_jd_header(job_requirements: JobRequirements) -> str:
    """Create job description header section."""
    header_parts = [f"# {job_requirements.job_title}"]

    if job_requirements.department:
        header_parts.append(f"**Department:** {job_requirements.department}")

    header_parts.append(
        f"**Employment Type:** {job_requirements.employment_type.value.replace('_', ' ').title()}"
    )

    # Location
    location_text = format_location_for_jd(job_requirements.location)
    header_parts.append(f"**Location:** {location_text}")

    return "\n".join(header_parts)


def create_job_overview(job_requirements: JobRequirements) -> str:
    """Create job overview section."""
    overview = "## Job Overview\n\n"

    # Generate contextual overview based on role
    role_type = categorize_job_role(job_requirements.job_title)

    if role_type == "technical":
        overview += generate_technical_overview(job_requirements)
    elif role_type == "marketing":
        overview += generate_marketing_overview(job_requirements)
    elif role_type == "sales":
        overview += generate_sales_overview(job_requirements)
    else:
        overview += generate_general_overview(job_requirements)

    return overview


def create_responsibilities_section(job_requirements: JobRequirements) -> str:
    """Create key responsibilities section."""
    section = "## Key Responsibilities\n\n"

    for i, responsibility in enumerate(job_requirements.responsibilities, 1):
        section += f"{i}. {responsibility}\n"

    return section


def create_requirements_section(job_requirements: JobRequirements) -> str:
    """Create requirements section."""
    section = "## Requirements\n\n"

    # Experience requirements
    experience_text = format_experience_for_jd(job_requirements.experience)
    section += f"• {experience_text}\n"

    # Skills requirements
    if job_requirements.skills.technical_skills:
        tech_skills = [
            skill.name
            for skill in job_requirements.skills.technical_skills
            if skill.is_required
        ]
        if tech_skills:
            section += f"• Technical skills: {', '.join(tech_skills)}\n"

    if job_requirements.skills.programming_languages:
        section += f"• Programming languages: {', '.join(job_requirements.skills.programming_languages)}\n"

    if job_requirements.skills.frameworks_tools:
        section += f"• Tools and frameworks: {', '.join(job_requirements.skills.frameworks_tools)}\n"

    # Soft skills
    if job_requirements.skills.soft_skills:
        soft_skills = [
            skill.name
            for skill in job_requirements.skills.soft_skills
            if skill.is_required
        ]
        if soft_skills:
            section += f"• Soft skills: {', '.join(soft_skills)}\n"

    return section


def create_qualifications_section(job_requirements: JobRequirements) -> str:
    """Create qualifications section."""
    section = "## Preferred Qualifications\n\n"

    # Education
    education_text = format_education_for_jd(job_requirements.education)
    section += f"• {education_text}\n"

    # Additional qualifications from non-required skills
    preferred_technical = [
        skill.name
        for skill in job_requirements.skills.technical_skills
        if not skill.is_required
    ]

    if preferred_technical:
        section += f"• Experience with: {', '.join(preferred_technical)}\n"

    if job_requirements.skills.certifications:
        section += (
            f"• Certifications: {', '.join(job_requirements.skills.certifications)}\n"
        )

    return section


def create_compensation_section(job_requirements: JobRequirements) -> str:
    """Create compensation section."""
    section = "## Compensation\n\n"
    section += f"• Salary: {job_requirements.salary.formatted_range}\n"

    if job_requirements.salary.equity_offered:
        section += "• Equity/stock options available\n"

    if job_requirements.salary.bonus_structure:
        section += f"• Bonus structure: {job_requirements.salary.bonus_structure}\n"

    return section


def create_benefits_section(job_requirements: JobRequirements) -> str:
    """Create benefits section."""
    section = "## Benefits & Perks\n\n"

    for benefit in job_requirements.benefits:
        section += f"• {benefit}\n"

    return section


def create_additional_section(job_requirements: JobRequirements) -> str:
    """Create additional requirements section."""
    section = "## Additional Requirements\n\n"
    section += job_requirements.additional_requirements

    return section


def create_jd_footer(job_requirements: JobRequirements) -> str:
    """Create job description footer."""
    return (
        "---\n\n"
        "*We are an equal opportunity employer committed to diversity and inclusion.*"
    )


# Helper functions for JD generation


def categorize_job_role(job_title: str) -> str:
    """Categorize job role for appropriate overview generation."""
    title_lower = job_title.lower()

    if any(
        word in title_lower
        for word in ["engineer", "developer", "programmer", "architect"]
    ):
        return "technical"
    elif any(word in title_lower for word in ["marketing", "growth", "content"]):
        return "marketing"
    elif any(
        word in title_lower for word in ["sales", "account", "business development"]
    ):
        return "sales"
    elif any(word in title_lower for word in ["manager", "director", "lead"]):
        return "management"
    else:
        return "general"


def generate_technical_overview(job_requirements: JobRequirements) -> str:
    """Generate overview for technical roles."""
    return (
        f"We are seeking a skilled {job_requirements.job_title} to join our "
        f"{job_requirements.department or 'team'}. In this role, you will be responsible "
        f"for designing, developing, and maintaining high-quality software solutions. "
        f"You'll work collaboratively with cross-functional teams to deliver "
        f"innovative products and services."
    )


def generate_marketing_overview(job_requirements: JobRequirements) -> str:
    """Generate overview for marketing roles."""
    return (
        f"We are looking for a creative and data-driven {job_requirements.job_title} "
        f"to join our {job_requirements.department or 'marketing team'}. You will "
        f"develop and execute marketing strategies, analyze campaign performance, "
        f"and drive growth through innovative marketing initiatives."
    )


def generate_sales_overview(job_requirements: JobRequirements) -> str:
    """Generate overview for sales roles."""
    return (
        f"We are seeking a motivated {job_requirements.job_title} to join our "
        f"{job_requirements.department or 'sales team'}. You will be responsible "
        f"for building relationships with prospects, driving revenue growth, "
        f"and contributing to our company's continued success."
    )


def generate_general_overview(job_requirements: JobRequirements) -> str:
    """Generate overview for general roles."""
    return (
        f"We are looking for a dedicated {job_requirements.job_title} to join "
        f"our {job_requirements.department or 'team'}. In this role, you will "
        f"contribute to our organization's mission and work collaboratively "
        f"to achieve our goals."
    )


def format_location_for_jd(location) -> str:
    """Format location for job description."""
    location_type = location.location_type.value.replace("_", " ").title()

    if location.location_type.value == "remote":
        if location.timezone_requirements:
            return f"Remote ({location.timezone_requirements})"
        return "Remote"

    if location.city and location.state:
        if location.location_type.value == "hybrid":
            return f"Hybrid - {location.city}, {location.state}"
        return f"{location.city}, {location.state}"

    return location_type


def format_experience_for_jd(experience) -> str:
    """Format experience for job description."""
    parts = []

    if experience.years_min > 0:
        parts.append(f"{experience.years_min}+ years of relevant experience")

    parts.append(f"{experience.level.value.title()}-level expertise")

    if experience.industry_experience:
        parts.append(f"experience in {experience.industry_experience}")

    if experience.leadership_required:
        parts.append("proven leadership experience")

    if experience.management_experience:
        parts.append("people management experience")

    return ", ".join(parts)


def format_education_for_jd(education) -> str:
    """Format education for job description."""
    level_text = education.level.value.replace("_", " ").title()

    if education.level.value == "none_required":
        return "No specific education requirements"

    base_text = f"{level_text} degree"

    if education.field_of_study:
        base_text += f" in {education.field_of_study}"

    if not education.is_required:
        base_text += " (preferred but not required)"

    if education.alternatives_accepted:
        alternatives = ", ".join(education.alternatives_accepted)
        base_text += f" or equivalent experience ({alternatives})"

    return base_text


def create_final_output_message(generated_jd: str) -> str:
    """Create final message with generated JD and export options.

    Args:
        generated_jd: Complete generated job description

    Returns:
        Final output message with JD and options
    """
    message_parts = [
        "🎉 **Your job description is ready!**\n",
        "Here's the complete job description I've generated based on our conversation:\n",
        "---\n",
        generated_jd,
        "\n---\n",
        "**What's next?**",
        "• Copy the text above to use in your job posting",
        "• Edit any sections that need adjustment",
        "• Post to your preferred job boards",
        "• Share with your hiring team for review\n",
        "Thank you for using the AI Job Description Generator! "
        "Feel free to start a new conversation to create another job description.",
    ]

    return "\n".join(message_parts)


def enhance_jd_with_context(base_jd: str, job_requirements: JobRequirements) -> str:
    """Enhance job description with additional context and formatting.

    Args:
        base_jd: Basic generated job description
        job_requirements: Source job requirements data

    Returns:
        Enhanced job description with improved formatting
    """
    # Add role-specific enhancements
    role_type = categorize_job_role(job_requirements.job_title)

    if role_type == "technical":
        return enhance_technical_jd(base_jd, job_requirements)
    elif role_type == "sales":
        return enhance_sales_jd(base_jd, job_requirements)
    elif role_type == "marketing":
        return enhance_marketing_jd(base_jd, job_requirements)

    return base_jd


def enhance_technical_jd(jd: str, job_requirements: JobRequirements) -> str:
    """Add technical role-specific enhancements."""
    # Could add technical growth opportunities, tech stack details, etc.
    return jd


def enhance_sales_jd(jd: str, job_requirements: JobRequirements) -> str:
    """Add sales role-specific enhancements."""
    # Could add quota information, commission structure, etc.
    return jd


def enhance_marketing_jd(jd: str, job_requirements: JobRequirements) -> str:
    """Add marketing role-specific enhancements."""
    # Could add campaign types, marketing channels, etc.
    return jd


def categorize_job_role(job_title: str) -> str:
    """Categorize job role type for enhancement."""
    title_lower = job_title.lower()

    if any(word in title_lower for word in ["engineer", "developer", "programmer"]):
        return "technical"
    elif any(
        word in title_lower for word in ["sales", "account", "business development"]
    ):
        return "sales"
    elif any(word in title_lower for word in ["marketing", "growth", "content"]):
        return "marketing"

    return "general"


def create_jd_summary_stats(job_requirements: JobRequirements) -> Dict[str, Any]:
    """Create summary statistics for the generated JD.

    Args:
        job_requirements: Source job requirements

    Returns:
        Dictionary of JD statistics
    """
    return {
        "total_responsibilities": len(job_requirements.responsibilities),
        "technical_skills_count": len(job_requirements.skills.technical_skills),
        "soft_skills_count": len(job_requirements.skills.soft_skills),
        "has_salary_range": job_requirements.salary is not None,
        "benefits_count": len(job_requirements.benefits),
        "experience_level": job_requirements.experience.level.value,
        "location_type": job_requirements.location.location_type.value,
        "is_remote_friendly": job_requirements.location.location_type.value
        in ["remote", "hybrid"],
    }


def validate_generated_jd(jd_text: str) -> tuple[bool, List[str]]:
    """Validate generated job description for quality and completeness.

    Args:
        jd_text: Generated job description text

    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []

    # Check minimum length
    if len(jd_text) < 500:
        issues.append("Job description seems too short (less than 500 characters)")

    # Check for required sections
    required_sections = ["responsibilities", "requirements", "qualifications"]
    for section in required_sections:
        if section.lower() not in jd_text.lower():
            issues.append(f"Missing {section} section")

    # Check for placeholder text
    placeholders = ["[placeholder]", "TODO", "TBD", "XXX"]
    for placeholder in placeholders:
        if placeholder in jd_text:
            issues.append(f"Contains placeholder text: {placeholder}")

    return len(issues) == 0, issues
