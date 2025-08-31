"""Question routing and generation nodes with functional patterns."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from ...models.graph_state import GraphState
from ...models.conversation import ConversationPhase


def question_router_node(state: GraphState) -> Dict[str, Any]:
    """Route to appropriate question based on current state.

    Pure function that analyzes conversation state and determines
    what field to collect next based on priority and completeness.

    Args:
        state: Current graph state

    Returns:
        State updates with next field to collect
    """
    # Get next field to collect
    next_field = _determine_next_field(state)

    if next_field is None:
        # All fields collected, mark as ready for generation
        return {
            **state,
            "current_field": None,
            "is_complete": True,
            "conversation_phase": ConversationPhase.REVIEWING_DATA.value,
        }

    # Update conversation phase based on field type
    phase = _determine_phase_for_field(next_field)

    return {**state, "current_field": next_field, "conversation_phase": phase.value}


def _determine_next_field(state: GraphState) -> Optional[str]:
    """Determine the next field to collect based on current state."""
    collected_fields = set(_get_collected_fields(state))

    # Also include fields extracted from evaluation
    extracted_fields = set(state.get("extracted_fields", {}).keys())
    all_collected_fields = collected_fields | extracted_fields

    # Define collection priority order for interviewing the candidate
    field_priority = [
        "background_intro",
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
        if field not in all_collected_fields:
            return field

    return None


def _get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been collected."""
    # Fields from job_data
    job_data_fields = [
        field
        for field, value in state.get("job_data", {}).items()
        if value is not None and value != "" and value != []
    ]

    # Fields from evaluation extraction
    extracted_fields = [
        field
        for field, value in state.get("extracted_fields", {}).items()
        if value is not None and value != "" and value != []
    ]

    # Combine and deduplicate
    all_fields = list(set(job_data_fields + extracted_fields))
    return all_fields


def question_generator_node(state: GraphState) -> Dict[str, Any]:
    """Generate contextual questions based on current field and state.

    Pure function that creates appropriate questions for the current
    field being collected, with context from previous responses.

    Args:
        state: Current graph state

    Returns:
        State updates with generated question message
    """
    current_field = state.get("current_field")
    if not current_field:
        return state

    # Check if this is a retry
    retry_count = state.get("retry_count", 0)
    validation_errors = state.get("validation_errors", [])

    if retry_count > 0 and validation_errors:
        question = _generate_retry_question(
            current_field, validation_errors, retry_count
        )
    else:
        question = _generate_initial_question(current_field, state)

    # Add question to conversation
    new_message = {
        "role": "assistant",
        "content": question,
        "message_type": "question",
        "timestamp": datetime.utcnow().isoformat(),
        "related_field": current_field,
    }

    new_messages = state.get("messages", []) + [new_message]
    return {**state, "messages": new_messages}


def _determine_phase_for_field(field_name: str) -> ConversationPhase:
    """Determine conversation phase based on field being collected.

    Args:
        field_name: Name of the field being collected

    Returns:
        Appropriate conversation phase
    """
    phase_mapping = {
        "background_intro": ConversationPhase.COLLECTING_BASIC_INFO,
        "job_title": ConversationPhase.COLLECTING_BASIC_INFO,
        "department": ConversationPhase.COLLECTING_BASIC_INFO,
        "employment_type": ConversationPhase.COLLECTING_BASIC_INFO,
        "location": ConversationPhase.COLLECTING_BASIC_INFO,
        "experience": ConversationPhase.COLLECTING_EXPERIENCE,
        "skills": ConversationPhase.COLLECTING_SKILLS,
        "responsibilities": ConversationPhase.COLLECTING_RESPONSIBILITIES,
        "education": ConversationPhase.COLLECTING_REQUIREMENTS,
        "salary": ConversationPhase.COLLECTING_REQUIREMENTS,
        "benefits": ConversationPhase.COLLECTING_REQUIREMENTS,
        "additional_requirements": ConversationPhase.COLLECTING_REQUIREMENTS,
    }

    return phase_mapping.get(field_name, ConversationPhase.COLLECTING_REQUIREMENTS)


def _generate_initial_question(field_name: str, state: GraphState) -> str:
    """Generate initial question for a specific field.

    Args:
        field_name: Field to generate question for
        state: Current graph state for context

    Returns:
        Generated question text
    """
    job_data = state.get("job_data", {})

    questions = {
        "background_intro": (
            "Let's start with your background! Tell me about yourself, your current role, "
            "and what kind of position you're looking for. This helps me understand your overall career direction."
        ),
        "job_title": (
            "What specific job title are you targeting? Please provide the exact title you'd want to see on your business card. "
            "Examples: 'Senior Software Engineer', 'Product Marketing Manager', 'Data Scientist'"
        ),
        "department": _get_department_question(job_data),
        "employment_type": (
            "What type of work arrangement are you seeking? Please choose from:\n"
            "• Full-time (permanent, 40+ hours/week)\n"
            "• Part-time (less than 30 hours/week)\n"
            "• Contract (project-based, fixed duration)\n"
            "• Internship (learning-focused, temporary)\n"
            "• Freelance (independent contractor)\n"
            "Please specify which type best matches what you want."
        ),
        "location": (
            "What are your location preferences? I need to know:\n"
            "• Work style: On-site, fully remote, or hybrid?\n"
            "• If on-site/hybrid: Which city and state do you prefer?\n"
            "• Are you open to relocation?\n"
            "Example: 'Hybrid in Austin, Texas' or 'Fully remote anywhere in US'"
        ),
        "experience": _get_experience_question(job_data),
        "responsibilities": (
            "What specific responsibilities and daily tasks do you want in your ideal role? "
            "Please list 3-10 specific duties you'd enjoy doing. "
            "Example: 'Design user interfaces, Write clean code, Mentor junior developers, Lead sprint planning'"
        ),
        "skills": _get_skills_question(job_data),
        "education": (
            "What's your educational background? Please tell me:\n"
            "• Your highest degree level (high school, associate, bachelor's, master's, doctorate)\n"
            "• Your field of study (if applicable)\n"
            "• Any relevant certifications or bootcamp experience\n"
            "Example: 'Bachelor's in Computer Science with AWS certification'"
        ),
        "salary": (
            "What salary range are you targeting? Please provide:\n"
            "• Minimum acceptable salary\n"
            "• Ideal target salary\n"
            "• Annual, monthly, or hourly rate\n"
            "Example: '$80,000 - $100,000 annual' or 'Skip if you prefer not to specify'"
        ),
        "benefits": (
            "What benefits and perks are most important to you? Please list specific benefits you value:\n"
            "Examples: Health insurance, 401k matching, Flexible PTO, Remote work, Professional development budget\n"
            "List as many as are important to you, separated by commas."
        ),
        "additional_requirements": (
            "Any other preferences or requirements for your ideal role? For example:\n"
            "• Specific certifications you have\n"
            "• Willingness to travel (percentage)\n"
            "• Security clearance requirements\n"
            "• Company size preferences\n"
            "• Work culture preferences\n"
            "(Optional - say 'none' if nothing else)"
        ),
    }

    return questions.get(field_name, f"Please provide information about {field_name}.")


def _get_department_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual department question."""
    job_title = job_data.get("job_title", "").lower()

    if "engineer" in job_title or "developer" in job_title:
        return (
            "What specific department or team do you want to work in? Please provide the exact team name:\n"
            "Examples: 'Engineering', 'Platform Engineering', 'DevOps', 'Frontend Engineering', 'Data Engineering'\n"
            "I need the specific department name for your job description."
        )
    elif "marketing" in job_title:
        return (
            "What specific marketing department interests you? Please provide the exact team name:\n"
            "Examples: 'Digital Marketing', 'Content Marketing', 'Growth Marketing', 'Brand Marketing', 'Product Marketing'\n"
            "I need the specific department name for your job description."
        )
    elif "sales" in job_title:
        return (
            "What specific sales department do you want to join? Please provide the exact team name:\n"
            "Examples: 'Inside Sales', 'Enterprise Sales', 'Business Development', 'Customer Success', 'Sales Development'\n"
            "I need the specific department name for your job description."
        )
    else:
        return (
            "What specific department or team do you want to work in? "
            "Please provide the exact department name (e.g., 'Operations', 'Finance', 'Human Resources', 'Legal')."
        )


def _get_location_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual location question."""
    return (
        "What's your preferred work location and setup?\n"
        "• On-site work (please specify your preferred city/state)\n"
        "• Fully remote\n"
        "• Hybrid (mix of remote and office)\n\n"
        "Where would you ideally like to work?"
    )


def _get_experience_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual experience question."""
    job_title = job_data.get("job_title", "").lower()

    if "senior" in job_title:
        return (
            "I need details about your professional experience for this senior role. Please provide:\n"
            "• EXACT number of years of experience (e.g., '7 years')\n"
            "• Specific industries you've worked in (e.g., 'fintech, healthcare')\n"
            "• Leadership experience: Yes/No and details if yes\n"
            "• Experience level: senior, lead, or principal\n"
            "Example: '8 years experience in fintech, led teams of 5+ developers, senior level'"
        )
    elif "junior" in job_title or "entry" in job_title:
        return (
            "Tell me about your background and experience level. Please provide:\n"
            "• EXACT years of professional experience (e.g., '1 year', '6 months', 'new graduate')\n"
            "• Relevant internships, projects, or bootcamp experience\n"
            "• Experience level: entry or junior\n"
            "• Any specific industry exposure\n"
            "Example: 'New graduate with 2 internships at tech startups, entry level'"
        )
    else:
        return (
            "Please tell me about your professional experience. I need:\n"
            "• EXACT number of years of experience (e.g., '5 years')\n"
            "• Industries or domains you've worked in\n"
            "• Experience level: entry/junior/mid/senior/lead/principal\n"
            "• Leadership experience: Yes/No with details\n"
            "Example: '5 years in e-commerce, mid-level, some team leadership experience'"
        )


def _get_skills_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual skills question."""
    job_title = job_data.get("job_title", "").lower()

    if any(
        tech_word in job_title
        for tech_word in ["engineer", "developer", "programmer", "analyst"]
    ):
        return (
            "What are your technical and soft skills? Please provide SPECIFIC lists:\n"
            "• Programming languages: List each one (e.g., Python, JavaScript, Java, SQL)\n"
            "• Frameworks/tools: List each one (e.g., React, Django, AWS, Docker)\n"
            "• Technical skills: List each one (e.g., System Design, API Development, Testing)\n"
            "• Soft skills: List each one (e.g., Communication, Problem-solving, Leadership)\n"
            "• Certifications: List any you have (e.g., AWS Certified, Scrum Master)\n"
            "Format: Separate each skill with commas within each category."
        )
    elif "marketing" in job_title:
        return (
            "What are your marketing skills and expertise? Please provide SPECIFIC lists:\n"
            "• Marketing tools: List each one (e.g., Google Analytics, HubSpot, Salesforce)\n"
            "• Technical skills: List each one (e.g., SEO, PPC, Email Marketing, Content Strategy)\n"
            "• Soft skills: List each one (e.g., Communication, Creativity, Analytical Thinking)\n"
            "• Certifications: List any you have (e.g., Google Ads, Facebook Blueprint)\n"
            "Format: Separate each skill with commas within each category."
        )
    elif "sales" in job_title:
        return (
            "What are your sales skills and experience? Please provide SPECIFIC lists:\n"
            "• Sales tools: List each one (e.g., Salesforce, HubSpot, Outreach)\n"
            "• Sales skills: List each one (e.g., Cold Calling, Negotiation, Account Management)\n"
            "• Soft skills: List each one (e.g., Communication, Relationship Building, Persistence)\n"
            "• Certifications: List any you have (e.g., Salesforce Certified, Sales methodology training)\n"
            "Format: Separate each skill with commas within each category."
        )
    else:
        return (
            "What are your key skills and expertise? Please provide SPECIFIC lists:\n"
            "• Technical skills: List tools, software, methods you know\n"
            "• Soft skills: List interpersonal and cognitive abilities\n"
            "• Certifications: List any professional certifications\n"
            "• Specialized knowledge: List domain expertise areas\n"
            "Format: Separate each skill with commas within each category."
        )


def _generate_retry_question(
    field_name: str, validation_errors: List[str], retry_count: int
) -> str:
    """Generate retry question when validation fails.

    Args:
        field_name: Field that failed validation
        validation_errors: List of validation error messages
        retry_count: Current retry attempt number

    Returns:
        Retry question with helpful guidance
    """
    error_context = " ".join(validation_errors[:2])  # Show first 2 errors

    base_messages = {
        "job_title": (
            f"I'd like to understand better what role you're seeking. {error_context} "
            "Could you be more specific about your ideal job title? (e.g., 'Software Engineer', 'Marketing Manager')"
        ),
        "responsibilities": (
            f"Help me understand what kind of work you enjoy most. {error_context} "
            "Could you tell me more about the types of tasks and responsibilities you'd want in your ideal role?"
        ),
        "skills": (
            f"I'd like to learn more about your skills and what you'd want to use. {error_context} "
            "Could you share more about your technical abilities and strengths?"
        ),
        "experience": (
            f"Let me learn more about your professional background. {error_context} "
            "Could you tell me more about your experience and what you've worked on?"
        ),
    }

    base_message = base_messages.get(
        field_name, f"Let me get more information about {field_name}. {error_context}"
    )

    if retry_count >= 2:
        return (
            f"{base_message}\n\n"
            "If you're not sure about this field, we can skip it for now and "
            "come back to it later. Just say 'skip' or 'not sure'."
        )

    return base_message


def get_field_examples(field_name: str) -> List[str]:
    """Get example responses for a specific field.

    Args:
        field_name: Field to get examples for

    Returns:
        List of example responses
    """
    examples = {
        "job_title": [
            "Senior Software Engineer",
            "Marketing Manager",
            "Sales Representative",
            "Data Scientist",
            "Product Manager",
        ],
        "department": ["Engineering", "Marketing", "Sales", "Product", "Operations"],
        "responsibilities": [
            "Design and develop web applications",
            "Manage marketing campaigns and analyze performance",
            "Build relationships with prospective clients",
            "Analyze data to drive business decisions",
        ],
        "skills": [
            "Python, React, PostgreSQL, problem-solving",
            "Google Analytics, content creation, project management",
            "CRM software, negotiation, relationship building",
            "SQL, Python, statistical analysis, data visualization",
        ],
    }

    return examples.get(field_name, [])


def should_provide_examples(field_name: str, retry_count: int) -> bool:
    """Determine if examples should be provided with question.

    Args:
        field_name: Current field being collected
        retry_count: Number of retry attempts

    Returns:
        True if examples should be included
    """
    # Provide examples on first attempt for complex fields
    complex_fields = {"responsibilities", "skills", "experience"}
    if field_name in complex_fields and retry_count == 0:
        return True

    # Always provide examples on retry
    return retry_count > 0
