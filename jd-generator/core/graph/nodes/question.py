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
            "conversation_phase": ConversationPhase.REVIEWING_DATA.value
        }
    
    # Update conversation phase based on field type
    phase = _determine_phase_for_field(next_field)
    
    return {
        **state,
        "current_field": next_field,
        "conversation_phase": phase.value
    }


def _determine_next_field(state: GraphState) -> Optional[str]:
    """Determine the next field to collect based on current state."""
    collected_fields = set(_get_collected_fields(state))
    
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
        "additional_requirements"
    ]
    
    # Find next uncollected field
    for field in field_priority:
        if field not in collected_fields:
            return field
    
    return None


def _get_collected_fields(state: GraphState) -> List[str]:
    """Get list of fields that have been collected."""
    return [
        field for field, value in state.get("job_data", {}).items()
        if value is not None and value != "" and value != []
    ]


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
        question = _generate_retry_question(current_field, validation_errors, retry_count)
    else:
        question = _generate_initial_question(current_field, state)
    
    # Add question to conversation
    new_message = {
        "role": "assistant",
        "content": question,
        "message_type": "question",
        "timestamp": datetime.utcnow().isoformat(),
        "related_field": current_field
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
        "additional_requirements": ConversationPhase.COLLECTING_REQUIREMENTS
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
        "job_title": "What is the job title for this position?",
        
        "department": _get_department_question(job_data),
        
        "employment_type": (
            "What type of employment is this? For example:\n"
            "• Full-time permanent position\n"
            "• Part-time\n" 
            "• Contract/consulting\n"
            "• Internship\n"
            "• Temporary"
        ),
        
        "location": _get_location_question(job_data),
        
        "experience": _get_experience_question(job_data),
        
        "responsibilities": (
            "What are the key responsibilities and day-to-day duties for this role? "
            "Please describe the main tasks and expectations."
        ),
        
        "skills": _get_skills_question(job_data),
        
        "education": (
            "What are the education requirements? For example:\n"
            "• Bachelor's degree in Computer Science\n"
            "• Master's preferred\n"
            "• High school diploma\n"
            "• No specific education required"
        ),
        
        "salary": (
            "What is the salary range for this position? "
            "(This is optional - you can skip if you prefer not to specify)"
        ),
        
        "benefits": (
            "What benefits and perks are offered? For example:\n"
            "• Health insurance\n"
            "• 401k matching\n"
            "• Flexible PTO\n"
            "• Remote work options\n"
            "(Optional - you can skip this)"
        ),
        
        "additional_requirements": (
            "Are there any additional requirements or special considerations? "
            "For example, security clearance, travel, certifications, etc. "
            "(Optional - you can skip this)"
        )
    }
    
    return questions.get(field_name, f"Please provide information about {field_name}.")


def _get_department_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual department question."""
    job_title = job_data.get("job_title", "").lower()
    
    if "engineer" in job_title or "developer" in job_title:
        return (
            "Which department or team will this role be part of? "
            "For example: Engineering, Product Development, DevOps, etc."
        )
    elif "marketing" in job_title:
        return (
            "Which department will this role be in? "
            "For example: Digital Marketing, Content Marketing, Growth, etc."
        )
    elif "sales" in job_title:
        return (
            "Which sales team or department? "
            "For example: Inside Sales, Enterprise Sales, Business Development, etc."
        )
    else:
        return "Which department or team will this role be part of?"


def _get_location_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual location question."""
    return (
        "What are the location requirements for this position?\n"
        "• On-site (please specify city/state)\n"
        "• Fully remote\n"
        "• Hybrid (mix of remote and office)\n\n"
        "If location-specific, please include the city and state."
    )


def _get_experience_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual experience question."""
    job_title = job_data.get("job_title", "").lower()
    
    if "senior" in job_title:
        return (
            "What experience requirements do you have for this senior role? "
            "Please include:\n"
            "• Years of experience needed\n"
            "• Specific industry experience\n"
            "• Leadership or management experience requirements"
        )
    elif "junior" in job_title or "entry" in job_title:
        return (
            "What experience requirements do you have? Since this appears to be "
            "an entry-level role, please specify:\n"
            "• Minimum years of experience (if any)\n"
            "• Internship or project experience\n"
            "• Any specific background needed"
        )
    else:
        return (
            "What experience requirements do you have for this role? "
            "Please specify:\n"
            "• Years of experience needed\n"
            "• Specific industry or domain experience\n"
            "• Any leadership or specialized experience"
        )


def _get_skills_question(job_data: Dict[str, Any]) -> str:
    """Generate contextual skills question."""
    job_title = job_data.get("job_title", "").lower()
    
    if any(tech_word in job_title for tech_word in ["engineer", "developer", "programmer", "analyst"]):
        return (
            "What technical skills and competencies are required? Please include:\n"
            "• Programming languages (e.g., Python, JavaScript, Java)\n"
            "• Frameworks and tools (e.g., React, Django, AWS)\n"
            "• Soft skills (e.g., communication, problem-solving)\n"
            "• Any certifications or specializations"
        )
    elif "marketing" in job_title:
        return (
            "What skills and competencies are required? Please include:\n"
            "• Marketing tools and platforms\n"
            "• Analytical and creative skills\n"
            "• Communication and presentation abilities\n"
            "• Any specific marketing certifications"
        )
    elif "sales" in job_title:
        return (
            "What skills and competencies are required? Please include:\n"
            "• Sales methodologies and techniques\n"
            "• CRM and sales tools experience\n"
            "• Communication and negotiation skills\n"
            "• Industry knowledge requirements"
        )
    else:
        return (
            "What skills and competencies are required for this role? "
            "Please include both technical skills (tools, software, methods) "
            "and soft skills (communication, leadership, etc.)"
        )


def _generate_retry_question(
    field_name: str, 
    validation_errors: List[str], 
    retry_count: int
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
            f"I need a bit more information about the job title. {error_context} "
            "Please provide a clear job title (e.g., 'Software Engineer', 'Marketing Manager')."
        ),
        
        "responsibilities": (
            f"Let me get more details about the responsibilities. {error_context} "
            "Please provide a more detailed description of the key duties and tasks."
        ),
        
        "skills": (
            f"I'd like to better understand the skill requirements. {error_context} "
            "Please provide more specific information about required skills and competencies."
        ),
        
        "experience": (
            f"Let me clarify the experience requirements. {error_context} "
            "Please provide more details about the background and experience needed."
        )
    }
    
    base_message = base_messages.get(
        field_name, 
        f"Let me get more information about {field_name}. {error_context}"
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
            "Product Manager"
        ],
        
        "department": [
            "Engineering",
            "Marketing",
            "Sales",
            "Product",
            "Operations"
        ],
        
        "responsibilities": [
            "Design and develop web applications",
            "Manage marketing campaigns and analyze performance",
            "Build relationships with prospective clients",
            "Analyze data to drive business decisions"
        ],
        
        "skills": [
            "Python, React, PostgreSQL, problem-solving",
            "Google Analytics, content creation, project management",
            "CRM software, negotiation, relationship building",
            "SQL, Python, statistical analysis, data visualization"
        ]
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