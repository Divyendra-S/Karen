"""Greeting node implementation with pure function design."""

from typing import Dict, Any, Optional
from datetime import datetime
from ...models.graph_state import GraphState


def greeting_node(state: GraphState) -> Dict[str, Any]:
    """Initial greeting node that welcomes user and explains the process.
    
    This pure function initiates the conversation with a friendly greeting
    and sets up the initial state for job description generation.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates dictionary with greeting message and phase change
    """
    greeting_message = (
        "Hi! I'm your AI assistant for creating comprehensive job descriptions. "
        "I'll guide you through a series of questions to gather all the necessary "
        "information about the position you want to post.\n\n"
        "The process typically takes 5-10 minutes and covers:\n"
        "• Job title and department\n"
        "• Experience and skill requirements\n"
        "• Key responsibilities\n"
        "• Location and employment details\n"
        "• Compensation and benefits\n\n"
        "Let's start with the basics! What is the job title for this position?"
    )
    
    # Create new message for state
    new_message = {
        "role": "assistant",
        "content": greeting_message,
        "message_type": "question",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Update state with greeting message
    new_messages = state.get("messages", []) + [new_message]
    
    return {
        **state,
        "messages": new_messages,
        "current_field": "job_title",
        "conversation_phase": "collecting_basic_info"
    }


def create_welcome_message() -> str:
    """Create a welcome message for the conversation.
    
    Returns:
        Formatted welcome message string
    """
    return (
        "Welcome to the AI Job Description Generator! 🎯\n\n"
        "I'll help you create a professional, comprehensive job description "
        "by asking you targeted questions about the role. The entire process "
        "is designed to be conversational and efficient.\n\n"
        "Ready to get started?"
    )


def get_process_explanation() -> str:
    """Get explanation of the JD generation process.
    
    Returns:
        Process explanation text
    """
    return (
        "Here's how this works:\n\n"
        "1. **Basic Information**: Job title, department, employment type\n"
        "2. **Experience & Skills**: Required background and competencies\n"
        "3. **Responsibilities**: Key duties and expectations\n"
        "4. **Requirements**: Education, location, and other specifics\n"
        "5. **Compensation**: Salary range and benefits (optional)\n"
        "6. **Generation**: I'll create a polished job description\n\n"
        "You can use voice input or typing - whatever feels more natural!"
    )


def should_show_process_explanation(state: GraphState) -> bool:
    """Determine if process explanation should be included.
    
    Args:
        state: Current graph state
        
    Returns:
        True if explanation should be shown
    """
    messages = state.get("messages", [])
    
    # Show explanation for new conversations
    if len(messages) == 0:
        return True
    
    # Don't repeat explanation
    return False


def get_initial_question() -> str:
    """Get the first question to ask the user.
    
    Returns:
        Initial question about job title
    """
    return "What is the job title for this position?"


def create_greeting_with_context(
    user_name: Optional[str] = None,
    company_name: Optional[str] = None
) -> str:
    """Create personalized greeting with available context.
    
    Args:
        user_name: Optional user name for personalization
        company_name: Optional company name for context
        
    Returns:
        Personalized greeting message
    """
    greeting_parts = []
    
    if user_name:
        greeting_parts.append(f"Hi {user_name}!")
    else:
        greeting_parts.append("Hi there!")
    
    if company_name:
        greeting_parts.append(
            f"I'm here to help you create a job description for {company_name}."
        )
    else:
        greeting_parts.append(
            "I'm here to help you create a professional job description."
        )
    
    greeting_parts.append(get_process_explanation())
    greeting_parts.append(get_initial_question())
    
    return "\n\n".join(greeting_parts)


def extract_session_context(state: GraphState) -> Dict[str, Any]:
    """Extract session context from state metadata.
    
    Args:
        state: Current graph state
        
    Returns:
        Dictionary of session context information
    """
    session_metadata = state.get("session_metadata", {})
    
    return {
        "session_id": session_metadata.get("session_id"),
        "user_name": session_metadata.get("user_name"),
        "company_name": session_metadata.get("company_name"),
        "voice_enabled": session_metadata.get("voice_enabled", False),
        "language": session_metadata.get("language", "en")
    }


def customize_greeting_by_context(state: GraphState) -> str:
    """Customize greeting message based on session context.
    
    Args:
        state: Current graph state
        
    Returns:
        Contextualized greeting message
    """
    context = extract_session_context(state)
    
    # Use context to personalize greeting
    return create_greeting_with_context(
        user_name=context.get("user_name"),
        company_name=context.get("company_name")
    )