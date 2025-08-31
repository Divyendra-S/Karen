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
        "Hi! I'm here to help you create a job description. What's the job title for this position?"
    )

    # Create new message for state
    new_message = {
        "role": "assistant",
        "content": greeting_message,
        "message_type": "question",
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Update state with greeting message
    new_messages = state.get("messages", []) + [new_message]

    return {
        **state,
        "messages": new_messages,
        "current_field": "job_title",
        "conversation_phase": "collecting_basic_info",
    }


def create_welcome_message() -> str:
    """Create a welcome message for the conversation.

    Returns:
        Formatted welcome message string
    """
    return (
        "Welcome to your personalized career interview! 🎯\n\n"
        "I'm here to understand your background, skills, and career goals "
        "so I can create a job description tailored to what you're looking for. "
        "This conversation is designed to be natural and focused on you.\n\n"
        "Ready to tell me about your career aspirations?"
    )


def get_process_explanation() -> str:
    """Get explanation of the JD generation process.

    Returns:
        Process explanation text
    """
    return (
        "Here's how our interview works:\n\n"
        "1. **Your Background**: Tell me about your current role and experience\n"
        "2. **Your Skills**: What technical and soft skills do you want to use?\n"
        "3. **Your Preferences**: What responsibilities do you enjoy most?\n"
        "4. **Your Requirements**: Location, work style, and education background\n"
        "5. **Your Goals**: Compensation expectations and benefits you value\n"
        "6. **Your Job Description**: I'll create a personalized JD based on your input\n\n"
        "Feel free to speak naturally - this is your interview!"
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
        Initial question about their background
    """
    return "Could you tell me a bit about yourself and what kind of role you're interested in?"


def create_greeting_with_context(
    user_name: Optional[str] = None, company_name: Optional[str] = None
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
            f"I'm here to interview you and understand what kind of role you'd be perfect for at {company_name}."
        )
    else:
        greeting_parts.append(
            "I'm here to interview you and understand what kind of role would be perfect for your career goals."
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
        "language": session_metadata.get("language", "en"),
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
        user_name=context.get("user_name"), company_name=context.get("company_name")
    )
