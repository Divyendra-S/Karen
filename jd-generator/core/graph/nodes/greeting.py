"""Greeting node implementation with dynamic LLM generation."""

from typing import Dict, Any, Optional
from datetime import datetime
from ...models.graph_state import GraphState
import streamlit as st
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def greeting_node(state: GraphState) -> Dict[str, Any]:
    """Initial greeting node that dynamically generates welcome message.
    
    Uses LLM to create personalized, context-aware greeting messages
    based on user input and session context.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates dictionary with dynamic greeting message
    """
    # Get last user message to understand how they initiated conversation
    last_user_message = _get_last_user_message_content(state)
    
    # Generate dynamic greeting response
    greeting_message = generate_dynamic_greeting(last_user_message, state)
    
    # Create new message for state
    new_message = {
        "role": "assistant",
        "content": greeting_message,
        "message_type": "greeting",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Update state with greeting message and start job_title collection
    new_messages = state.get("messages", []) + [new_message]
    
    return {
        **state,
        "messages": new_messages,
        "current_field": "job_title",
        "conversation_phase": "collecting_basic_info"
    }


def generate_dynamic_greeting(user_input: Optional[str], state: GraphState) -> str:
    """Generate dynamic greeting using LLM based on user's initial message."""
    try:
        # Extract any name mentioned in user's input
        user_name = _extract_name_from_input(user_input) if user_input else None
        
        greeting_prompt = f"""The user just started a conversation by saying: "{user_input or 'Hello'}"

You are an AI interviewer conducting a profile assessment. Generate a warm, professional greeting that:
1. Responds naturally to their message
2. {f"Uses their name ({user_name})" if user_name else "Greets them warmly"}
3. Explains you're here to understand their professional profile and experience
4. Shows interest in learning about their background
5. Keep it conversational and under 40 words

Make it feel natural and engaging, not robotic.

Example responses:
- If they said "Hi": "Hi there! I'm here to learn about your professional background and experience. Tell me about yourself!"
- If they said "Hello, I'm John": "Hello John! Nice to meet you. I'd love to learn about your professional background and experience."
- If they said "I'm a software developer": "Great to meet you! I'd love to learn more about your experience as a software developer."

Generate greeting response:"""

        if hasattr(st, 'session_state') and 'groq_client' in st.session_state:
            logger.info("🔄 Calling Groq API for dynamic greeting generation")
            completion = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": greeting_prompt}],
                model=settings.llm_model,
                max_tokens=60,
                temperature=0.4
            )
            
            generated_greeting = completion.choices[0].message.content.strip()
            logger.info(f"✅ LLM Generated greeting: '{generated_greeting}'")
            return generated_greeting
        else:
            # Minimal fallback
            logger.warning("⚠️ Using hardcoded fallback greeting - LLM client not available")
            return _get_minimal_greeting_fallback(user_name)
            
    except Exception as e:
        logger.error(f"Error generating dynamic greeting: {e}")
        return _get_minimal_greeting_fallback(user_name)


def _extract_name_from_input(user_input: str) -> Optional[str]:
    """Extract user's name from their input."""
    if not user_input:
        return None
        
    import re
    
    input_lower = user_input.lower()
    
    # Common introduction patterns
    patterns = [
        r"i am (\w+)",
        r"i'm (\w+)", 
        r"my name is (\w+)",
        r"call me (\w+)",
        r"this is (\w+)",
        r"hi.* i'm (\w+)",
        r"hello.* i'm (\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_lower)
        if match:
            name = match.group(1)
            # Filter out common non-name words
            non_names = ["there", "here", "good", "fine", "well", "ok", "yes", "no", "looking", "trying"]
            if name not in non_names and len(name) > 1:
                return name.title()
    
    return None


def _get_last_user_message_content(state: GraphState) -> Optional[str]:
    """Get the last user message content."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def _get_minimal_greeting_fallback(user_name: Optional[str]) -> str:
    """Minimal greeting fallback when LLM fails."""
    if user_name:
        return f"Hi {user_name}! I'd love to learn about your professional background and experience."
    else:
        return "Hello! I'm here to understand your professional profile and experience. Tell me about yourself!"


def should_personalize_greeting(state: GraphState) -> bool:
    """Determine if greeting should be personalized based on context."""
    session_metadata = state.get("session_metadata", {})
    
    # Personalize if we have user information
    return bool(
        session_metadata.get("user_name") or 
        session_metadata.get("company_name")
    )


def extract_session_context(state: GraphState) -> Dict[str, Any]:
    """Extract session context for greeting personalization."""
    session_metadata = state.get("session_metadata", {})
    
    return {
        "user_name": session_metadata.get("user_name"),
        "company_name": session_metadata.get("company_name"),
        "voice_enabled": session_metadata.get("voice_enabled", False),
        "language": session_metadata.get("language", "en")
    }


def create_greeting_with_company_context(
    user_name: Optional[str],
    company_name: Optional[str],
    user_input: str
) -> str:
    """Create greeting with company context using LLM."""
    try:
        company_prompt = f"""Generate a greeting for someone from {company_name or 'a company'} who said: "{user_input}"

Include:
1. Warm greeting {f"using name {user_name}" if user_name else ""}
2. {f"Reference to {company_name}" if company_name else "Professional tone"}
3. Explain you'll help create their job description
4. Ask what position they're hiring for

Keep it under 35 words and professional.
"""

        if hasattr(st, 'session_state') and 'groq_client' in st.session_state:
            completion = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": company_prompt}],
                model=settings.llm_model,
                max_tokens=50,
                temperature=0.3
            )
            
            return completion.choices[0].message.content.strip()
        else:
            base = f"Hi {user_name}!" if user_name else "Hello!"
            company_part = f" Thanks for reaching out from {company_name}." if company_name else ""
            return f"{base}{company_part} I'd love to learn about your professional background and experience."
            
    except Exception as e:
        logger.error(f"Error with company context greeting: {e}")
        return _get_minimal_greeting_fallback(user_name)


def is_greeting_message(user_input: str) -> bool:
    """Check if user input appears to be an initial greeting."""
    if not user_input:
        return False
        
    greeting_indicators = [
        "hi", "hello", "hey", "good morning", "good afternoon", 
        "good evening", "greetings", "yo", "sup"
    ]
    
    input_lower = user_input.lower().strip()
    
    # Check if input starts with or contains greeting words
    return (
        any(input_lower.startswith(greeting) for greeting in greeting_indicators) or
        any(greeting in input_lower for greeting in greeting_indicators) and len(input_lower) < 50
    )