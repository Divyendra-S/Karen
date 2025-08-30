"""Question routing and generation nodes with pure LLM-driven approach."""

from typing import Dict, Any, Optional
from datetime import datetime
from ...models.graph_state import GraphState
from ...models.conversation import ConversationPhase
import streamlit as st
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def question_router_node(state: GraphState) -> Dict[str, Any]:
    """Route to appropriate question based on current state.
    
    Pure function that analyzes conversation state and determines
    next action based on completion and clarification needs.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with routing decisions
    """
    current_field = state.get("current_field")
    clarification_needed = state.get("clarification_needed", [])
    processed_data = state.get("processed_data", {})
    is_complete = state.get("is_complete", False)
    
    # If conversation is complete, mark it
    if is_complete or not current_field:
        return {
            **state,
            "conversation_phase": ConversationPhase.REVIEWING_DATA.value
        }
    
    # If current field needs clarification, keep asking about it
    if current_field in clarification_needed:
        return {
            **state,
            "conversation_phase": _determine_phase_for_field(current_field).value
        }
    
    # Otherwise, continue with current field collection
    return {
        **state,
        "conversation_phase": _determine_phase_for_field(current_field).value
    }


def _determine_phase_for_field(field_name: str) -> ConversationPhase:
    """Determine conversation phase based on field being collected."""
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


def generate_dynamic_question(
    field_name: str, 
    state: GraphState,
    retry_count: int = 0
) -> str:
    """Generate question dynamically using LLM based on context."""
    try:
        processed_data = state.get("processed_data", {})
        raw_responses = state.get("raw_responses", {})
        clarification_needed = state.get("clarification_needed", [])
        
        # Build context for question generation
        context_summary = _build_context_summary(processed_data)
        
        # Check if this is a clarification request
        if field_name in clarification_needed and field_name in raw_responses:
            return generate_clarification_question_for_field(
                field_name, 
                raw_responses[field_name], 
                state
            )
        
        # Generate initial question for field
        # Get recent conversation for better context
        recent_messages = state.get("messages", [])[-4:]  # Last 4 messages for context
        conversation_context = ""
        if recent_messages:
            conversation_context = "\\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in recent_messages
            ])
        
        # Check if we should acknowledge previous information
        acknowledgment = _generate_acknowledgment(processed_data, field_name)
        
        # Check if this is a follow-up or first question for this field
        is_clarification = field_name in state.get("clarification_needed", [])
        raw_response = state.get("raw_responses", {}).get(field_name, "")
        
        if is_clarification and raw_response:
            # Generate clarification question
            question_prompt = f"""The candidate said "{raw_response}" when asked about their {field_name.replace('_', ' ')}.

This needs clarification. Generate a friendly, specific follow-up question that:
1. Acknowledges what they said
2. Asks for the specific detail you need
3. Keeps it conversational and under 20 words

Examples:
- "You mentioned 'some experience' - how many years would you say?"
- "You said 'tech stuff' - what specific technologies do you work with?"
- "You mentioned 'local area' - which city or region?"

Generate clarification question:"""
        else:
            # Generate initial question with context
            question_prompt = f"""You are a friendly recruiter having a natural conversation.

WHAT YOU KNOW ABOUT THEM:
{context_summary}

{acknowledgment}

TASK: Ask about their {field_name.replace('_', ' ')} naturally.

REQUIREMENTS:
1. Sound like a real person, not a robot
2. Build on what you already know
3. Keep it under 20 words
4. Be specific about what you want to know

QUESTION STYLES BY FIELD:
- job_title: "What's your current role?" or "What do you do for work?"
- experience: "How many years of experience do you have?"
- skills: "What technologies do you work with day-to-day?"
- responsibilities: "What does a typical day look like for you?"
- location: "Where are you based?" or "Are you looking for remote work?"
- department: "Which team or department do you work in?"
- education: "What's your educational background?"

Generate a natural question about their {field_name.replace('_', ' ')}:"""

        if hasattr(st, 'session_state') and 'groq_client' in st.session_state:
            logger.info(f"🔄 Calling Groq API for {field_name} question generation")
            completion = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": question_prompt}],
                model=settings.llm_model,
                max_tokens=60,
                temperature=0.4
            )
            
            generated_question = completion.choices[0].message.content.strip()
            logger.info(f"✅ LLM Generated question for {field_name}: '{generated_question}'")
            return generated_question
        else:
            # Very minimal fallback
            logger.warning(f"⚠️ Using hardcoded fallback question for {field_name} - LLM client not available")
            return _get_minimal_fallback_question(field_name)
            
    except Exception as e:
        logger.error(f"Error generating question for {field_name}: {e}")
        return _get_minimal_fallback_question(field_name)


def generate_clarification_question_for_field(
    field_name: str,
    original_response: str, 
    state: GraphState
) -> str:
    """Generate clarification question when user response was unclear."""
    try:
        clarification_prompt = f"""The candidate said "{original_response}" when I asked about their {field_name.replace('_', ' ')}.

Generate a natural clarification question that:
1. References what they said
2. Asks for specific details about their {field_name.replace('_', ' ')}
3. Keep it conversational and under 20 words

Examples:
- "You mentioned 'some remote work' - is your current role fully remote, hybrid, or office-based?"
- "You said 'a few years' - could you share how many years of experience you have?"
- "You mentioned 'tech work' - what specific technologies do you work with?"

Generate clarification for their {field_name.replace('_', ' ')}:"""

        if hasattr(st, 'session_state') and 'groq_client' in st.session_state:
            completion = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": clarification_prompt}],
                model=settings.llm_model,
                max_tokens=50,
                temperature=0.3
            )
            
            return completion.choices[0].message.content.strip()
        else:
            return f"You mentioned '{original_response}' - could you clarify the {field_name.replace('_', ' ')}?"
            
    except Exception as e:
        logger.error(f"Error generating clarification: {e}")
        return f"Could you clarify your response about {field_name.replace('_', ' ')}?"


def _build_context_summary(processed_data: dict) -> str:
    """Build summary of collected data for context."""
    if not processed_data:
        return "Nothing collected yet"
    
    # Prioritize showing key context fields
    key_fields = ["job_title", "department", "employment_type"]
    context_items = []
    
    for field in key_fields:
        if field in processed_data:
            value = processed_data[field]
            display_name = field.replace('_', ' ').title()
            context_items.append(f"{display_name}: {value}")
    
    # Add count of other fields
    other_fields = len([f for f in processed_data.keys() if f not in key_fields])
    if other_fields > 0:
        context_items.append(f"+ {other_fields} other fields")
    
    return "; ".join(context_items) if context_items else "Basic info"


def _generate_acknowledgment(processed_data: dict, field_name: str) -> str:
    """Generate natural acknowledgment of recently collected information."""
    # Get the most recent field that was collected
    important_fields = ["name", "job_title", "experience", "skills"]
    recent_field = None
    recent_value = None
    
    for field in important_fields:
        if field in processed_data and field != field_name:
            recent_field = field
            recent_value = processed_data[field]
            break
    
    if recent_field and recent_value:
        if recent_field == "name":
            return f"CONTEXT: Their name is {recent_value}. Be personable."
        elif recent_field == "job_title":
            return f"CONTEXT: They're a {recent_value}. Reference this naturally."
        elif recent_field == "experience":
            return f"CONTEXT: They have {recent_value} experience. Build on this."
        elif recent_field == "skills":
            return f"CONTEXT: They mentioned {recent_value}. Acknowledge briefly."
    
    return ""


def _get_minimal_fallback_question(field_name: str) -> str:
    """Ultra-minimal fallback for when LLM fails."""
    field_display = field_name.replace('_', ' ')
    return f"What about the {field_display}?"


def should_provide_examples(field_name: str, retry_count: int) -> bool:
    """Determine if examples should be provided with question."""
    # Provide examples on retry or for complex fields
    complex_fields = {"responsibilities", "skills", "experience"}
    return retry_count > 0 or field_name in complex_fields


def get_field_context_hints(field_name: str, state: GraphState) -> str:
    """Get contextual hints for field based on already collected data."""
    processed_data = state.get("processed_data", {})
    job_title = processed_data.get("job_title", "").lower()
    
    # Provide context-aware hints
    if field_name == "skills" and "engineer" in job_title:
        return " (programming languages, frameworks, tools)"
    elif field_name == "skills" and "marketing" in job_title:
        return " (marketing tools, analytics, creative skills)"
    elif field_name == "experience" and "senior" in job_title:
        return " (including leadership experience)"
    elif field_name == "location" and "remote" in str(processed_data.get("employment_type", "")):
        return " (fully remote, hybrid, or specific location)"
    
    return ""


def adapt_question_to_job_type(field_name: str, state: GraphState) -> Optional[str]:
    """Adapt question based on job type context."""
    processed_data = state.get("processed_data", {})
    job_title = processed_data.get("job_title", "").lower()
    
    # Job-specific adaptations could be added here
    # For now, return None to use default generation
    return None


def question_generator_node(state: GraphState) -> Dict[str, Any]:
    """Generate question as a proper graph node.
    
    This is the missing node function that wraps question generation
    for use in the LangGraph flow.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with generated question message
    """
    current_field = state.get("current_field")
    retry_count = state.get("retry_count", 0)
    
    if not current_field:
        return state
    
    # Generate question for current field
    question_text = generate_dynamic_question(current_field, state, retry_count)
    
    # Create question message
    question_message = {
        "role": "assistant",
        "content": question_text,
        "message_type": "question",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add message to state
    new_messages = state.get("messages", []) + [question_message]
    
    return {
        **state,
        "messages": new_messages
    }


def is_ready_for_jd_generation(state: GraphState) -> bool:
    """Check if we have sufficient data for JD generation."""
    required_fields = {"job_title", "responsibilities", "skills"}
    collected_fields = set(state.get("processed_data", {}).keys())
    
    return required_fields.issubset(collected_fields)