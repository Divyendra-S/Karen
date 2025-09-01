"""Example of functional voice conversation flow.

This example demonstrates how to use the Pipecat + Gemini integration
in a functional programming style for voice-to-voice job description interviews.
"""

import asyncio
import os
from typing import Dict, Any
from dataclasses import replace

from loguru import logger

# Import our functional voice components
from core.types.voice_types import (
    VoiceConfig,
    VoiceModel, 
    ConversationState,
    ConversationMode,
)
from core.services.pipecat_voice_service import (
    FunctionalVoiceService,
    create_voice_service_factory,
)
from core.services.voice_events import (
    VoiceEvent,
    VoiceEventType,
    create_voice_state_reducer,
    create_conversation_state_initial,
)
from core.services.voice_integration import (
    create_complete_voice_integration,
    pipe_voice_to_graph,
)
from core.models.graph_state import create_initial_graph_state


async def demonstrate_functional_voice_conversation():
    """Demonstrate complete functional voice conversation flow."""
    
    print("🎤 Starting Functional Voice Conversation Demo")
    print("=" * 50)
    
    # 1. Create voice configuration functionally
    voice_config = VoiceConfig(
        google_api_key=os.getenv("GOOGLE_API_KEY", "demo_key"),
        voice_model=VoiceModel.PUCK,
        sample_rate=16000,
        enable_interruption=True,
        vad_threshold=0.5,
        response_timeout_ms=5000
    )
    
    print(f"✅ Voice config created: {voice_config.voice_model.value} voice")
    
    # 2. Initialize voice service
    voice_service = FunctionalVoiceService(voice_config)
    print(f"✅ Voice service initialized")
    
    # 3. Create initial conversation state
    initial_voice_state = create_conversation_state_initial()
    initial_graph_state = create_initial_graph_state()
    
    print(f"✅ Initial states created")
    print(f"   Voice mode: {initial_voice_state.mode.value}")
    print(f"   Graph phase: {initial_graph_state['conversation_phase']}")
    
    # 4. Simulate conversation flow with pure functions
    conversation_context = {
        "current_field": "job_title",
        "job_data": {},
        "messages": []
    }
    
    # Start voice conversation
    if await voice_service.start_voice_conversation(conversation_context):
        print("✅ Voice conversation started")
    else:
        print("❌ Failed to start voice conversation")
        return
    
    # 5. Simulate voice interactions using functional approach
    voice_interactions = [
        "I'm looking for a software engineer",
        "Engineering department", 
        "Three to five years experience",
        "Full time position",
        "Remote work is fine"
    ]
    
    # Create state reducer
    state_reducer = create_voice_state_reducer()
    current_voice_state = initial_voice_state
    current_graph_state = initial_graph_state
    
    # Process each interaction functionally
    for i, user_input in enumerate(voice_interactions):
        print(f"\n👤 User says: '{user_input}'")
        
        # Create voice event
        transcript_event = VoiceEvent(
            VoiceEventType.TRANSCRIPT_READY,
            {
                "text": user_input,
                "confidence": 0.9,
                "is_final": True
            }
        )
        
        # Update voice state functionally
        current_voice_state = state_reducer(current_voice_state, transcript_event)
        print(f"   Voice state updated: transcript='{current_voice_state.current_transcript}'")
        
        # Simulate AI response
        ai_response = generate_mock_hr_response(user_input, conversation_context["current_field"])
        
        response_event = VoiceEvent(
            VoiceEventType.RESPONSE_GENERATED,
            {
                "text": ai_response,
                "original_transcript": user_input
            }
        )
        
        # Update states
        current_voice_state = state_reducer(current_voice_state, response_event)
        
        # Integrate with graph state
        integration_funcs = create_complete_voice_integration()
        event_processor = integration_funcs["event_processor"]
        
        current_graph_state = event_processor(transcript_event, current_graph_state)
        current_graph_state = event_processor(response_event, current_graph_state)
        
        print(f"🤖 AI responds: '{ai_response}'")
        print(f"   Current field: {current_graph_state.get('current_field')}")
        print(f"   Job data fields: {list(current_graph_state.get('job_data', {}).keys())}")
        
        # Update conversation context for next iteration
        conversation_context = {
            "current_field": current_graph_state.get("current_field"),
            "job_data": current_graph_state.get("job_data", {}),
            "messages": current_graph_state.get("messages", [])
        }
    
    # 6. Demonstrate conversation completion
    print(f"\n🎯 Final Results:")
    print(f"   Completed fields: {len(current_graph_state.get('job_data', {}))}")
    print(f"   Conversation complete: {current_graph_state.get('is_complete', False)}")
    print(f"   Total messages: {len(current_graph_state.get('messages', []))}")
    
    # Stop voice service
    await voice_service.stop_voice_conversation()
    print(f"✅ Voice conversation stopped")
    
    print("\n🎉 Functional Voice Conversation Demo Complete!")


def generate_mock_hr_response(user_input: str, current_field: str) -> str:
    """Generate mock HR response for demo purposes."""
    
    field_responses = {
        "job_title": "Great! Which department will this software engineer be working in?",
        "department": "Perfect! How many years of experience should candidates have?",
        "experience": "Excellent! Is this a full-time position?",
        "employment_type": "Got it! Where will they be working - remote, hybrid, or on-site?",
        "location": "Wonderful! What will be their main responsibilities?",
        "responsibilities": "That's helpful! What key skills should they have?",
        "skills": "Great list! What education level do you prefer?",
        "education": "Perfect! What's the salary range for this role?",
        "salary": "Excellent! Any other special requirements?",
        "additional_requirements": "Perfect! I have all the information needed to create your job description!"
    }
    
    return field_responses.get(current_field, "Thank you! What else can you tell me?")


def demonstrate_functional_composition():
    """Demonstrate functional composition patterns used in voice pipeline."""
    
    print("\n🔧 Functional Composition Demo")
    print("=" * 30)
    
    # Example 1: Function composition
    from functools import reduce
    
    def compose(*functions):
        return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
    
    # Voice processing pipeline
    clean_text = lambda text: text.strip().lower()
    extract_keywords = lambda text: [word for word in text.split() if len(word) > 3]
    format_output = lambda keywords: f"Keywords: {', '.join(keywords)}"
    
    voice_processor = compose(format_output, extract_keywords, clean_text)
    result = voice_processor("  I am looking for a SOFTWARE ENGINEER position  ")
    print(f"✅ Composed processing result: {result}")
    
    # Example 2: State transformation with immutability
    initial_state = create_conversation_state_initial()
    print(f"✅ Initial state mode: {initial_state.mode.value}")
    
    # Pure state transformation
    updated_state = initial_state.with_transcript("Hello, I need to hire someone")
    print(f"✅ Updated state transcript: '{updated_state.current_transcript}'")
    print(f"✅ Original state unchanged: '{initial_state.current_transcript}'")
    
    # Example 3: Higher-order functions
    def create_field_validator(required_length: int):
        def validate_field(text: str) -> bool:
            return len(text.strip()) >= required_length
        return validate_field
    
    job_title_validator = create_field_validator(3)
    department_validator = create_field_validator(2)
    
    print(f"✅ Job title valid: {job_title_validator('Software Engineer')}")
    print(f"✅ Department valid: {department_validator('IT')}")
    
    print("\n🎯 Functional patterns demonstrated successfully!")


if __name__ == "__main__":
    # Run the functional composition demo
    demonstrate_functional_composition()
    
    # Run the voice conversation demo (async)
    try:
        asyncio.run(demonstrate_functional_voice_conversation())
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Voice demo error: {e}")