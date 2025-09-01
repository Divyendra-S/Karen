"""Enhanced voice interface for seamless voice-to-voice conversation.

This module provides an improved voice conversation experience with continuous
interaction, voice feedback, and better user guidance.
"""

import time
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
import asyncio

from core.services.simple_voice_service import SimpleFunctionalVoiceService
from core.services.tts_service import FunctionalTTSService


@dataclass(frozen=True)
class VoiceConversationState:
    """State for voice conversation session."""
    is_listening: bool = False
    is_speaking: bool = False
    last_transcript: str = ""
    last_response: str = ""
    turn_count: int = 0
    total_processing_time: float = 0.0
    
    def with_listening(self, listening: bool) -> 'VoiceConversationState':
        return dataclass.replace(self, is_listening=listening)
    
    def with_speaking(self, speaking: bool) -> 'VoiceConversationState':
        return dataclass.replace(self, is_speaking=speaking)
    
    def with_turn_completed(self, transcript: str, response: str, processing_time: float) -> 'VoiceConversationState':
        return dataclass.replace(
            self,
            last_transcript=transcript,
            last_response=response,
            turn_count=self.turn_count + 1,
            total_processing_time=self.total_processing_time + processing_time,
            is_listening=False,
            is_speaking=False
        )


def create_enhanced_voice_processor() -> Callable:
    """Create enhanced voice processor with TTS integration."""
    
    async def process_voice_with_speech(
        transcript: str,
        voice_service: SimpleFunctionalVoiceService,
        tts_service: FunctionalTTSService,
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process voice input and generate spoken response."""
        
        start_time = time.time()
        
        try:
            # Generate AI response
            result = await voice_service.process_voice_input(transcript, conversation_context)
            
            if not result.success:
                return {
                    "success": False,
                    "error": result.error_message,
                    "transcript": transcript,
                    "response": "",
                    "speech_played": False
                }
            
            response_text = result.response_text
            
            # Generate and play speech
            speech_success = False
            if tts_service and response_text:
                try:
                    speech_success = await tts_service.speak_text(response_text)
                except Exception as e:
                    logger.error(f"TTS error: {e}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "transcript": transcript,
                "response": response_text,
                "speech_played": speech_success,
                "processing_time": processing_time,
                "ai_processing_time": result.processing_time_ms,
                "tts_time": processing_time - result.processing_time_ms
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcript": transcript,
                "response": "",
                "speech_played": False
            }
    
    return process_voice_with_speech


def create_voice_conversation_guide() -> Callable:
    """Create conversation guide for voice interactions."""
    
    def get_conversation_guidance(
        current_field: str,
        job_data: Dict[str, Any],
        voice_mode: bool = True
    ) -> Dict[str, str]:
        """Pure function to provide conversation guidance."""
        
        field_guidance = {
            "job_title": {
                "prompt": "What job position are you looking to fill?",
                "example": "Say: 'Senior Software Engineer' or 'Marketing Manager'"
            },
            "department": {
                "prompt": "Which department will this role be in?", 
                "example": "Say: 'Engineering' or 'Marketing team'"
            },
            "experience": {
                "prompt": "How much experience should candidates have?",
                "example": "Say: 'Three to five years' or 'Entry level'"
            },
            "employment_type": {
                "prompt": "What type of employment is this?",
                "example": "Say: 'Full-time' or 'Part-time contract'"
            },
            "location": {
                "prompt": "Where will they work?",
                "example": "Say: 'Remote' or 'San Francisco office'"
            },
            "responsibilities": {
                "prompt": "What will be their main responsibilities?",
                "example": "Say: 'Build web apps and lead projects'"
            },
            "skills": {
                "prompt": "What skills should they have?",
                "example": "Say: 'Python, React, and team leadership'"
            },
            "education": {
                "prompt": "What education is required?", 
                "example": "Say: 'Bachelor's degree in Computer Science'"
            },
            "salary": {
                "prompt": "What's the salary range?",
                "example": "Say: 'Eighty to one twenty thousand annually'"
            },
            "additional_requirements": {
                "prompt": "Any other requirements?",
                "example": "Say: 'None' or 'Must have security clearance'"
            }
        }
        
        return field_guidance.get(current_field, {
            "prompt": "Tell me more about this role",
            "example": "Speak naturally about the job requirements"
        })
    
    return get_conversation_guidance


def create_voice_session_analyzer() -> Callable:
    """Create voice session analyzer."""
    
    def analyze_voice_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voice conversation session performance."""
        
        messages = session_data.get("messages", [])
        voice_messages = [m for m in messages if m.get("source") == "voice"]
        
        if not voice_messages:
            return {
                "voice_turns": 0,
                "avg_processing_time": 0.0,
                "conversation_efficiency": 0.0,
                "speech_success_rate": 0.0
            }
        
        # Calculate metrics
        total_processing_time = sum(
            m.get("processing_time", 0) for m in voice_messages 
            if m.get("role") == "assistant"
        )
        
        ai_voice_messages = [m for m in voice_messages if m.get("role") == "assistant"]
        avg_processing = total_processing_time / max(len(ai_voice_messages), 1)
        
        # Efficiency: fields collected per voice turn
        job_data = session_data.get("job_data", {})
        completed_fields = len([k for k, v in job_data.items() if v])
        efficiency = completed_fields / max(len(voice_messages) / 2, 1)  # User + AI = 1 turn
        
        return {
            "voice_turns": len(voice_messages) // 2,  # User + AI pairs
            "avg_processing_time": avg_processing,
            "conversation_efficiency": min(1.0, efficiency),
            "fields_completed": completed_fields,
            "total_voice_messages": len(voice_messages)
        }
    
    return analyze_voice_session


def create_voice_conversation_tips() -> List[str]:
    """Create helpful tips for voice conversation."""
    return [
        "🎤 **Speak clearly** and at normal pace",
        "⏱️ **Wait for AI response** before speaking again", 
        "🔊 **Listen to AI questions** - they guide the conversation",
        "💬 **Be specific** - 'Senior Software Engineer' vs 'Engineer'",
        "🔄 **Say 'go back'** to return to previous question",
        "✋ **Say 'text mode'** to switch back to typing",
        "📝 **Say 'skip this'** to move to next question",
        "🎯 **Speak naturally** - no need for formal language"
    ]


def create_field_progress_indicator() -> Callable:
    """Create function to show voice conversation progress."""
    
    def show_voice_progress(
        current_field: str,
        job_data: Dict[str, Any],
        conversation_state: Dict[str, Any]
    ) -> Dict[str, str]:
        """Show progress in voice conversation."""
        
        field_order = [
            'job_title', 'department', 'experience', 'employment_type',
            'location', 'responsibilities', 'skills', 'education', 
            'salary', 'additional_requirements'
        ]
        
        try:
            current_index = field_order.index(current_field) if current_field else 0
        except ValueError:
            current_index = 0
        
        completed_count = len([f for f in field_order if f in job_data])
        
        # Create progress message
        progress_msg = f"Field {current_index + 1} of {len(field_order)}: {current_field.replace('_', ' ').title()}"
        completion_msg = f"{completed_count}/{len(field_order)} fields completed"
        
        # Next field preview
        next_field = None
        if current_index + 1 < len(field_order):
            next_field = field_order[current_index + 1].replace('_', ' ').title()
        
        return {
            "current_progress": progress_msg,
            "completion_status": completion_msg, 
            "next_field": f"Next: {next_field}" if next_field else "Almost done!",
            "percent_complete": (completed_count / len(field_order)) * 100
        }
    
    return show_voice_progress


# Enhanced voice interface for Streamlit
def create_enhanced_streamlit_voice_interface() -> Dict[str, Callable]:
    """Create enhanced voice interface for better UX."""
    
    voice_processor = create_enhanced_voice_processor()
    conversation_guide = create_voice_conversation_guide()
    session_analyzer = create_voice_session_analyzer()
    progress_indicator = create_field_progress_indicator()
    
    async def process_voice_turn(
        transcript: str,
        voice_service: SimpleFunctionalVoiceService,
        tts_service: FunctionalTTSService,
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process complete voice conversation turn."""
        
        # Process voice input with speech output
        result = await voice_processor(
            transcript,
            voice_service,
            tts_service,
            conversation_context
        )
        
        # Add conversation guidance
        if result["success"]:
            current_field = conversation_context.get("current_field")
            job_data = conversation_context.get("job_data", {})
            
            guidance = conversation_guide(current_field, job_data)
            progress = progress_indicator(current_field, job_data, conversation_context)
            
            result.update({
                "guidance": guidance,
                "progress": progress
            })
        
        return result
    
    def handle_streamlit_voice_input(
        transcript: str,
        session_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle voice input in Streamlit with enhanced processing."""
        
        voice_service = session_state.get("voice_service")
        tts_service = session_state.get("tts_service")
        
        if not voice_service:
            return {"success": False, "error": "Voice service not available"}
        
        conversation_context = {
            "current_field": session_state["conversation_state"].get("current_field"),
            "job_data": session_state["conversation_state"].get("job_data", {}),
            "messages": session_state["conversation_state"].get("messages", [])
        }
        
        try:
            # Process voice turn with speech
            result = asyncio.run(process_voice_turn(
                transcript,
                voice_service,
                tts_service,
                conversation_context
            ))
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcript": transcript,
                "response": ""
            }
    
    return {
        "process_voice_turn": handle_streamlit_voice_input,
        "get_conversation_tips": create_voice_conversation_tips,
        "analyze_session": session_analyzer,
        "show_progress": progress_indicator
    }


if __name__ == "__main__":
    # Test the enhanced interface
    print("🎤 Testing Enhanced Voice Interface")
    print("-" * 35)
    
    interface = create_enhanced_streamlit_voice_interface()
    tips = interface["get_conversation_tips"]()
    
    print("✅ Enhanced interface created")
    print(f"✅ Conversation tips: {len(tips)} available")
    
    for tip in tips[:3]:
        print(f"   {tip}")
    
    print("\n🎉 Enhanced voice interface ready!")