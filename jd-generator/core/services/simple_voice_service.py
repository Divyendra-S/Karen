"""Simplified functional voice service without Daily API dependency.

This implementation provides voice-to-voice functionality using Google Gemini
and the existing Groq Whisper integration, following functional programming principles.
"""

import asyncio
import os
import time
from typing import Dict, Any, Optional, Callable
from functools import partial
from dataclasses import replace

from core.types.voice_types import (
    VoiceConfig,
    VoiceModel,
    ConversationState,
    ConversationMode,
    VoicePipelineResult,
    TranscriptFrame,
    ResponseFrame,
)


class SimpleFunctionalVoiceService:
    """Simplified voice service using functional patterns without Daily API."""
    
    def __init__(self, voice_config: VoiceConfig):
        self.config = voice_config
        self._is_active = False
        self._gemini_client = None
    
    def initialize_gemini(self) -> bool:
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.google_api_key)
            self._gemini_client = genai.GenerativeModel('gemini-1.5-flash')
            return True
        except ImportError:
            print("Warning: google-generativeai not installed, using mock responses")
            return False
        except Exception as e:
            print(f"Gemini initialization error: {e}")
            return False
    
    async def start_conversation(self, context: Dict[str, Any]) -> bool:
        """Start voice conversation."""
        if self.initialize_gemini():
            self._is_active = True
            return True
        
        # Fallback mode without Gemini
        self._is_active = True
        return True
    
    async def process_voice_input(
        self,
        transcript: str,
        conversation_context: Dict[str, Any]
    ) -> VoicePipelineResult:
        """Process voice input functionally."""
        
        if not self._is_active:
            return VoicePipelineResult.error_result("Voice service not active")
        
        start_time = time.time()
        
        try:
            # Generate response using Gemini or fallback
            if self._gemini_client:
                response_text = await self._generate_gemini_response(transcript, conversation_context)
            else:
                response_text = self._generate_fallback_response(transcript, conversation_context)
            
            processing_time = (time.time() - start_time) * 1000
            
            return VoicePipelineResult.success_result(
                transcript=transcript,
                response_text=response_text,
                response_audio=b"",  # No TTS for simplified version
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return VoicePipelineResult.error_result(str(e), processing_time)
    
    async def _generate_gemini_response(
        self,
        transcript: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate response using Gemini."""
        
        current_field = context.get("current_field", "job_title")
        job_data = context.get("job_data", {})
        
        # Create HR conversation prompt
        prompt = self._create_hr_prompt(transcript, current_field, job_data)
        
        try:
            response = await asyncio.to_thread(
                self._gemini_client.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 100,
                }
            )
            
            return self._optimize_for_voice(response.text)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_fallback_response(transcript, context)
    
    def _generate_fallback_response(
        self,
        transcript: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate fallback response without Gemini."""
        
        current_field = context.get("current_field", "job_title")
        
        # Simple field-based responses
        field_responses = {
            "job_title": "Great! Which department will this role be in?",
            "department": "Perfect! How many years of experience should candidates have?",
            "experience": "Excellent! Is this a full-time position?",
            "employment_type": "Got it! Where will they be working?",
            "location": "Wonderful! What will be their main responsibilities?",
            "responsibilities": "That's helpful! What key skills should they have?",
            "skills": "Great! What education level do you prefer?",
            "education": "Perfect! What's the salary range?",
            "salary": "Excellent! Any other requirements?",
            "additional_requirements": "Perfect! I have everything needed for your job description!"
        }
        
        acknowledgment = "Thank you for that information."
        next_question = field_responses.get(current_field, "What else can you tell me?")
        
        return f"{acknowledgment} {next_question}"
    
    def _create_hr_prompt(
        self,
        user_input: str,
        current_field: str,
        job_data: Dict[str, Any]
    ) -> str:
        """Create HR conversation prompt for Gemini."""
        
        collected_fields = list(job_data.keys())
        
        return f"""You are an HR professional conducting a voice interview for a job description.

Current field: {current_field}
Already collected: {', '.join(collected_fields) if collected_fields else 'none'}

User just said: "{user_input}"

Respond briefly (15-25 words) like a natural HR conversation:
1. Acknowledge what they said
2. Ask about the next field needed

Keep it conversational and brief for voice interaction."""
    
    def _optimize_for_voice(self, text: str) -> str:
        """Optimize response for voice interaction."""
        # Remove markdown and special characters
        cleaned = text.replace('*', '').replace('#', '').replace('`', '')
        
        # Ensure reasonable length for voice
        if len(cleaned) > 150:
            sentences = cleaned.split('. ')
            cleaned = '. '.join(sentences[:2])
            if not cleaned.endswith('.'):
                cleaned += '.'
        
        return cleaned.strip()
    
    async def stop_conversation(self) -> bool:
        """Stop voice conversation."""
        self._is_active = False
        return True
    
    def is_active(self) -> bool:
        """Check if service is active."""
        return self._is_active


def create_simple_voice_service_factory() -> Callable:
    """Factory for simple voice service."""
    
    def create_service(env_vars: Dict[str, str]) -> Optional[SimpleFunctionalVoiceService]:
        """Create simple voice service from environment."""
        
        google_api_key = env_vars.get("GOOGLE_API_KEY")
        if not google_api_key:
            print("Warning: No Google API key found, using fallback responses")
            google_api_key = "demo_key"
        
        voice_model_str = env_vars.get("VOICE_MODEL", "Puck")
        try:
            voice_model = VoiceModel(voice_model_str)
        except ValueError:
            voice_model = VoiceModel.PUCK
        
        voice_config = VoiceConfig(
            google_api_key=google_api_key,
            voice_model=voice_model,
            sample_rate=int(env_vars.get("AUDIO_SAMPLE_RATE", "16000")),
            enable_interruption=True,
            vad_threshold=0.5,
            response_timeout_ms=5000
        )
        
        return SimpleFunctionalVoiceService(voice_config)
    
    return create_service


def create_simple_streamlit_interface() -> Dict[str, Callable]:
    """Create simplified Streamlit interface functions."""
    
    def process_audio_simple(
        transcript: str,
        voice_service: SimpleFunctionalVoiceService,
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process transcript through voice service."""
        
        if not voice_service or not voice_service.is_active():
            return {
                "success": False,
                "error": "Voice service not active",
                "transcript": transcript,
                "response": ""
            }
        
        try:
            # Run async processing
            result = asyncio.run(
                voice_service.process_voice_input(transcript, conversation_context)
            )
            
            if result.success:
                return {
                    "success": True,
                    "transcript": result.transcript,
                    "response": result.response_text,
                    "processing_time": result.processing_time_ms,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message,
                    "transcript": transcript,
                    "response": ""
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcript": transcript,
                "response": ""
            }
    
    def start_voice_simple(
        voice_service: SimpleFunctionalVoiceService,
        conversation_context: Dict[str, Any]
    ) -> bool:
        """Start voice mode."""
        if not voice_service:
            return False
        
        try:
            return asyncio.run(voice_service.start_conversation(conversation_context))
        except Exception as e:
            print(f"Failed to start voice mode: {e}")
            return False
    
    def stop_voice_simple(voice_service: SimpleFunctionalVoiceService) -> bool:
        """Stop voice mode."""
        if not voice_service:
            return True
        
        try:
            return asyncio.run(voice_service.stop_conversation())
        except Exception as e:
            print(f"Failed to stop voice mode: {e}")
            return False
    
    return {
        "process_audio": process_audio_simple,
        "start_voice": start_voice_simple,
        "stop_voice": stop_voice_simple
    }


# Test the simplified service
def test_simple_voice_service():
    """Test the simplified voice service."""
    print("🧪 Testing Simplified Voice Service")
    print("-" * 35)
    
    # Create service
    factory = create_simple_voice_service_factory()
    env_vars = {
        "GOOGLE_API_KEY": "your_google_api_key_here",
        "VOICE_MODEL": "Puck"
    }
    
    service = factory(env_vars)
    print("✅ Voice service created")
    
    # Test conversation
    context = {
        "current_field": "job_title",
        "job_data": {},
        "messages": []
    }
    
    # Start conversation
    started = asyncio.run(service.start_conversation(context))
    assert started == True
    print("✅ Conversation started")
    
    # Process voice input
    result = asyncio.run(service.process_voice_input("Senior Software Engineer", context))
    assert result.success == True
    assert "Senior Software Engineer" in result.transcript
    print(f"✅ Voice processing: '{result.response_text}'")
    
    # Test interface functions
    interface = create_simple_streamlit_interface()
    audio_result = interface["process_audio"](
        "Engineering department",
        service,
        {"current_field": "department", "job_data": {}, "messages": []}
    )
    
    assert audio_result["success"] == True
    print(f"✅ Interface processing: '{audio_result['response']}'")
    
    print("\n🎉 Simplified voice service test completed successfully!")
    return True


if __name__ == "__main__":
    test_simple_voice_service()