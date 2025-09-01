"""Pure functional voice pipeline implementation.

This module implements a functional approach to voice processing using Pipecat and Gemini,
following immutable data patterns and pure function composition.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable
from functools import partial, reduce
from dataclasses import replace

from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    LLMMessagesFrame,
    TTSAudioRawFrame,
)
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer

from core.types.voice_types import (
    AudioFrame,
    TranscriptFrame,
    ResponseFrame,
    VoicePipelineResult,
    VoiceConfig,
    ConversationState,
    ConversationMode,
    Pipeline,
    StateReducer,
)


def compose(*functions: Callable) -> Callable:
    """Compose multiple functions into a single function."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def pipe(value: Any, *functions: Callable) -> Any:
    """Apply functions to value in sequence (left-to-right composition)."""
    return reduce(lambda acc, func: func(acc), functions, value)


def create_transcript_processor(confidence_threshold: float = 0.7) -> Callable[[str], TranscriptFrame]:
    """Create a transcript processor with given confidence threshold."""
    def process_transcript(text: str) -> TranscriptFrame:
        """Pure function to process transcript text."""
        confidence = min(1.0, len(text.strip()) / 50.0 + 0.5)  # Simple confidence metric
        return TranscriptFrame(
            text=text.strip(),
            confidence=confidence,
            timestamp=time.time(),
            is_final=confidence >= confidence_threshold
        )
    return process_transcript


def create_audio_frame_converter() -> Callable[[bytes], AudioFrame]:
    """Create converter from raw audio bytes to AudioFrame."""
    def convert_audio(audio_bytes: bytes) -> AudioFrame:
        """Pure function to convert audio bytes to AudioFrame."""
        return AudioFrame(
            data=audio_bytes,
            timestamp=time.time(),
            sample_rate=16000,
            format="wav"
        )
    return convert_audio


def create_response_validator() -> Callable[[str], bool]:
    """Create response validator function."""
    def validate_response(text: str) -> bool:
        """Pure function to validate AI response."""
        return (
            len(text.strip()) > 0 and
            len(text) < 1000 and  # Reasonable length for voice
            not any(char in text for char in ['<', '>', '{', '}'])  # No markup
        )
    return validate_response


async def transcribe_audio_async(
    audio_frame: AudioFrame,
    whisper_service: Any  # Groq client
) -> TranscriptFrame:
    """Async function to transcribe audio using Whisper (functional approach)."""
    try:
        # Create temporary file for Whisper API
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_frame.data)
            temp_file_path = temp_file.name
        
        try:
            # Call Whisper API
            with open(temp_file_path, "rb") as audio_file:
                transcription = whisper_service.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language="en"
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Create transcript processor and apply
            processor = create_transcript_processor()
            return processor(transcription.text)
            
        finally:
            # Ensure cleanup
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return TranscriptFrame(
            text="",
            confidence=0.0,
            timestamp=time.time(),
            is_final=False
        )


def create_gemini_response_generator(
    gemini_service: GeminiMultimodalLiveLLMService
) -> Callable[[TranscriptFrame, Dict[str, Any]], ResponseFrame]:
    """Create Gemini response generator function."""
    
    async def generate_response(
        transcript: TranscriptFrame,
        conversation_context: Dict[str, Any]
    ) -> ResponseFrame:
        """Generate response using Gemini (functional approach)."""
        try:
            if not transcript.text or transcript.confidence < 0.5:
                return ResponseFrame(
                    text="I didn't catch that. Could you repeat?",
                    audio_data=None,
                    timestamp=time.time(),
                    metadata={"error": "low_confidence_transcript"}
                )
            
            # Prepare context for Gemini
            messages = conversation_context.get("messages", [])
            current_field = conversation_context.get("current_field", "job_title")
            job_data = conversation_context.get("job_data", {})
            
            # Create system prompt
            system_prompt = create_hr_system_prompt(current_field, job_data)
            
            # Prepare messages for Gemini
            gemini_messages = [
                {"role": "user", "content": system_prompt},
                *messages[-4:],  # Last 4 messages for context
                {"role": "user", "content": transcript.text}
            ]
            
            # Generate response (this would integrate with actual Gemini service)
            # For now, using a placeholder that matches the expected interface
            response_text = f"Thank you for that information about {current_field}. Let me ask about the next detail..."
            
            return ResponseFrame(
                text=response_text,
                audio_data=None,  # TTS would be added here
                timestamp=time.time(),
                metadata={
                    "field": current_field,
                    "confidence": transcript.confidence,
                    "processing_time": time.time() - transcript.timestamp
                }
            )
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return ResponseFrame(
                text="I'm having trouble processing that. Could you try again?",
                audio_data=None,
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
    
    return generate_response


def create_hr_system_prompt(current_field: str, job_data: Dict[str, Any]) -> str:
    """Pure function to create system prompt for HR conversation."""
    collected_fields = list(job_data.keys())
    
    base_prompt = f"""You're an HR professional helping create a job description. 
Keep responses brief (1-2 sentences) and conversational for voice interaction.

Current field: {current_field}
Already collected: {collected_fields}

VOICE CONVERSATION STYLE:
- Natural spoken language, not formal text
- Ask one focused question at a time
- Use acknowledgments: "Great!", "Perfect!", "Got it!"
- Keep responses under 50 words for voice clarity
- Avoid special characters or formatting

Examples:
❌ "Please provide the employment type for this position"
✅ "Got it! Is this a full-time role?"

Focus on gathering complete job details through natural conversation."""

    return base_prompt


def create_state_reducer() -> StateReducer:
    """Create pure state reducer function."""
    
    def reduce_state(state: ConversationState, event: Dict[str, Any]) -> ConversationState:
        """Pure function to reduce conversation state based on events."""
        event_type = event.get("type", "unknown")
        
        if event_type == "transcript_received":
            return pipe(
                state,
                lambda s: s.with_transcript(event.get("transcript", "")),
                lambda s: s.with_speaking(False)
            )
        
        elif event_type == "response_generated":
            return state.with_speaking(True)
        
        elif event_type == "mode_changed":
            new_mode = ConversationMode(event.get("mode", ConversationMode.TEXT))
            return state.with_mode(new_mode)
        
        elif event_type == "audio_received":
            return replace(state, audio_buffer=event.get("audio_data", b""))
        
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return state
    
    return reduce_state


def create_voice_pipeline(
    voice_config: VoiceConfig,
    whisper_service: Any,
    conversation_context: Dict[str, Any]
) -> Pipeline:
    """Create complete voice pipeline using functional composition."""
    
    # Create individual processors
    audio_converter = create_audio_frame_converter()
    transcript_processor = create_transcript_processor()
    
    async def pipeline(audio_bytes: bytes, context: Dict[str, Any]) -> VoicePipelineResult:
        """Complete voice processing pipeline."""
        start_time = time.time()
        
        try:
            # Convert audio bytes to AudioFrame
            audio_frame = audio_converter(audio_bytes)
            
            # Transcribe audio
            transcript = await transcribe_audio_async(audio_frame, whisper_service)
            
            if not transcript.is_final:
                return VoicePipelineResult.error_result(
                    "Transcription confidence too low",
                    (time.time() - start_time) * 1000
                )
            
            # Generate response (this would use actual Gemini service)
            response_text = await generate_contextual_response(
                transcript.text, 
                context
            )
            
            # Validate response
            validator = create_response_validator()
            if not validator(response_text):
                return VoicePipelineResult.error_result(
                    "Generated response validation failed",
                    (time.time() - start_time) * 1000
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            return VoicePipelineResult.success_result(
                transcript=transcript.text,
                response_text=response_text,
                response_audio=b"",  # TTS would generate this
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return VoicePipelineResult.error_result(
                f"Pipeline processing failed: {str(e)}",
                (time.time() - start_time) * 1000
            )
    
    return pipeline


async def generate_contextual_response(
    transcript_text: str,
    conversation_context: Dict[str, Any]
) -> str:
    """Generate contextual response based on transcript and conversation state."""
    current_field = conversation_context.get("current_field", "job_title")
    job_data = conversation_context.get("job_data", {})
    
    # This is a simplified response generator
    # In the actual implementation, this would call Gemini
    field_prompts = {
        "job_title": "What's the job title you're looking for?",
        "department": "Which department will this role be in?",
        "experience": "How many years of experience should candidates have?",
        "employment_type": "Is this a full-time position?",
        "location": "Where will this person be working?",
        "responsibilities": "What will be their main responsibilities?",
        "skills": "What key skills should they have?",
        "education": "What education level do you prefer?",
        "salary": "What's the salary range?",
        "additional_requirements": "Any other requirements?"
    }
    
    acknowledgment = "Great!" if transcript_text else "I didn't catch that."
    next_question = field_prompts.get(current_field, "Tell me more about this role.")
    
    return f"{acknowledgment} {next_question}"


def create_conversation_state_initial() -> ConversationState:
    """Create initial conversation state (pure function)."""
    return ConversationState(
        mode=ConversationMode.VOICE,
        is_speaking=False,
        current_transcript="",
        audio_buffer=b"",
        last_response=None,
        metadata={}
    )


def should_interrupt_conversation(
    state: ConversationState,
    new_audio_confidence: float
) -> bool:
    """Pure function to determine if conversation should be interrupted."""
    return (
        state.is_speaking and
        new_audio_confidence > 0.8 and
        len(state.current_transcript) > 10
    )


def extract_voice_commands(transcript: str) -> Dict[str, Any]:
    """Pure function to extract voice commands from transcript."""
    lower_text = transcript.lower().strip()
    
    commands = {}
    
    # Mode switching commands
    if any(phrase in lower_text for phrase in ["switch to text", "text mode", "stop voice"]):
        commands["switch_mode"] = ConversationMode.TEXT
    elif any(phrase in lower_text for phrase in ["switch to voice", "voice mode", "start voice"]):
        commands["switch_mode"] = ConversationMode.VOICE
    
    # Navigation commands
    if any(phrase in lower_text for phrase in ["go back", "previous", "last question"]):
        commands["navigation"] = "back"
    elif any(phrase in lower_text for phrase in ["skip this", "next", "skip question"]):
        commands["navigation"] = "skip"
    
    # Control commands
    if any(phrase in lower_text for phrase in ["start over", "reset", "clear all"]):
        commands["control"] = "reset"
    elif any(phrase in lower_text for phrase in ["finish", "done", "complete"]):
        commands["control"] = "complete"
    
    return commands


# Higher-order functions for pipeline customization
def with_retry(max_attempts: int) -> Callable:
    """Higher-order function to add retry capability to any async function."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    logger.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
            return None
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float) -> Callable:
    """Higher-order function to add timeout to any async function."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"Timeout in {func.__name__} after {timeout_seconds}s")
                raise
        return wrapper
    return decorator


def create_enhanced_pipeline(
    base_pipeline: Pipeline,
    max_retries: int = 3,
    timeout_seconds: float = 10.0
) -> Pipeline:
    """Enhance pipeline with retry and timeout capabilities."""
    enhanced = compose(
        with_timeout(timeout_seconds),
        with_retry(max_retries)
    )(base_pipeline)
    return enhanced


# Functional state transformation utilities
def update_conversation_context(
    context: Dict[str, Any],
    transcript: str,
    response: str
) -> Dict[str, Any]:
    """Pure function to update conversation context."""
    updated_messages = context.get("messages", []) + [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": response}
    ]
    
    return {
        **context,
        "messages": updated_messages,
        "last_interaction": time.time()
    }


def calculate_conversation_metrics(
    conversation_context: Dict[str, Any]
) -> Dict[str, float]:
    """Pure function to calculate conversation performance metrics."""
    messages = conversation_context.get("messages", [])
    job_data = conversation_context.get("job_data", {})
    
    total_messages = len(messages)
    completed_fields = len([k for k, v in job_data.items() if v is not None])
    
    # Simple metrics calculation
    efficiency = completed_fields / max(total_messages / 2, 1)  # Fields per conversation turn
    completeness = completed_fields / 9.0  # Assuming 9 total fields
    
    return {
        "efficiency": min(1.0, efficiency),
        "completeness": completeness,
        "total_turns": total_messages / 2,
        "fields_collected": completed_fields
    }


def filter_relevant_context(
    full_context: Dict[str, Any],
    max_messages: int = 6
) -> Dict[str, Any]:
    """Pure function to filter context for optimal performance."""
    messages = full_context.get("messages", [])
    
    # Keep only recent messages for context
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    return {
        **full_context,
        "messages": recent_messages,
        "context_size": len(recent_messages)
    }


# Factory function for creating configured pipelines
def create_configured_voice_pipeline(
    voice_config: VoiceConfig,
    services: Dict[str, Any]
) -> Pipeline:
    """Factory function to create a fully configured voice pipeline."""
    
    # Get services
    whisper_service = services.get("whisper")
    gemini_service = services.get("gemini")
    
    if not whisper_service or not gemini_service:
        raise ValueError("Required services (whisper, gemini) not provided")
    
    # Create base pipeline
    base_pipeline = create_voice_pipeline(
        voice_config,
        whisper_service,
        {}  # Initial context
    )
    
    # Enhance with reliability features
    enhanced_pipeline = create_enhanced_pipeline(
        base_pipeline,
        max_retries=3,
        timeout_seconds=voice_config.response_timeout_ms / 1000.0
    )
    
    return enhanced_pipeline


# Utility functions for functional data processing
def merge_conversation_states(*states: ConversationState) -> ConversationState:
    """Merge multiple conversation states (functional approach)."""
    if not states:
        return create_conversation_state_initial()
    
    # Start with first state and merge others
    result = states[0]
    for state in states[1:]:
        result = replace(
            result,
            current_transcript=state.current_transcript or result.current_transcript,
            last_response=state.last_response or result.last_response,
            metadata={**result.metadata, **state.metadata}
        )
    
    return result


def is_conversation_complete(job_data: Dict[str, Any]) -> bool:
    """Pure function to check if conversation is complete."""
    required_fields = [
        'job_title', 'department', 'experience', 'employment_type',
        'location', 'responsibilities', 'skills', 'education', 'salary'
    ]
    
    return all(
        field in job_data and job_data[field] is not None
        for field in required_fields
    )