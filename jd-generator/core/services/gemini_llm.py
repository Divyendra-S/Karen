"""Functional Gemini LLM service implementation.

This module provides a pure functional interface to Google's Gemini multimodal model
for voice-to-voice conversations in the job description generator.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, NamedTuple
from functools import partial, lru_cache
from dataclasses import dataclass, replace

import google.generativeai as genai
from loguru import logger

from core.types.voice_types import (
    TranscriptFrame,
    ResponseFrame,
    VoiceConfig,
    VoiceModel,
)


class GeminiMessage(NamedTuple):
    """Immutable message structure for Gemini."""
    role: str
    content: str
    timestamp: float


@dataclass(frozen=True)
class GeminiConfig:
    """Immutable Gemini service configuration."""
    api_key: str
    model_name: str = "gemini-1.5-flash"
    voice_model: VoiceModel = VoiceModel.PUCK
    temperature: float = 0.7
    max_tokens: int = 150
    safety_settings: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize safety settings if not provided."""
        if self.safety_settings is None:
            object.__setattr__(self, 'safety_settings', {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
            })
    
    def with_temperature(self, temp: float) -> 'GeminiConfig':
        """Return new config with different temperature."""
        return replace(self, temperature=temp)
    
    def with_voice(self, voice: VoiceModel) -> 'GeminiConfig':
        """Return new config with different voice model."""
        return replace(self, voice_model=voice)


@lru_cache(maxsize=1)
def create_gemini_client(api_key: str) -> genai.GenerativeModel:
    """Create cached Gemini client (pure function)."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')


def create_hr_conversation_prompt() -> str:
    """Pure function to create the base HR conversation prompt."""
    return """You are an expert HR professional conducting a voice interview to gather job description details. 

VOICE CONVERSATION RULES:
- Keep responses very brief (20-40 words max)
- Use natural, conversational language
- Ask one focused question at a time
- Acknowledge what the user said before asking next question
- Avoid special characters, formatting, or bullets
- Sound like a real HR person, not a chatbot

CONVERSATION FLOW:
1. Job Title → 2. Department → 3. Experience → 4. Employment Type → 5. Location → 
6. Responsibilities → 7. Skills → 8. Education → 9. Salary → 10. Additional Requirements

RESPONSE EXAMPLES:
✅ "Got it, Software Engineer! Which department will this be in?"
✅ "Perfect! How many years of experience should candidates have?"
❌ "Thank you for providing the job title. Please specify the department where this position will be located within the organizational structure."

Be conversational, efficient, and helpful. Focus on gathering complete information naturally."""


def create_message_formatter() -> Callable[[List[Dict[str, Any]]], List[GeminiMessage]]:
    """Create function to format messages for Gemini."""
    
    def format_messages(messages: List[Dict[str, Any]]) -> List[GeminiMessage]:
        """Pure function to format conversation messages for Gemini."""
        formatted = []
        current_time = time.time()
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", current_time)
            
            # Convert timestamp string to float if needed
            if isinstance(timestamp, str):
                timestamp = current_time
            
            # Map roles for Gemini
            gemini_role = "user" if role in ["user", "human"] else "model"
            
            formatted.append(GeminiMessage(
                role=gemini_role,
                content=content,
                timestamp=timestamp
            ))
        
        return formatted
    
    return format_messages


def create_context_enricher(
    current_field: str,
    job_data: Dict[str, Any]
) -> Callable[[str], str]:
    """Create function to enrich user input with context."""
    
    def enrich_context(user_input: str) -> str:
        """Pure function to add context to user input."""
        collected_fields = list(job_data.keys())
        context_info = f"""
Current field being collected: {current_field}
Previously collected: {', '.join(collected_fields) if collected_fields else 'none'}

User said: "{user_input}"

Based on this context, provide a brief conversational response that acknowledges what they said and asks for the next piece of information needed."""
        
        return context_info
    
    return enrich_context


async def generate_gemini_response(
    config: GeminiConfig,
    transcript: TranscriptFrame,
    conversation_context: Dict[str, Any]
) -> ResponseFrame:
    """Generate response using Gemini with functional approach."""
    try:
        # Create Gemini client
        client = create_gemini_client(config.api_key)
        
        # Format conversation context
        message_formatter = create_message_formatter()
        messages = conversation_context.get("messages", [])
        formatted_messages = message_formatter(messages)
        
        # Create context enricher
        current_field = conversation_context.get("current_field", "job_title")
        job_data = conversation_context.get("job_data", {})
        context_enricher = create_context_enricher(current_field, job_data)
        
        # Prepare prompt
        system_prompt = create_hr_conversation_prompt()
        enriched_input = context_enricher(transcript.text)
        
        # Combine system prompt with enriched context
        full_prompt = f"{system_prompt}\n\n{enriched_input}"
        
        # Generate response
        response = await asyncio.to_thread(
            client.generate_content,
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
        )
        
        response_text = response.text.strip()
        
        # Validate response for voice interaction
        if not is_valid_voice_response(response_text):
            response_text = "Could you tell me more about that?"
        
        return ResponseFrame(
            text=response_text,
            audio_data=None,  # TTS would be added here
            timestamp=time.time(),
            metadata={
                "model": config.model_name,
                "field": current_field,
                "confidence": transcript.confidence,
                "voice_optimized": True
            }
        )
        
    except Exception as e:
        logger.error(f"Gemini response generation error: {e}")
        return ResponseFrame(
            text="I'm having trouble processing that. Could you repeat?",
            audio_data=None,
            timestamp=time.time(),
            metadata={"error": str(e)}
        )


def is_valid_voice_response(text: str) -> bool:
    """Pure function to validate response for voice interaction."""
    return (
        1 <= len(text.strip()) <= 200 and  # Appropriate length for voice
        not any(char in text for char in ['*', '#', '[', ']', '```']) and  # No markdown
        '.' in text or '?' in text or '!' in text  # Has proper punctuation
    )


def create_field_specific_prompter() -> Callable[[str, Dict[str, Any]], str]:
    """Create function to generate field-specific prompts."""
    
    field_contexts = {
        "job_title": "Ask for the specific job title they want to hire for",
        "department": "Ask which department or team this role belongs to", 
        "experience": "Ask about years of experience required",
        "employment_type": "Ask if it's full-time, part-time, contract, etc.",
        "location": "Ask about work location - remote, hybrid, or on-site",
        "responsibilities": "Ask for 3-4 key responsibilities for this role",
        "skills": "Ask for required technical and soft skills",
        "education": "Ask about education requirements",
        "salary": "Ask about salary range or compensation",
        "additional_requirements": "Ask if there are any other requirements"
    }
    
    def create_field_prompt(field: str, job_data: Dict[str, Any]) -> str:
        """Pure function to create field-specific prompt."""
        context = field_contexts.get(field, "Ask for more details about this role")
        collected = list(job_data.keys())
        
        return f"""
Context: {context}
Already collected: {', '.join(collected) if collected else 'none'}

Be conversational and brief. Acknowledge any previous information and ask your question naturally."""
    
    return create_field_prompt


def optimize_for_voice_interaction(text: str) -> str:
    """Pure function to optimize text for voice interaction."""
    # Remove markdown formatting
    cleaned = text.replace('*', '').replace('#', '').replace('`', '')
    
    # Replace written abbreviations with spoken equivalents
    replacements = {
        'e.g.': 'for example',
        'i.e.': 'that is',
        'etc.': 'and so on',
        '&': 'and',
        '@': 'at',
        '#': 'number',
        '%': 'percent'
    }
    
    for abbrev, spoken in replacements.items():
        cleaned = cleaned.replace(abbrev, spoken)
    
    # Ensure proper sentence endings
    if cleaned and not cleaned.rstrip().endswith(('.', '!', '?')):
        cleaned = cleaned.rstrip() + '.'
    
    return cleaned


def create_conversation_flow_controller() -> Callable[[Dict[str, Any]], str]:
    """Create function to control conversation flow."""
    
    def control_flow(context: Dict[str, Any]) -> str:
        """Pure function to determine next conversation step."""
        current_field = context.get("current_field")
        job_data = context.get("job_data", {})
        
        field_order = [
            'job_title', 'department', 'experience', 'employment_type',
            'location', 'responsibilities', 'skills', 'education', 
            'salary', 'additional_requirements'
        ]
        
        if not current_field:
            return field_order[0]
        
        try:
            current_index = field_order.index(current_field)
            # Check if current field is complete
            if current_field in job_data and job_data[current_field]:
                # Move to next field
                next_index = current_index + 1
                if next_index < len(field_order):
                    return field_order[next_index]
                else:
                    return "complete"
            else:
                # Stay on current field
                return current_field
        except ValueError:
            return field_order[0]
    
    return control_flow


# Factory for creating voice-optimized LLM service
def create_voice_llm_service(config: GeminiConfig) -> Callable:
    """Factory function to create voice-optimized LLM service."""
    
    flow_controller = create_conversation_flow_controller()
    field_prompter = create_field_specific_prompter()
    
    async def voice_llm_service(
        transcript: TranscriptFrame,
        context: Dict[str, Any]
    ) -> ResponseFrame:
        """Voice-optimized LLM service function."""
        
        # Determine conversation flow
        next_step = flow_controller(context)
        
        if next_step == "complete":
            return ResponseFrame(
                text="Perfect! I have all the information needed. Your job description is ready!",
                audio_data=None,
                timestamp=time.time(),
                metadata={"conversation_complete": True}
            )
        
        # Generate field-specific prompt
        field_prompt = field_prompter(next_step, context.get("job_data", {}))
        
        # Generate response with Gemini
        response = await generate_gemini_response(config, transcript, context)
        
        # Optimize response for voice
        optimized_text = optimize_for_voice_interaction(response.text)
        
        return replace(
            response,
            text=optimized_text,
            metadata={**response.metadata, "next_field": next_step}
        )
    
    return voice_llm_service