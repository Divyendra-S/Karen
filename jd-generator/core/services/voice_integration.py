"""Functional integration between voice pipeline and LangGraph state.

This module provides pure functions to bridge the voice processing pipeline
with the existing LangGraph conversation state management.
"""

import time
from typing import Dict, Any, Optional, Callable, List
from functools import partial, reduce
from dataclasses import replace

from loguru import logger

from core.types.voice_types import (
    ConversationState as VoiceConversationState,
    ConversationMode,
    VoiceEvent,
    VoiceEventType,
    TranscriptFrame,
    ResponseFrame,
)
from core.models.graph_state import GraphState, create_initial_graph_state
from core.services.voice_events import (
    create_voice_state_reducer,
    extract_conversation_commands,
    apply_conversation_commands,
)


def create_voice_state_bridge() -> Callable[[GraphState], VoiceConversationState]:
    """Create function to convert GraphState to VoiceConversationState."""
    
    def bridge_to_voice_state(graph_state: GraphState) -> VoiceConversationState:
        """Pure function to convert GraphState to voice conversation state."""
        # Extract relevant data from graph state
        messages = graph_state.get("messages", [])
        current_field = graph_state.get("current_field")
        job_data = graph_state.get("job_data", {})
        is_complete = graph_state.get("is_complete", False)
        
        # Determine current transcript from last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        
        # Create voice conversation state
        return VoiceConversationState(
            mode=ConversationMode.VOICE,
            is_speaking=False,
            current_transcript=last_user_message or "",
            audio_buffer=b"",
            last_response=None,
            metadata={
                "current_field": current_field,
                "job_data": job_data,
                "is_complete": is_complete,
                "graph_state_sync": True
            }
        )
    
    return bridge_to_voice_state


def create_graph_state_bridge() -> Callable[[VoiceConversationState, GraphState], GraphState]:
    """Create function to update GraphState from VoiceConversationState."""
    
    def bridge_to_graph_state(
        voice_state: VoiceConversationState,
        current_graph_state: GraphState
    ) -> GraphState:
        """Pure function to update GraphState with voice conversation data."""
        
        # Extract voice metadata
        voice_metadata = voice_state.metadata
        current_field = voice_metadata.get("current_field")
        
        # Prepare updates
        updates = {}
        
        # Update current field if changed
        if current_field and current_field != current_graph_state.get("current_field"):
            updates["current_field"] = current_field
        
        # Update session metadata with voice info
        session_meta = current_graph_state.get("session_metadata", {})
        updates["session_metadata"] = {
            **session_meta,
            "voice_mode_active": voice_state.mode == ConversationMode.VOICE,
            "last_voice_interaction": time.time(),
            "voice_transcript_confidence": voice_metadata.get("last_transcript_confidence", 0.0)
        }
        
        # Apply updates
        return {**current_graph_state, **updates}
    
    return bridge_to_graph_state


def create_voice_message_processor() -> Callable[[str, str, GraphState], GraphState]:
    """Create function to process voice messages and update graph state."""
    
    def process_voice_message(
        transcript: str,
        ai_response: str,
        graph_state: GraphState
    ) -> GraphState:
        """Pure function to process voice interaction and update graph state."""
        
        # Create user message
        user_message = {
            "role": "user",
            "content": transcript,
            "message_type": "voice_input",
            "timestamp": time.time(),
            "source": "voice"
        }
        
        # Create assistant message
        assistant_message = {
            "role": "assistant", 
            "content": ai_response,
            "message_type": "voice_response",
            "timestamp": time.time(),
            "source": "voice"
        }
        
        # Update messages
        new_messages = graph_state["messages"] + [user_message, assistant_message]
        
        return {**graph_state, "messages": new_messages}
    
    return process_voice_message


def create_field_extractor_bridge() -> Callable[[str, str, Dict[str, Any]], Dict[str, Any]]:
    """Create function to extract job data from voice input."""
    
    def extract_field_from_voice(
        transcript: str,
        current_field: str,
        current_job_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pure function to extract job field data from voice transcript."""
        
        if not transcript or not current_field:
            return current_job_data
        
        # Use field-specific extraction logic
        extracted_value = extract_field_value_functional(transcript, current_field)
        
        if extracted_value:
            return {
                **current_job_data,
                current_field: {
                    "value": extracted_value,
                    "source": "voice",
                    "confidence": calculate_extraction_confidence(transcript, current_field),
                    "timestamp": time.time()
                }
            }
        
        return current_job_data
    
    return extract_field_from_voice


def extract_field_value_functional(transcript: str, field_name: str) -> Optional[str]:
    """Pure function to extract field value from transcript."""
    text_lower = transcript.lower().strip()
    
    # Field-specific extraction rules
    if field_name == "job_title":
        # Remove common prefixes/suffixes
        prefixes = ["looking for", "need", "want", "hiring", "position"]
        suffixes = ["role", "position", "job"]
        
        cleaned = transcript.strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        for suffix in suffixes:
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
        
        return cleaned.title() if len(cleaned) >= 3 else None
    
    elif field_name == "department":
        # Common department keywords
        dept_keywords = {
            "engineering": "Engineering",
            "product": "Product",
            "marketing": "Marketing", 
            "sales": "Sales",
            "hr": "Human Resources",
            "finance": "Finance",
            "operations": "Operations",
            "design": "Design",
            "data": "Data Science"
        }
        
        for keyword, dept_name in dept_keywords.items():
            if keyword in text_lower:
                return dept_name
        
        return transcript.strip().title()
    
    elif field_name == "employment_type":
        type_mappings = {
            "full": "Full-time",
            "part": "Part-time", 
            "contract": "Contract",
            "intern": "Internship",
            "temporary": "Temporary",
            "freelance": "Freelance"
        }
        
        for key, emp_type in type_mappings.items():
            if key in text_lower:
                return emp_type
        
        return None
    
    elif field_name == "location":
        if any(word in text_lower for word in ["remote", "work from home", "anywhere"]):
            return "Remote"
        elif any(word in text_lower for word in ["hybrid", "flexible", "mix"]):
            return "Hybrid"
        elif any(word in text_lower for word in ["office", "on-site", "in-person"]):
            return "On-site"
        else:
            return transcript.strip().title()  # Assume it's a city/location
    
    else:
        # Default: return cleaned transcript
        return transcript.strip()


def calculate_extraction_confidence(transcript: str, field_name: str) -> float:
    """Pure function to calculate confidence in field extraction."""
    base_confidence = min(0.9, len(transcript.strip()) / 20.0 + 0.3)
    
    # Field-specific confidence adjustments
    confidence_adjustments = {
        "job_title": 0.1 if any(word in transcript.lower() for word in ["engineer", "manager", "analyst"]) else 0.0,
        "department": 0.1 if any(word in transcript.lower() for word in ["engineering", "marketing", "sales"]) else 0.0,
        "employment_type": 0.2 if any(word in transcript.lower() for word in ["full", "part", "contract"]) else 0.0,
        "location": 0.15 if any(word in transcript.lower() for word in ["remote", "hybrid", "office"]) else 0.0,
    }
    
    adjustment = confidence_adjustments.get(field_name, 0.0)
    return min(1.0, base_confidence + adjustment)


def create_voice_to_graph_processor() -> Callable[[VoiceEvent, GraphState], GraphState]:
    """Create processor to handle voice events and update graph state."""
    
    field_extractor = create_field_extractor_bridge()
    message_processor = create_voice_message_processor()
    
    def process_voice_to_graph(event: VoiceEvent, graph_state: GraphState) -> GraphState:
        """Pure function to process voice event and update graph state."""
        
        if event.type == VoiceEventType.TRANSCRIPT_READY:
            transcript = event.get("text", "")
            current_field = graph_state.get("current_field")
            
            if transcript and current_field:
                # Extract job data from transcript
                updated_job_data = field_extractor(
                    transcript,
                    current_field,
                    graph_state.get("job_data", {})
                )
                
                # Update graph state with new job data
                return {**graph_state, "job_data": updated_job_data}
        
        elif event.type == VoiceEventType.RESPONSE_GENERATED:
            transcript = event.data.get("original_transcript", "")
            response_text = event.get("text", "")
            
            if transcript and response_text:
                # Add conversation messages to graph state
                return message_processor(transcript, response_text, graph_state)
        
        return graph_state
    
    return process_voice_to_graph


def create_field_completion_detector() -> Callable[[GraphState], bool]:
    """Create function to detect when a field is complete from voice input."""
    
    def is_field_complete_from_voice(graph_state: GraphState) -> bool:
        """Pure function to check if current field is complete."""
        current_field = graph_state.get("current_field")
        job_data = graph_state.get("job_data", {})
        
        if not current_field or current_field not in job_data:
            return False
        
        field_data = job_data[current_field]
        
        # Check if we have value and good confidence
        if isinstance(field_data, dict):
            has_value = bool(field_data.get("value"))
            good_confidence = field_data.get("confidence", 0.0) >= 0.5
            return has_value and good_confidence
        
        return bool(field_data)
    
    return is_field_complete_from_voice


def create_next_field_calculator() -> Callable[[GraphState], Optional[str]]:
    """Create function to calculate next field to collect."""
    
    field_order = [
        'job_title', 'department', 'experience', 'employment_type',
        'location', 'responsibilities', 'skills', 'education',
        'salary', 'additional_requirements'
    ]
    
    def calculate_next_field(graph_state: GraphState) -> Optional[str]:
        """Pure function to determine next field to collect."""
        job_data = graph_state.get("job_data", {})
        completion_detector = create_field_completion_detector()
        
        # Find first incomplete field
        for field in field_order:
            field_state = {**graph_state, "current_field": field}
            if not completion_detector(field_state):
                return field
        
        return None  # All fields complete
    
    return calculate_next_field


def create_voice_conversation_controller() -> Callable[[VoiceEvent, GraphState], GraphState]:
    """Create main controller for voice conversation flow."""
    
    voice_processor = create_voice_to_graph_processor()
    completion_detector = create_field_completion_detector()
    next_field_calc = create_next_field_calculator()
    
    def control_voice_conversation(event: VoiceEvent, graph_state: GraphState) -> GraphState:
        """Pure function to control voice conversation flow."""
        
        # Process the voice event
        updated_state = voice_processor(event, graph_state)
        
        # Check for field completion and advance if needed
        if completion_detector(updated_state):
            next_field = next_field_calc(updated_state)
            
            if next_field:
                # Move to next field
                updated_state = {**updated_state, "current_field": next_field}
                logger.info(f"Voice conversation: moved to field {next_field}")
            else:
                # Mark conversation complete
                updated_state = {
                    **updated_state,
                    "is_complete": True,
                    "current_field": None,
                    "conversation_phase": "complete"
                }
                logger.info("Voice conversation: marked complete")
        
        return updated_state
    
    return control_voice_conversation


def create_voice_command_processor() -> Callable[[str, GraphState], GraphState]:
    """Create processor for voice commands that affect conversation flow."""
    
    def process_voice_commands(transcript: str, graph_state: GraphState) -> GraphState:
        """Pure function to process voice commands and update graph state."""
        
        commands = extract_conversation_commands(transcript)
        
        if not commands:
            return graph_state
        
        updates = {}
        
        # Handle navigation commands
        if "navigation" in commands:
            nav_command = commands["navigation"]
            
            if nav_command == "previous":
                # Go to previous field
                current_field = graph_state.get("current_field")
                if current_field:
                    field_order = [
                        'job_title', 'department', 'experience', 'employment_type',
                        'location', 'responsibilities', 'skills', 'education',
                        'salary', 'additional_requirements'
                    ]
                    try:
                        current_index = field_order.index(current_field)
                        if current_index > 0:
                            updates["current_field"] = field_order[current_index - 1]
                    except ValueError:
                        pass
            
            elif nav_command == "next":
                # Force move to next field
                next_field_calc = create_next_field_calculator()
                next_field = next_field_calc(graph_state)
                if next_field:
                    updates["current_field"] = next_field
            
            elif nav_command == "reset":
                # Reset conversation
                return create_initial_graph_state()
            
            elif nav_command == "complete":
                # Mark as complete
                updates["is_complete"] = True
                updates["current_field"] = None
        
        # Handle mode switches in session metadata
        if "mode_switch" in commands:
            session_meta = graph_state.get("session_metadata", {})
            updates["session_metadata"] = {
                **session_meta,
                "preferred_mode": commands["mode_switch"].value,
                "mode_switch_time": time.time()
            }
        
        return {**graph_state, **updates}
    
    return process_voice_commands


def create_voice_data_synchronizer() -> Callable[[VoiceConversationState, GraphState], GraphState]:
    """Create function to synchronize voice conversation data with graph state."""
    
    def synchronize_voice_data(
        voice_state: VoiceConversationState,
        graph_state: GraphState
    ) -> GraphState:
        """Pure function to sync voice state changes to graph state."""
        
        updates = {}
        voice_metadata = voice_state.metadata
        
        # Sync job data if available
        if "job_data" in voice_metadata:
            voice_job_data = voice_metadata["job_data"]
            current_job_data = graph_state.get("job_data", {})
            
            # Merge job data (voice data takes precedence)
            merged_job_data = {**current_job_data, **voice_job_data}
            updates["job_data"] = merged_job_data
        
        # Sync current field
        if "current_field" in voice_metadata:
            updates["current_field"] = voice_metadata["current_field"]
        
        # Sync completion status
        if "is_complete" in voice_metadata:
            updates["is_complete"] = voice_metadata["is_complete"]
        
        # Update session metadata with voice metrics
        session_meta = graph_state.get("session_metadata", {})
        updates["session_metadata"] = {
            **session_meta,
            "voice_state_sync": time.time(),
            "voice_conversation_active": voice_state.mode == ConversationMode.VOICE,
            "last_transcript": voice_state.current_transcript,
            "voice_buffer_size": len(voice_state.audio_buffer)
        }
        
        return {**graph_state, **updates}
    
    return synchronize_voice_data


def create_unified_event_processor() -> Callable[[VoiceEvent, GraphState], GraphState]:
    """Create unified processor for voice events affecting graph state."""
    
    voice_controller = create_voice_conversation_controller()
    command_processor = create_voice_command_processor()
    
    def process_unified_event(event: VoiceEvent, graph_state: GraphState) -> GraphState:
        """Process voice event with full graph state integration."""
        
        # First apply basic voice event processing
        updated_state = voice_controller(event, graph_state)
        
        # Then check for and apply voice commands
        if event.type == VoiceEventType.TRANSCRIPT_READY:
            transcript = event.get("text", "")
            updated_state = command_processor(transcript, updated_state)
        
        return updated_state
    
    return process_unified_event


def create_voice_session_manager() -> Callable:
    """Create session manager for voice conversations."""
    
    def manage_voice_session(
        events: List[VoiceEvent],
        initial_graph_state: GraphState
    ) -> GraphState:
        """Manage voice session by processing all events."""
        
        event_processor = create_unified_event_processor()
        
        # Process all events in sequence
        final_state = reduce(
            lambda state, event: event_processor(event, state),
            events,
            initial_graph_state
        )
        
        return final_state
    
    return manage_voice_session


def create_voice_state_validator() -> Callable[[GraphState], bool]:
    """Create validator for voice-enhanced graph state."""
    
    def validate_voice_state(graph_state: GraphState) -> bool:
        """Pure function to validate voice-enhanced graph state."""
        
        # Basic graph state validation
        required_keys = ["messages", "job_data", "current_field", "is_complete"]
        if not all(key in graph_state for key in required_keys):
            return False
        
        # Voice-specific validation
        session_meta = graph_state.get("session_metadata", {})
        
        # Check voice metadata consistency
        if "voice_conversation_active" in session_meta:
            voice_active = session_meta["voice_conversation_active"]
            if voice_active and not session_meta.get("last_transcript"):
                logger.warning("Voice conversation marked active but no transcript")
        
        return True
    
    return validate_voice_state


# Factory function for complete voice integration
def create_complete_voice_integration() -> Dict[str, Callable]:
    """Factory to create all voice integration functions."""
    
    return {
        "state_bridge": create_voice_state_bridge(),
        "graph_bridge": create_graph_state_bridge(),
        "message_processor": create_voice_message_processor(),
        "field_extractor": create_field_extractor_bridge(),
        "event_processor": create_unified_event_processor(),
        "session_manager": create_voice_session_manager(),
        "state_validator": create_voice_state_validator(),
        "command_processor": create_voice_command_processor(),
        "data_synchronizer": create_voice_data_synchronizer()
    }


def pipe_voice_to_graph(
    voice_events: List[VoiceEvent],
    graph_state: GraphState
) -> GraphState:
    """High-level function to pipe voice events to graph state."""
    
    integration_funcs = create_complete_voice_integration()
    session_manager = integration_funcs["session_manager"]
    
    return session_manager(voice_events, graph_state)


def create_conversation_health_monitor() -> Callable[[GraphState], Dict[str, Any]]:
    """Create function to monitor conversation health in voice mode."""
    
    def monitor_health(graph_state: GraphState) -> Dict[str, Any]:
        """Pure function to assess conversation health."""
        
        session_meta = graph_state.get("session_metadata", {})
        job_data = graph_state.get("job_data", {})
        messages = graph_state.get("messages", [])
        
        # Calculate health metrics
        now = time.time()
        last_interaction = session_meta.get("last_voice_interaction", 0.0)
        time_since_interaction = now - last_interaction
        
        voice_messages = [m for m in messages if m.get("source") == "voice"]
        message_frequency = len(voice_messages) / max((now - session_meta.get("voice_state_sync", now)) / 60.0, 1.0)
        
        completed_fields = len([k for k, v in job_data.items() if v])
        completion_rate = completed_fields / 9.0  # Assuming 9 total fields
        
        return {
            "is_responsive": time_since_interaction < 60.0,
            "message_frequency_per_minute": message_frequency,
            "completion_rate": completion_rate,
            "voice_active": session_meta.get("voice_conversation_active", False),
            "health_score": min(1.0, (
                (1.0 if time_since_interaction < 30.0 else 0.5) +
                min(0.5, message_frequency / 2.0) +
                completion_rate
            ) / 2.0)
        }
    
    return monitor_health