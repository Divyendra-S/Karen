"""Functional voice event processors with immutable state management.

This module handles voice interaction events using pure functions and immutable data structures,
integrating with the Pipecat pipeline for real-time voice processing.
"""

import time
from typing import Dict, Any, List, Callable, Optional, Union
from functools import reduce, partial
from dataclasses import replace
from enum import Enum

from loguru import logger

from core.types.voice_types import (
    ConversationState,
    ConversationMode,
    AudioFrame,
    TranscriptFrame,
    ResponseFrame,
    VoicePipelineResult,
    StateReducer,
)


class VoiceEventType(str, Enum):
    """Types of voice events in the pipeline."""
    AUDIO_RECEIVED = "audio_received"
    TRANSCRIPT_READY = "transcript_ready"
    RESPONSE_GENERATED = "response_generated"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    MODE_SWITCHED = "mode_switched"
    ERROR_OCCURRED = "error_occurred"
    CONVERSATION_COMPLETE = "conversation_complete"


class VoiceEvent:
    """Immutable voice event data."""
    
    def __init__(
        self,
        event_type: VoiceEventType,
        data: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        self._type = event_type
        self._data = data.copy()  # Defensive copy
        self._timestamp = timestamp or time.time()
    
    @property
    def type(self) -> VoiceEventType:
        """Get event type."""
        return self._type
    
    @property 
    def data(self) -> Dict[str, Any]:
        """Get event data (immutable view)."""
        return self._data.copy()
    
    @property
    def timestamp(self) -> float:
        """Get event timestamp."""
        return self._timestamp
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get data value safely."""
        return self._data.get(key, default)


def create_audio_event(audio_data: bytes, sample_rate: int = 16000) -> VoiceEvent:
    """Pure function to create audio event."""
    return VoiceEvent(
        VoiceEventType.AUDIO_RECEIVED,
        {
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "size_bytes": len(audio_data)
        }
    )


def create_transcript_event(transcript: TranscriptFrame) -> VoiceEvent:
    """Pure function to create transcript event."""
    return VoiceEvent(
        VoiceEventType.TRANSCRIPT_READY,
        {
            "text": transcript.text,
            "confidence": transcript.confidence,
            "is_final": transcript.is_final
        }
    )


def create_response_event(response: ResponseFrame) -> VoiceEvent:
    """Pure function to create response event."""
    return VoiceEvent(
        VoiceEventType.RESPONSE_GENERATED,
        {
            "text": response.text,
            "audio_data": response.audio_data,
            "metadata": response.metadata
        }
    )


def create_error_event(error_message: str, context: Dict[str, Any] = None) -> VoiceEvent:
    """Pure function to create error event."""
    return VoiceEvent(
        VoiceEventType.ERROR_OCCURRED,
        {
            "error": error_message,
            "context": context or {}
        }
    )


# State reducers for different event types
def reduce_audio_event(state: ConversationState, event: VoiceEvent) -> ConversationState:
    """Pure function to reduce audio received event."""
    audio_data = event.get("audio_data", b"")
    return replace(state, audio_buffer=audio_data)


def reduce_transcript_event(state: ConversationState, event: VoiceEvent) -> ConversationState:
    """Pure function to reduce transcript ready event."""
    transcript_text = event.get("text", "")
    confidence = event.get("confidence", 0.0)
    
    # Only update if confidence is acceptable
    if confidence >= 0.5:
        return replace(
            state,
            current_transcript=transcript_text,
            metadata={**state.metadata, "last_transcript_confidence": confidence}
        )
    return state


def reduce_response_event(state: ConversationState, event: VoiceEvent) -> ConversationState:
    """Pure function to reduce response generated event."""
    response_text = event.get("text", "")
    return replace(
        state,
        last_response=response_text,
        is_speaking=True,
        metadata={**state.metadata, "last_response_time": time.time()}
    )


def reduce_speech_started_event(state: ConversationState, event: VoiceEvent) -> ConversationState:
    """Pure function to reduce speech started event."""
    return replace(state, is_speaking=True)


def reduce_speech_ended_event(state: ConversationState, event: VoiceEvent) -> ConversationState:
    """Pure function to reduce speech ended event."""
    return replace(
        state,
        is_speaking=False,
        audio_buffer=b"",  # Clear buffer after speech
        current_transcript=""  # Clear transcript after processing
    )


def reduce_mode_switch_event(state: ConversationState, event: VoiceEvent) -> ConversationState:
    """Pure function to reduce mode switch event."""
    new_mode = event.get("mode", ConversationMode.VOICE)
    if isinstance(new_mode, str):
        new_mode = ConversationMode(new_mode)
    return replace(state, mode=new_mode)


def reduce_error_event(state: ConversationState, event: VoiceEvent) -> ConversationState:
    """Pure function to reduce error event."""
    error_message = event.get("error", "Unknown error")
    return replace(
        state,
        metadata={
            **state.metadata,
            "last_error": error_message,
            "error_time": time.time()
        }
    )


# Event reducer registry (functional approach)
def create_event_reducer_registry() -> Dict[VoiceEventType, Callable]:
    """Create registry of event reducers."""
    return {
        VoiceEventType.AUDIO_RECEIVED: reduce_audio_event,
        VoiceEventType.TRANSCRIPT_READY: reduce_transcript_event,
        VoiceEventType.RESPONSE_GENERATED: reduce_response_event,
        VoiceEventType.SPEECH_STARTED: reduce_speech_started_event,
        VoiceEventType.SPEECH_ENDED: reduce_speech_ended_event,
        VoiceEventType.MODE_SWITCHED: reduce_mode_switch_event,
        VoiceEventType.ERROR_OCCURRED: reduce_error_event,
    }


def create_voice_state_reducer() -> StateReducer:
    """Create the main voice state reducer function."""
    reducer_registry = create_event_reducer_registry()
    
    def reduce_voice_state(state: ConversationState, event: Union[VoiceEvent, Any]) -> ConversationState:
        """Pure function to reduce voice state based on events."""
        if isinstance(event, VoiceEvent):
            reducer_func = reducer_registry.get(event.type)
            if reducer_func:
                return reducer_func(state, event)
            else:
                logger.warning(f"No reducer for event type: {event.type}")
                return state
        
        # Handle raw event dictionaries (backward compatibility)
        elif isinstance(event, dict):
            event_type_str = event.get("type", "unknown")
            try:
                event_type = VoiceEventType(event_type_str)
                voice_event = VoiceEvent(event_type, event)
                return reduce_voice_state(state, voice_event)
            except ValueError:
                logger.warning(f"Unknown event type in dict: {event_type_str}")
                return state
        
        return state
    
    return reduce_voice_state


def create_event_processor(
    state_reducer: StateReducer
) -> Callable[[List[VoiceEvent], ConversationState], ConversationState]:
    """Create function to process multiple events."""
    
    def process_events(events: List[VoiceEvent], initial_state: ConversationState) -> ConversationState:
        """Pure function to process list of events against state."""
        return reduce(state_reducer, events, initial_state)
    
    return process_events


def filter_events_by_type(
    events: List[VoiceEvent],
    event_type: VoiceEventType
) -> List[VoiceEvent]:
    """Pure function to filter events by type."""
    return [event for event in events if event.type == event_type]


def filter_events_by_timerange(
    events: List[VoiceEvent],
    start_time: float,
    end_time: float
) -> List[VoiceEvent]:
    """Pure function to filter events by time range."""
    return [
        event for event in events 
        if start_time <= event.timestamp <= end_time
    ]


def create_event_validator() -> Callable[[VoiceEvent], bool]:
    """Create event validator function."""
    
    def validate_event(event: VoiceEvent) -> bool:
        """Pure function to validate voice events."""
        # Basic validation rules
        if not isinstance(event.type, VoiceEventType):
            return False
        
        data = event.data
        
        # Type-specific validation
        if event.type == VoiceEventType.AUDIO_RECEIVED:
            return "audio_data" in data and isinstance(data["audio_data"], bytes)
        
        elif event.type == VoiceEventType.TRANSCRIPT_READY:
            return (
                "text" in data and
                "confidence" in data and
                isinstance(data["confidence"], (int, float))
            )
        
        elif event.type == VoiceEventType.RESPONSE_GENERATED:
            return "text" in data and isinstance(data["text"], str)
        
        return True  # Allow other event types by default
    
    return validate_event


def create_event_metrics_calculator() -> Callable[[List[VoiceEvent]], Dict[str, Any]]:
    """Create function to calculate metrics from events."""
    
    def calculate_metrics(events: List[VoiceEvent]) -> Dict[str, Any]:
        """Pure function to calculate event processing metrics."""
        if not events:
            return {"total_events": 0}
        
        event_counts = {}
        total_processing_time = 0.0
        error_count = 0
        
        first_event_time = min(event.timestamp for event in events)
        last_event_time = max(event.timestamp for event in events)
        
        for event in events:
            event_type = event.type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event.type == VoiceEventType.ERROR_OCCURRED:
                error_count += 1
            
            # Calculate processing time for response events
            if event.type == VoiceEventType.RESPONSE_GENERATED:
                metadata = event.get("metadata", {})
                proc_time = metadata.get("processing_time", 0.0)
                total_processing_time += proc_time
        
        return {
            "total_events": len(events),
            "event_counts": event_counts,
            "error_count": error_count,
            "total_duration": last_event_time - first_event_time,
            "average_processing_time": total_processing_time / max(event_counts.get("response_generated", 1), 1),
            "error_rate": error_count / len(events) if events else 0.0
        }
    
    return calculate_metrics


# High-level event processing utilities
def create_voice_event_stream_processor(
    state_reducer: StateReducer,
    event_validator: Callable[[VoiceEvent], bool]
) -> Callable[[List[VoiceEvent], ConversationState], ConversationState]:
    """Create stream processor for voice events."""
    
    def process_event_stream(
        events: List[VoiceEvent],
        initial_state: ConversationState
    ) -> ConversationState:
        """Process stream of voice events with validation."""
        # Filter valid events
        valid_events = [event for event in events if event_validator(event)]
        
        if len(valid_events) != len(events):
            logger.warning(f"Filtered out {len(events) - len(valid_events)} invalid events")
        
        # Process valid events
        return reduce(state_reducer, valid_events, initial_state)
    
    return process_event_stream


def should_interrupt_speech(
    current_state: ConversationState,
    new_audio_confidence: float
) -> bool:
    """Pure function to determine if current speech should be interrupted."""
    return (
        current_state.is_speaking and
        new_audio_confidence > 0.8 and
        current_state.mode == ConversationMode.VOICE
    )


def extract_conversation_commands(transcript_text: str) -> Dict[str, Any]:
    """Pure function to extract voice commands from transcript."""
    text_lower = transcript_text.lower().strip()
    commands = {}
    
    # Mode switching
    if any(phrase in text_lower for phrase in ["text mode", "switch to text", "type instead"]):
        commands["mode_switch"] = ConversationMode.TEXT
    elif any(phrase in text_lower for phrase in ["voice mode", "talk to me", "speak"]):
        commands["mode_switch"] = ConversationMode.VOICE
    
    # Flow control
    if any(phrase in text_lower for phrase in ["go back", "previous question", "last one"]):
        commands["navigation"] = "previous"
    elif any(phrase in text_lower for phrase in ["skip", "next", "skip this"]):
        commands["navigation"] = "next"
    elif any(phrase in text_lower for phrase in ["start over", "reset", "begin again"]):
        commands["navigation"] = "reset"
    elif any(phrase in text_lower for phrase in ["we're done", "finished", "that's all"]):
        commands["navigation"] = "complete"
    
    # Voice control
    if any(phrase in text_lower for phrase in ["speak slower", "slow down"]):
        commands["speech_control"] = "slower"
    elif any(phrase in text_lower for phrase in ["speak faster", "speed up"]):
        commands["speech_control"] = "faster"
    elif any(phrase in text_lower for phrase in ["repeat", "say that again"]):
        commands["speech_control"] = "repeat"
    
    return commands


def apply_conversation_commands(
    state: ConversationState,
    commands: Dict[str, Any]
) -> ConversationState:
    """Pure function to apply conversation commands to state."""
    updated_state = state
    
    # Apply mode switch
    if "mode_switch" in commands:
        updated_state = updated_state.with_mode(commands["mode_switch"])
    
    # Apply speech control
    if "speech_control" in commands:
        control_action = commands["speech_control"]
        metadata_update = {**updated_state.metadata, "speech_control": control_action}
        updated_state = replace(updated_state, metadata=metadata_update)
    
    # Navigation commands affect metadata but don't change core state
    if "navigation" in commands:
        nav_action = commands["navigation"]
        metadata_update = {**updated_state.metadata, "navigation_command": nav_action}
        updated_state = replace(updated_state, metadata=metadata_update)
    
    return updated_state


def create_smart_event_processor() -> Callable[[VoiceEvent, ConversationState], ConversationState]:
    """Create intelligent event processor that handles commands and state updates."""
    
    base_reducer = create_voice_state_reducer()
    
    def process_smart_event(event: VoiceEvent, state: ConversationState) -> ConversationState:
        """Process event with command detection and smart state updates."""
        # First apply basic state reduction
        updated_state = base_reducer(state, event)
        
        # For transcript events, check for commands
        if event.type == VoiceEventType.TRANSCRIPT_READY:
            transcript_text = event.get("text", "")
            commands = extract_conversation_commands(transcript_text)
            
            if commands:
                logger.info(f"Detected voice commands: {commands}")
                updated_state = apply_conversation_commands(updated_state, commands)
        
        return updated_state
    
    return process_smart_event


def create_event_logger() -> Callable[[VoiceEvent], VoiceEvent]:
    """Create event logging function (pure - returns same event)."""
    
    def log_event(event: VoiceEvent) -> VoiceEvent:
        """Log voice event for debugging (pure function)."""
        event_data = {
            "type": event.type.value,
            "timestamp": event.timestamp,
            "data_keys": list(event.data.keys())
        }
        
        # Add specific details based on event type
        if event.type == VoiceEventType.TRANSCRIPT_READY:
            event_data["text_length"] = len(event.get("text", ""))
            event_data["confidence"] = event.get("confidence", 0.0)
        elif event.type == VoiceEventType.AUDIO_RECEIVED:
            event_data["audio_size"] = len(event.get("audio_data", b""))
        elif event.type == VoiceEventType.ERROR_OCCURRED:
            event_data["error"] = event.get("error", "Unknown")
        
        logger.debug(f"Voice event processed: {event_data}")
        return event  # Return unchanged (pure function)
    
    return log_event


def create_event_pipeline(
    processors: List[Callable[[VoiceEvent], VoiceEvent]]
) -> Callable[[VoiceEvent], VoiceEvent]:
    """Create event processing pipeline from list of processors."""
    
    def process_event(event: VoiceEvent) -> VoiceEvent:
        """Apply all processors to event in sequence."""
        return reduce(lambda e, processor: processor(e), processors, event)
    
    return process_event


def create_conversation_state_analyzer() -> Callable[[ConversationState], Dict[str, Any]]:
    """Create function to analyze conversation state."""
    
    def analyze_state(state: ConversationState) -> Dict[str, Any]:
        """Pure function to analyze conversation state."""
        metadata = state.metadata
        
        analysis = {
            "mode": state.mode.value,
            "is_active_conversation": bool(state.current_transcript or state.is_speaking),
            "has_audio_buffer": len(state.audio_buffer) > 0,
            "last_activity": metadata.get("last_response_time", 0.0),
            "transcript_confidence": metadata.get("last_transcript_confidence", 0.0),
            "speech_controls": metadata.get("speech_control"),
            "navigation_commands": metadata.get("navigation_command"),
            "error_status": metadata.get("last_error")
        }
        
        # Calculate conversation health
        now = time.time()
        last_activity = analysis["last_activity"]
        time_since_activity = now - last_activity if last_activity else float('inf')
        
        analysis["conversation_health"] = {
            "is_responsive": time_since_activity < 30.0,  # Active within 30 seconds
            "is_healthy": (
                analysis["transcript_confidence"] > 0.5 and
                not analysis["error_status"] and
                analysis["is_active_conversation"]
            )
        }
        
        return analysis
    
    return analyze_state


def create_event_history_manager(max_events: int = 100) -> Callable:
    """Create event history manager with functional approach."""
    
    def manage_event_history(
        events: List[VoiceEvent],
        new_event: VoiceEvent
    ) -> List[VoiceEvent]:
        """Pure function to manage event history with size limit."""
        updated_events = events + [new_event]
        
        # Keep only recent events
        if len(updated_events) > max_events:
            return updated_events[-max_events:]
        
        return updated_events
    
    return manage_event_history


# Integration with job data extraction
def extract_job_data_from_voice_event(
    event: VoiceEvent,
    current_job_data: Dict[str, Any],
    current_field: str
) -> Dict[str, Any]:
    """Pure function to extract job data from voice events."""
    if event.type != VoiceEventType.TRANSCRIPT_READY:
        return current_job_data
    
    transcript_text = event.get("text", "")
    if not transcript_text or len(transcript_text.strip()) < 3:
        return current_job_data
    
    # Use existing extraction logic from main.py
    # This would integrate with the existing rule-based and LLM extraction
    return {
        **current_job_data,
        current_field: {
            "value": transcript_text,
            "confidence": event.get("confidence", 0.0),
            "source": "voice",
            "timestamp": event.timestamp
        }
    }