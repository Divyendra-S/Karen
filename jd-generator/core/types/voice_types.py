"""Functional type definitions for voice pipeline integration.

This module defines immutable data structures and types for the Pipecat voice pipeline
following functional programming principles.
"""

from typing import NamedTuple, Optional, List, Dict, Any, Callable, Protocol
from dataclasses import dataclass, replace
from enum import Enum
from datetime import datetime


class VoiceModel(str, Enum):
    """Available voice models for TTS."""
    PUCK = "Puck"
    AOEDE = "Aoede" 
    CHARON = "Charon"
    FENRIR = "Fenrir"
    KORE = "Kore"


class ConversationMode(str, Enum):
    """Conversation interaction modes."""
    TEXT = "text"
    VOICE = "voice"
    HYBRID = "hybrid"


class AudioFrame(NamedTuple):
    """Immutable audio frame data."""
    data: bytes
    timestamp: float
    sample_rate: int
    format: str


class TranscriptFrame(NamedTuple):
    """Immutable transcript data."""
    text: str
    confidence: float
    timestamp: float
    is_final: bool


class ResponseFrame(NamedTuple):
    """Immutable AI response data."""
    text: str
    audio_data: Optional[bytes]
    timestamp: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class VoiceConfig:
    """Immutable voice pipeline configuration."""
    google_api_key: str
    voice_model: VoiceModel
    sample_rate: int = 16000
    enable_interruption: bool = True
    vad_threshold: float = 0.5
    response_timeout_ms: int = 5000
    
    def with_voice(self, voice: VoiceModel) -> 'VoiceConfig':
        """Return new config with different voice model."""
        return replace(self, voice_model=voice)


@dataclass(frozen=True)
class ConversationState:
    """Immutable conversation state for voice interactions."""
    mode: ConversationMode
    is_speaking: bool
    current_transcript: str
    audio_buffer: bytes
    last_response: Optional[str]
    metadata: Dict[str, Any]
    
    def with_mode(self, mode: ConversationMode) -> 'ConversationState':
        """Return new state with different mode."""
        return replace(self, mode=mode)
    
    def with_transcript(self, transcript: str) -> 'ConversationState':
        """Return new state with updated transcript."""
        return replace(self, current_transcript=transcript)
    
    def with_speaking(self, speaking: bool) -> 'ConversationState':
        """Return new state with updated speaking status."""
        return replace(self, is_speaking=speaking)


class AudioProcessor(Protocol):
    """Protocol for functional audio processing."""
    
    def __call__(self, audio: AudioFrame) -> TranscriptFrame:
        """Process audio frame to transcript."""
        ...


class LLMProcessor(Protocol):
    """Protocol for functional LLM processing."""
    
    def __call__(self, transcript: TranscriptFrame, context: Dict[str, Any]) -> ResponseFrame:
        """Process transcript with context to generate response."""
        ...


class TTSProcessor(Protocol):
    """Protocol for functional text-to-speech processing."""
    
    def __call__(self, response: ResponseFrame) -> AudioFrame:
        """Convert response text to audio."""
        ...


class StateTransformer(Protocol):
    """Protocol for functional state transformations."""
    
    def __call__(self, old_state: ConversationState, event: Any) -> ConversationState:
        """Transform state based on event."""
        ...


@dataclass(frozen=True)
class VoicePipelineResult:
    """Immutable result from voice pipeline processing."""
    success: bool
    transcript: Optional[str]
    response_text: Optional[str]
    response_audio: Optional[bytes]
    error_message: Optional[str]
    processing_time_ms: float
    
    @classmethod
    def success_result(
        cls,
        transcript: str,
        response_text: str,
        response_audio: bytes,
        processing_time: float
    ) -> 'VoicePipelineResult':
        """Create successful pipeline result."""
        return cls(
            success=True,
            transcript=transcript,
            response_text=response_text,
            response_audio=response_audio,
            error_message=None,
            processing_time_ms=processing_time
        )
    
    @classmethod
    def error_result(cls, error: str, processing_time: float = 0.0) -> 'VoicePipelineResult':
        """Create error pipeline result."""
        return cls(
            success=False,
            transcript=None,
            response_text=None,
            response_audio=None,
            error_message=error,
            processing_time_ms=processing_time
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the complete voice pipeline."""
    voice_config: VoiceConfig
    stt_processor: AudioProcessor
    llm_processor: LLMProcessor
    tts_processor: TTSProcessor
    state_transformer: StateTransformer
    
    def with_voice_config(self, config: VoiceConfig) -> 'PipelineConfig':
        """Return new pipeline config with different voice config."""
        return replace(self, voice_config=config)


# Type aliases for functional composition
AudioToTranscript = Callable[[AudioFrame], TranscriptFrame]
TranscriptToResponse = Callable[[TranscriptFrame, Dict[str, Any]], ResponseFrame]
ResponseToAudio = Callable[[ResponseFrame], AudioFrame]
StateReducer = Callable[[ConversationState, Any], ConversationState]
Pipeline = Callable[[AudioFrame, Dict[str, Any]], VoicePipelineResult]