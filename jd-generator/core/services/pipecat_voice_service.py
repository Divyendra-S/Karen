"""Pipecat voice service integration for Streamlit.

This module provides a functional interface to integrate Pipecat's real-time voice
capabilities with the Streamlit-based JD generator application.
"""

import asyncio
import os
import tempfile
from typing import Dict, Any, Optional, Callable, List
from functools import partial
from dataclasses import replace

from loguru import logger
from PIL import Image

# Pipecat imports
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    OutputImageRawFrame,
    SpriteFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
# Note: For basic voice functionality, we'll use a simplified approach without Daily transport
# from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
# from pipecat.audio.vad.silero import SileroVADAnalyzer

# Simplified imports for non-Daily integration
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from core.types.voice_types import (
    VoiceConfig,
    ConversationState,
    VoiceModel,
    AudioFrame,
    VoicePipelineResult,
)
from core.services.voice_events import VoiceEvent, VoiceEventType
from core.services.voice_integration import create_complete_voice_integration


class FunctionalAnimationProcessor(FrameProcessor):
    """Functional animation processor for bot visual feedback."""
    
    def __init__(self, sprite_frames: List[OutputImageRawFrame]):
        super().__init__()
        self._sprites = sprite_frames
        self._is_talking = False
        self._quiet_frame = sprite_frames[0] if sprite_frames else None
        self._talking_frame = SpriteFrame(images=sprite_frames) if sprite_frames else None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process animation frames functionally."""
        await super().process_frame(frame, direction)
        
        # Pure function approach to state changes
        new_talking_state = self._calculate_talking_state(frame, self._is_talking)
        
        if new_talking_state != self._is_talking:
            self._is_talking = new_talking_state
            animation_frame = self._select_animation_frame(new_talking_state)
            if animation_frame:
                await self.push_frame(animation_frame)
        
        await self.push_frame(frame, direction)
    
    def _calculate_talking_state(self, frame: Frame, current_talking: bool) -> bool:
        """Pure function to calculate new talking state."""
        if isinstance(frame, BotStartedSpeakingFrame):
            return True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            return False
        return current_talking
    
    def _select_animation_frame(self, is_talking: bool) -> Optional[Frame]:
        """Pure function to select appropriate animation frame."""
        if is_talking and self._talking_frame:
            return self._talking_frame
        elif not is_talking and self._quiet_frame:
            return self._quiet_frame
        return None


def load_animation_sprites(assets_dir: str) -> List[OutputImageRawFrame]:
    """Pure function to load animation sprites."""
    sprites = []
    
    try:
        for i in range(1, 26):
            sprite_path = os.path.join(assets_dir, f"robot0{i:02d}.png")
            if os.path.exists(sprite_path):
                with Image.open(sprite_path) as img:
                    sprite = OutputImageRawFrame(
                        image=img.tobytes(),
                        size=img.size,
                        format=img.format
                    )
                    sprites.append(sprite)
        
        # Create smooth animation with reverse
        if sprites:
            return sprites + sprites[::-1]
        
    except Exception as e:
        logger.error(f"Error loading sprites: {e}")
    
    return []


def create_hr_conversation_context(
    current_field: str,
    job_data: Dict[str, Any],
    conversation_history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Pure function to create conversation context for Gemini."""
    
    # Base system message for HR conversation
    system_message = {
        "role": "user",
        "content": f"""You are an expert HR professional conducting a voice interview to create a job description.

CURRENT STATUS:
- Collecting: {current_field}
- Already have: {', '.join(job_data.keys()) if job_data else 'nothing yet'}

VOICE CONVERSATION RULES:
- Keep responses very brief (15-30 words)
- Sound natural and conversational
- Ask one focused question at a time
- Acknowledge what they said first
- No special characters or formatting
- Sound like a real HR person

FIELD ORDER:
Job Title → Department → Experience → Employment Type → Location → 
Responsibilities → Skills → Education → Salary → Additional Requirements

Current focus: {current_field}

Be friendly, efficient, and professional. Generate natural conversation that gathers complete job information."""
    }
    
    # Combine system message with recent conversation history
    context = [system_message]
    
    # Add recent messages (last 6 for context)
    recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
    context.extend(recent_messages)
    
    return context


class FunctionalVoiceService:
    """Functional voice service using Pipecat and Gemini."""
    
    def __init__(self, voice_config: VoiceConfig):
        self.config = voice_config
        self.integration_funcs = create_complete_voice_integration()
        self._pipeline_task: Optional[PipelineTask] = None
        self._is_running = False
    
    async def initialize_pipeline(
        self,
        conversation_context: Dict[str, Any]
    ) -> PipelineTask:
        """Initialize Pipecat pipeline functionally."""
        
        # Create Gemini service
        gemini_service = GeminiMultimodalLiveLLMService(
            api_key=self.config.google_api_key,
            voice_id=self.config.voice_model.value
        )
        
        # Create conversation context
        hr_context = create_hr_conversation_context(
            conversation_context.get("current_field", "job_title"),
            conversation_context.get("job_data", {}),
            conversation_context.get("messages", [])
        )
        
        # Set up context aggregator
        context = OpenAILLMContext(hr_context)
        context_aggregator = gemini_service.create_context_aggregator(context)
        
        # Load animation sprites
        script_dir = os.path.dirname(__file__)
        assets_dir = os.path.join(script_dir, "..", "..", "assets")
        sprites = load_animation_sprites(assets_dir)
        
        # Create animation processor
        animation_processor = FunctionalAnimationProcessor(sprites)
        
        # Create pipeline
        pipeline = Pipeline([
            # Input processing
            context_aggregator.user(),
            
            # LLM processing
            gemini_service,
            
            # Animation processing
            animation_processor,
            
            # Output processing
            context_aggregator.assistant(),
        ])
        
        # Create task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            )
        )
        
        # Set initial animation frame
        if sprites:
            await task.queue_frame(sprites[0])
        
        return task
    
    async def start_voice_conversation(
        self,
        conversation_context: Dict[str, Any]
    ) -> bool:
        """Start voice conversation pipeline."""
        try:
            self._pipeline_task = await self.initialize_pipeline(conversation_context)
            
            # Start with greeting
            await self._pipeline_task.queue_frames([LLMRunFrame()])
            
            self._is_running = True
            logger.info("Voice conversation started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice conversation: {e}")
            return False
    
    async def process_audio_input(
        self,
        audio_bytes: bytes,
        conversation_context: Dict[str, Any]
    ) -> VoicePipelineResult:
        """Process audio input through the pipeline."""
        
        if not self._is_running or not self._pipeline_task:
            return VoicePipelineResult.error_result("Voice pipeline not running")
        
        try:
            start_time = time.time()
            
            # Convert audio to Pipecat frame
            audio_frame = AudioRawFrame(
                audio=audio_bytes,
                sample_rate=self.config.sample_rate,
                num_channels=1
            )
            
            # Queue audio frame
            await self._pipeline_task.queue_frame(audio_frame)
            
            # Process through pipeline (this is simplified)
            # In actual implementation, you'd wait for pipeline response
            
            processing_time = (time.time() - start_time) * 1000
            
            # Placeholder response
            return VoicePipelineResult.success_result(
                transcript="[Audio processed through Pipecat]",
                response_text="Thank you for that information. What else can you tell me?",
                response_audio=b"",  # TTS audio would be here
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return VoicePipelineResult.error_result(str(e))
    
    async def stop_voice_conversation(self) -> bool:
        """Stop voice conversation pipeline."""
        try:
            if self._pipeline_task:
                await self._pipeline_task.cancel()
                self._pipeline_task = None
            
            self._is_running = False
            logger.info("Voice conversation stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop voice conversation: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if voice service is running."""
        return self._is_running
    
    def get_config(self) -> VoiceConfig:
        """Get current voice configuration."""
        return self.config
    
    def with_config(self, new_config: VoiceConfig) -> 'FunctionalVoiceService':
        """Create new service with different configuration."""
        return FunctionalVoiceService(new_config)


def create_voice_service_factory() -> Callable[[Dict[str, str]], FunctionalVoiceService]:
    """Factory function to create voice service from environment variables."""
    
    def create_service(env_vars: Dict[str, str]) -> FunctionalVoiceService:
        """Create voice service from environment configuration."""
        
        google_api_key = env_vars.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        voice_model_str = env_vars.get("VOICE_MODEL", "Puck")
        try:
            voice_model = VoiceModel(voice_model_str)
        except ValueError:
            voice_model = VoiceModel.PUCK
            logger.warning(f"Invalid voice model {voice_model_str}, using default: Puck")
        
        voice_config = VoiceConfig(
            google_api_key=google_api_key,
            voice_model=voice_model,
            sample_rate=int(env_vars.get("AUDIO_SAMPLE_RATE", "16000")),
            enable_interruption=env_vars.get("ENABLE_VOICE_MODE", "true").lower() == "true",
            vad_threshold=float(env_vars.get("VAD_THRESHOLD", "0.5")),
            response_timeout_ms=int(env_vars.get("VOICE_TIMEOUT", "5000"))
        )
        
        return FunctionalVoiceService(voice_config)
    
    return create_service


# Streamlit integration helpers
def create_streamlit_voice_interface() -> Dict[str, Callable]:
    """Create functional interface for Streamlit voice integration."""
    
    def initialize_voice_service(env_vars: Dict[str, str]) -> Optional[FunctionalVoiceService]:
        """Initialize voice service for Streamlit session."""
        try:
            factory = create_voice_service_factory()
            return factory(env_vars)
        except Exception as e:
            logger.error(f"Failed to initialize voice service: {e}")
            return None
    
    def process_streamlit_audio(
        audio_bytes: bytes,
        voice_service: FunctionalVoiceService,
        conversation_context: Dict[str, Any]
    ) -> Optional[VoicePipelineResult]:
        """Process audio from Streamlit audio recorder."""
        if not voice_service or not voice_service.is_running():
            return None
        
        try:
            # Run async processing in sync context
            return asyncio.run(
                voice_service.process_audio_input(audio_bytes, conversation_context)
            )
        except Exception as e:
            logger.error(f"Streamlit audio processing error: {e}")
            return VoicePipelineResult.error_result(str(e))
    
    def start_voice_mode(
        voice_service: FunctionalVoiceService,
        conversation_context: Dict[str, Any]
    ) -> bool:
        """Start voice mode for Streamlit session."""
        if not voice_service:
            return False
        
        try:
            return asyncio.run(
                voice_service.start_voice_conversation(conversation_context)
            )
        except Exception as e:
            logger.error(f"Failed to start voice mode: {e}")
            return False
    
    def stop_voice_mode(voice_service: FunctionalVoiceService) -> bool:
        """Stop voice mode for Streamlit session."""
        if not voice_service:
            return True
        
        try:
            return asyncio.run(voice_service.stop_voice_conversation())
        except Exception as e:
            logger.error(f"Failed to stop voice mode: {e}")
            return False
    
    return {
        "initialize": initialize_voice_service,
        "process_audio": process_streamlit_audio,
        "start_voice": start_voice_mode,
        "stop_voice": stop_voice_mode
    }


def create_voice_session_manager() -> Callable:
    """Create session manager for voice interactions in Streamlit."""
    
    def manage_voice_session(
        session_state: Dict[str, Any],
        voice_interface: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Manage voice session state in Streamlit."""
        
        # Initialize voice service if not present
        if "voice_service" not in session_state:
            env_vars = {
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                "VOICE_MODEL": os.getenv("VOICE_MODEL", "Puck"),
                "AUDIO_SAMPLE_RATE": os.getenv("AUDIO_SAMPLE_RATE", "16000"),
                "ENABLE_VOICE_MODE": os.getenv("ENABLE_VOICE_MODE", "true"),
            }
            
            voice_service = voice_interface["initialize"](env_vars)
            session_state["voice_service"] = voice_service
            session_state["voice_mode_active"] = False
        
        return session_state
    
    return manage_voice_session


def create_audio_processor_for_streamlit() -> Callable:
    """Create audio processor specifically for Streamlit integration."""
    
    def process_streamlit_audio_bytes(
        audio_bytes: bytes,
        conversation_context: Dict[str, Any],
        voice_service: FunctionalVoiceService
    ) -> Dict[str, Any]:
        """Process audio bytes from Streamlit and return structured result."""
        
        if not audio_bytes or len(audio_bytes) == 0:
            return {
                "success": False,
                "error": "No audio data received",
                "transcript": "",
                "response": ""
            }
        
        # Process through voice service
        streamlit_interface = create_streamlit_voice_interface()
        result = streamlit_interface["process_audio"](
            audio_bytes,
            voice_service,
            conversation_context
        )
        
        if result and result.success:
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
                "error": result.error_message if result else "Processing failed",
                "transcript": "",
                "response": ""
            }
    
    return process_streamlit_audio_bytes


def create_voice_mode_toggle() -> Callable:
    """Create function to toggle voice mode in Streamlit."""
    
    def toggle_voice_mode(
        current_mode: bool,
        session_state: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Toggle voice mode and update session state."""
        
        voice_service = session_state.get("voice_service")
        if not voice_service:
            return {
                "success": False,
                "error": "Voice service not initialized",
                "mode_active": False
            }
        
        streamlit_interface = create_streamlit_voice_interface()
        
        if not current_mode:
            # Start voice mode
            success = streamlit_interface["start_voice"](voice_service, conversation_context)
            if success:
                session_state["voice_mode_active"] = True
                return {
                    "success": True,
                    "mode_active": True,
                    "message": "Voice mode activated"
                }
        else:
            # Stop voice mode
            success = streamlit_interface["stop_voice"](voice_service)
            if success:
                session_state["voice_mode_active"] = False
                return {
                    "success": True,
                    "mode_active": False,
                    "message": "Voice mode deactivated"
                }
        
        return {
            "success": False,
            "error": "Failed to toggle voice mode",
            "mode_active": current_mode
        }
    
    return toggle_voice_mode


def create_conversation_context_adapter() -> Callable:
    """Create adapter to convert GraphState to conversation context for voice."""
    
    def adapt_context(graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Pure function to adapt graph state for voice conversation."""
        
        return {
            "messages": graph_state.get("messages", []),
            "current_field": graph_state.get("current_field", "job_title"),
            "job_data": graph_state.get("job_data", {}),
            "conversation_phase": graph_state.get("conversation_phase", "greeting"),
            "is_complete": graph_state.get("is_complete", False),
            "session_metadata": graph_state.get("session_metadata", {})
        }
    
    return adapt_context


# High-level integration function
def integrate_voice_with_streamlit_session(
    session_state: Dict[str, Any],
    audio_input: Optional[bytes] = None
) -> Dict[str, Any]:
    """High-level function to integrate voice processing with Streamlit session."""
    
    # Create interface functions
    voice_interface = create_streamlit_voice_interface()
    session_manager = create_voice_session_manager()
    audio_processor = create_audio_processor_for_streamlit()
    context_adapter = create_conversation_context_adapter()
    
    # Manage voice session
    updated_session = session_manager(session_state, voice_interface)
    
    # Process audio if provided
    if audio_input and updated_session.get("voice_service"):
        conversation_context = context_adapter(
            updated_session.get("conversation_state", {})
        )
        
        audio_result = audio_processor(
            audio_input,
            conversation_context,
            updated_session["voice_service"]
        )
        
        # Update session with audio result
        updated_session["last_voice_result"] = audio_result
        
        return {
            **updated_session,
            "voice_processing_result": audio_result
        }
    
    return updated_session


def create_voice_metrics_calculator() -> Callable:
    """Create function to calculate voice interaction metrics."""
    
    def calculate_voice_metrics(session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate voice interaction performance metrics."""
        
        voice_results = session_state.get("voice_processing_results", [])
        conversation_state = session_state.get("conversation_state", {})
        
        if not voice_results:
            return {
                "total_voice_interactions": 0,
                "average_processing_time": 0.0,
                "success_rate": 0.0,
                "voice_mode_usage": 0.0
            }
        
        successful_results = [r for r in voice_results if r.get("success", False)]
        total_processing_time = sum(r.get("processing_time", 0) for r in successful_results)
        
        messages = conversation_state.get("messages", [])
        voice_messages = [m for m in messages if m.get("source") == "voice"]
        
        return {
            "total_voice_interactions": len(voice_results),
            "successful_interactions": len(successful_results),
            "average_processing_time": total_processing_time / max(len(successful_results), 1),
            "success_rate": len(successful_results) / len(voice_results),
            "voice_mode_usage": len(voice_messages) / max(len(messages), 1),
            "last_interaction_time": max(
                (r.get("timestamp", 0) for r in voice_results),
                default=0.0
            )
        }
    
    return calculate_voice_metrics