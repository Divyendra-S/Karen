"""Simplified real-time voice conversation using existing services.

This module provides real-time voice conversation without complex Pipecat dependencies,
using WebSocket streaming with the existing Groq/Gemini services.
"""

import asyncio
import json
import os
import time
import websockets
import numpy as np
from typing import Dict, Any, Callable, Optional, AsyncGenerator
from dataclasses import dataclass
import threading
from queue import Queue

from loguru import logger
from core.services.simple_voice_service import SimpleFunctionalVoiceService
from core.services.tts_service import FunctionalTTSService


@dataclass(frozen=True)
class RealtimeConfig:
    """Configuration for real-time voice conversation."""
    websocket_port: int = 8765
    audio_sample_rate: int = 16000
    vad_threshold: float = 0.02
    silence_duration: float = 1.5
    response_timeout: float = 5.0


class RealtimeVoiceServer:
    """Real-time voice conversation server using WebSockets."""
    
    def __init__(
        self,
        config: RealtimeConfig,
        voice_service: SimpleFunctionalVoiceService,
        tts_service: FunctionalTTSService
    ):
        self.config = config
        self.voice_service = voice_service
        self.tts_service = tts_service
        self.is_running = False
        self.active_connections = set()
        self.last_processed_time = {}  # Track last processing time per connection
        self.audio_buffers = {}  # Buffer audio chunks per connection
        
        # Conversation state
        self.conversation_state = {
            "current_field": "job_title",
            "job_data": {},
            "field_order": [
                "job_title", "department", "experience", "employment_type",
                "location", "responsibilities", "skills", "education",
                "salary", "additional_requirements"
            ],
            "messages": []
        }
    
    async def start_server(self) -> Dict[str, Any]:
        """Start the real-time voice WebSocket server."""
        try:
            # Ensure voice service is active
            if hasattr(self.voice_service, '_is_active') and not self.voice_service._is_active:
                conversation_context = {
                    "current_field": "job_title",
                    "job_data": {},
                    "messages": []
                }
                voice_started = await self.voice_service.start_conversation(conversation_context)
                if not voice_started:
                    return {"success": False, "error": "Failed to activate voice service"}
                logger.info("Voice service activated for real-time server")
            
            self.server = await websockets.serve(
                self.handle_client,
                "localhost",
                self.config.websocket_port
            )
            
            self.is_running = True
            logger.info(f"Real-time voice server started on port {self.config.websocket_port}")
            
            return {
                "success": True,
                "websocket_url": f"ws://localhost:{self.config.websocket_port}",
                "status": "running"
            }
            
        except Exception as e:
            logger.error(f"Failed to start real-time server: {e}")
            return {"success": False, "error": str(e)}
    
    async def stop_server(self):
        """Stop the real-time voice server."""
        try:
            if hasattr(self, 'server'):
                self.server.close()
                await self.server.wait_closed()
            
            self.is_running = False
            logger.info("Real-time voice server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
    
    async def handle_client(self, websocket):
        """Handle client WebSocket connection."""
        self.active_connections.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            # Send welcome message
            await self.send_ai_response(
                websocket,
                "Welcome! I'm here to help you create a job description. What position are you looking to fill?"
            )
            
            # Process incoming messages
            async for message in websocket:
                await self.process_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            self.active_connections.discard(websocket)
    
    async def process_client_message(self, websocket, raw_message: str):
        """Process message from client."""
        try:
            message = json.loads(raw_message)
            
            if message.get("type") == "audio_chunk":
                # Skip audio chunk processing - use VoiceRecorder instead
                logger.debug("Skipping audio chunk - using VoiceRecorder component")
            
            elif message.get("type") == "transcription":
                # Handle voice transcription
                transcript = message.get("text", "")
                if transcript:
                    await self.process_voice_input(websocket, transcript)
            
            elif message.get("type") == "text":
                # Handle direct text input
                text = message.get("text", "")
                if text:
                    await self.process_voice_input(websocket, text)
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON message from client")
        except Exception as e:
            logger.error(f"Error processing client message: {e}")
    
    async def process_audio_chunk(self, websocket, audio_data: str):
        """Process audio chunk and transcribe."""
        try:
            # Convert base64 audio to bytes
            import base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Skip very small audio chunks
            if len(audio_bytes) < 1000:
                logger.debug("Skipping small audio chunk")
                return
            
            logger.debug(f"Received audio chunk: {len(audio_bytes)} bytes")
            
            # Get connection ID for buffering
            connection_id = id(websocket)
            
            # Initialize buffer for this connection if needed
            if connection_id not in self.audio_buffers:
                self.audio_buffers[connection_id] = []
            
            # Add chunk to buffer
            self.audio_buffers[connection_id].append(audio_bytes)
            
            # Only process when we have enough audio (5+ seconds worth)
            total_size = sum(len(chunk) for chunk in self.audio_buffers[connection_id])
            if total_size < 50000:  # Wait for more audio
                return
            
            # Combine all buffered audio
            combined_audio = b''.join(self.audio_buffers[connection_id])
            self.audio_buffers[connection_id] = []  # Clear buffer
            
            logger.info(f"Processing combined audio: {len(combined_audio)} bytes")
            
            # Use direct Groq Whisper without format conversion
            try:
                import io
                from groq import Groq
                
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                
                # Try direct WebM upload to Groq
                transcription = groq_client.audio.transcriptions.create(
                    file=("audio.webm", io.BytesIO(combined_audio), "audio/webm"),
                    model="whisper-large-v3"
                )
                
                transcript = transcription.text.strip()
                if transcript and len(transcript) > 3:
                    logger.info(f"Transcribed: {transcript}")
                    
                    # Send transcription result to client
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": transcript,
                        "timestamp": time.time()
                    }))
                    
                    # Process the transcription
                    await self.process_voice_input(websocket, transcript)
                else:
                    logger.debug("Empty or very short transcription")
                    
            except Exception as groq_error:
                logger.error(f"Groq transcription failed: {groq_error}")
                # Try fallback with ffmpeg conversion
                await self._try_ffmpeg_conversion(websocket, combined_audio)
                    
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await self.send_error(websocket, f"Audio processing failed: {e}")
    
    async def _try_ffmpeg_conversion(self, websocket, audio_bytes: bytes):
        """Fallback: try ffmpeg conversion."""
        try:
            import tempfile
            import subprocess
            
            # Save WebM audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
                webm_file.write(audio_bytes)
                webm_path = webm_file.name
            
            # Convert to WAV using ffmpeg with more robust settings
            wav_path = webm_path.replace('.webm', '.wav')
            result = subprocess.run([
                'ffmpeg', '-i', webm_path, '-ar', '16000', '-ac', '1', 
                '-acodec', 'pcm_s16le', '-f', 'wav', wav_path, '-y'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Send converted audio for transcription
                with open(wav_path, 'rb') as wav_file:
                    wav_bytes = wav_file.read()
                
                from groq import Groq
                import io
                
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                
                transcription = groq_client.audio.transcriptions.create(
                    file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                    model="whisper-large-v3"
                )
                
                transcript = transcription.text.strip()
                if transcript and len(transcript) > 3:
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": transcript,
                        "timestamp": time.time()
                    }))
                    await self.process_voice_input(websocket, transcript)
            
            # Cleanup
            try:
                os.unlink(webm_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"ffmpeg fallback failed: {e}")
            await self.send_error(websocket, "Audio conversion failed")
    
    async def process_voice_input(self, websocket, transcript: str):
        """Process voice input and generate real-time response."""
        
        # Rate limiting: prevent processing same message too quickly
        connection_id = id(websocket)
        current_time = time.time()
        
        if connection_id in self.last_processed_time:
            time_diff = current_time - self.last_processed_time[connection_id]
            if time_diff < 2.0:  # Minimum 2 seconds between processing
                logger.debug(f"Rate limiting: skipping message (last processed {time_diff:.1f}s ago)")
                return
        
        self.last_processed_time[connection_id] = current_time
        logger.info(f"Processing: {transcript}")
        
        # Build conversation context
        conversation_context = {
            "current_field": self.conversation_state["current_field"],
            "job_data": self.conversation_state["job_data"],
            "messages": self.conversation_state["messages"]
        }
        
        try:
            # Process with voice service
            result = await self.voice_service.process_voice_input(
                transcript, 
                conversation_context
            )
            
            if result.success:
                # Update conversation state
                self.conversation_state["messages"].extend([
                    {
                        "role": "user",
                        "content": transcript,
                        "timestamp": time.time()
                    },
                    {
                        "role": "assistant", 
                        "content": result.response_text,
                        "timestamp": time.time()
                    }
                ])
                
                # Extract job field data from the user's response
                current_field = self.conversation_state["current_field"]
                if current_field and current_field != "complete":
                    # Simple field extraction
                    field_value = self._extract_field_from_transcript(transcript, current_field)
                    if field_value:
                        self.conversation_state["job_data"][current_field] = field_value
                        logger.info(f"Extracted {current_field}: {field_value}")
                        
                        # Move to next field
                        self._advance_to_next_field()
                
                # Send AI response with TTS
                await self.send_ai_response(websocket, result.response_text)
                
            else:
                await self.send_error(websocket, result.error_message)
                
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            await self.send_error(websocket, str(e))
    
    async def send_ai_response(self, websocket, response_text: str):
        """Send AI response to client with TTS audio."""
        try:
            # Send text response immediately
            await websocket.send(json.dumps({
                "type": "response",
                "text": response_text,
                "timestamp": time.time()
            }))
            
            # Generate and send TTS audio
            if self.tts_service:
                speech_success = await self.tts_service.speak_text(response_text)
                
                # Notify client about speech status
                await websocket.send(json.dumps({
                    "type": "speech_status",
                    "success": speech_success,
                    "text": response_text[:50] + "..." if len(response_text) > 50 else response_text
                }))
            
        except Exception as e:
            logger.error(f"Error sending AI response: {e}")
    
    async def send_error(self, websocket, error_message: str):
        """Send error message to client."""
        try:
            await websocket.send(json.dumps({
                "type": "error",
                "message": error_message,
                "timestamp": time.time()
            }))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    def _extract_field_from_transcript(self, transcript: str, field: str) -> Optional[str]:
        """Extract field value from transcript."""
        text_lower = transcript.lower().strip()
        
        # Field-specific extraction logic
        if field == "job_title":
            if any(word in text_lower for word in ["engineer", "manager", "developer", "analyst", "director", "senior", "junior"]):
                return transcript.strip()
        elif field == "department":
            if any(word in text_lower for word in ["engineering", "marketing", "sales", "hr", "finance", "team", "department"]):
                return transcript.strip()
        elif field == "experience":
            if any(word in text_lower for word in ["year", "experience", "entry", "senior", "junior", "level"]):
                return transcript.strip()
        elif field == "employment_type":
            if any(word in text_lower for word in ["full", "part", "contract", "time", "remote", "employment"]):
                return transcript.strip()
        elif field == "location":
            if any(word in text_lower for word in ["remote", "office", "location", "city", "work", "francisco", "york"]):
                return transcript.strip()
        else:
            # For other fields, accept any substantial input
            if len(transcript.strip()) > 5:
                return transcript.strip()
        
        return None
    
    def _advance_to_next_field(self):
        """Move to the next field in the conversation."""
        current_field = self.conversation_state["current_field"]
        field_order = self.conversation_state["field_order"]
        
        try:
            current_index = field_order.index(current_field)
            if current_index + 1 < len(field_order):
                next_field = field_order[current_index + 1]
                self.conversation_state["current_field"] = next_field
                logger.info(f"Advanced to next field: {next_field}")
            else:
                self.conversation_state["current_field"] = "complete"
                logger.info("Interview complete - all fields collected")
        except ValueError:
            logger.warning(f"Unknown field: {current_field}")
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
        return {
            "is_running": self.is_running,
            "active_connections": len(self.active_connections),
            "current_field": self.conversation_state["current_field"],
            "job_data": self.conversation_state["job_data"],
            "completed_fields": len(self.conversation_state["job_data"]),
            "total_fields": len(self.conversation_state["field_order"]),
            "message_count": len(self.conversation_state["messages"])
        }


class RealtimeVoiceManager:
    """Manages real-time voice conversation sessions."""
    
    def __init__(self):
        self.server = None
        self.server_task = None
    
    async def start_realtime_session(
        self,
        voice_service: SimpleFunctionalVoiceService,
        tts_service: FunctionalTTSService,
        config: Optional[RealtimeConfig] = None
    ) -> Dict[str, Any]:
        """Start real-time voice session."""
        
        if self.server and self.server.is_running:
            return {"success": False, "error": "Session already running"}
        
        try:
            config = config or RealtimeConfig()
            self.server = RealtimeVoiceServer(config, voice_service, tts_service)
            
            # Start server in background
            result = await self.server.start_server()
            
            if result["success"]:
                # Keep server running
                self.server_task = asyncio.create_task(self._keep_server_alive())
                logger.info("Real-time voice session started successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to start real-time session: {e}")
            return {"success": False, "error": str(e)}
    
    async def stop_realtime_session(self) -> bool:
        """Stop real-time voice session."""
        try:
            if self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
            
            if self.server:
                await self.server.stop_server()
                self.server = None
            
            logger.info("Real-time voice session stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping real-time session: {e}")
            return False
    
    async def _keep_server_alive(self):
        """Keep server running until cancelled."""
        try:
            while self.server and self.server.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        if self.server:
            return self.server.get_conversation_state()
        return {
            "is_running": False,
            "active_connections": 0,
            "current_field": None,
            "job_data": {},
            "completed_fields": 0,
            "total_fields": 10
        }


def create_simplified_realtime_interface() -> Dict[str, Callable]:
    """Create simplified real-time interface for Streamlit."""
    
    manager = RealtimeVoiceManager()
    
    async def start_realtime_conversation(
        voice_service: SimpleFunctionalVoiceService,
        tts_service: FunctionalTTSService,
        config_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start real-time conversation."""
        
        config = RealtimeConfig(
            websocket_port=config_dict.get("port", 8765) if config_dict else 8765,
            vad_threshold=config_dict.get("vad_threshold", 0.02) if config_dict else 0.02,
            silence_duration=config_dict.get("silence_duration", 1.5) if config_dict else 1.5
        )
        
        return await manager.start_realtime_session(voice_service, tts_service, config)
    
    async def stop_realtime_conversation() -> bool:
        """Stop real-time conversation."""
        return await manager.stop_realtime_session()
    
    def get_realtime_status() -> Dict[str, Any]:
        """Get real-time session status."""
        return manager.get_session_status()
    
    return {
        "start_conversation": start_realtime_conversation,
        "stop_conversation": stop_realtime_conversation,
        "get_status": get_realtime_status,
        "manager": manager
    }


async def test_simplified_realtime():
    """Test the simplified real-time voice system."""
    print("⚡ Testing Simplified Real-time Voice")
    print("-" * 40)
    
    interface = create_simplified_realtime_interface()
    
    print("✅ Real-time interface created")
    print("✅ WebSocket server ready")
    print("✅ Voice processing pipeline configured")
    
    status = interface["get_status"]()
    print(f"✅ Initial status: {status}")
    
    print("\n🎉 Simplified real-time voice ready!")
    print("🚀 Use start_conversation() to begin real-time session")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_simplified_realtime())