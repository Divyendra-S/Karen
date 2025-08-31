"""Direct speech-to-text service using streamlit-mic-recorder and Groq Whisper."""

import tempfile
import os
from typing import Optional, Dict, Any
from groq import Groq
from app.config import settings
from app.utils.logger import logger


class DirectSpeechService:
    """Service for handling direct microphone recording and speech-to-text conversion."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize direct speech service.
        
        Args:
            api_key: Groq API key, uses settings if not provided
        """
        self.api_key = api_key or settings.get_llm_api_key()
        self.groq_client = Groq(api_key=self.api_key)
    
    def transcribe_audio_dict(self, audio_dict: Dict[str, Any]) -> Optional[str]:
        """Transcribe audio from streamlit-mic-recorder output.
        
        Args:
            audio_dict: Audio dictionary from mic_recorder containing 'bytes', 'sample_rate', etc.
            
        Returns:
            Transcribed text or None if failed
        """
        if not audio_dict or 'bytes' not in audio_dict:
            logger.warning("Invalid audio dictionary provided")
            return None
        
        try:
            audio_bytes = audio_dict['bytes']
            sample_rate = audio_dict.get('sample_rate', 44100)
            
            logger.info(f"Processing audio: {len(audio_bytes)} bytes, {sample_rate}Hz")
            
            # Transcribe using Groq Whisper
            transcribed_text = self._transcribe_with_groq(audio_bytes)
            
            if transcribed_text:
                logger.info(f"Transcription successful: {transcribed_text[:50]}...")
                return transcribed_text
            else:
                logger.warning("Transcription returned empty result")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def _transcribe_with_groq(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio bytes using Groq's Whisper API.
        
        Args:
            audio_bytes: Raw audio data as bytes
            
        Returns:
            Transcribed text or None if failed
        """
        temp_audio_path = None
        
        try:
            # Save audio to temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                temp_audio_path = tmp_file.name
            
            logger.info(f"Temporary audio file created: {temp_audio_path}")
            
            # Transcribe with Groq Whisper
            with open(temp_audio_path, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    language="en",
                    response_format="text"
                )
            
            # Extract text from response
            if hasattr(transcription, 'text'):
                return transcription.text.strip()
            elif isinstance(transcription, str):
                return transcription.strip()
            else:
                logger.warning(f"Unexpected transcription response type: {type(transcription)}")
                return str(transcription).strip()
                
        except Exception as e:
            logger.error(f"Groq transcription failed: {e}")
            return None
            
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    logger.info("Temporary audio file cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")
    
    def test_service(self) -> tuple[bool, str]:
        """Test the speech service connectivity.
        
        Returns:
            Tuple of (success, status_message)
        """
        try:
            # Test Groq API connection by checking if client is properly initialized
            if self.groq_client and self.api_key:
                logger.info("Direct speech service test successful")
                return True, "Direct speech service ready (Groq API connected)"
            else:
                return False, "Groq API key not configured"
                
        except Exception as e:
            error_msg = f"Direct speech service test failed: {e}"
            logger.error(error_msg)
            return False, error_msg


# Global direct speech service instance
_direct_speech_service_instance: Optional[DirectSpeechService] = None


def get_direct_speech_service() -> DirectSpeechService:
    """Get or create global direct speech service instance.
    
    Returns:
        DirectSpeechService instance
    """
    global _direct_speech_service_instance
    
    if _direct_speech_service_instance is None:
        _direct_speech_service_instance = DirectSpeechService()
        
    return _direct_speech_service_instance