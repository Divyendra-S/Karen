"""Functional Text-to-Speech service for voice responses.

This module provides a pure functional interface for converting AI text responses
to speech audio using Google TTS, following functional programming principles.
"""

import asyncio
import tempfile
import os
import time
from typing import Optional, Callable, Dict, Any
from functools import lru_cache
from dataclasses import dataclass

from gtts import gTTS
import pygame
from loguru import logger

from core.types.voice_types import ResponseFrame


@dataclass(frozen=True)
class TTSConfig:
    """Immutable TTS configuration."""
    language: str = "en"
    slow_speech: bool = False
    voice_speed: float = 1.0
    volume: float = 0.8
    
    def with_speed(self, speed: float) -> 'TTSConfig':
        """Return new config with different speech speed."""
        return dataclass.replace(self, voice_speed=speed)
    
    def with_volume(self, volume: float) -> 'TTSConfig':
        """Return new config with different volume."""
        return dataclass.replace(self, volume=volume)


def create_tts_processor(config: TTSConfig) -> Callable[[str], bytes]:
    """Create TTS processor function."""
    
    def process_text_to_speech(text: str) -> Optional[bytes]:
        """Pure function to convert text to speech audio."""
        if not text or len(text.strip()) == 0:
            return None
        
        try:
            # Clean text for better TTS
            cleaned_text = clean_text_for_speech(text)
            
            # Create TTS
            tts = gTTS(
                text=cleaned_text,
                lang=config.language,
                slow=config.slow_speech
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                tts.save(temp_file.name)
                temp_file_path = temp_file.name
            
            # Read audio bytes
            with open(temp_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"TTS processing error: {e}")
            return None
    
    return process_text_to_speech


def clean_text_for_speech(text: str) -> str:
    """Pure function to clean text for better speech synthesis."""
    # Remove markdown formatting
    cleaned = text.replace('*', '').replace('#', '').replace('`', '')
    
    # Replace abbreviations with spoken equivalents
    replacements = {
        'e.g.': 'for example',
        'i.e.': 'that is',
        'etc.': 'and so on',
        '&': 'and',
        '@': 'at',
        '#': 'number',
        '%': 'percent',
        '$': 'dollars'
    }
    
    for abbrev, spoken in replacements.items():
        cleaned = cleaned.replace(abbrev, spoken)
    
    # Ensure proper sentence endings for natural speech
    cleaned = cleaned.strip()
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    
    return cleaned


def create_audio_player() -> Callable[[bytes], bool]:
    """Create audio player function."""
    
    def play_audio_bytes(audio_bytes: bytes) -> bool:
        """Pure function to play audio bytes."""
        if not audio_bytes:
            return False
        
        try:
            # Initialize pygame mixer if not already done
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Save bytes to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Load and play audio
                pygame.mixer.music.load(temp_file_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                return True
                
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False
    
    return play_audio_bytes


class FunctionalTTSService:
    """Functional text-to-speech service."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.tts_processor = create_tts_processor(config)
        self.audio_player = create_audio_player()
    
    async def speak_response(self, response: ResponseFrame) -> bool:
        """Convert response to speech and play it."""
        try:
            # Generate speech audio
            audio_bytes = await asyncio.to_thread(
                self.tts_processor,
                response.text
            )
            
            if audio_bytes:
                # Play audio
                return await asyncio.to_thread(
                    self.audio_player,
                    audio_bytes
                )
            
            return False
            
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            return False
    
    async def speak_text(self, text: str) -> bool:
        """Directly speak text."""
        if not text:
            return False
        
        response_frame = ResponseFrame(
            text=text,
            audio_data=None,
            timestamp=time.time(),
            metadata={}
        )
        
        return await self.speak_response(response_frame)
    
    def with_config(self, new_config: TTSConfig) -> 'FunctionalTTSService':
        """Create new TTS service with different config."""
        return FunctionalTTSService(new_config)


def create_tts_service_factory() -> Callable[[Dict[str, Any]], FunctionalTTSService]:
    """Factory function to create TTS service."""
    
    def create_service(config_dict: Dict[str, Any]) -> FunctionalTTSService:
        """Create TTS service from configuration."""
        
        tts_config = TTSConfig(
            language=config_dict.get("language", "en"),
            slow_speech=config_dict.get("slow_speech", False),
            voice_speed=float(config_dict.get("voice_speed", 1.0)),
            volume=float(config_dict.get("volume", 0.8))
        )
        
        return FunctionalTTSService(tts_config)
    
    return create_service


def create_voice_response_handler() -> Callable:
    """Create handler for voice responses with TTS."""
    
    def handle_voice_response(
        ai_response: str,
        tts_service: FunctionalTTSService,
        should_speak: bool = True
    ) -> Dict[str, Any]:
        """Handle AI response with optional speech output."""
        
        result = {
            "text": ai_response,
            "spoken": False,
            "error": None
        }
        
        if should_speak and tts_service:
            try:
                # Speak the response
                spoken = asyncio.run(tts_service.speak_text(ai_response))
                result["spoken"] = spoken
                
                if spoken:
                    logger.info(f"AI response spoken: '{ai_response[:50]}...'")
                else:
                    logger.warning("Failed to speak AI response")
                    
            except Exception as e:
                result["error"] = str(e)
                logger.error(f"TTS error: {e}")
        
        return result
    
    return handle_voice_response


# Streamlit integration for voice output
def create_streamlit_voice_output() -> Dict[str, Callable]:
    """Create Streamlit integration for voice output."""
    
    def initialize_tts_service() -> FunctionalTTSService:
        """Initialize TTS service for Streamlit."""
        factory = create_tts_service_factory()
        return factory({
            "language": "en",
            "slow_speech": False,
            "voice_speed": 1.0,
            "volume": 0.8
        })
    
    def play_ai_response(
        response_text: str,
        tts_service: FunctionalTTSService
    ) -> bool:
        """Play AI response as speech in Streamlit."""
        if not response_text or not tts_service:
            return False
        
        try:
            return asyncio.run(tts_service.speak_text(response_text))
        except Exception as e:
            logger.error(f"Streamlit TTS error: {e}")
            return False
    
    def create_voice_message_with_speech(
        text: str,
        tts_service: FunctionalTTSService,
        auto_play: bool = True
    ) -> Dict[str, Any]:
        """Create voice message with optional auto-play."""
        
        result = {
            "text": text,
            "timestamp": time.time(),
            "has_audio": False,
            "played": False
        }
        
        if auto_play and tts_service:
            # Generate and play speech
            played = play_ai_response(text, tts_service)
            result["played"] = played
            result["has_audio"] = True
        
        return result
    
    return {
        "initialize_tts": initialize_tts_service,
        "play_response": play_ai_response,
        "create_voice_message": create_voice_message_with_speech
    }


# Test the TTS functionality
def test_tts_functionality():
    """Test text-to-speech functionality."""
    print("🔊 Testing Text-to-Speech Functionality")
    print("-" * 40)
    
    # Create TTS service
    factory = create_tts_service_factory()
    tts_service = factory({
        "language": "en",
        "voice_speed": 1.2,  # Slightly faster
        "volume": 0.8
    })
    
    print("✅ TTS service created")
    
    # Test speech generation
    test_texts = [
        "Great! Which department will this role be in?",
        "Perfect! How many years of experience should candidates have?",
        "Excellent! Is this a full-time position?"
    ]
    
    async def test_speech():
        for i, text in enumerate(test_texts, 1):
            print(f"{i}. 🤖 Speaking: '{text}'")
            
            start_time = time.time()
            success = await tts_service.speak_text(text)
            duration = time.time() - start_time
            
            if success:
                print(f"   ✅ Spoken successfully in {duration:.1f}s")
            else:
                print(f"   ❌ Speech failed")
            
            # Small pause between utterances
            await asyncio.sleep(1)
    
    try:
        asyncio.run(test_speech())
        print("\n🎉 TTS test completed!")
        return True
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False


if __name__ == "__main__":
    test_tts_functionality()