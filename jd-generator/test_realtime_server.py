#!/usr/bin/env python3
"""Test script to start the real-time voice server."""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.simplified_realtime_voice import create_simplified_realtime_interface
from core.services.simple_voice_service import create_simple_voice_service_factory
from core.services.tts_service import create_tts_service_factory
from loguru import logger

async def main():
    """Test starting real-time voice server."""
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not available, using system environment")
    
    # Get API keys
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment")
        return
    
    print("🚀 Starting Real-time Voice Server Test")
    print("-" * 40)
    
    try:
        # Create voice services
        voice_factory = create_simple_voice_service_factory()
        voice_service = voice_factory({
            "GOOGLE_API_KEY": google_api_key,
            "VOICE_MODEL": "Puck",
            "AUDIO_SAMPLE_RATE": "16000",
            "ENABLE_VOICE_MODE": "true"
        })
        
        # Activate the voice service
        conversation_context = {
            "current_field": "job_title",
            "job_data": {},
            "messages": []
        }
        voice_started = await voice_service.start_conversation(conversation_context)
        if not voice_started:
            logger.error("Failed to start voice service")
            return
        
        print("✅ Voice service activated")
        
        tts_factory = create_tts_service_factory()
        tts_service = tts_factory({
            "language": "en",
            "voice_speed": 1.1,
            "volume": 0.8
        })
        
        print("✅ Voice services created")
        
        # Create real-time interface
        realtime_interface = create_simplified_realtime_interface()
        
        print("✅ Real-time interface created")
        
        # Start real-time conversation
        result = await realtime_interface["start_conversation"](
            voice_service,
            tts_service,
            {"port": 8765}
        )
        
        if result["success"]:
            print(f"✅ Real-time server started: {result.get('websocket_url')}")
            print("🎤 Ready for WebSocket connections on port 8765")
            
            # Keep server running
            try:
                while True:
                    await asyncio.sleep(1)
                    status = realtime_interface["get_status"]()
                    if status["is_running"]:
                        if status["active_connections"] > 0:
                            print(f"🔊 Active connections: {status['active_connections']}")
                    else:
                        print("❌ Server stopped running")
                        break
                        
            except KeyboardInterrupt:
                print("\n⏹️ Stopping server...")
                await realtime_interface["stop_conversation"]()
                print("✅ Server stopped")
        else:
            print(f"❌ Failed to start server: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())