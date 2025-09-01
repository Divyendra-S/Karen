#!/usr/bin/env python3
"""Simple transcription server for VoiceRecorder component."""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import io

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from groq import Groq
from loguru import logger

app = Flask(__name__)
CORS(app)

# Initialize Groq client
groq_client = None

def init_groq():
    global groq_client
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
        logger.info("Groq client initialized")
        return True
    else:
        logger.error("GROQ_API_KEY not found")
        return False

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file using Groq Whisper."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        logger.info(f"Received audio file: {len(audio_data)} bytes, type: {audio_file.content_type}")
        
        if not groq_client:
            return jsonify({'error': 'Groq client not initialized'}), 500
        
        # Use Groq Whisper for transcription
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, io.BytesIO(audio_data), audio_file.content_type),
            model="whisper-large-v3"
        )
        
        transcript = transcription.text.strip()
        logger.info(f"Transcribed: {transcript}")
        
        return jsonify({
            'transcript': transcript,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'groq_available': groq_client is not None
    })

if __name__ == '__main__':
    print("🎤 Starting Transcription Server")
    print("-" * 30)
    
    if init_groq():
        print("✅ Groq client ready")
        print("🚀 Starting server on http://localhost:8504")
        app.run(host='0.0.0.0', port=8504, debug=False)
    else:
        print("❌ Failed to initialize Groq client")
        sys.exit(1)