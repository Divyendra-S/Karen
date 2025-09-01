'use client';

import { useState, useRef, useCallback } from 'react';

interface VoiceRecorderProps {
  onTranscript: (transcript: string) => void;
  isConnected: boolean;
}

export default function VoiceRecorder({ onTranscript, isConnected }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      streamRef.current = stream;

      // Setup audio level monitoring
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      analyser.fftSize = 256;
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      // Monitor audio levels
      const updateAudioLevel = () => {
        if (analyser) {
          analyser.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((acc, val) => acc + val, 0) / bufferLength;
          setAudioLevel(average / 255);
          
          if (isRecording) {
            requestAnimationFrame(updateAudioLevel);
          }
        }
      };

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      const audioChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        
        // Send to server for transcription (using existing proven method)
        try {
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.webm');
          
          const response = await fetch('http://localhost:8504/transcribe', {
            method: 'POST',
            body: formData
          });
          
          if (response.ok) {
            const result = await response.json();
            if (result.transcript) {
              onTranscript(result.transcript);
            }
          } else {
            console.error('Transcription failed:', response.statusText);
          }
        } catch (error) {
          console.error('Transcription request failed:', error);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
      updateAudioLevel();

    } catch (error) {
      console.error('Failed to start recording:', error);
    }
  }, [onTranscript, isRecording]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    setIsRecording(false);
    setAudioLevel(0);
  }, []);

  const getRecordingStyle = () => {
    if (!isRecording) return 'bg-green-500 hover:bg-green-600';
    
    const intensity = Math.floor(audioLevel * 5);
    const colors = [
      'bg-red-300',
      'bg-red-400', 
      'bg-red-500',
      'bg-red-600',
      'bg-red-700'
    ];
    
    return colors[intensity] || 'bg-red-500';
  };

  return (
    <div className="flex items-center gap-4">
      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={!isConnected}
        className={`px-6 py-3 text-white rounded-lg transition-all duration-200 ${getRecordingStyle()} disabled:bg-gray-400`}
      >
        {isRecording ? '⏹️ Stop Recording' : '🎤 Start Recording'}
      </button>
      
      {isRecording && (
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse" />
          <span className="text-sm text-red-600 font-medium">Recording...</span>
          <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-red-500 transition-all duration-100"
              style={{ width: `${audioLevel * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}