"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import VoiceRecorder from "./components/VoiceRecorder";

interface VoiceMessage {
  type: "user" | "assistant";
  text: string;
  timestamp: number;
}

interface ServerMessage {
  type: "response" | "transcription" | "speech_status" | "error";
  text?: string;
  message?: string;
  success?: boolean;
  timestamp: number;
}

export default function RealtimeVoiceClient() {
  const [isConnected, setIsConnected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [isClient, setIsClient] = useState(false);
  const [textInput, setTextInput] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    setIsClient(true);
    setMessages([
      {
        type: "assistant",
        text: "Welcome! I'll help you create a job description. Click connect to start our voice conversation.",
        timestamp: Date.now(),
      },
    ]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const connect = useCallback(async () => {
    try {
      console.log("Attempting to connect to ws://localhost:8765...");
      const ws = new WebSocket("ws://localhost:8765");

      ws.onopen = () => {
        setIsConnected(true);
        console.log("✅ Connected to real-time voice server");
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            text: "✅ Connected to real-time voice server!",
            timestamp: Date.now(),
          },
        ]);
      };

      ws.onmessage = (event) => {
        try {
          console.log("📨 Received message:", event.data);
          const data: ServerMessage = JSON.parse(event.data);
          handleServerMessage(data);
        } catch (error) {
          console.error("Error parsing server message:", error);
        }
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        console.log(
          "🔌 Disconnected from voice server. Code:",
          event.code,
          "Reason:",
          event.reason
        );
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            text: "🔌 Disconnected from voice server",
            timestamp: Date.now(),
          },
        ]);
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            text: "❌ WebSocket connection failed. Make sure the voice server is running.",
            timestamp: Date.now(),
          },
        ]);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("❌ Failed to connect:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: "❌ Failed to connect to voice server",
          timestamp: Date.now(),
        },
      ]);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const handleServerMessage = useCallback((data: ServerMessage) => {
    switch (data.type) {
      case "response":
        if (data.text) {
          setMessages((prev) => [
            ...prev,
            {
              type: "assistant",
              text: data.text as string,
              timestamp: data.timestamp || Date.now(),
            },
          ]);
        }
        break;

      case "transcription":
        if (data.text) {
          setMessages((prev) => [
            ...prev,
            {
              type: "user",
              text: data.text as string,
              timestamp: data.timestamp || Date.now(),
            },
          ]);
        }
        break;

      case "speech_status":
        if (data.success) {
          setIsSpeaking(true);
          // Reset speaking status and restart listening in continuous mode
          setTimeout(() => setIsSpeaking(false), 3000);
        }
        break;

      case "error":
        const errorText = data.message || "Unknown error";
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            text: `❌ Error: ${errorText}`,
            timestamp: data.timestamp || Date.now(),
          },
        ]);
        break;
    }
  }, []);

  const sendTestMessage = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "text",
          text: "I want to hire a Senior Software Engineer",
        })
      );
    }
  }, []);

  const sendTextMessage = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && textInput.trim()) {
      wsRef.current.send(
        JSON.stringify({
          type: "text",
          text: textInput.trim(),
        })
      );
      setTextInput("");
    }
  }, [textInput]);

  const handleTextKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendTextMessage();
      }
    },
    [sendTextMessage]
  );

  const handleVoiceTranscript = useCallback((transcript: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && transcript.trim()) {
      console.log("📝 Sending voice transcript:", transcript);
      wsRef.current.send(
        JSON.stringify({
          type: "text",
          text: transcript.trim(),
        })
      );
    }
  }, []);

  const getStatusColor = () => {
    if (!isConnected) return "bg-red-100 text-red-800 border-red-200";
    if (isSpeaking) return "bg-yellow-100 text-yellow-800 border-yellow-200";
    return "bg-green-100 text-green-800 border-green-200";
  };

  const getStatusText = () => {
    if (!isConnected) return "🔴 Disconnected";
    if (isSpeaking) return "🔊 AI Speaking";
    return "🟢 Connected - Ready for Voice";
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              ⚡ Real-time Voice JD Generator
            </h1>
            <p className="text-gray-600">
              Continuous voice conversation for creating job descriptions
            </p>
          </div>

          {/* Status */}
          <div className={`rounded-lg border p-4 mb-6 ${getStatusColor()}`}>
            <div className="font-semibold">{getStatusText()}</div>
            {isConnected && (
              <div className="text-sm mt-1">
                WebSocket connected to localhost:8765
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex flex-wrap gap-3 mb-6">
            <button
              onClick={connect}
              disabled={isConnected}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-400 hover:bg-blue-600 transition-colors"
            >
              🔗 Connect
            </button>

            <button
              onClick={disconnect}
              disabled={!isConnected}
              className="px-4 py-2 bg-red-500 text-white rounded-lg disabled:bg-gray-400 hover:bg-red-600 transition-colors"
            >
              ❌ Disconnect
            </button>

            <div className="flex items-center gap-2">
              <VoiceRecorder
                onTranscript={handleVoiceTranscript}
                isConnected={isConnected}
              />
              <span className="text-sm text-gray-600">
                ← Uses proven /transcribe endpoint
              </span>
            </div>

            <button
              onClick={sendTestMessage}
              disabled={!isConnected}
              className="px-4 py-2 bg-purple-500 text-white rounded-lg disabled:bg-gray-400 hover:bg-purple-600 transition-colors"
            >
              🧪 Test Message
            </button>
          </div>

          {/* Conversation */}
          <div className="bg-gray-50 rounded-lg p-4 h-96 overflow-y-auto mb-6">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.type === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.type === "user"
                        ? "bg-blue-500 text-white"
                        : "bg-white text-gray-800 border border-gray-200"
                    }`}
                  >
                    <div className="font-semibold text-sm mb-1">
                      {message.type === "user" ? "👤 You" : "🤖 AI"}
                    </div>
                    <div>{message.text}</div>
                    {isClient && (
                      <div className="text-xs opacity-70 mt-1">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <div ref={messagesEndRef} />
          </div>

          {/* Text Input Interface */}
          {isConnected && (
            <div className="mb-6 p-4 bg-green-50 rounded-lg border border-green-200">
              <h3 className="font-semibold text-green-800 mb-3">
                💬 Text Input (Alternative to Voice)
              </h3>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyDown={handleTextKeyPress}
                  placeholder="Type your message here and press Enter..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                  disabled={!isConnected}
                />
                <button
                  onClick={sendTextMessage}
                  disabled={!isConnected || !textInput.trim()}
                  className="px-4 py-2 bg-green-500 text-white rounded-lg disabled:bg-gray-400 hover:bg-green-600 transition-colors"
                >
                  Send
                </button>
              </div>
              <p className="text-sm text-green-600 mt-2">
                Use this text interface while we fix the voice recording issues
              </p>
            </div>
          )}

          {/* Instructions */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
            <h3 className="font-semibold text-blue-800 mb-2">
              🎯 How to Use Real-time Voice:
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-blue-700">
              <div>• Click "Connect" to establish WebSocket connection</div>
              <div>• Click "Start Listening" to begin voice conversation</div>
              <div>• Speak naturally about the job you want to create</div>
              <div>• AI responds instantly with voice and text</div>
              <div>• Adjust microphone sensitivity as needed</div>
              <div>• Use "Test Message" to verify connection works</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
