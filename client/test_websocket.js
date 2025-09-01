const WebSocket = require('ws');

console.log('🧪 Testing WebSocket connection to localhost:8765');

const ws = new WebSocket('ws://localhost:8765');

ws.on('open', function open() {
  console.log('✅ Connected to WebSocket server');
  
  // Send test message
  ws.send(JSON.stringify({
    type: 'text',
    text: 'I want to hire a Senior Software Engineer'
  }));
  
  console.log('📤 Sent test message');
});

ws.on('message', function message(data) {
  console.log('📨 Received:', data.toString());
});

ws.on('error', function error(err) {
  console.error('❌ WebSocket error:', err.message);
});

ws.on('close', function close() {
  console.log('🔌 Connection closed');
});

// Close after 10 seconds
setTimeout(() => {
  console.log('⏰ Closing connection...');
  ws.close();
}, 10000);