#!/usr/bin/env python3
"""
Simple HTTP server to test streaming with curl
"""

import time
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

class StreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        # Simulate download progress
        messages = [
            "📁 File: angelo-custode.ogg (3.9MB)\n",
            "🤖 Model: distil-large-v3\n\n",
            "📦 Model 'distil-large-v3' already cached (1.2GB)\n",
            "🔄 Loading cached model...\n",
            "📥 Downloaded: 50.0MB (4.2%)\n",
            "📥 Downloaded: 100.0MB (8.3%)\n",
            "📥 Downloaded: 150.0MB (12.5%)\n",
            "📥 Downloaded: 200.0MB (16.7%)\n",
            "📥 Downloaded: 250.0MB (20.8%)\n",
            "📥 Downloaded: 300.0MB (25.0%)\n",
            "📥 Downloaded: 350.0MB (29.2%)\n",
            "📥 Downloaded: 400.0MB (33.3%)\n",
            "📥 Downloaded: 450.0MB (37.5%)\n",
            "📥 Downloaded: 500.0MB (41.7%)\n",
            "📥 Downloaded: 550.0MB (45.8%)\n",
            "📥 Downloaded: 600.0MB (50.0%)\n",
            "📥 Downloaded: 650.0MB (54.2%)\n",
            "📥 Downloaded: 700.0MB (58.3%)\n",
            "📥 Downloaded: 750.0MB (62.5%)\n",
            "📥 Downloaded: 800.0MB (66.7%)\n",
            "📥 Downloaded: 850.0MB (70.8%)\n",
            "📥 Downloaded: 900.0MB (75.0%)\n",
            "📥 Downloaded: 950.0MB (79.2%)\n",
            "📥 Downloaded: 1000.0MB (83.3%)\n",
            "📥 Downloaded: 1050.0MB (87.5%)\n",
            "📥 Downloaded: 1100.0MB (91.7%)\n",
            "📥 Downloaded: 1150.0MB (95.8%)\n",
            "📥 Downloaded: 1200.0MB (100.0%)\n",
            "✅ Model 'distil-large-v3' loaded successfully!\n\n",
            "🎯 Starting transcription...\n",
            "✅ Transcription completed!\n",
            "⏱️ Processing time: 25.47s\n",
            "⚡ Real-time factor: 8.2x\n\n",
            "="*50 + "\n",
            "📝 TRANSCRIPTION RESULT:\n",
            "="*50 + "\n\n",
            "Benissimo, allora dire di procedere in questo modo...\n"
        ]

        for i, message in enumerate(messages):
            self.wfile.write(message.encode('utf-8'))
            self.wfile.flush()

            # Different delays for different types of messages
            if "Downloaded:" in message:
                time.sleep(1.0)  # 1 second for each download chunk
            elif message.startswith("🔄") or message.startswith("🎯"):
                time.sleep(0.5)  # Short delay for status messages
            elif message.startswith("✅"):
                time.sleep(0.2)  # Quick for completion messages
            else:
                time.sleep(0.1)  # Very quick for other messages

    def log_message(self, format, *args):
        # Suppress default logging
        pass

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8001), StreamHandler)
    print("🚀 Starting test server on http://localhost:8001")
    print("📞 Test with: curl --no-buffer -X POST http://localhost:8001/")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.shutdown()