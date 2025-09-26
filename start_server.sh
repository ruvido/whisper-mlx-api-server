#!/bin/bash

# Lightning MLX Whisper API Startup Script
# Optimized for Apple Silicon with UV package manager

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Lightning MLX Whisper API${NC}"
echo -e "${BLUE}================================${NC}"

# Check if we're on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Not on Apple Silicon (${ARCH}). MLX performance will be limited.${NC}"
fi

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ UV not found. Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies if needed
echo -e "${BLUE}📦 Installing dependencies...${NC}"
export PATH="$HOME/.cargo/bin:$PATH"  # Add Rust to PATH for tiktoken
uv run --with lightning-whisper-mlx --with fastapi --with uvicorn --with python-multipart --with requests echo "Dependencies ready"

# Check MLX availability
echo -e "${BLUE}🔍 Checking MLX availability...${NC}"
if uv run python -c "import mlx.core as mx; print(f'✅ MLX available on {mx.default_device()}')" 2>/dev/null; then
    echo -e "${GREEN}✅ MLX hardware acceleration available${NC}"
else
    echo -e "${YELLOW}⚠️  MLX not available. Performance will be limited.${NC}"
fi

# Set environment variables
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-"info"}

echo -e "${BLUE}🌐 Starting server on ${HOST}:${PORT}${NC}"

# Check if port is available
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}⚠️  Port ${PORT} is already in use. Trying port $((PORT + 1))...${NC}"
    export PORT=$((PORT + 1))
fi

echo -e "${GREEN}🎯 API will be available at:${NC}"
echo -e "   ${BLUE}http://localhost:${PORT}${NC} (local)"
echo -e "   ${BLUE}http://$(hostname -I | awk '{print $1}'):${PORT}${NC} (network)"
echo ""
echo -e "${GREEN}📖 API Documentation:${NC}"
echo -e "   ${BLUE}http://localhost:${PORT}/docs${NC} (Swagger UI)"
echo -e "   ${BLUE}http://localhost:${PORT}/redoc${NC} (ReDoc)"
echo ""
echo -e "${GREEN}🔧 Test the API:${NC}"
echo -e "   ${BLUE}curl http://localhost:${PORT}/health${NC}"
echo ""

# Start the server
echo -e "${BLUE}🚀 Starting Lightning MLX Whisper API...${NC}"
export PATH="$HOME/.cargo/bin:$PATH"  # Ensure Rust is in PATH
uv run --with lightning-whisper-mlx --with fastapi --with uvicorn --with python-multipart --with requests \
    uvicorn whisper_api:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level "${LOG_LEVEL}" \
    --no-access-log \
    --workers 1