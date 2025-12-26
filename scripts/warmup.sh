#!/bin/bash
#
# Ollama Warmup Script
# This script ensures Ollama is running with the llama3 model loaded into memory,
# making all subsequent lab operations faster.
#

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="llama3"
MAX_WAIT_SERVER=60
MAX_WAIT_PULL=300

echo "=== Ollama Warmup Script ==="
echo ""

# Function to check if Ollama server is responding
check_server() {
    curl -s -o /dev/null -w "%{http_code}" "${OLLAMA_HOST}/api/tags" 2>/dev/null || echo "000"
}

# Function to check if model is available
check_model() {
    curl -s "${OLLAMA_HOST}/api/tags" 2>/dev/null | grep -q "\"${MODEL}\""
}

# Step 1: Check if Ollama is installed
echo "[1/5] Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "  -> Ollama not found, installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  -> Ollama installed successfully"
else
    echo "  -> Ollama is already installed"
fi

# Step 2: Start Ollama server if not running
echo ""
echo "[2/5] Checking Ollama server..."
if [ "$(check_server)" != "200" ]; then
    echo "  -> Starting Ollama server..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!

    # Wait for server to be ready
    echo "  -> Waiting for server to be ready..."
    WAITED=0
    while [ "$(check_server)" != "200" ] && [ $WAITED -lt $MAX_WAIT_SERVER ]; do
        sleep 1
        WAITED=$((WAITED + 1))
        printf "\r  -> Waiting... %ds" $WAITED
    done
    echo ""

    if [ "$(check_server)" != "200" ]; then
        echo "  -> ERROR: Server failed to start after ${MAX_WAIT_SERVER}s"
        exit 1
    fi
    echo "  -> Server is ready"
else
    echo "  -> Server is already running"
fi

# Step 3: Pull the model if not present
echo ""
echo "[3/5] Checking ${MODEL} model..."
if ! check_model; then
    echo "  -> Model not found, pulling ${MODEL}..."
    echo "  -> This may take several minutes on first run..."
    ollama pull ${MODEL}
    echo "  -> Model pulled successfully"
else
    echo "  -> Model ${MODEL} is already available"
fi

# Step 4: Warm up the model by sending a simple request
echo ""
echo "[4/5] Warming up ${MODEL} model (loading into memory)..."
echo "  -> Sending warmup request..."

WARMUP_START=$(date +%s)
RESPONSE=$(curl -s -X POST "${OLLAMA_HOST}/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL}"'",
        "prompt": "Reply with only the word: ready",
        "stream": false,
        "options": {
            "num_predict": 10
        }
    }')
WARMUP_END=$(date +%s)
WARMUP_TIME=$((WARMUP_END - WARMUP_START))

if echo "$RESPONSE" | grep -q '"response"'; then
    echo "  -> Model loaded and responding (warmup took ${WARMUP_TIME}s)"
else
    echo "  -> WARNING: Warmup request may have failed"
    echo "  -> Response: $RESPONSE"
fi

# Step 5: Verify everything is working
echo ""
echo "[5/5] Verifying setup..."

# Quick verification request
VERIFY_START=$(date +%s)
VERIFY_RESPONSE=$(curl -s -X POST "${OLLAMA_HOST}/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL}"'",
        "prompt": "Say hello",
        "stream": false,
        "options": {
            "num_predict": 5
        }
    }')
VERIFY_END=$(date +%s)
VERIFY_TIME=$((VERIFY_END - VERIFY_START))

if echo "$VERIFY_RESPONSE" | grep -q '"response"'; then
    echo "  -> Verification successful (response time: ${VERIFY_TIME}s)"
else
    echo "  -> WARNING: Verification may have failed"
fi

# Summary
echo ""
echo "=== Warmup Complete ==="
echo ""
echo "Ollama is ready with the following configuration:"
echo "  - Server: ${OLLAMA_HOST}"
echo "  - Model: ${MODEL}"
echo "  - Initial warmup time: ${WARMUP_TIME}s"
echo "  - Subsequent response time: ${VERIFY_TIME}s"
echo ""
echo "You can now run the labs with faster startup and response times!"
echo ""
echo "Quick test command:"
echo "  curl -s ${OLLAMA_HOST}/api/generate -d '{\"model\":\"${MODEL}\",\"prompt\":\"Hi\",\"stream\":false}'"
echo ""
