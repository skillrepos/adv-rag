#!/bin/bash
# startup_ollama.sh - Start and warm up Ollama for lab exercises

echo "========================================"
echo "Ollama Startup & Warmup Script"
echo "========================================"
echo ""

# Step 1: Check and install Ollama if needed
echo "Step 1: Checking for Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is already installed"
else
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
fi
echo ""

# Step 2: Start Ollama service
echo "Step 2: Starting Ollama service..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!
echo "✓ Ollama started (PID: $OLLAMA_PID)"
echo ""

# Wait for Ollama to be ready
echo "Step 3: Waiting for Ollama to be ready..."
sleep 3
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "  Waiting for Ollama server..."
    sleep 1
done
echo "✓ Ollama server is ready"
echo ""

# Step 4: Pull llama3.2:3b model if not present
echo "Step 4: Checking for llama3.2:3b model..."
if ollama list | grep -q "llama3.2:3b"; then
    echo "✓ llama3.2:3b model already present"
else
    echo "  Pulling llama3.2:3b model (this may take a few minutes)..."
    ollama pull llama3.2:3b
    echo "✓ llama3.2:3b model downloaded"
fi
echo ""

# Step 5: Warm up the model with a simple query
echo "Step 5: Warming up llama3.2:3b model..."
echo "  Running test inference to load model into memory..."
curl -s -X POST http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Hello",
  "stream": false
}' > /dev/null
echo "✓ Model warmed up and ready"
echo ""

# Step 6: Set up color wrapper and aliases for lab exercises
echo "Step 6: Setting up color aliases for lab exercises..."

# Create a dedicated file for ollama colors
cat > ~/.ollama_colors.sh << 'EOF'
# ======================================
# Ollama Color Wrapper for Labs
# ======================================

# Main wrapper function - reads color AND context from environment variables
ollama() {
    if [ "$1" = "run" ]; then
        local color="${OLLAMA_COLOR:-\033[1;37m}"
        local context="${OLLAMA_CONTEXT:-}"

        # Print context label if set
        if [ -n "$context" ]; then
            echo -e "\n${color}${context}\033[0m"
        fi

        # Print model output in color
        echo -e "${color}"
        command ollama "$@"
        echo -e "\033[0m"
    else
        command ollama "$@"
    fi
}

# Quick color switchers - just type the color name!
alias cyan="export OLLAMA_COLOR='\033[1;36m'; export OLLAMA_CONTEXT=''"
alias green="export OLLAMA_COLOR='\033[1;32m'; export OLLAMA_CONTEXT=''"
alias yellow="export OLLAMA_COLOR='\033[1;33m'; export OLLAMA_CONTEXT=''"
alias blue="export OLLAMA_COLOR='\033[1;34m'; export OLLAMA_CONTEXT=''"
alias magenta="export OLLAMA_COLOR='\033[1;35m'; export OLLAMA_CONTEXT=''"
alias white="export OLLAMA_COLOR='\033[1;37m'; export OLLAMA_CONTEXT=''"

# Discipline-specific colors with context labels
alias ds-color="export OLLAMA_COLOR='\033[1;34m'; export OLLAMA_CONTEXT='📊 Context: Data Science'"
alias ai-color="export OLLAMA_COLOR='\033[1;35m'; export OLLAMA_CONTEXT='📋 Context: Traditional AI'"
alias ml-color="export OLLAMA_COLOR='\033[1;32m'; export OLLAMA_CONTEXT='🎯 Context: Machine Learning'"
alias dl-color="export OLLAMA_COLOR='\033[1;36m'; export OLLAMA_CONTEXT='🧠 Context: Deep Learning'"
alias gen-color="export OLLAMA_COLOR='\033[1;33m'; export OLLAMA_CONTEXT='✨ Context: Generative AI'"
EOF

# Add source line to .bashrc if not already present
if ! grep -q ".ollama_colors.sh" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Source Ollama color aliases for labs" >> ~/.bashrc
    echo "[ -f ~/.ollama_colors.sh ] && source ~/.ollama_colors.sh" >> ~/.bashrc
    echo "✓ Color aliases configured in ~/.bashrc"
else
    echo "✓ Color aliases already configured in ~/.bashrc"
fi

# Source the file NOW to make aliases available immediately
source ~/.ollama_colors.sh
echo "✓ Color aliases loaded and ready to use"
echo ""

# Step 7: Display status
echo "========================================"
echo "Status: Ollama Ready for Labs"
echo "========================================"
echo ""
echo "Available models:"
ollama list
echo ""
echo "Ollama API endpoint: http://localhost:11434"
echo "Ollama PID: $OLLAMA_PID"
echo ""
echo "Color aliases available:"
echo "  ds-color  - Data Science (Blue)"
echo "  ai-color  - Traditional AI (Magenta)"
echo "  ml-color  - Machine Learning (Green)"
echo "  dl-color  - Deep Learning (Cyan)"
echo "  gen-color - Generative AI (Yellow)"
echo ""
echo "Usage: Type 'ds-color' then run your ollama commands"
echo "       Context labels will appear before model output"
echo ""
echo "To stop Ollama later, run:"
echo "  kill $OLLAMA_PID"
echo "  or use: pkill ollama"
echo ""

# Step 8: Stop Ollama process to allow postAttach command to restart it
echo "Stopping Ollama process ($OLLAMA_PID) so it can be managed by postAttach in devcontainer.json..."
kill $OLLAMA_PID
sleep 1
if ps -p $OLLAMA_PID > /dev/null 2>&1; then
    echo "  Ollama did not exit cleanly, forcing kill..."
    kill -9 $OLLAMA_PID
else
    echo "✓ Ollama process stopped"
fi
echo ""

echo "Ready for lab exercises! (Ollama will be started automatically by devcontainer postAttach)"
echo "========================================"
