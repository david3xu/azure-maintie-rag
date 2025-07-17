#!/bin/bash

# MaintIE Enhanced RAG Startup Script

echo "🚀 Starting MaintIE Enhanced RAG system..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,indices} logs

# Check for environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your OpenAI API key and other settings"
    exit 1
fi

# Check for raw text data (Universal RAG)
TEXT_FILES=$(find data/raw/ -name "*.txt" -o -name "*.md" 2>/dev/null | wc -l)
if [ "$TEXT_FILES" -eq 0 ]; then
    echo "⚠️  No text data found in data/raw/"
    echo "Please place .txt or .md files in data/raw/ directory for Universal RAG processing"
    echo "Universal RAG works with any raw text files - no specific JSON format required"
fi

# Start the API server
echo "Starting API server..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

echo "✅ MaintIE Enhanced RAG is running at http://localhost:8000"
echo "📚 API Documentation available at http://localhost:8000/docs"
