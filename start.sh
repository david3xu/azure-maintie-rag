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

# Check for MaintIE data
if [ ! -f "data/raw/gold_release.json" ]; then
    echo "⚠️  MaintIE data not found in data/raw/"
    echo "Please place gold_release.json and silver_release.json in data/raw/ directory"
    echo "You can create sample data for testing if needed"
fi

# Start the API server
echo "Starting API server..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

echo "✅ MaintIE Enhanced RAG is running at http://localhost:8000"
echo "📚 API Documentation available at http://localhost:8000/docs"
