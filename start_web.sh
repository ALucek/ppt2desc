#!/bin/bash

# Start PPT2Desc Web Application
# This script provides an easy way to start the web interface

echo "🚀 Starting PPT2Desc Web Application..."
echo ""

# Parse arguments
REBUILD=false
if [ "$1" = "--rebuild" ]; then
    REBUILD=true
fi

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker compose &> /dev/null; then
    echo "✓ Docker found"
    echo ""

    if [ "$REBUILD" = true ]; then
        echo "Rebuilding containers..."
        docker compose down
        docker compose build --no-cache
    fi

    echo "Starting services with Docker Compose..."
    echo ""

    docker compose up -d

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Services started successfully!"
        echo ""
        echo "🌐 Web Interface: http://localhost:8000"
        echo "🔧 LibreOffice Converter: http://localhost:2002"
        echo ""
        echo "To view logs: docker compose logs -f"
        echo "To stop: docker compose down"
        echo ""
        echo "Waiting for services to be healthy..."
        sleep 3
        docker compose ps
    else
        echo "❌ Failed to start services"
        echo ""
        echo "💡 Try rebuilding with: ./start_web.sh --rebuild"
        exit 1
    fi
else
    echo "⚠ Docker not found, starting locally..."
    echo ""

    # Check if UV is installed
    if ! command -v uv &> /dev/null; then
        echo "❌ UV package manager not found. Please install UV first:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    echo "Installing dependencies..."
    uv sync

    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi

    echo ""
    echo "Starting web application..."
    echo ""
    echo "✅ Server starting..."
    echo ""
    echo "🌐 Web Interface: http://localhost:8000"
    echo ""
    echo "Note: For local mode, make sure LibreOffice is installed or"
    echo "      run 'docker compose up -d libreoffice-converter' separately"
    echo ""

    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    uv run python -m uvicorn src.webapp:app --host 0.0.0.0 --port 8000
fi
