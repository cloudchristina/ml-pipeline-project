#!/bin/bash

# ML Pipeline Services Startup Script

set -e

echo "🚀 Starting ML Pipeline Services..."

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ Please review and update the .env file with your configuration."
fi

# Start core services
echo "🐳 Starting core services (PostgreSQL, MLflow, API)..."
docker-compose up -d postgres mlflow api

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker-compose ps | grep -q "Up (healthy)"; then
        echo "✅ Core services are healthy!"
        break
    fi

    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts - waiting for services..."
    sleep 10
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Services did not become healthy in time. Check logs:"
    docker-compose logs
    exit 1
fi

# Start monitoring services
echo "📊 Starting monitoring services (Prometheus, Grafana)..."
docker-compose up -d prometheus grafana

echo "🎉 All services started successfully!"
echo ""
echo "📋 Service URLs:"
echo "  - API Documentation: http://localhost:8000/docs"
echo "  - MLflow UI: http://localhost:5000"
echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "🔧 To run training:"
echo "  docker-compose --profile training up training"
echo ""
echo "📊 To view logs:"
echo "  docker-compose logs -f [service_name]"
echo ""
echo "🛑 To stop all services:"
echo "  docker-compose down"