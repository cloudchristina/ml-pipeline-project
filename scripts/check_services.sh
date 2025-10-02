#!/bin/bash
# Service Health Check Script
# Verifies that all ML pipeline services are running and healthy

echo "🔍 ML Pipeline Service Health Check"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service status
check_service() {
    local service_name=$1
    local url=$2
    local description=$3

    echo -n "Checking $description... "

    if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed${NC}"
        return 1
    fi
}

# Function to check docker service
check_docker_service() {
    local service_name=$1
    local description=$2

    echo -n "Checking $description... "

    local status=$(docker-compose ps --services --filter "status=running" | grep "^$service_name$")
    if [ -n "$status" ]; then
        echo -e "${GREEN}✓ Running${NC}"
        return 0
    else
        echo -e "${RED}✗ Not Running${NC}"
        return 1
    fi
}

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose not found${NC}"
    exit 1
fi

# Check Docker services
echo "📦 Docker Services:"
check_docker_service "postgres" "PostgreSQL Database"
check_docker_service "mlflow" "MLflow Tracking Server"
check_docker_service "api" "ML API Service"

echo ""

# Check service endpoints
echo "🌐 Service Endpoints:"
check_service "postgres" "localhost:5432" "PostgreSQL (Port 5432)" || true
check_service "mlflow" "http://localhost:5001" "MLflow UI (Port 5001)" || true
check_service "api" "http://localhost:8000/docs" "API Documentation (Port 8000)" || true

echo ""

# Display service URLs
echo "📋 Service URLs:"
echo "• API Documentation: http://localhost:8000/docs"
echo "• MLflow UI: http://localhost:5001"
echo "• PostgreSQL: localhost:5432"
echo ""

# Show container status
echo "🐳 Container Status:"
docker-compose ps