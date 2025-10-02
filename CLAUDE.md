# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enterprise-grade ML pipeline for sentiment analysis using Hugging Face Transformers. The architecture implements comprehensive MLOps practices including model training, serving, monitoring, and infrastructure as code.

## Development Commands

### Development Environment Setup

#### Option 1: Dev Container (Recommended)
The project includes a VS Code dev container configuration for isolated development:

```bash
# Open in VS Code and select "Reopen in Container"
# All dependencies are automatically installed in a clean Python 3.12 environment
# Services automatically start on container launch
# Ports forwarded: 8000 (API), 5001 (MLflow), 5432 (PostgreSQL), 3000 (Grafana), 9090 (Prometheus)
```

#### Option 2: Local Development
```bash
# Create virtual environment (recommended to avoid dependency conflicts)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Core Development
```bash
# Start all services locally
./scripts/start_services.sh

# Run training pipeline
python scripts/train_pipeline.py

# Run training with hyperparameter optimization
python scripts/train_pipeline.py --optimize --n-trials 20

# Quick test training run
python scripts/train_pipeline.py --quick
```

### Testing & Quality
```bash
# Run full test suite with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run specific test types
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests only
pytest -m "not slow"           # Skip slow tests
pytest -m database            # Database-related tests only

# Run single test file/function
pytest tests/unit/test_config.py::test_config_validation -v

# Linting and formatting
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=100
mypy src/ --ignore-missing-imports --no-strict-optional
```

### Docker Services
```bash
# Start core services (DB, MLflow, API)
docker-compose up -d postgres mlflow api

# Start frontend application
docker-compose up -d frontend

# Run training in container
docker-compose --profile training up training

# Start monitoring stack
docker-compose up -d prometheus grafana

# View service logs
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f training

# Stop specific services
docker-compose stop api              # Stop API service only
docker-compose stop frontend         # Stop frontend only
docker-compose stop mlflow           # Stop MLflow only
docker-compose stop postgres         # Stop database only

# Stop all services (containers remain, can restart quickly)
docker-compose stop

# Stop and remove all services (full cleanup)
docker-compose down

# Stop and remove with volumes (WARNING: deletes all data)
docker-compose down -v

# Check service status
docker-compose ps

# Run comprehensive health check
./scripts/check_services.sh
```

### Frontend Application
```bash
# Frontend is built with React + Vite + TailwindCSS
# Served via Nginx with API proxy on port 5173

# Access frontend
open http://localhost:5173  # macOS
# or navigate to http://localhost:5173 in browser

# Frontend features:
# - Real-time sentiment analysis interface
# - Nginx proxy: /api/* → http://localhost:8000/*
# - No CORS issues (same-origin requests via proxy)

# Rebuild frontend after code changes
docker-compose build frontend
docker-compose up -d frontend

# Frontend development (local, not containerized)
cd frontend
npm install
npm run dev  # Development server with hot reload
npm run build  # Production build
```

### Service Health Verification

After starting services, verify everything is working:

```bash
# Quick health check
./scripts/check_services.sh

# Manual verification
curl http://localhost:8000/docs    # API documentation
curl http://localhost:5001         # MLflow UI
docker-compose logs api            # Check API logs
docker-compose logs mlflow         # Check MLflow logs
docker-compose logs postgres       # Check database logs
```

## Architecture Overview

### Core Components Structure
- `src/data/` - Data pipeline and dataset loading with Hugging Face integration
- `src/models/` - Model training pipeline with MLflow tracking and Optuna optimization
- `src/api/` - FastAPI service with prediction endpoints, health checks, and metrics
- `src/database/` - SQLAlchemy models and repositories with security hardening
- `src/monitoring/` - Evidently-based drift detection and alerting system
- `src/utils/` - Shared configuration, logging, and custom exception hierarchy

### Key Integrations
- **MLflow**: Experiment tracking and model registry (http://localhost:5001)
- **FastAPI**: RESTful API with automatic OpenAPI docs (http://localhost:8000/docs)
- **PostgreSQL**: Metadata storage with connection pooling and security features
- **Prometheus/Grafana**: Metrics collection and visualization (http://localhost:3000)
- **Evidently AI**: Data and model drift detection with automated alerts

### Security Features Implemented
- SQL injection prevention with table name validation and whitelisting
- CORS configuration with environment-specific allowed origins
- Secrets management without hardcoded passwords
- Multi-stage Docker builds with non-root users
- Custom security exception handling throughout the application

## Configuration System

The project uses environment-based configuration via `src/utils/config.py`:

### Required Environment Variables
```bash
# Database (no defaults - must be provided)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ml_pipeline
DB_USER=postgres
DB_PASSWORD=<required>

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# API Configuration
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### Model Configuration
- Default model: `distilbert-base-uncased`
- Configurable via `MODEL_NAME` environment variable
- Training hyperparameters controlled through config or CLI arguments

## Training Pipeline Architecture

The training system uses a modular pipeline approach:

1. **Data Pipeline** (`src/data/data_pipeline.py`): Loads and preprocesses Hugging Face datasets
2. **Model Training** (`src/models/training_pipeline.py`): Handles model training with experiment tracking
3. **Hyperparameter Optimization**: Optuna integration for automated hyperparameter tuning
4. **Model Serving** (`src/api/services.py`): Inference service with monitoring and feedback collection

## API Service Details

### Key Endpoints
- `POST /predict` - Single predictions
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check with dependency validation
- `GET /metrics` - Prometheus metrics
- `POST /feedback` - Model feedback collection
- `GET /model/info` - Model metadata and performance stats

### Error Handling
Custom exception hierarchy provides structured error responses:
- `ConfigurationError` - Configuration issues
- `SecurityError` - Security violations
- `ModelError` - Model-related failures
- `DatabaseError` - Database connectivity issues

## Infrastructure & Deployment

### Local Development
Use docker-compose for local development with hot-reloading and volume mounts for the API service.

### Production Deployment
- **Infrastructure**: Terraform configurations in `infrastructure/` directory
- **Container Registry**: Uses multi-stage Docker builds for optimized production images
- **Database**: RDS PostgreSQL with encryption and automated backups
- **Compute**: ECS Fargate with auto-scaling policies
- **Monitoring**: CloudWatch integration with custom metrics

## Testing Strategy

### Test Organization
- `tests/unit/` - Isolated unit tests with mocks
- `tests/integration/` - Database and service integration tests
- `tests/conftest.py` - Shared fixtures and test configuration

### Coverage Requirements
- Minimum 80% test coverage enforced in pytest.ini
- Focus on critical paths: security features, model training, API endpoints

## Monitoring & Observability

### Metrics Collection
- Custom Prometheus metrics for predictions, errors, and performance
- MLflow experiment tracking for model performance over time
- Evidently AI for data and model drift detection

### Alerting
- Automated drift detection with configurable thresholds
- Health check failures trigger alerts
- Performance degradation monitoring

## Common Development Patterns

### Adding New Models
1. Extend `TrainingPipeline` class in `src/models/training_pipeline.py`
2. Update model configuration in `src/utils/config.py`
3. Add model-specific preprocessing in `src/data/data_pipeline.py`
4. Update API service tests to include new model validation

### Database Changes
1. Create new SQLAlchemy models in `src/database/models.py`
2. Generate Alembic migration: `alembic revision --autogenerate -m "description"`
3. Apply migration: `alembic upgrade head`
4. Update security validation in `database.py` if adding new tables

### Security Considerations
- Always validate user inputs using Pydantic models
- New database tables must be added to the security whitelist
- Environment variables containing secrets should not have defaults
- Container images should run as non-root users

## 📚 Additional Documentation

For more detailed information, refer to:

- **[README.md](README.md)** - User guide, quick start, deployment instructions
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Architecture diagrams, design decisions, detailed component explanations
- **[docs/MONITORING.md](docs/MONITORING.md)** - Prometheus metrics, Grafana dashboards, drift detection, troubleshooting

These documents provide comprehensive coverage of:
- System architecture with Mermaid diagrams
- Training and serving pipeline flows
- Production deployment on AWS
- Complete monitoring setup and alerting