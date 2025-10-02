# ML Pipeline Design Document

This document captures the architectural decisions, design rationale, and technical design of the ML Pipeline project. It explains **WHY** key decisions were made, not just what was implemented.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Design Decisions](#core-design-decisions)
3. [MLflow: Training vs Serving](#mlflow-training-vs-serving)
4. [Database Design](#database-design)
5. [Security Architecture](#security-architecture)
6. [Testing Strategy](#testing-strategy)
7. [Monitoring & Observability](#monitoring--observability)
8. [Performance & Scalability](#performance--scalability)

---

## Architecture Overview

### System Architecture

The pipeline implements a production-ready MLOps architecture with clear separation between training and serving:

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                       │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Hugging   │→ │    Data      │→ │    Training      │    │
│  │   Face     │  │   Pipeline   │  │    Pipeline      │    │
│  │  Datasets  │  │              │  │  + MLflow Track  │    │
│  └────────────┘  └──────────────┘  └──────────────────┘    │
│                                              ↓               │
│                                    ┌──────────────────┐     │
│                                    │  Model Registry  │     │
│                                    │   (MLflow)       │     │
│                                    └──────────────────┘     │
│                                              ↓               │
└──────────────────────────────────────────────────────────────┘
                                              ↓
                                    ┌──────────────────┐
                                    │  Save to Disk    │
                                    │ (models/final/)  │
                                    └──────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      SERVING PIPELINE                        │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │   User     │→ │   FastAPI    │→ │  Prediction      │    │
│  │  Request   │  │   Endpoint   │  │   Service        │    │
│  └────────────┘  └──────────────┘  └──────────────────┘    │
│                                              ↓               │
│                          ┌───────────────────┴────────────┐  │
│                          ↓                                ↓  │
│                  ┌──────────────┐              ┌──────────┐ │
│                  │  PostgreSQL  │              │  Return  │ │
│                  │  (Logging)   │              │  Result  │ │
│                  └──────────────┘              └──────────┘ │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  ┌──────────────┐
                  │  Monitoring  │
                  │  (Evidently) │
                  └──────────────┘
```

### Key Architectural Principles

**1. Separation of Concerns**
- Training uses MLflow for experiment tracking and versioning
- Serving uses saved model files (no MLflow dependency)
- API layer is completely independent of training infrastructure

**2. Production-First Design**
- Multi-stage Docker builds for security and size optimization
- Non-root container users
- Health checks with graceful degradation
- Comprehensive error handling with custom exception hierarchy

**3. Observability by Default**
- All predictions logged to PostgreSQL
- Prometheus metrics for API performance
- Drift detection with Evidently AI
- MLflow UI for experiment visualization

---

## Core Design Decisions

### Why FastAPI?

**Decision**: Use FastAPI for the serving layer

**Rationale**:
- **Automatic OpenAPI documentation** - Essential for team collaboration and API consumers
- **Type validation** - Pydantic models provide runtime validation with zero effort
- **Performance** - ASGI-based, one of the fastest Python frameworks
- **Modern async support** - Ready for high-concurrency workloads
- **Developer experience** - Interactive docs at `/docs` endpoint

**Alternatives Considered**:
- Flask: More mature but slower, no async, manual validation
- Django: Too heavyweight for ML serving
- gRPC: Better performance but harder to debug and consume

### Why DistilBERT?

**Decision**: Use DistilBERT as the default model

**Rationale**:
- **60% faster** than BERT with 97% of its accuracy
- **40% smaller** model size - critical for Docker images
- **Lower latency** - <100ms inference on CPU
- **Production-ready** - Battle-tested on millions of deployments
- **Good balance** - Performance vs accuracy tradeoff optimized

**Alternatives Considered**:
- BERT: Better accuracy but 2x slower
- RoBERTa: Marginal accuracy gains, significantly slower
- TinyBERT: Faster but accuracy drops too much (10-15%)

### Why Docker Compose for Local Development?

**Decision**: Use Docker Compose for all local services

**Rationale**:
- **Reproducibility** - Same environment on every developer's machine
- **Service orchestration** - Start all dependencies (DB, MLflow, API) with one command
- **Network isolation** - Services communicate securely within Docker network
- **Easy cleanup** - `docker-compose down` removes everything
- **Production parity** - Mimics ECS/Kubernetes patterns locally

**Key Services**:
```yaml
services:
  postgres:    # Metadata storage
  mlflow:      # Experiment tracking (training only)
  api:         # Production serving
  prometheus:  # Metrics collection
  grafana:     # Visualization
```

---

## MLflow: Training vs Serving

### The Critical Distinction

**MLflow is ONLY used during training/experimentation - NOT during production serving.**

This is a crucial architectural decision that many ML systems get wrong.

### During Training: MLflow's Four Roles

#### 1. Experiment Tracking

**Problem**: Training a model 50 times with different hyperparameters creates chaos
- Which learning rate gave best accuracy?
- What was the exact configuration from 2 weeks ago?
- How does batch_size=16 compare to batch_size=32?

**Solution**: MLflow automatically logs everything
```python
with mlflow.start_run(run_name="experiment_1"):
    mlflow.log_params({
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3
    })

    mlflow.log_metrics({
        "accuracy": 0.92,
        "f1_score": 0.91
    })
```

**Result**: Visual comparison in MLflow UI (http://localhost:5001)

| Experiment | Accuracy | F1 Score | Learning Rate | Batch Size | Time |
|------------|----------|----------|---------------|------------|------|
| Quick Test | 48.6%    | 0.33     | 2e-5          | 16         | 68s  |
| Full Train | 91.2%    | 0.91     | 2e-5          | 16         | 45m  |
| Optimized  | 92.8%    | 0.93     | 3e-5          | 32         | 6h   |

#### 2. Model Registry & Versioning

**Problem**:
- Where to store model versions?
- How to track which data trained which model?
- How to roll back if new model fails?

**Solution**: MLflow provides version control for models
```python
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="sentiment_classifier"
)
# Creates: sentiment_classifier v1, v2, v3...
# Tracks: training data, code version, parent experiment
```

#### 3. Hyperparameter Optimization Tracking

**Problem**: Optuna tries 100 hyperparameter combinations - how to track all of them?

**Solution**: MLflow + Optuna integration
```python
with mlflow.start_run(run_name="hyperparameter_optimization"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_f1_score", study.best_value)
```

#### 4. Model Performance Over Time

**Problem**: Weekly retraining - is the model improving or degrading?

**Solution**: MLflow tracks historical performance
```python
# Week 1
with mlflow.start_run(run_name="weekly_training_2025_W01"):
    mlflow.log_metric("accuracy", 0.91)

# Week 2
with mlflow.start_run(run_name="weekly_training_2025_W02"):
    mlflow.log_metric("accuracy", 0.93)
```

### During Serving: Why MLflow is NOT Needed

**Production Inference Flow**:
```
User Request
    ↓
FastAPI Endpoint (/predict)
    ↓
Load model from disk (models/final_model/)
    ↓
Tokenize input text
    ↓
Run inference (model.forward())
    ↓
Return prediction
    ↓
Log to PostgreSQL
```

**Notice**: No MLflow anywhere!

**What API Actually Needs**:
1. **Model files** (saved during training):
   ```
   models/final_model/
   ├── config.json
   ├── model.safetensors
   ├── tokenizer_config.json
   └── vocab.txt
   ```

2. **Database connection** (for logging predictions)

3. **Environment config** (.env file)

**Why This Design?**
- ✅ **Simpler deployment** - No MLflow server needed in production
- ✅ **Lower latency** - Direct model loading is faster
- ✅ **Fewer dependencies** - Reduces attack surface and failure points
- ✅ **Cost effective** - MLflow server costs money in cloud
- ✅ **Independence** - API works even if MLflow is down

### Health Check Design

The API reports MLflow status but doesn't require it:

```python
{
    "status": "healthy",           # Overall status
    "model_loaded": true,          # CRITICAL - required
    "database_connected": true,    # CRITICAL - required
    "mlflow_connected": false      # INFORMATIONAL - not required
}
```

**Status Logic**:
- `healthy`: Model loaded AND database connected (MLflow optional)
- `degraded`: One critical dependency failing
- `unhealthy`: Multiple critical failures

---

## Database Design

### Schema Architecture

The database uses PostgreSQL with five core tables designed for ML operations:

#### 1. prediction_logs

**Purpose**: Track every prediction made by the API

**Schema**:
```sql
CREATE TABLE prediction_logs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id       UUID REFERENCES model_deployments(id),
    request_id          VARCHAR(255),
    input_text          TEXT NOT NULL,
    input_hash          VARCHAR(255),              -- MD5 for deduplication
    predicted_label     INTEGER NOT NULL,          -- 0 or 1
    predicted_sentiment VARCHAR(50) NOT NULL,      -- 'POSITIVE' or 'NEGATIVE'
    confidence_score    DOUBLE PRECISION NOT NULL, -- 0.0 to 1.0
    probabilities       JSONB,                     -- Full distribution
    prediction_time     DOUBLE PRECISION,          -- Milliseconds
    model_version       VARCHAR(50),
    predicted_at        TIMESTAMP DEFAULT NOW(),
    feedback_score      DOUBLE PRECISION,          -- User feedback
    is_correct          BOOLEAN,                   -- Correctness feedback
    drift_score         DOUBLE PRECISION           -- Data drift score
);
```

**Design Decisions**:
- **UUID for ID** - Distributed-system friendly, no collisions
- **JSONB for probabilities** - Flexible schema, indexed queries
- **input_hash** - Enables deduplication and analysis
- **Separate label and sentiment** - Supports both numeric and string APIs
- **Nullable feedback fields** - Future-proofing for human-in-the-loop

**Indexes**:
```sql
CREATE INDEX idx_prediction_logs_predicted_at ON prediction_logs(predicted_at);
CREATE INDEX idx_prediction_logs_sentiment ON prediction_logs(predicted_sentiment);
CREATE INDEX idx_prediction_logs_deployment ON prediction_logs(deployment_id);
CREATE INDEX idx_prediction_logs_hash ON prediction_logs(input_hash);
```

#### 2. model_deployments

**Purpose**: Track which model versions are deployed

**Schema**:
```sql
CREATE TABLE model_deployments (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name       VARCHAR(255) NOT NULL,
    model_version    VARCHAR(50) NOT NULL,
    deployment_type  VARCHAR(50),       -- 'production', 'staging', 'canary'
    deployed_at      TIMESTAMP DEFAULT NOW(),
    deployed_by      VARCHAR(255),
    model_path       TEXT,
    model_metrics    JSONB,            -- Training metrics
    is_active        BOOLEAN DEFAULT true,
    deactivated_at   TIMESTAMP
);
```

**Design Decisions**:
- **deployment_type** - Supports A/B testing and canary deployments
- **model_metrics** - JSONB stores training results for comparison
- **is_active flag** - Soft deletes preserve history

#### 3. experiment_runs

**Purpose**: Mirror MLflow experiments for analytics

**Schema**:
```sql
CREATE TABLE experiment_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   VARCHAR(255),
    run_id          VARCHAR(255) UNIQUE,
    run_name        VARCHAR(255),
    status          VARCHAR(50),
    start_time      TIMESTAMP,
    end_time        TIMESTAMP,
    parameters      JSONB,
    metrics         JSONB,
    tags            JSONB,
    artifacts_uri   TEXT
);
```

#### 4. dataset_metrics

**Purpose**: Track data quality and distribution over time

**Schema**:
```sql
CREATE TABLE dataset_metrics (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name       VARCHAR(255) NOT NULL,
    metric_type        VARCHAR(100),
    metric_value       DOUBLE PRECISION,
    metric_details     JSONB,
    calculated_at      TIMESTAMP DEFAULT NOW()
);
```

#### 5. monitoring_alerts

**Purpose**: Store drift detection and anomaly alerts

**Schema**:
```sql
CREATE TABLE monitoring_alerts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type      VARCHAR(100) NOT NULL,   -- 'data_drift', 'concept_drift', etc.
    severity        VARCHAR(50),             -- 'low', 'medium', 'high', 'critical'
    description     TEXT,
    alert_details   JSONB,
    triggered_at    TIMESTAMP DEFAULT NOW(),
    resolved_at     TIMESTAMP,
    resolved_by     VARCHAR(255)
);
```

### Security Hardening

#### SQL Injection Prevention

**Problem**: Direct string interpolation creates SQL injection vulnerabilities
```python
# VULNERABLE CODE (example of what NOT to do)
query = f"SELECT * FROM {table_name}"  # DANGEROUS!
```

**Solution**: Multi-layer validation
```python
def _validate_table_name(self, table_name: str) -> str:
    # Layer 1: Regex validation
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
        raise SecurityError(f"Invalid table name format: {table_name}")

    # Layer 2: Whitelist validation
    valid_tables = {
        'prediction_logs',
        'model_deployments',
        'experiment_runs',
        'dataset_metrics',
        'monitoring_alerts'
    }
    if table_name not in valid_tables:
        raise SecurityError(f"Table name not allowed: {table_name}")

    return table_name
```

**Why Two Layers?**
- Regex catches injection attempts ('; DROP TABLE --)
- Whitelist ensures only known tables are accessed
- Defense in depth - both must pass

#### Connection Security

**Design**:
```python
# No default passwords
DB_PASSWORD = os.getenv("DB_PASSWORD")  # No default!
if not DB_PASSWORD:
    raise ConfigurationError("DB_PASSWORD must be set")

# Connection pooling
engine = create_engine(
    connection_string,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Validate connections before use
)
```

### Analytics Queries

**Sentiment Distribution**:
```sql
SELECT
    predicted_sentiment,
    COUNT(*) as total,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence
FROM prediction_logs
GROUP BY predicted_sentiment;
```

**Performance Over Time**:
```sql
SELECT
    DATE_TRUNC('hour', predicted_at) as hour,
    COUNT(*) as predictions,
    AVG(prediction_time) as avg_time_ms,
    AVG(confidence_score) as avg_confidence
FROM prediction_logs
GROUP BY hour
ORDER BY hour DESC;
```

**Drift Detection**:
```sql
SELECT
    DATE(predicted_at) as date,
    AVG(drift_score) as avg_drift,
    MAX(drift_score) as max_drift,
    COUNT(CASE WHEN drift_score > 0.5 THEN 1 END) as high_drift_count
FROM prediction_logs
WHERE drift_score IS NOT NULL
GROUP BY date;
```

---

## Security Architecture

### Defense in Depth Strategy

The system implements multiple security layers:

#### 1. Application Layer Security

**CORS Configuration**:
```python
# Environment-specific allowed origins
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # No wildcards!
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)
```

**Why**:
- Prevents cross-site request forgery
- Environment-specific (dev vs prod origins)
- No wildcard `*` in production

**Input Validation**:
```python
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    return_probabilities: bool = False

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
```

#### 2. Container Security

**Multi-stage Docker Build**:
```dockerfile
# Stage 1: Build dependencies
FROM python:3.9-slim as builder
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Production runtime
FROM python:3.9-slim as production
# Non-root user
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -s /bin/false appuser

# Copy only wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Run as non-root
USER appuser
```

**Why Multi-stage?**
- Smaller final image (no build tools)
- Reduced attack surface
- Build cache optimization

**Why Non-root?**
- Container breakout is less dangerous
- Follows principle of least privilege
- Kubernetes security policies require it

#### 3. Secrets Management

**Environment-based Secrets**:
```python
# config.py
class Config:
    # NO DEFAULTS for sensitive values
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")  # Required!

    @validator("DB_PASSWORD")
    def validate_password(cls, v):
        if not v or v == "changeme":
            raise ValueError("DB_PASSWORD must be set to a secure value")
        return v
```

**Docker Compose (Dev)**:
```yaml
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}  # From .env file
```

**Terraform (Prod)**:
```hcl
resource "aws_secretsmanager_secret" "db_password" {
  name = "ml-pipeline-db-password"
  recovery_window_in_days = 7
}

resource "aws_db_instance" "postgres" {
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
}
```

#### 4. Database Security

**Encrypted Storage**:
```hcl
resource "aws_db_instance" "postgres" {
  storage_encrypted = true
  kms_key_id       = aws_kms_key.db_encryption.arn
}
```

**Network Isolation**:
```hcl
# Database in private subnet only
resource "aws_db_subnet_group" "main" {
  subnet_ids = module.vpc.private_subnets
}

# Only allow access from ECS tasks
resource "aws_security_group_rule" "rds_ingress" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs_tasks.id
  security_group_id        = aws_security_group.rds.id
}
```

### Custom Exception Hierarchy

**Design**:
```python
class MLPipelineException(Exception):
    """Base exception for all ML pipeline errors"""
    pass

class ConfigurationError(MLPipelineException):
    """Configuration and environment issues"""
    pass

class SecurityError(MLPipelineException):
    """Security violations"""
    pass

class DatabaseError(MLPipelineException):
    """Database connectivity and query errors"""
    pass

class ModelError(MLPipelineException):
    """Model-related failures"""
    pass
```

**Why Custom Exceptions?**
- **Specific error handling** - Catch exact error types
- **Better debugging** - Error type reveals problem location
- **Structured logging** - Different severities for different errors
- **Client-friendly messages** - Security errors don't leak details

**Usage**:
```python
try:
    table_name = self._validate_table_name(user_input)
except SecurityError as e:
    logger.error(f"Security violation: {e}")
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid table name"}  # Don't leak details
    )
```

---

## Testing Strategy

### Test Architecture

The testing strategy balances thoroughness with practicality:

#### Test Pyramid

```
        ╱╲
       ╱  ╲
      ╱ E2E ╲          ← Few (5-10 tests)
     ╱________╲
    ╱          ╲
   ╱Integration╲       ← Some (20-30 tests)
  ╱______________╲
 ╱                ╲
╱   Unit Tests     ╲   ← Many (50+ tests)
╱____________________╲
```

#### 1. Unit Tests (`tests/unit/`)

**Focus**: Isolated component testing with mocks

**Example**:
```python
# tests/unit/test_config.py
def test_config_validation():
    """Test configuration validation catches invalid inputs"""
    with pytest.raises(ConfigurationError):
        Config(DB_PASSWORD=None)  # Should fail

    config = Config(DB_PASSWORD="secure_password")
    assert config.DB_PASSWORD == "secure_password"

# tests/unit/test_database.py
def test_table_name_validation():
    """Test SQL injection prevention"""
    db = Database(mock_session)

    # Valid table name
    assert db._validate_table_name("prediction_logs") == "prediction_logs"

    # SQL injection attempts
    with pytest.raises(SecurityError):
        db._validate_table_name("users; DROP TABLE users--")

    with pytest.raises(SecurityError):
        db._validate_table_name("../../etc/passwd")
```

**Coverage Target**: 80% minimum (enforced in pytest.ini)

#### 2. Integration Tests (`tests/integration/`)

**Focus**: Component interaction and database operations

**Example**:
```python
# tests/integration/test_api_database.py
@pytest.mark.integration
def test_prediction_logging(test_client, test_db):
    """Test predictions are logged to database"""
    # Make prediction
    response = test_client.post("/predict", json={
        "text": "Great movie!"
    })

    assert response.status_code == 200

    # Verify database log
    logs = test_db.query(PredictionLog).all()
    assert len(logs) == 1
    assert logs[0].input_text == "Great movie!"
    assert logs[0].predicted_sentiment in ["POSITIVE", "NEGATIVE"]
```

#### 3. API Tests (`tests/integration/test_api.py`)

**Focus**: End-to-end API behavior

**Health Check Design**:
```python
def test_health_endpoint(test_client):
    """Test health check reports correct status"""
    response = test_client.get("/health")

    assert response.status_code == 200
    health = response.json()

    # Critical dependencies must be true
    assert health["model_loaded"] is True
    assert health["database_connected"] is True

    # Overall status based on critical dependencies
    if health["model_loaded"] and health["database_connected"]:
        assert health["status"] == "healthy"
    else:
        assert health["status"] in ["degraded", "unhealthy"]

    # MLflow is informational only
    # (can be true or false, doesn't affect status)
```

**Error Handling Tests**:
```python
def test_invalid_input_handling(test_client):
    """Test API handles invalid inputs gracefully"""
    # Missing required field
    response = test_client.post("/predict", json={
        "wrong_field": "value"
    })
    assert response.status_code == 422

    # Empty text
    response = test_client.post("/predict", json={
        "text": ""
    })
    assert response.status_code == 422

    # Invalid JSON
    response = test_client.post("/predict", data="invalid json")
    assert response.status_code == 422
```

### Test Fixtures

**Shared fixtures** (`tests/conftest.py`):
```python
@pytest.fixture
def test_db():
    """Provide clean database for tests"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()

@pytest.fixture
def test_client(test_db):
    """Provide FastAPI test client"""
    app.dependency_overrides[get_db] = lambda: test_db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()

@pytest.fixture
def mock_model():
    """Provide mock model for testing without GPU"""
    class MockModel:
        def predict(self, text):
            return {"label": 1, "score": 0.95}

    return MockModel()
```

### CI/CD Testing Pipeline

**GitHub Actions workflow**:
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

### Performance Testing

**Load testing** (using Apache Bench):
```bash
# Test single prediction endpoint
ab -n 100 -c 10 -p test_request.json -T application/json \
  http://localhost:8000/predict

# Success criteria:
# - Requests per second > 10
# - Mean time per request < 100ms
# - No failed requests
```

**Stress testing**:
```python
# tests/performance/test_load.py
@pytest.mark.performance
def test_concurrent_predictions(test_client):
    """Test API handles concurrent requests"""
    import concurrent.futures

    def make_request():
        return test_client.post("/predict", json={
            "text": "Test movie review"
        })

    # 50 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [f.result() for f in futures]

    # All should succeed
    assert all(r.status_code == 200 for r in results)
```

### Test Configuration

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Coverage settings
addopts =
    --strict-markers
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80

# Custom markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests requiring database
    performance: marks performance/load tests
    security: marks security-related tests
```

---

## Monitoring & Observability

### Multi-Layer Monitoring Strategy

#### 1. Application Metrics (Prometheus)

**Custom metrics**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Prediction metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['sentiment', 'model_version']
)

prediction_latency = Histogram(
    'prediction_duration_seconds',
    'Time spent processing predictions',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

# Model confidence distribution
confidence_gauge = Gauge(
    'prediction_confidence',
    'Model prediction confidence score',
    ['sentiment']
)

# Usage in API
@app.post("/predict")
async def predict(request: PredictionRequest):
    with prediction_latency.time():
        result = model.predict(request.text)

    prediction_counter.labels(
        sentiment=result['sentiment'],
        model_version=MODEL_VERSION
    ).inc()

    confidence_gauge.labels(
        sentiment=result['sentiment']
    ).set(result['confidence'])

    return result
```

**Why Prometheus?**
- Industry standard for ML systems
- Pull-based model (services don't need to know about monitoring)
- Powerful query language (PromQL)
- Excellent Grafana integration

#### 2. Data Drift Detection (Evidently AI)

**Reference data setup**:
```python
from evidently.metrics import DataDriftMetric
from evidently.report import Report

# Store reference data distribution from training
reference_data = pd.DataFrame({
    'text_length': [len(t) for t in training_texts],
    'confidence': training_confidences
})

# Compare current predictions
current_data = pd.DataFrame({
    'text_length': [len(t) for t in recent_texts],
    'confidence': recent_confidences
})

# Generate drift report
report = Report(metrics=[DataDriftMetric()])
report.run(reference_data=reference_data, current_data=current_data)

# Check drift
drift_score = report.as_dict()['metrics'][0]['result']['drift_score']
if drift_score > 0.5:
    create_alert('data_drift', severity='high')
```

**Drift metrics tracked**:
- Input text length distribution
- Confidence score distribution
- Sentiment ratio changes
- Prediction latency patterns

#### 3. Model Performance Monitoring

**Concept drift detection**:
```python
def check_concept_drift(predictions_df, window_size=100):
    """Detect if model behavior is changing over time"""

    # Calculate rolling confidence
    rolling_confidence = predictions_df['confidence'].rolling(
        window=window_size
    ).mean()

    # Calculate rolling sentiment ratio
    rolling_positive_ratio = predictions_df['predicted_label'].rolling(
        window=window_size
    ).mean()

    # Detect significant changes
    confidence_drift = abs(
        rolling_confidence.iloc[-1] - rolling_confidence.iloc[0]
    ) > 0.1

    sentiment_drift = abs(
        rolling_positive_ratio.iloc[-1] - rolling_positive_ratio.iloc[0]
    ) > 0.2

    if confidence_drift or sentiment_drift:
        create_alert('concept_drift', {
            'confidence_change': rolling_confidence.iloc[-1] - rolling_confidence.iloc[0],
            'sentiment_ratio_change': rolling_positive_ratio.iloc[-1] - rolling_positive_ratio.iloc[0]
        })
```

#### 4. Alerting System

**Alert design**:
```python
class MonitoringAlert:
    """Structured alert for monitoring events"""

    SEVERITY_LEVELS = ['low', 'medium', 'high', 'critical']

    def __init__(
        self,
        alert_type: str,
        severity: str,
        description: str,
        details: dict
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.description = description
        self.details = details
        self.triggered_at = datetime.now()

    def save_to_db(self):
        """Persist alert to database"""
        db.execute(
            """
            INSERT INTO monitoring_alerts
            (alert_type, severity, description, alert_details, triggered_at)
            VALUES (:type, :severity, :desc, :details, :time)
            """,
            {
                'type': self.alert_type,
                'severity': self.severity,
                'desc': self.description,
                'details': json.dumps(self.details),
                'time': self.triggered_at
            }
        )

    def send_notification(self):
        """Send alert via configured channels"""
        if self.severity in ['high', 'critical']:
            send_slack_alert(self)
            send_email_alert(self)
```

**Alert types**:
- `data_drift` - Input distribution changed significantly
- `concept_drift` - Model behavior changed
- `performance_degradation` - Latency or accuracy dropped
- `database_error` - DB connectivity issues
- `model_error` - Model prediction failures

### Grafana Dashboards

**Key dashboard panels**:

1. **Prediction Overview**
   - Total predictions (24h)
   - Positive/Negative ratio
   - Average confidence
   - Prediction latency (p50, p95, p99)

2. **Model Performance**
   - Confidence distribution histogram
   - Predictions per minute (time series)
   - Error rate
   - Model version distribution

3. **Data Quality**
   - Drift score over time
   - Text length distribution
   - Duplicate detection rate
   - Low confidence alerts

4. **Infrastructure**
   - API response times
   - Database connection pool
   - Memory usage
   - CPU utilization

---

## Performance & Scalability

### Performance Optimization Strategies

#### 1. Model Optimization

**Choice of DistilBERT**:
- 40% smaller than BERT
- 60% faster inference
- Minimal accuracy loss (3%)

**Quantization** (optional):
```python
# Convert to ONNX for faster inference
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "models/final_model",
    export=True,
    provider="CPUExecutionProvider"  # or "CUDAExecutionProvider"
)

# 2-3x faster inference with minimal accuracy loss
```

#### 2. Database Connection Pooling

**Configuration**:
```python
engine = create_engine(
    connection_string,
    pool_size=10,          # Connections kept open
    max_overflow=20,       # Additional connections under load
    pool_pre_ping=True,    # Validate before use
    pool_recycle=3600      # Recycle after 1 hour
)
```

**Why pooling?**
- Reuse connections (avoid TCP handshake overhead)
- Handle bursts (overflow connections)
- Automatic recovery (pre-ping validation)

#### 3. Caching Strategy

**Model caching** (in memory):
```python
class PredictionService:
    _model = None
    _tokenizer = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                "models/final_model"
            )
        return cls._model
```

**Response caching** (Redis, optional):
```python
@lru_cache(maxsize=1000)
def get_prediction(text_hash: str):
    """Cache predictions for identical inputs"""
    # Only compute once per unique input
    return model.predict(text)
```

### Scalability Design

#### 1. Horizontal Scaling (ECS Fargate)

**Auto-scaling configuration**:
```hcl
resource "aws_appautoscaling_target" "api" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "api-cpu-scaling"
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

**Load balancing**:
```hcl
resource "aws_lb" "api" {
  name               = "ml-pipeline-alb"
  internal           = false
  load_balancer_type = "application"

  enable_deletion_protection = true
  enable_http2              = true
}

resource "aws_lb_target_group" "api" {
  health_check {
    enabled             = true
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
  }
}
```

#### 2. Database Scaling

**Read replicas** (for analytics):
```hcl
resource "aws_db_instance" "replica" {
  replicate_source_db = aws_db_instance.postgres.id
  instance_class      = "db.t3.medium"

  # Route read queries here
  tags = {
    Role = "read-replica"
  }
}
```

**Connection pooling at scale**:
```python
# Use pgbouncer for connection pooling
# docker-compose.yml
services:
  pgbouncer:
    image: pgbouncer/pgbouncer
    environment:
      DATABASES_HOST: postgres
      DATABASES_PORT: 5432
      PGBOUNCER_POOL_MODE: transaction
      PGBOUNCER_MAX_CLIENT_CONN: 1000
      PGBOUNCER_DEFAULT_POOL_SIZE: 25
```

#### 3. Batch Processing Optimization

**Batching for throughput**:
```python
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    # Process in batches of 32 for optimal GPU utilization
    batch_size = 32
    results = []

    for i in range(0, len(request.texts), batch_size):
        batch = request.texts[i:i + batch_size]

        # Batch inference is 5-10x faster than sequential
        batch_results = model.predict_batch(batch)
        results.extend(batch_results)

    return results
```

---

## Summary

This design document captures the key architectural decisions and their rationale:

1. **MLflow Separation** - Training tool only, not needed for serving (reduces production complexity)

2. **Database-First Logging** - PostgreSQL for all predictions (enables analytics and drift detection)

3. **Security in Depth** - Multiple layers (input validation, SQL injection prevention, container security, secrets management)

4. **Health Check Design** - Critical dependencies required, optional dependencies informational

5. **Testing Strategy** - Pyramid approach with 80% coverage minimum

6. **Monitoring Architecture** - Prometheus metrics, Evidently drift detection, structured alerting

7. **Scalability Design** - Horizontal scaling with ECS, connection pooling, caching strategies

**Key Principle**: Every design decision prioritizes production readiness, security, and operational simplicity over theoretical perfection.
