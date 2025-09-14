# End-to-End ML Pipeline with Hugging Face

A comprehensive MLOps pipeline demonstrating enterprise-level machine learning practices using Hugging Face datasets for sentiment analysis.

## 🏗️ Architecture Overview

This project implements a complete ML pipeline covering:

- **Data Pipelines & Orchestration**: Reproducible pipelines with MLflow
- **Model Building**: Training with experiment tracking
- **Model Production**: FastAPI service with monitoring
- **Database Technologies**: PostgreSQL for metadata storage
- **Backend Development**: RESTful API with proper software engineering
- **Monitoring & Observability**: Data/model drift detection
- **Automation & CI/CD**: GitHub Actions pipeline
- **Container Orchestration**: Docker & docker-compose
- **Infrastructure Provisioning**: Terraform for AWS
- **Cloud Technology**: End-to-end AWS deployment

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd ml-pipeline-project

# Start services
docker-compose up -d

# Run training pipeline
python scripts/train_pipeline.py

# Access services
# - MLflow UI: http://localhost:5000
# - API docs: http://localhost:8000/docs
# - Monitoring: http://localhost:3000
```

## 📊 Project Structure

```
ml-pipeline-project/
├── src/
│   ├── data/           # Data pipeline modules
│   ├── models/         # Model training & inference
│   ├── api/            # FastAPI backend
│   ├── monitoring/     # Drift detection & alerts
│   └── utils/          # Shared utilities
├── infrastructure/     # Terraform configs
├── scripts/           # Pipeline orchestration
├── tests/             # Unit & integration tests
├── docker/            # Dockerfiles
└── .github/           # CI/CD workflows
```

## 🛠️ Technology Stack

- **ML Framework**: Hugging Face Transformers, PyTorch
- **Orchestration**: MLflow, Apache Airflow
- **API**: FastAPI, Pydantic
- **Database**: PostgreSQL, SQLAlchemy
- **Monitoring**: Evidently AI, Grafana
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Cloud**: AWS (ECS, RDS, S3, CloudWatch)
- **IaC**: Terraform