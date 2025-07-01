# Wykop Social Media Analytics Platform

A comprehensive distributed data processing platform for analyzing Polish social media content from Wykop.pl using MLOps practices, containerization, and machine learning.

## 🚀 Features

- Real-time data ingestion system that automatically fetches posts from Wykop API using scheduled tasks, processes them through a distributed pipeline, and handles rate limiting and error recovery to ensure continuous data flow
- Distributed processing architecture built on Celery task queue that enables horizontal scaling across multiple worker nodes, automatic task retry mechanisms, and efficient resource utilization for high-throughput data processing
- NLP pipeline incorporating language detection and text vectorization using the HerBERT transformer model specifically trained for Polish language understanding, enabling semantic analysis of social media content
- Regression models designed to predict post popularity based on content features, user engagement patterns, and temporal factors, with automated model training and evaluation capabilities
- Comprehensive monitoring and observability stack featuring Prometheus metrics collection, custom dashboards in Grafana, real-time alerting, and performance tracking across all system components
- Full containerization support with Docker images, Kubernetes deployment manifests, Helm charts for easy installation, and production-ready configuration for cloud environments
- Robust data storage solution using MongoDB for structured document storage, efficient indexing for fast queries, and data persistence with backup and recovery mechanisms

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Wykop API     │────│  Celery Tasks   │────│   MongoDB       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   HerBERT       │────│   Spark ML      │
                    │   Vectorizer    │    │   Training      │
                    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   Prometheus    │────│   Grafana       │
                    │   Metrics       │    │   Dashboard     │
                    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.9 or higher installed with pip package manager for dependency management and virtual environment support
- Docker and Docker Compose for containerized development environment, local testing of services, and consistent deployment across different platforms
- Kubernetes cluster access for production deployment, either through cloud providers like AWS EKS, Google GKE, or local development clusters like Minikube
- Valid Wykop API token with appropriate permissions for accessing social media data, rate limiting considerations, and API endpoint access

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd k8s-wykop-main
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create `.env` file:

```bash
WYKOP_API_TOKEN=your_api_token_here
REDIS_HOST=redis://localhost:6379/0
MONGO_HOST=mongodb://localhost:27017
PTG_HOST=localhost:9091
```

### 4. Start Services (Development)

```bash
# Start Redis and MongoDB
docker-compose up -d redis mongodb

# Start Celery worker
celery -A celery_app worker --loglevel=info

# Start Celery beat (scheduler)
celery -A celery_app beat --loglevel=info
```

### 5. Run Machine Learning Pipeline

```bash
python app/regression.py
```

## 🐳 Docker Deployment

### Build Images

```bash
docker build -f Dockerfile.celery -t wykop-celery .
docker build -f Dockerfile.spark -t wykop-spark .
```

### Run with Docker Compose

```bash
docker-compose up -d
```

## ☸️ Kubernetes Deployment

### Deploy with Helm

```bash
helm install wykop-analytics ./helm-chart
```

### Monitor with Prometheus

```bash
kubectl port-forward svc/prometheus 9090:9090
```

## 📊 Monitoring

The platform includes comprehensive monitoring:

- Application metrics tracking covers post processing rates, language detection accuracy, task completion times, and error frequencies to ensure optimal pipeline performance
- Performance metrics monitoring includes task execution times, memory usage patterns, CPU utilization, and throughput measurements across all system components
- Infrastructure metrics collection encompasses CPU usage, memory consumption, disk I/O operations, network traffic, and container health status for proactive system management
- Business metrics analysis provides insights into content popularity trends, user engagement patterns, viral content characteristics, and platform usage statistics for strategic decision making

Access dashboards:

- Prometheus metrics collection interface available at `http://localhost:9090` for querying time-series data, creating custom queries, and monitoring system health
- Grafana visualization platform accessible at `http://localhost:3000` featuring interactive dashboards, alerting capabilities, and comprehensive system overview

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest test_app.py -v

# Type checking
mypy app/

# Code formatting
black app/

# Linting
flake8 app/
```

## 📁 Project Structure

```
k8s-wykop-main/
├── app/                    # Main application code
│   ├── __init__.py        # Package initialization
│   ├── celery_app.py      # Celery configuration
│   ├── tasks.py           # Distributed tasks
│   ├── models.py          # Data models
│   ├── mongo_client.py    # MongoDB utilities
│   ├── utils.py           # Helper functions
│   ├── metrics.py         # Prometheus metrics
│   └── regression.py      # ML model training
├── requirements.txt       # Python dependencies
├── mypy.ini              # Type checking configuration
├── test_app.py           # Unit tests
├── docker-compose.yml    # Local development setup
├── Dockerfile.celery     # Celery worker image
├── Dockerfile.spark      # Spark processing image
└── README.md             # Project documentation
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WYKOP_API_TOKEN` | Wykop.pl API authentication token | Required |
| `REDIS_HOST` | Redis connection string | `redis://localhost:6379/0` |
| `MONGO_HOST` | MongoDB connection string | `mongodb://localhost:27017` |
| `PTG_HOST` | Prometheus Push Gateway host | `localhost:9091` |

### Celery Configuration

- Broker configuration uses Redis as the message broker for reliable task queuing, supporting task priority, routing, and dead letter queues for failed tasks
- Backend storage utilizes Redis for result persistence, enabling task result retrieval, monitoring task status, and implementing complex workflow patterns
- Task routes employ automatic discovery mechanism for task registration, supporting dynamic task loading and modular application architecture
- Beat schedule operates on 1-minute intervals for data fetching tasks, with configurable timing, timezone support, and cron-like scheduling capabilities

## 🤝 Contributing

1. Fork the repository to your GitHub account and clone it locally for development
2. Create feature branch using descriptive naming convention (`git checkout -b feature/amazing-feature`) to isolate your changes
3. Add comprehensive tests for new functionality, ensuring both unit and integration test coverage for robust code quality
4. Ensure code passes type checking and linting by running mypy, flake8, and black formatters before committing changes
5. Commit changes with clear, descriptive commit messages (`git commit -m 'Add amazing feature'`) following conventional commit format
6. Push to branch and create detailed pull request (`git push origin feature/amazing-feature`) with description of changes and testing approach
7. Open Pull Request with comprehensive description, including motivation, implementation details, and any breaking changes

### Code Standards

- Follow PEP 8 style guidelines strictly for consistent code formatting, proper naming conventions, line length limits, and import organization
- Add comprehensive type hints to all functions, methods, and variables to improve code documentation, enable better IDE support, and catch type-related errors early
- Include detailed docstrings for all modules, classes, and functions following Google or NumPy documentation style for comprehensive API documentation
- Write thorough unit tests for new features, including edge cases, error conditions, and integration scenarios to ensure code reliability
- Maintain test coverage above 80% for all new code, with focus on critical paths, business logic, and error handling mechanisms

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- Real-time streaming implementation with Apache Kafka for immediate data processing, enabling live sentiment analysis, trending topic detection, and real-time recommendation systems
- Advanced NLP models integration including sentiment analysis for emotional content understanding, topic modeling for content categorization, and named entity recognition for data enrichment
- GraphQL API development for flexible data access patterns, enabling efficient client-server communication, reducing over-fetching, and providing type-safe query interfaces
- A/B testing framework implementation for data-driven decision making, supporting multivariate testing, statistical significance calculation, and automated result interpretation
- Automated model retraining pipeline with performance monitoring, concept drift detection, continuous integration for ML models, and automated deployment of improved versions
- Multi-language support expansion beyond Polish to include English, German, and other European languages with corresponding transformer models and language-specific processing
