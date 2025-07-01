"""
Wykop Social Media Data Processing Application.

This package provides a complete pipeline for fetching, processing, and analyzing
social media posts from Wykop.pl using distributed task processing, machine learning,
and monitoring capabilities.

Modules:
    celery_app: Celery application configuration
    tasks: Distributed tasks for data processing
    models: Data models and schemas
    mongo_client: MongoDB connection utilities
    utils: Helper functions for text processing
    metrics: Prometheus metrics collection
    regression: Machine learning models for prediction
"""

__version__ = "1.0.0"
__author__ = "Adam Bednarski"
__description__ = "Distributed social media data processing and analytics platform"

try:
    from .celery_app import app as celery_app
    from .models import PostData
    from .mongo_client import get_mongo_collection
except ImportError:
    pass
