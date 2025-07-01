"""
Celery application configuration for Wykop social media data processing.

This module configures Celery for distributed task processing, including
periodic tasks for fetching posts from Wykop API.
"""

import os
from datetime import timedelta
from celery import Celery


def get_redis_url() -> str:
    """Get Redis URL from environment variables with fallback."""
    redis_host = os.environ.get("REDIS_HOST", "redis://localhost:6379/0")
    return redis_host


app: Celery = Celery(
    "celery_app",
    broker=get_redis_url(),
    backend=get_redis_url(),
)

app.conf.beat_schedule = {
    "fetch-posts": {
        "task": "tasks.fetch_wykop_posts",
        "schedule": timedelta(minutes=1),
    }
}

app.conf.timezone = "UTC"

import tasks
