"""
Prometheus metrics collection and management.

This module handles metrics collection for monitoring Wykop data processing,
including post counts, language detection, and performance metrics.
"""

import os
from typing import Optional
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import redis


registry: CollectorRegistry = CollectorRegistry()

redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """
    Get or create Redis client for metrics storage.

    Returns:
        redis.Redis: Configured Redis client

    Raises:
        ConnectionError: If unable to connect to Redis
    """
    global redis_client

    if redis_client is None:
        redis_host = os.environ.get("REDIS_HOST", "mini-project-redis-master")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))

        try:
            redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    return redis_client


post_gauge = Gauge(
    "wykop_posts_total", "Total number of processed posts", registry=registry
)

lang_gauge = Gauge(
    "wykop_language_detected_total",
    "Number of posts by detected language",
    ["lang"],
    registry=registry,
)

fetch_duration = Gauge(
    "wykop_fetch_duration_seconds",
    "Duration of last fetch operation in seconds",
    registry=registry,
)

fetched_count = Gauge(
    "wykop_last_fetch_post_count",
    "Number of posts fetched in the last request",
    registry=registry,
)

plus_metric = Gauge(
    "wykop_post_pluses",
    "Number of pluses per post and author",
    ["post_id", "author"],
    registry=registry,
)

comment_metric = Gauge(
    "wykop_post_comments",
    "Number of comments per post and author",
    ["post_id", "author"],
    registry=registry,
)

POST_KEY: str = "posts_total"
LANG_KEY: str = "posts_by_lang"


def increment_metrics(language: str) -> None:
    """
    Increment metrics for processed posts.

    Args:
        language: Detected language code for the post

    Raises:
        ConnectionError: If Redis connection fails
    """
    try:
        client = get_redis_client()
        client.incr(POST_KEY)
        client.hincrby(LANG_KEY, language, 1)
    except redis.ConnectionError as e:
        raise ConnectionError(f"Failed to increment metrics in Redis: {e}")


def push_metrics() -> None:
    """
    Push current metrics to Prometheus Gateway.

    Raises:
        ConnectionError: If unable to connect to Redis or Prometheus Gateway
    """
    try:
        client = get_redis_client()

        total = int(client.get(POST_KEY) or 0)
        lang_counts = client.hgetall(LANG_KEY)

        post_gauge.set(total)
        for lang, count in lang_counts.items():
            lang_gauge.labels(lang=lang).set(int(count))

        gateway_host = os.environ.get("PTG_HOST")
        if not gateway_host:
            raise ValueError("PTG_HOST environment variable is required")

        push_to_gateway(gateway_host, job="vectorizer", registry=registry)

    except (redis.ConnectionError, ValueError) as e:
        raise ConnectionError(f"Failed to push metrics: {e}")


def reset_metrics() -> None:
    """Reset all metrics in Redis (useful for testing)."""
    try:
        client = get_redis_client()
        client.delete(POST_KEY, LANG_KEY)
    except redis.ConnectionError as e:
        raise ConnectionError(f"Failed to reset metrics: {e}")
