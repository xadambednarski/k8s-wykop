"""
Celery tasks for Wykop social media data processing pipeline.

This module contains distributed tasks for fetching posts from Wykop API,
processing text content, detecting languages, vectorizing text using
transformer MODELs, and storing results in MongoDB.
"""

import os
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
import requests
import torch
from transformers import AutoTOKENIZER, AutoMODEL
from celery_app import app
from celery import shared_task
from MODELs import PostData
from mongo_client import get_mongo_collection
from utils import (
    load_last_timestamp,
    save_last_timestamp,
    detect_language,
    clean_text,
)
from metrics import (
    fetch_duration,
    fetched_count,
    plus_metric,
    comment_metric,
    registry,
    increment_metrics,
    push_metrics,
)
from prometheus_client import pushadd_to_gateway
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
torch.set_num_threads(1)

try:
    MODEL = AutoMODEL.from_pretrained("../MODELs/herbert", use_safetensors=True)
    TOKENIZER = AutoTOKENIZER.from_pretrained("../MODELs/herbert", use_safetensors=True)
except Exception as e:
    logging.error(f"Failed to load transformer MODEL: {e}")
    MODEL = None
    TOKENIZER = None

WYKOP_API_TOKEN = os.environ.get("WYKOP_API_TOKEN")
if not WYKOP_API_TOKEN:
    raise ValueError("WYKOP_API_TOKEN environment variable is required")

logger = logging.getLogger(__name__)

WYKOP_API_PARAMS = {
    "page": 1,
    "limit": 100,
    "sort": "all",
    "type": "all",
    "multimedia": "false",
}
WYKOP_API_URL = "https://wykop.pl/api/v3/tags/heheszki/stream"


def _make_api_request() -> Optional[Dict[str, Any]]:
    """
    Make request to Wykop API with proper error handling.

    Returns:
        Dict containing API response data, or None if request fails
    """
    try:
        headers = {"Authorization": f"Bearer {WYKOP_API_TOKEN}"}
        response = requests.get(
            WYKOP_API_URL, headers=headers, timeout=10, params=WYKOP_API_PARAMS
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Wykop API request failed: {e}")
        return None
    except ValueError as e:
        logger.error(f"Failed to parse API response: {e}")
        return None


def _parse_entry_data(entry: Dict[str, Any]) -> Optional[PostData]:
    """
    Parse raw API entry data into PostData object.

    Args:
        entry: Raw entry data from Wykop API

    Returns:
        PostData object or None if parsing fails
    """
    try:
        post_id = str(entry.get("id", ""))
        if not post_id:
            logger.warning("Entry missing ID, skipping")
            return None

        author = entry.get("author", {}).get("username", "")
        if not author:
            logger.warning(f"Entry {post_id} missing author, skipping")
            return None

        raw_text = str(entry.get("content", "")).strip()
        text = clean_text(raw_text)

        votes = entry.get("votes", {})
        pluses = int(votes.get("up", 0)) if votes else 0

        comments_data = entry.get("comments", {})
        comments = int(comments_data.get("count", 0)) if comments_data else 0

        created_at_str = entry.get("created_at", "")
        if not created_at_str:
            logger.warning(f"Entry {post_id} missing created_at, skipping")
            return None

        created_at = datetime.datetime.fromisoformat(created_at_str).isoformat()

        return PostData(
            post_id=post_id,
            author=author,
            text=text,
            pluses=pluses,
            comments=comments,
            created_at=created_at,
        )

    except (ValueError, KeyError) as e:
        logger.error(f"Failed to parse entry data: {e}")
        return None


def _process_entries(
    entries: List[Dict[str, Any]], last_timestamp: float
) -> Tuple[int, float]:
    """
    Process API entries and queue tasks for new posts.

    Args:
        entries: List of entry data from API
        last_timestamp: Last processed timestamp

    Returns:
        Tuple of (new_posts_count, latest_timestamp)
    """
    new_posts_count = 0
    latest_timestamp = last_timestamp

    for entry in entries:
        try:
            entry_timestamp = datetime.datetime.fromisoformat(
                entry["created_at"]
            ).timestamp()

            if entry_timestamp <= last_timestamp:
                continue

            latest_timestamp = max(latest_timestamp, entry_timestamp)

            post_data = _parse_entry_data(entry)
            if not post_data:
                continue

            plus_metric.labels(post_id=post_data.post_id, author=post_data.author).set(
                post_data.pluses
            )

            comment_metric.labels(
                post_id=post_data.post_id, author=post_data.author
            ).set(post_data.comments)

            new_posts_count += 1

            detect_language_task.delay(post_data.__dict__)

        except Exception as e:
            logger.error(f"Error processing entry: {e}")
            continue

    return new_posts_count, latest_timestamp


def _push_metrics_to_gateway() -> None:
    """Push metrics to Prometheus Gateway with error handling."""
    try:
        gateway_host = os.environ.get("PTG_HOST")
        if gateway_host:
            pushadd_to_gateway(
                gateway_host,
                job="wykop_scraper",
                grouping_key={"instance": "celery-worker"},
                registry=registry,
            )
    except Exception as e:
        logger.error(f"Failed to push metrics to Prometheus Gateway: {e}")


@shared_task()
def fetch_wykop_posts() -> Dict[str, Any]:
    """
    Fetch new posts from Wykop API and process them.

    This task is scheduled to run periodically and fetches new posts
    from the Wykop API, processes them, and queues follow-up tasks.

    Returns:
        Dict with processing statistics
    """
    start_time = datetime.datetime.now()
    last_timestamp = load_last_timestamp()

    logger.info(f"Starting fetch task, last timestamp: {last_timestamp}")

    api_data = _make_api_request()
    if not api_data or "data" not in api_data:
        logger.error("Failed to fetch data from Wykop API")
        return {"error": "API request failed", "new_posts": 0}

    entries = api_data["data"]
    if not entries:
        logger.info("No entries in API response")
        return {"new_posts": 0}

    new_posts_count, latest_timestamp = _process_entries(entries, last_timestamp)

    duration = (datetime.datetime.now() - start_time).total_seconds()
    fetch_duration.set(duration)
    fetched_count.set(new_posts_count)

    push_metrics()
    _push_metrics_to_gateway()

    if latest_timestamp > last_timestamp:
        save_last_timestamp(latest_timestamp)

    logger.info(f"Fetch completed: {new_posts_count} new posts in {duration:.2f}s")

    return {
        "new_posts": new_posts_count,
        "duration_seconds": duration,
        "last_timestamp": latest_timestamp,
    }


@shared_task()
def detect_language_task(post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Detect language of post content and queue vectorization if Polish.

    Args:
        post_data: Dictionary containing post data

    Returns:
        Dict with language detection results or None if processing fails
    """
    try:
        post = PostData(**post_data)
        post.lang = detect_language(post.text)

        increment_metrics(post.lang)
        _push_metrics_to_gateway()

        if post.lang != "pl":
            logger.info(f"Post ID={post.post_id} skipped (language: {post.lang})")
            return {"post_id": post.post_id, "language": post.lang, "processed": False}

        vectorize_post_batch.delay(post.__dict__)

        return {"post_id": post.post_id, "language": post.lang, "processed": True}

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return None


@shared_task()
def vectorize_post_batch(post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate vector embeddings for post text using transformer MODEL.

    Args:
        post_data: Dictionary containing post data

    Returns:
        Dict with vectorization results or None if processing fails
    """
    if MODEL is None or TOKENIZER is None:
        logger.error("Transformer MODEL not available for vectorization")
        return None

    try:
        post = PostData(**post_data)

        inputs = TOKENIZER(
            post.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = MODEL(**inputs)

        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask

        post.vector = sentence_embedding.cpu().numpy().tolist()[0]

        save_post.delay(post.__dict__)

        logger.info(f"Vectorized post ID={post.post_id}")

        return {
            "post_id": post.post_id,
            "vector_dim": len(post.vector),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        return None


@shared_task()
def save_post(post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Save processed post data to MongoDB.

    Args:
        post_data: Dictionary containing complete post data with vectors

    Returns:
        Dict with save operation results or None if save fails
    """
    try:
        collection = get_mongo_collection()

        post_data["created_at"] = datetime.datetime.fromisoformat(
            post_data["created_at"]
        )

        result = collection.insert_one(post_data)

        logger.info(f"Saved post ID={post_data['post_id']} to MongoDB")

        return {
            "post_id": post_data["post_id"],
            "mongodb_id": str(result.inserted_id),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Failed to save post to MongoDB: {e}")
        return None
