import os
import datetime
import logging
import requests
import torch

from transformers import AutoTokenizer, AutoModel
from celery_app import app
from celery import shared_task
from models import PostData
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

model = AutoModel.from_pretrained("../models/herbert", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("../models/herbert", use_safetensors=True)

token = os.environ["WYKOP_API_TOKEN"]
logger = logging.getLogger(__name__)

PARAMS = {"page": 1, "limit": 100, "sort": "all", "type": "all", "multimedia": "false"}
URL = "https://wykop.pl/api/v3/tags/heheszki/stream"


@shared_task()
def fetch_wykop_posts():
    new_fetched_count = 0
    last_timestamp = load_last_timestamp()
    start_time = datetime.datetime.now()

    def process_entries(entries, last_timestamp):
        nonlocal new_fetched_count
        for entry in entries:
            try:
                timestamp = datetime.datetime.fromisoformat(
                    entry["created_at"]
                ).timestamp()

                if timestamp <= last_timestamp:
                    continue

                raw_text = str(entry.get("content", "")).strip()
                text = clean_text(raw_text)
                author = str(entry["author"]["username"])
                post_id = str(entry["id"])
                pluses = int(entry["votes"]["up"]) if entry.get("votes") else 0
                comments = (
                    int(entry["comments"]["count"]) if entry.get("comments") else 0
                )
                created_at = datetime.datetime.fromisoformat(
                    entry["created_at"]
                ).isoformat()

                post = PostData(
                    post_id=post_id,
                    author=author,
                    text=text,
                    pluses=pluses,
                    comments=comments,
                    created_at=created_at,
                )

                plus_metric.labels(post_id=post_id, author=author).set(pluses)
                comment_metric.labels(post_id=post_id, author=author).set(comments)
                new_fetched_count += 1

                detect_language_task.delay(post.__dict__)

            except Exception as e:
                logger.error("Error processing entry: %s", e)
                continue

    try:
        response = requests.get(
            URL, headers={"Authorization": f"Bearer {token}"}, timeout=10, params=PARAMS
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Initial request failed: {e}")
        return

    process_entries(data["data"], last_timestamp)

    duration = (datetime.datetime.now() - start_time).total_seconds()
    fetch_duration.set(duration)
    fetched_count.set(new_fetched_count)
    push_metrics()

    save_last_timestamp(
        datetime.datetime.fromisoformat(data["data"][-1]["created_at"]).timestamp()
    )

    try:
        pushadd_to_gateway(
            os.environ.get("PTG_HOST"),
            job="wykop_scraper",
            grouping_key={"instance": "celery-worker"},
            registry=registry,
        )
    except Exception as e:
        logger.error("Failed to push metrics to Prometheus: %s", e)


@shared_task()
def detect_language_task(post: dict):
    post_data = PostData(**post)
    post_data.lang = detect_language(post_data.text)
    increment_metrics(post_data.lang)

    try:
        pushadd_to_gateway(
            os.environ.get("PTG_HOST"),
            job="wykop_scraper",
            grouping_key={"instance": "celery-worker"},
            registry=registry,
        )
    except Exception as e:
        logger.error("Failed to push metrics to Prometheus: %s", e)

    if post_data.lang != "pl":
        logger.info(
            "Post ID=%s skipped due to non-Polish language: %s",
            post_data.post_id,
            post_data.lang,
        )
        return

    vectorize_post_batch.delay(post)


@shared_task()
def vectorize_post_batch(post: dict):
    try:
        inputs = tokenizer(
            post["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask
        post["vector"] = sentence_embedding.cpu().numpy().tolist()

        save_post.delay(post)
    except Exception as e:
        logger.error("Batch vectorization failed: %s", str(e))


@shared_task()
def save_post(post: dict):
    try:
        collection = get_mongo_collection()
        post["created_at"] = datetime.datetime.fromisoformat(post["created_at"])
        collection.insert_one(post)
    except Exception as e:
        logger.error("Failed to save post vectors to MongoDB: %s", str(e))
