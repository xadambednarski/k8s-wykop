from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import redis
import os


registry = CollectorRegistry()
redis_client = redis.Redis(host="mini-project-redis-master", port=6379, decode_responses=True)

post_gauge = Gauge("wykop_posts_total", "Total posts", registry=registry)
lang_gauge = Gauge("wykop_language_detected_total", "Detected languages", ["lang"], registry=registry)

POST_KEY = "posts_total"
LANG_KEY = "posts_by_lang" 


def increment_metrics(lang: str):
    redis_client.incr(POST_KEY)
    redis_client.hincrby(LANG_KEY, lang, 1)


def push_metrics():
    total = int(redis_client.get(POST_KEY) or 0)
    lang_counts = redis_client.hgetall(LANG_KEY)

    post_gauge.set(total)
    for lang, count in lang_counts.items():
        lang_gauge.labels(lang=lang).set(int(count))

    push_to_gateway(os.environ.get("PTG_HOST"), job="vectorizer", registry=registry)


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
