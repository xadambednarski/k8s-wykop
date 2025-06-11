from datetime import timedelta
from celery import Celery
import os


app = Celery(
    "celery_app",
    broker=os.environ.get("REDIS_HOST"),
    backend=os.environ.get("REDIS_HOST"),
)
app.conf.beat_schedule = {
    "fetch-posts":  {
        "task": "tasks.fetch_wykop_posts",
        "schedule": timedelta(minutes=1),
    }
}

import tasks
