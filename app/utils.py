from langdetect import detect
import re


POSTS_TIMESTAMP_PATH = ".last_timestamp"


def load_last_timestamp():
    try:
        with open(POSTS_TIMESTAMP_PATH, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def save_last_timestamp(ts):
    with open(POSTS_TIMESTAMP_PATH, "w", encoding="utf-8") as f:
        f.write(f"{ts:.6f}")


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def clean_text(text: str) -> str:
    text = re.sub(r"http\\S+|www\\.\\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
