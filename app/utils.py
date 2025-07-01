"""
Utility functions for text processing and timestamp management.

This module provides helper functions for language detection, text cleaning,
and managing timestamp persistence for the Wykop data pipeline.
"""

import re
from langdetect import detect


POSTS_TIMESTAMP_PATH: str = ".last_timestamp"


def load_last_timestamp() -> float:
    """
    Load the last processed timestamp from file.

    Returns:
        float: Last processed timestamp, or 0.0 if file doesn't exist
    """
    try:
        with open(POSTS_TIMESTAMP_PATH, "r", encoding="utf-8") as f:
            timestamp_str = f.read().strip()
            return float(timestamp_str)
    except (FileNotFoundError, ValueError) as e:
        return 0.0


def save_last_timestamp(timestamp: float) -> None:
    """
    Save the last processed timestamp to file.

    Args:
        timestamp: Unix timestamp to save

    Raises:
        IOError: If unable to write to timestamp file
    """
    try:
        with open(POSTS_TIMESTAMP_PATH, "w", encoding="utf-8") as f:
            f.write(f"{timestamp:.6f}")
    except IOError as e:
        raise IOError(f"Failed to save timestamp: {e}")


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text: Input text to analyze

    Returns:
        str: Detected language code (e.g., 'pl', 'en') or 'unknown' if detection fails
    """
    if not text or not text.strip():
        return "unknown"

    try:
        return detect(text)
    except Exception:
        return "unknown"


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    Args:
        text: Raw text to clean

    Returns:
        str: Cleaned text with URLs removed and normalized whitespace
    """
    if not text:
        return ""

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
