"""
Data models for Wykop social media data processing.

This module defines dataclasses and type definitions used throughout
the application for handling Wykop post data.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PostData:
    """
    Data model for a Wykop post.

    Attributes:
        post_id: Unique identifier for the post
        author: Username of the post author
        text: Cleaned text content of the post
        pluses: Number of upvotes/likes the post received
        comments: Number of comments on the post
        created_at: ISO formatted timestamp when post was created
        lang: Detected language of the post content (optional)
        vector: Numerical vector representation of the post text (optional)
    """

    post_id: str
    author: str
    text: str
    pluses: int
    comments: int
    created_at: str
    lang: Optional[str] = None
    vector: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate post data after initialization."""
        if not self.post_id:
            raise ValueError("post_id cannot be empty")
        if not self.author:
            raise ValueError("author cannot be empty")
        if self.pluses < 0:
            raise ValueError("pluses cannot be negative")
        if self.comments < 0:
            raise ValueError("comments cannot be negative")
