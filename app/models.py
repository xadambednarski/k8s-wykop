from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PostData:
    post_id: str
    author: str
    text: str
    pluses: int
    comments: int
    created_at: str
    lang: Optional[str] = None
    vector: Optional[List[float]] = None
