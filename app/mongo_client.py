"""
MongoDB client configuration and utilities.

This module provides functions for connecting to MongoDB and retrieving
collections for storing Wykop post data.
"""

import os
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection


_MONGO_CLIENT: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    """
    Get or create a MongoDB client instance.

    Returns:
        MongoClient: Configured MongoDB client

    Raises:
        ValueError: If MONGO_HOST environment variable is not set
    """
    global _MONGO_CLIENT

    if _MONGO_CLIENT is None:
        mongo_host = os.environ.get("MONGO_HOST")
        if not mongo_host:
            raise ValueError("MONGO_HOST environment variable is required")

        _MONGO_CLIENT = MongoClient(mongo_host)

    return _MONGO_CLIENT


def get_mongo_collection(
    database_name: str = "wykopdb",
    collection_name: str = "posts"
) -> Collection:
    """
    Get MongoDB collection for storing posts.

    Args:
        database_name: Name of the MongoDB database (default: "wykopdb")
        collection_name: Name of the collection (default: "posts")

    Returns:
        Collection: MongoDB collection object
    """
    client = get_mongo_client()
    return client[database_name][collection_name]


def close_mongo_connection() -> None:
    """Close the MongoDB connection if it exists."""
    global _MONGO_CLIENT
    if _MONGO_CLIENT is not None:
        _MONGO_CLIENT.close()
        _MONGO_CLIENT = None
