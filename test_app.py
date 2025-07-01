"""
Unit tests for Wykop data processing application.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from app.models import PostData
from app.utils import (
    clean_text,
    detect_language,
    load_last_timestamp,
    save_last_timestamp,
)


class TestPostData(unittest.TestCase):
    def test_valid_post_data(self) -> None:
        post = PostData(
            post_id="12345",
            author="test_user",
            text="Test post",
            pluses=10,
            comments=5,
            created_at="2024-01-01T10:00:00",
        )

        self.assertEqual(post.post_id, "12345")
        self.assertEqual(post.author, "test_user")
        self.assertEqual(post.pluses, 10)
        self.assertEqual(post.comments, 5)
        self.assertIsNone(post.lang)
        self.assertIsNone(post.vector)

    def test_post_data_validation(self) -> None:
        with self.assertRaises(ValueError):
            PostData(
                post_id="",
                author="test_user",
                text="test",
                pluses=0,
                comments=0,
                created_at="2024-01-01T10:00:00",
            )

        with self.assertRaises(ValueError):
            PostData(
                post_id="12345",
                author="test_user",
                text="test",
                pluses=-1,
                comments=0,
                created_at="2024-01-01T10:00:00",
            )


class TestUtils(unittest.TestCase):
    def test_clean_text(self) -> None:
        text_with_url = "Check this out http://example.com great stuff"
        cleaned = clean_text(text_with_url)
        self.assertNotIn("http://example.com", cleaned)

        text_with_spaces = "Too    many     spaces"
        cleaned = clean_text(text_with_spaces)
        self.assertEqual(cleaned, "Too many spaces")

        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")

    @patch("app.utils.detect")
    def test_detect_language(self, mock_detect: Mock) -> None:
        mock_detect.return_value = "pl"
        result = detect_language("To jest polski tekst")
        self.assertEqual(result, "pl")

        mock_detect.side_effect = Exception("Detection failed")
        result = detect_language("Some text")
        self.assertEqual(result, "unknown")

        result = detect_language("")
        self.assertEqual(result, "unknown")

    @patch("builtins.open")
    def test_timestamp_operations(self, mock_open: Mock) -> None:
        mock_file = MagicMock()
        mock_file.read.return_value = "1234567890.123456"
        mock_open.return_value.__enter__.return_value = mock_file

        timestamp = load_last_timestamp()
        self.assertEqual(timestamp, 1234567890.123456)

        save_last_timestamp(1234567890.123456)
        mock_open.assert_called_with(".last_timestamp", "w", encoding="utf-8")


class TestTaskHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_entry = {
            "id": "12345",
            "content": "Test post content",
            "author": {"username": "test_user"},
            "votes": {"up": 10},
            "comments": {"count": 5},
            "created_at": "2024-01-01T10:00:00",
        }

    def test_parse_entry_data_valid(self) -> None:
        from app.tasks import _parse_entry_data

        result = _parse_entry_data(self.sample_entry)

        self.assertIsNotNone(result)
        self.assertEqual(result.post_id, "12345")
        self.assertEqual(result.author, "test_user")
        self.assertEqual(result.pluses, 10)
        self.assertEqual(result.comments, 5)

    def test_parse_entry_data_missing_fields(self) -> None:
        from app.tasks import _parse_entry_data

        entry_no_id = self.sample_entry.copy()
        del entry_no_id["id"]
        result = _parse_entry_data(entry_no_id)
        self.assertIsNone(result)

        entry_no_author = self.sample_entry.copy()
        entry_no_author["author"] = {}
        result = _parse_entry_data(entry_no_author)
        self.assertIsNone(result)


class TestMongoClient(unittest.TestCase):
    @patch.dict("os.environ", {"MONGO_HOST": "mongodb://localhost:27017"})
    @patch("app.mongo_client.MongoClient")
    def test_get_mongo_client(self, mock_mongo_client: Mock) -> None:
        from app.mongo_client import get_mongo_client

        import app.mongo_client

        app.mongo_client._MONGO_CLIENT = None

        client = get_mongo_client()
        mock_mongo_client.assert_called_once_with("mongodb://localhost:27017")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_mongo_client_no_host(self) -> None:
        from app.mongo_client import get_mongo_client

        import app.mongo_client

        app.mongo_client._MONGO_CLIENT = None

        with self.assertRaises(ValueError):
            get_mongo_client()


class TestMetrics(unittest.TestCase):
    @patch("app.metrics.redis.Redis")
    def test_increment_metrics(self, mock_redis_class: Mock) -> None:
        from app.metrics import increment_metrics

        import app.metrics

        app.metrics.redis_client = None

        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True

        increment_metrics("pl")

        mock_redis.incr.assert_called_once_with("posts_total")
        mock_redis.hincrby.assert_called_once_with("posts_by_lang", "pl", 1)


if __name__ == "__main__":
    unittest.main()
