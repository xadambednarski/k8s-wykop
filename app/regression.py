"""
Machine learning regression model for predicting Wykop post popularity.

This module implements a PySpark-based pipeline for training and using
regression models to predict the number of pluses (upvotes) a post will receive
based on its text content vectorized using transformer models.
"""

import os
from typing import Any
import pandas as pd
import logging
import torch
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    MODEL = AutoModel.from_pretrained("../models/herbert", use_safetensors=True)
    TOKENIZER = AutoTokenizer.from_pretrained("../models/herbert", use_safetensors=True)
    logger.info("Successfully loaded Herbert transformer model")
except Exception as e:
    logger.error("Failed to load transformer model: %s", e)
    MODEL = None
    TOKENIZER = None


def get_mongo_client() -> MongoClient:
    """
    Create MongoDB client from environment configuration.

    Returns:
        MongoClient: Configured MongoDB client

    Raises:
        ValueError: If MONGO_HOST environment variable is not set
    """
    mongo_host = os.environ.get("MONGO_HOST")
    if not mongo_host:
        raise ValueError("MONGO_HOST environment variable is required")
    return MongoClient(mongo_host)


def load_from_database() -> list[dict[str, Any]]:
    """
    Load post data with vectors from MongoDB.

    Returns:
        List of post documents containing vectors and target values

    Raises:
        ConnectionError: If unable to connect to MongoDB
    """
    try:
        client = get_mongo_client()
        db = client["wykopdb"]
        collection = db["posts"]

        query = {"vector": {"$exists": True, "$ne": None}}
        projection = {"vector": 1, "pluses": 1, "comments": 1, "_id": 0}

        posts = list(collection.find(query, projection))
        logger.info("Loaded %d posts with vectors from database", len(posts))

        return posts

    except Exception as e:
        logger.error("Failed to load data from MongoDB: %s", e)
        raise ConnectionError("Database connection failed: %s", e)


def get_features_for_model(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Transform post data into features suitable for machine learning.

    Args:
        posts: List of post documents from MongoDB

    Returns:
        List of feature dictionaries ready for training
    """
    filtered_posts = []

    for post in posts:
        try:
            if "vector" not in post or "pluses" not in post:
                continue

            vector = post["vector"]
            if not isinstance(vector, list) or len(vector) == 0:
                continue

            sample = {str(i): float(v) for i, v in enumerate(vector)}

            sample["comments"] = int(post.get("comments", 0))

            sample["label"] = int(post["pluses"])

            filtered_posts.append(sample)

        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid post data: {e}")
            continue

    logger.info("Prepared %d samples for training", len(filtered_posts))
    return filtered_posts


def load_and_save_posts() -> str:
    """
    Load posts from database and save as CSV for Spark processing.

    Returns:
        str: Path to the saved CSV file

    Raises:
        IOError: If unable to save CSV file
    """
    try:
        posts = load_from_database()
        if not posts:
            raise ValueError("No posts found in database")

        filtered_posts = get_features_for_model(posts)
        if not filtered_posts:
            raise ValueError("No valid posts after feature extraction")

        df = pd.DataFrame(filtered_posts)

        data_dir = "../data"
        os.makedirs(data_dir, exist_ok=True)

        csv_path = os.path.join(data_dir, "posts.csv")
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved {len(df)} samples to {csv_path}")
        return csv_path

    except Exception as e:
        logger.error("Failed to save posts data: %s", e)
        raise IOError("Data export failed: %s", e)


def load_model(model_path: str = "../models/linear_regression_model") -> PipelineModel:
    """
    Load trained regression model from disk.

    Args:
        model_path: Path to the saved model directory

    Returns:
        PipelineModel: Loaded Spark ML pipeline model

    Raises:
        FileNotFoundError: If model doesn't exist at specified path
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        model = PipelineModel.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def vectorize_text(text: str) -> list[float]:
    """
    Generate vector embedding for input text using transformer model.

    Args:
        text: Input text to vectorize

    Returns:
        List of float values representing the text embedding

    Raises:
        RuntimeError: If transformer model is not available
    """
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("Transformer model not available")

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    try:
        inputs = TOKENIZER(
            text,
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

        return sentence_embedding.cpu().numpy().tolist()[0]

    except Exception as e:
        logger.error(f"Text vectorization failed: {e}")
        raise RuntimeError(f"Vectorization error: {e}")


def predict(model: PipelineModel, sample_text: str, comments: int = 0) -> float:
    """
    Predict number of pluses for given text using trained model.

    Args:
        model: Trained Spark ML pipeline model
        sample_text: Text content to predict popularity for
        comments: Number of comments (default: 0)

    Returns:
        Predicted number of pluses

    Raises:
        RuntimeError: If prediction fails
    """
    try:
        sample_vector = vectorize_text(sample_text)

        spark = SparkSession.builder.appName("WykopPredict").getOrCreate()

        sample_dict = {str(i): float(val) for i, val in enumerate(sample_vector)}
        sample_dict["comments"] = int(comments)

        sample_df = spark.createDataFrame([sample_dict])

        predictions = model.transform(sample_df)
        prediction_value = predictions.select("prediction").collect()[0][0]

        logger.info(f"Prediction for text: {prediction_value:.2f} pluses")
        return float(prediction_value)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction error: {e}")


def train_model(
    csv_path: str = "../data/posts.csv",
    model_save_path: str = "../models/linear_regression_model",
) -> tuple[float, str]:
    """
    Train regression model on post data using Spark ML.

    Args:
        csv_path: Path to CSV file with training data
        model_save_path: Path where to save trained model

    Returns:
        Tuple of (RMSE score, model_save_path)

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        RuntimeError: If training fails
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data not found at {csv_path}")

    try:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("WykopPostRegression")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )

        df = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_path)
        )

        logger.info(f"Loaded training data: {df.count()} samples")

        feature_cols = [col for col in df.columns if col != "label"]

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")

        scaler = StandardScaler(
            inputCol="raw_features", outputCol="features", withMean=True, withStd=True
        )

        lr = GeneralizedLinearRegression(
            featuresCol="features",
            labelCol="label",
            family="poisson",
            link="log",
            maxIter=50,
            regParam=0.8,
        )

        pipeline = Pipeline(stages=[assembler, scaler, lr])

        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

        logger.info(
            f"Training set: {train_data.count()} samples, "
            f"Test set: {test_data.count()} samples"
        )

        logger.info("Starting model training...")
        trained_model = pipeline.fit(train_data)

        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse"
        )

        predictions = trained_model.transform(test_data)
        rmse = evaluator.evaluate(predictions)

        logger.info(f"Model training completed. RMSE: {rmse:.4f}")

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        trained_model.write().overwrite().save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        return rmse, model_save_path

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise RuntimeError(f"Training error: {e}")


def main() -> None:
    """
    Main function to run the complete training and prediction pipeline.
    """
    try:
        logger.info("Starting Wykop post regression pipeline...")

        csv_path = load_and_save_posts()

        rmse, model_path = train_model(csv_path)

        model = load_model(model_path)

        sample_text = "Nawrocki to kibol i sutener, ale to dopiero poczatek jego zalet"
        prediction = predict(model, sample_text)

        logger.info(f"Sample prediction for text: {prediction:.2f} pluses")
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
