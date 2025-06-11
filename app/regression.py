import os
import pandas as pd
from typing import List
import torch
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression

from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator


_model = AutoModel.from_pretrained("../models/herbert", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("../models/herbert", use_safetensors=True)


def load_from_database() -> List[dict]:
    client = MongoClient(os.environ.get("MONGO_HOST"))
    db = client["wykopdb"]
    collection = db["posts"]
    posts = collection.find({}, {"vector": 1, "pluses": 1, "comments": 1})
    return list(posts)


def get_features_for_model(posts: List[dict]) -> List[dict]:
    filtered_posts = []
    for post in posts:
        sample = {str(i): v for i, v in enumerate(post["vector"][0])}
        sample["comments"] = post.get("comments", 0)
        sample["label"] = post["pluses"]
        filtered_posts.append(sample)
    return filtered_posts


def load_and_save_posts():
    posts = load_from_database()
    filtered_posts = get_features_for_model(posts)
    df = pd.DataFrame(filtered_posts)

    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/posts.csv", index=False)


def load_model():
    model_path = "../models/linear_regression_model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = PipelineModel.load(model_path)
    return model


def predict(model, sample_text: str):
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = _model(**inputs)

    last_hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentence_embedding = sum_embeddings / sum_mask
    sample_vector = sentence_embedding.cpu().numpy().tolist()

    spark = SparkSession.builder.appName("Predict").getOrCreate()

    sample_dict = {str(i): val for i, val in enumerate(sample_vector[0])}
    sample_dict["comments"] = 0
    sample_df = spark.createDataFrame([sample_dict])

    predictions = model.transform(sample_df)
    return predictions.select("prediction").collect()[0][0]


def train_model():
    spark = (
        SparkSession.builder.master("local[*]").appName("PostRegression").getOrCreate()
    )

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv("../data/posts.csv")
    )

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

    trained_model = pipeline.fit(train_data)

    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    predictions = trained_model.transform(test_data)
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")
    print("Training completed successfully.")

    model_path = "../models/linear_regression_model"
    trained_model.write().overwrite().save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    load_and_save_posts()
    train_model()
    model = load_model()
    sample_text = "Nawrocki to kibol i sutener, ale to dopiero poczatek jego zalet"
    prediction = predict(model, sample_text)
    print(f"Predicted pluses for the sample text: {prediction}")
