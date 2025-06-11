import os
from pymongo import MongoClient


MONGO_CLIENT = None


def get_mongo_collection():
    global MONGO_CLIENT
    if MONGO_CLIENT is None:
        MONGO_CLIENT = MongoClient(os.environ.get("MONGO_HOST"),)["wykopdb"]["posts"]

    return MONGO_CLIENT
