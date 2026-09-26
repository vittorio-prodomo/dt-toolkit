"""Shared pytest configuration and fixtures."""

import pytest


def _mongod_available() -> bool:
    """Return True if a local MongoDB instance is reachable."""
    try:
        import os

        import pymongo

        uri = os.environ.get("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=500)
        client.server_info()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip tests marked requires_mongod when MongoDB is not reachable."""
    if _mongod_available():
        return
    skip = pytest.mark.skip(reason="requires a running mongod on localhost:27017")
    for item in items:
        if item.get_closest_marker("requires_mongod"):
            item.add_marker(skip)
