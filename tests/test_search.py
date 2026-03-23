"""
Integration tests for search tools.
Requires a running pgvector DB and Ollama instance.
Set env vars before running — see .env.example
"""
import pytest

from app.tools.db import close_pool, init_pool
from app.tools.search import get_service_details, search_services


@pytest.fixture(autouse=True)
async def db_pool():
    await init_pool()
    yield
    await close_pool()


async def test_search_services_returns_results():
    results = await search_services("emergency shelter", limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        assert "service_id" in results[0]
        assert "category_names" in results[0]
        # Embedding fields must NOT be returned (context window protection)
        assert "embedding" not in results[0]
        assert "embedding_text" not in results[0]


async def test_search_services_respects_limit():
    results = await search_services("food bank", limit=2)
    assert len(results) <= 2


async def test_search_services_empty_query_still_returns():
    results = await search_services("", limit=1)
    assert isinstance(results, list)


async def test_get_service_details_unknown_id():
    result = await get_service_details(999_999_999)
    assert result is None


async def test_get_service_details_includes_prose():
    # Get a real service_id from search first
    results = await search_services("shelter", limit=1)
    if not results:
        pytest.skip("No data in DB")
    service_id = results[0]["service_id"]
    detail = await get_service_details(service_id)
    assert detail is not None
    assert "embedding_text" in detail
    assert "embedding" not in detail
