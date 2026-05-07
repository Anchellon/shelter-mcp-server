"""
Integration tests for search tools.
Requires a running pgvector DB and Ollama instance.
Set env vars before running — see .env.example

⚠️ These tests are currently broken — fix in a follow-up. Two issues:

1. `search_services()` is called with `limit=N` in four tests below, but the
   function signature does not accept a `limit` keyword argument. Calls raise
   TypeError. The function caps internally at _MAX_RESULTS = 50 instead.
   Either add a `limit` parameter to search_services, or drop the kwargs and
   adjust assertions to match the internal cap.

2. `test_search_services_returns_results` asserts `"embedding_text" not in
   results[0]`, with a comment about "context window protection". But the
   current `_SEARCH_FIELDS` in app/tools/search.py explicitly includes
   `"embedding_text"` and search_services returns it. Either the code drifted
   from the intended schema or the test is stale. format_results in shelter-
   chat-api currently relies on embedding_text being present in search_services
   results, so removing it would be a cross-repo break — most likely the test
   should be updated, not the code.

TODO: triage and fix these in a dedicated PR. None of the production behaviour
depends on these tests passing right now, but they are misleading as-is.
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
