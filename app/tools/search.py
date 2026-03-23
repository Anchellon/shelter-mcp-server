import logging

from app.config import settings
from app.embeddings import embed_query
from app.tools.db import get_pool

logger = logging.getLogger(__name__)

# Fields returned to the LLM — deliberately excludes embedding and embedding_text
# to keep context window usage low. Claude doesn't need the raw prose or vector.
_SEARCH_FIELDS = [
    "id",
    "service_id",
    "resource_id",
    "latitude",
    "longitude",
    "category_names",
    "parent_category_names",
    "sfsg_category_names",
    "ucsf_top_category_names",
    "ucsf_sub_category_names",
    "our415_category_names",
    "eligibility_age",
    "eligibility_gender",
    "eligibility_housing",
    "eligibility_health",
    "eligibility_financial",
    "eligibility_immigration",
    "eligibility_all",
    "schedule",
]

# Details include the prose text for a single service
_DETAIL_FIELDS = _SEARCH_FIELDS + ["embedding_text"]


async def search_services(query: str, limit: int = 5) -> list[dict]:
    """
    Semantic similarity search for social services matching the user's need.
    Returns up to `limit` services ranked by relevance to the query.
    Each result includes category, eligibility, location, and schedule metadata.
    Do not include raw embeddings or prose text — use get_service_details for full detail.
    """
    logger.info(f"search_services: query='{query[:80]}', limit={limit}")
    vector = await embed_query(query)
    pool = await get_pool()

    fields = ", ".join(_SEARCH_FIELDS)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {fields}
            FROM {settings.pgvector_table}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            str(vector),
            limit,
        )

    results = [dict(r) for r in rows]
    logger.info(f"search_services: returned {len(results)} results")
    return results


async def get_service_details(service_id: int) -> dict | None:
    """
    Returns full details for a specific service by its service_id, including
    the prose description used for search. Use this after search_services to
    surface complete information about a specific result.
    """
    logger.info(f"get_service_details: service_id={service_id}")
    pool = await get_pool()

    fields = ", ".join(_DETAIL_FIELDS)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT {fields}
            FROM {settings.pgvector_table}
            WHERE service_id = $1
            LIMIT 1
            """,
            service_id,
        )

    if row is None:
        logger.warning(f"get_service_details: service_id={service_id} not found")
        return None

    return dict(row)
