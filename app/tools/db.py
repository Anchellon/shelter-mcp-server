import logging

import asyncpg

from app.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def init_pool() -> None:
    global _pool
    logger.info(f"Initializing DB pool ({settings.db_pool_min}-{settings.db_pool_max} connections)...")
    _pool = await asyncpg.create_pool(
        host=settings.pgvector_host,
        port=settings.pgvector_port,
        database=settings.pgvector_db,
        user=settings.pgvector_user,
        password=settings.pgvector_password,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        timeout=settings.db_pool_timeout,
        command_timeout=30.0,
    )
    logger.info("DB pool initialized.")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("DB pool closed.")
