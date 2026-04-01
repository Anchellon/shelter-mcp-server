import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from app.tools.db import close_pool, init_pool
from app.tools.geocode import geocode_location
from app.tools.search import get_service_details, search_services

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP):
    await init_pool()
    logger.info("MCP server ready")
    yield
    await close_pool()
    logger.info("MCP server shutdown")


mcp = FastMCP("shelter-search", lifespan=lifespan)

mcp.add_tool(search_services)
mcp.add_tool(get_service_details)
mcp.add_tool(geocode_location)

# Entry point for uvicorn:
# uvicorn app.server:http_app --host 0.0.0.0 --port 8001
http_app = mcp.http_app()
