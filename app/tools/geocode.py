import logging

import httpx

logger = logging.getLogger(__name__)

# SF bounding box: west, south, east, north
_SF_VIEWBOX = "-122.52,37.70,-122.35,37.84"

# Nominatim requires a descriptive User-Agent per usage policy
_USER_AGENT = "shelter-navigator/1.0 (sheltertech.org)"

# Request timeout in seconds
_TIMEOUT = 5.0


async def geocode_location(location_text: str) -> dict | None:
    """
    Converts a location string (neighborhood, zipcode, or address) to
    latitude/longitude coordinates, scoped to San Francisco.

    Returns {lat, lng, display_name} or None if the location cannot be resolved.
    """
    query = f"{location_text}, San Francisco, CA"
    logger.info(f"geocode_location: query='{query}'")

    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "us",
        "viewbox": _SF_VIEWBOX,
        "bounded": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
            results = response.json()

        if not results:
            logger.warning(f"geocode_location: no results for '{location_text}'")
            return None

        top = results[0]
        return {
            "lat": float(top["lat"]),
            "lng": float(top["lon"]),
            "display_name": top["display_name"],
        }

    except httpx.TimeoutException:
        logger.warning(f"geocode_location: timeout for '{location_text}'")
        return None
    except Exception as e:
        logger.error(f"geocode_location: error for '{location_text}': {e}")
        return None
