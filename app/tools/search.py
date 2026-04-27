import logging
import math

from app.config import settings
from app.embeddings import embed_query
from app.tools.db import get_pool

logger = logging.getLogger(__name__)

# Fields returned to the LLM — deliberately excludes embedding and embedding_text
# to keep context window usage low. Claude doesn't need the raw prose or vector.
_SEARCH_FIELDS = [
    "service_id",
    "latitude",
    "longitude",
    "schedule",
    "category_names",
    "sfsg_category_names",
    "eligibility_age",
    "eligibility_employment",
    "eligibility_ethnicity",
    "eligibility_family_status",
    "eligibility_financial",
    "eligibility_gender",
    "eligibility_health",
    "eligibility_immigration",
    "eligibility_housing",
    "eligibility_other",
    "eligibility_all",
    "embedding_text",
]

# Details include the prose text for a single service
_DETAIL_FIELDS = _SEARCH_FIELDS + ["embedding_text"]


def _parse_when(when: str) -> tuple[str, int] | None:
    """
    Parse 'Monday 14:00' → ('Monday', 840).
    Returns None if unparseable.
    """
    try:
        parts = when.strip().split()
        day = parts[0].capitalize()
        h, m = parts[1].split(":")
        return day, int(h) * 60 + int(m)
    except Exception:
        return None



def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lng points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


_DEFAULT_RADIUS_FT = 2500.0
_FT_TO_KM = 0.0003048  # 1 ft = 0.3048 m = 0.0003048 km
_MAX_RESULTS = 50


async def search_services(
    query: str,
    categories: list[str] | None = None,
    eligibilities: list[str] | None = None,
    lat: float | None = None,
    lng: float | None = None,
    radius_ft: float = _DEFAULT_RADIUS_FT,
    when: str | None = None,
) -> list[dict]:
    """
    Semantic similarity search for social services matching the user's need.
    When lat/lng are provided, results are restricted to within radius_ft feet
    (default 2500 ft ≈ 0.76 km). Results are sorted by:
      1. similarity score (primary) — with small tag nudges applied
      2. category match nudge: -0.10 applied to score if service has requested category
      3. eligibility match nudge: -0.05 applied to score if eligibility overlaps
      4. distance ascending as tiebreaker
    All results are returned — no hard filters on category or eligibility.
    """
    logger.info(f"search_services: query='{query[:80]}', categories={categories}, eligibilities={eligibilities}, lat={lat}, lng={lng}")
    vector = await embed_query(query)
    pool = await get_pool()

    fields = ", ".join(_SEARCH_FIELDS)

    cat_match = "(CASE WHEN $2::text[] IS NOT NULL AND sfsg_category_names && $2::text[] THEN 1 ELSE 0 END)"
    elig_match = "(CASE WHEN $3::text[] IS NOT NULL AND eligibility_all && $3::text[] THEN 1 ELSE 0 END)"

    dist_expr = f"({lat} - latitude::float8)^2 + ({lng} - longitude::float8)^2" if lat is not None and lng is not None else "0"

    # Schedule filter — only apply if when is provided
    parsed_when = _parse_when(when) if when else None
    if parsed_when:
        when_day, when_mins = parsed_when
        schedule_filter = f"""
            AND (
                schedule IS NULL
                OR jsonb_array_length(schedule) = 0
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements(schedule) e
                    WHERE e->>'day' = '{when_day}'
                    AND (e->>'open_mins')::int <= {when_mins}
                    AND (e->>'close_mins')::int >= {when_mins}
                )
            )
        """
    else:
        schedule_filter = ""

    sql = f"""
        SELECT DISTINCT ON (service_id) {fields},
               {cat_match} AS category_match,
               {elig_match} AS eligibility_match,
               similarity
        FROM (
            SELECT *, embedding <=> $1::vector AS similarity
            FROM {settings.pgvector_table}
            WHERE embedding <=> $1::vector < 0.5 {schedule_filter}
            ORDER BY similarity
        ) ranked
        ORDER BY service_id{f', {dist_expr}' if lat is not None and lng is not None else ''}
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, str(vector), categories or None, eligibilities or None)

    results = [dict(r) for r in rows]

    # Attach distance_km
    for r in results:
        r_lat = r.get("latitude")
        r_lng = r.get("longitude")
        if lat is not None and lng is not None and r_lat is not None and r_lng is not None:
            r["distance_km"] = round(_haversine_km(lat, lng, float(r_lat), float(r_lng)), 2)
        else:
            r["distance_km"] = None

    # Region filter — restrict to radius_ft when coordinates are provided
    if lat is not None and lng is not None:
        radius_km = radius_ft * _FT_TO_KM
        results = [r for r in results if r["distance_km"] is not None and r["distance_km"] <= radius_km]


    # Sort: similarity as primary signal, with tag nudges and distance as tiebreaker
    results.sort(key=lambda r: (
        r["similarity"] - (0.10 * r["category_match"]) - (0.05 * r["eligibility_match"]),
        r["distance_km"] if r["distance_km"] is not None else float("inf"),
    ))

    results = results[:_MAX_RESULTS]
    logger.info(f"search_services: returned {len(results)} results")
    return results


async def list_categories() -> list[str]:
    """
    Returns the available top-level service categories.
    Use this to map a user's 'what' to a known category before searching.
    """
    return [
        "sfsg-domesticviolence",
        "sfsg-health",
        "sfsg-finance",
        "sfsg-food",
        "sfsg-housing",
        "sfsg-hygiene",
        "sfsg-internet",
        "sfsg-jobs",
        "sfsg-lgbtqa",
        "sfsg-substanceuse",
        "sfsg-shelter",
        "sfsg-longterm",
        "sfsg-familyservices",
    ]


async def list_eligibilities() -> dict[str, list[str]]:
    """
    Returns eligible population values grouped by dimension.
    Use this to map a user's 'who' to known eligibility values before searching.
    """
    return {
        "age": [
            "All Ages", "Infants", "Toddlers", "Children", "Teens",
            "Transitional Aged Youth (TAY)", "Adults", "Senior",
        ],
        "housing": [
            "Home Owners", "Home Renters", "Experiencing Homelessness",
            "In Jail", "Near Homeless",
        ],
        "gender": ["LGBTQ+", "Men", "Women"],
        "family_status": [
            "Individuals", "Single Parent", "Married no children",
            "Families with children below 18 years old",
        ],
        "employment": ["Employed", "Retired", "Veterans", "Unemployed"],
        "financial": ["Low-Income", "Uninsured"],
        "health": [
            "HIV/AIDS", "Pregnant", "Special Needs/Disabilities", "Substance Dependency",
            "Visual Impairment", "Deaf or Hard of Hearing",
        ],
        "ethnicity": [
            "African/Black", "API (Asian/Pacific Islander)", "Chinese", "Filipino/a",
            "Jewish", "Latinx", "Middle Eastern and North African",
            "Native American", "Pacific Islander", "Samoan",
        ],
        "immigration": ["Immigrants", "Undocumented"],
        "other": [
            "Anyone in Need", "Disaster Victim", "Domestic Violence Survivors",
            "Gender-Based Violence", "Human Trafficking Survivors",
            "San Francisco Residents", "Sexual Assault Survivors",
            "Trauma Survivors", "Abuse or Neglect Survivors",
        ],
    }


async def get_service_details_batch(service_ids: list[int]) -> list[dict]:
    """
    Returns card-level details for a batch of services by their service_ids.
    Each result includes: service_id, name, long_description, org_name,
    address_1, city, state_province, postal_code, latitude, longitude, phone.
    Address prefers service-level (via addresses_services join table), falling
    back to resource-level. Phone is the first phone on the resource.
    """
    if not service_ids:
        return []

    logger.info(f"get_service_details_batch: {len(service_ids)} service_ids")
    pool = await get_pool()

    sql = """
        SELECT
            s.id                                                AS service_id,
            s.name,
            s.long_description,
            r.id                                                AS resource_id,
            r.name                                              AS org_name,
            COALESCE(sa.address_1,    ra.address_1)            AS address_1,
            COALESCE(sa.city,         ra.city)                 AS city,
            COALESCE(sa.state_province, ra.state_province)     AS state_province,
            COALESCE(sa.postal_code,  ra.postal_code)          AS postal_code,
            COALESCE(sa.latitude,     ra.latitude)             AS latitude,
            COALESCE(sa.longitude,    ra.longitude)            AS longitude,
            p.number                                            AS phone
        FROM services s
        JOIN resources r ON r.id = s.resource_id
        LEFT JOIN LATERAL (
            SELECT a.address_1, a.city, a.state_province, a.postal_code,
                   a.latitude, a.longitude
            FROM addresses a
            JOIN addresses_services ads ON a.id = ads.address_id
            WHERE ads.service_id = s.id
            ORDER BY a.id
            LIMIT 1
        ) sa ON true
        LEFT JOIN LATERAL (
            SELECT address_1, city, state_province, postal_code,
                   latitude, longitude
            FROM addresses
            WHERE resource_id = r.id
            ORDER BY id
            LIMIT 1
        ) ra ON true
        LEFT JOIN LATERAL (
            SELECT number
            FROM phones
            WHERE resource_id = r.id
            ORDER BY id
            LIMIT 1
        ) p ON true
        WHERE s.id = ANY($1::int[])
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, service_ids)

    results = [dict(r) for r in rows]
    logger.info(f"get_service_details_batch: returned {len(results)} results")
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
