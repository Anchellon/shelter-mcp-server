import logging
import os

from langchain_aws import BedrockEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

_embeddings: BedrockEmbeddings | None = None


def get_embeddings() -> BedrockEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return _embeddings


async def embed_query(text: str) -> list[float]:
    vector = await get_embeddings().aembed_query(text)
    logger.debug(f"Embedded query (dim: {len(vector)})")
    return vector
