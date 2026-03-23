import logging

from langchain_ollama import OllamaEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

_embeddings: OllamaEmbeddings | None = None


def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
    return _embeddings


async def embed_query(text: str) -> list[float]:
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(text)
    logger.debug(f"Embedded query (dim: {len(vector)})")
    return vector
