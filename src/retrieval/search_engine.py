"""Vector-based retrieval engine for multimodal embeddings.

This module provides a lightweight retrieval client that embeds text queries
with Vertex AI and performs nearest-neighbor search in PostgreSQL/pgvector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from psycopg2.extras import RealDictCursor

from src.config.settings import load_settings
from src.embedding.db_manager import DatabaseError, get_connection
from src.embedding.embedder import VertexClientConfig, VertexEmbeddingError, VertexMultimodalEmbedder


logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 1408


class SearchEngineError(RuntimeError):
    """Raised when retrieval fails in VectorSearchEngine."""


class VectorSearchEngine:
    """Search engine that performs pgvector similarity retrieval.

    The engine embeds a text query into the same 1408-dimensional vector space
    as the indexed multimodal records, and then ranks rows by cosine distance
    using pgvector's ``<=>`` operator.
    """

    def __init__(self) -> None:
        """Initialize the search engine and Vertex embedder.

        Raises:
            KeyError: If required settings are missing from environment
                variables.
            SearchEngineError: If the embedder cannot be initialized.
        """
        settings = load_settings()
        config = VertexClientConfig(
            project_id=settings.gcp_project_id,
            location=settings.gcp_region,
        )
        try:
            self._embedder = VertexMultimodalEmbedder(config=config)
        except Exception as exc:
            logger.exception("Failed to initialize VertexMultimodalEmbedder.")
            raise SearchEngineError("Failed to initialize embedding client.") from exc

    @staticmethod
    def _format_vector_literal(vector: List[float]) -> str:
        """Format an embedding vector into pgvector literal syntax.

        Args:
            vector: Embedding values.

        Returns:
            A vector literal such as ``[0.1,0.2,...]`` accepted by pgvector.
        """
        values = ",".join(f"{v:.8f}" for v in vector)
        return f"[{values}]"

    def search(
        self,
        query_text: str,
        top_k_image: int = 3,
        top_k_text: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search relevant records for the given text query via federated tracks.

        Retrieval flow:
        1. Embed ``query_text`` to a 1408-dimensional vector.
        2. Run two independent pgvector tracks:
           - image track (``content_type = 'image'``)
           - text track (``content_type IN ('text', 'transcript')``)
        3. Merge tracks with ``UNION ALL`` and sort by similarity.
        4. Return ranked rows as dictionaries.

        Args:
            query_text: Natural-language query describing desired content.
            top_k_image: Number of image results to return from image track.
            top_k_text: Number of text/transcript results to return from text track.

        Returns:
            A list of result dictionaries. Each item includes:
            ``id``, ``similarity``, ``source_file``, ``image_path``,
            ``timestamp_or_page``, and ``content_payload``.

        Raises:
            ValueError: If query text is empty or top-k values are invalid.
            SearchEngineError: If embedding generation or database query fails.
        """
        if not query_text.strip():
            raise ValueError("query_text must not be empty.")
        if top_k_image <= 0:
            raise ValueError("top_k_image must be a positive integer.")
        if top_k_text <= 0:
            raise ValueError("top_k_text must be a positive integer.")

        try:
            query_vector = self._embedder.get_text_embedding(query_text)
        except VertexEmbeddingError as exc:
            logger.exception("Failed to generate query embedding.")
            raise SearchEngineError("Failed to generate query embedding.") from exc
        except Exception as exc:
            logger.exception("Unexpected embedding error.")
            raise SearchEngineError("Unexpected embedding error.") from exc

        if len(query_vector) != EXPECTED_EMBEDDING_DIM:
            logger.warning(
                "Query embedding dimension mismatch. expected=%d actual=%d",
                EXPECTED_EMBEDDING_DIM,
                len(query_vector),
            )

        vector_literal = self._format_vector_literal(query_vector)

        sql = """
        (
            SELECT
                id,
                content_type,
                source_file,
                timestamp AS timestamp_or_page,
                content_payload,
                1 - (embedding <=> %s::vector) AS similarity
            FROM multimodal_embeddings
            WHERE content_type = 'image'
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        )
        UNION ALL
        (
            SELECT
                id,
                content_type,
                source_file,
                timestamp AS timestamp_or_page,
                content_payload,
                1 - (embedding <=> %s::vector) AS similarity
            FROM multimodal_embeddings
            WHERE content_type IN ('text', 'transcript')
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        )
        ORDER BY similarity DESC;
        """

        try:
            with get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        sql,
                        (
                            vector_literal,
                            vector_literal,
                            top_k_image,
                            vector_literal,
                            vector_literal,
                            top_k_text,
                        ),
                    )
                    rows = cur.fetchall()
        except DatabaseError as exc:
            logger.exception("Database connection failed during retrieval.")
            raise SearchEngineError("Database connection failed during retrieval.") from exc
        except Exception as exc:
            logger.exception("Database query failed during retrieval.")
            raise SearchEngineError("Database query failed during retrieval.") from exc

        results: List[Dict[str, Any]] = []
        for row in rows:
            source_file = str(row["source_file"])
            results.append(
                {
                    "id": int(row["id"]),
                    "content_type": str(row["content_type"]),
                    "similarity": float(row["similarity"]),
                    "source_file": source_file,
                    "image_path": source_file,
                    "timestamp_or_page": str(row["timestamp_or_page"]),
                    "content_payload": row["content_payload"],
                }
            )

        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    test_query = "skier leaning heavily into a carving turn"

    try:
        engine = VectorSearchEngine()
        results = engine.search(test_query, top_k_image=3, top_k_text=3)
    except (SearchEngineError, ValueError, KeyError):
        logger.exception("Search test failed.")
        raise SystemExit(1)

    if not results:
        logger.info("No retrieval results found for query: %s", test_query)
        raise SystemExit(0)

    logger.info("Top %d results for query: %s", len(results), test_query)
    for idx, item in enumerate(results, start=1):
        logger.info(
            "Top %d | image_path=%s | similarity=%.6f",
            idx,
            item["image_path"],
            item["similarity"],
        )
