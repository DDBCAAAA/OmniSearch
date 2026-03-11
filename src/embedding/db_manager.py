"""Database manager for storing multimodal embeddings.

This module provides helpers to initialize the PostgreSQL schema and insert
multimodal embeddings produced by the embedding layer.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Iterable, Optional

import psycopg2
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import RealDictCursor

from src.config.settings import load_settings
from src.embedding.embedder import VertexClientConfig, VertexMultimodalEmbedder, VertexEmbeddingError

from pathlib import Path
import os


logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised when a database operation fails."""


@contextmanager
def get_connection() -> Generator[PGConnection, None, None]:
    """Yield a PostgreSQL connection using settings from environment variables.

    The connection is always closed when the context exits.

    Yields:
        An open psycopg2 connection.

    Raises:
        DatabaseError: If connecting to the database fails.
    """
    settings = load_settings()
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        logger.debug(
            "Opened PostgreSQL connection to %s:%s/%s",
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
        )
    except Exception as exc:
        logger.exception("Failed to open PostgreSQL connection.")
        raise DatabaseError("Failed to open PostgreSQL connection.") from exc

    try:
        yield conn
    finally:
        try:
            conn.close()
            logger.debug("Closed PostgreSQL connection.")
        except Exception:
            logger.exception("Error while closing PostgreSQL connection.")


def init_db() -> None:
    """Initialize the database schema for multimodal embeddings.

    This function ensures that the ``vector`` extension is enabled and that the
    ``multimodal_embeddings`` table exists with the expected schema.

    The table schema:

    - id: serial primary key
    - content_type: varchar
    - source_file: varchar
    - timestamp: varchar
    - content_payload: text
    - embedding: vector(1408)

    Raises:
        DatabaseError: If any database operation fails.
    """
    logger.info("Initializing PostgreSQL schema for multimodal embeddings.")
    ddl_extension = "CREATE EXTENSION IF NOT EXISTS vector;"
    ddl_table = """
    CREATE TABLE IF NOT EXISTS multimodal_embeddings (
        id SERIAL PRIMARY KEY,
        content_type VARCHAR NOT NULL,
        source_file VARCHAR NOT NULL,
        timestamp VARCHAR NOT NULL,
        content_payload TEXT,
        embedding VECTOR(1408) NOT NULL
    );
    """

    with get_connection() as conn:
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute(ddl_extension)
                cur.execute(ddl_table)
            conn.commit()
            logger.info("Database schema initialized successfully.")
        except Exception as exc:
            conn.rollback()
            logger.exception("Failed to initialize database schema.")
            raise DatabaseError("Failed to initialize database schema.") from exc


def _format_embedding(embedding: Iterable[float], expected_dim: Optional[int] = 1408) -> str:
    """Format an embedding vector for insertion into a pgvector column.

    pgvector expects vector literals in bracket form, e.g. ``[1.0,2.0,3.0]``.

    Args:
        embedding: Iterable of float values representing the embedding.
        expected_dim: Expected dimensionality of the embedding. If provided and
            the length does not match, a warning is logged.

    Returns:
        A string representation suitable for casting to a ``vector`` type in
        PostgreSQL, e.g. ``'[1.0,2.0,3.0]'``.
    """
    values = list(embedding)
    if expected_dim is not None and len(values) != expected_dim:
        logger.warning(
            "Embedding dimensionality mismatch. expected=%d actual=%d",
            expected_dim,
            len(values),
        )
    inner = ",".join(f"{v:.8f}" for v in values)
    return f"[{inner}]"


def insert_embedding(
    content_type: str,
    source_file: str,
    timestamp: str,
    content_payload: str,
    embedding: Iterable[float],
) -> int:
    """Insert a multimodal embedding record into the database.

    Args:
        content_type: Logical type of the content (e.g., "image", "text").
        source_file: Source file path or identifier.
        timestamp: Logical timestamp associated with the content (e.g., frame
            time or document section identifier).
        content_payload: Raw or lightly processed payload (e.g., text snippet
            or description).
        embedding: Embedding vector values.

    Returns:
        The auto-generated primary key ID of the inserted row.

    Raises:
        DatabaseError: If the insert operation fails.
    """
    logger.info(
        "Inserting embedding into database. content_type=%s source_file=%s timestamp=%s",
        content_type,
        source_file,
        timestamp,
    )
    embedding_str = _format_embedding(embedding)

    sql = """
    INSERT INTO multimodal_embeddings (
        content_type,
        source_file,
        timestamp,
        content_payload,
        embedding
    )
    VALUES (%s, %s, %s, %s, %s::vector)
    RETURNING id;
    """

    with get_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    sql,
                    (
                        content_type,
                        source_file,
                        timestamp,
                        content_payload,
                        embedding_str,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
            inserted_id = int(row["id"])
            logger.info("Successfully inserted embedding with id=%d", inserted_id)
            return inserted_id
        except Exception as exc:
            conn.rollback()
            logger.exception("Failed to insert embedding into database.")
            raise DatabaseError("Failed to insert embedding into database.") from exc


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize database schema.
    init_db()

    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_REGION", "us-central1")

    if not project_id:
        logger.error("Environment variable GCP_PROJECT_ID is required for the test.")
        raise SystemExit(1)

    config = VertexClientConfig(project_id=project_id, location=location)
    embedder = VertexMultimodalEmbedder(config=config)

    repo_root = Path(__file__).resolve().parents[2]
    frames_dir = repo_root / "data" / "frames"

    logger.info("Looking for frames under: %s", frames_dir)
    if not frames_dir.exists():
        logger.error("Frames directory does not exist: %s", frames_dir)
        raise SystemExit(2)

    candidates = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpeg"))
    )
    if not candidates:
        logger.error("No frame images found under: %s", frames_dir)
        raise SystemExit(3)

    test_image = candidates[0]
    logger.info("Selected test image for DB insert: %s", test_image)

    try:
        embedding = embedder.get_image_embedding(str(test_image))
    except VertexEmbeddingError:
        logger.exception("Failed to obtain embedding for test image.")
        raise SystemExit(4)
    except Exception:
        logger.exception("Unexpected error while obtaining embedding for test image.")
        raise SystemExit(5)

    inserted_id = insert_embedding(
        content_type="image",
        source_file=str(test_image),
        timestamp="0",
        content_payload="First ski frame sample",
        embedding=embedding,
    )

    logger.info(
        "成功插入向量到数据库！记录 id=%d，向量维度为: %d，前 5 个数值为: %s",
        inserted_id,
        len(embedding),
        embedding[:5],
    )

