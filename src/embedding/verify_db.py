"""Verification script for multimodal_embeddings table.

This module connects to the local PostgreSQL database, reports the total row
count of ``multimodal_embeddings``, inspects the latest inserted row, and
verifies that its embedding vector has dimension 1408.
"""

from __future__ import annotations

import logging
from typing import Any, List

from src.embedding.db_manager import DatabaseError, get_connection
from psycopg2.extras import RealDictCursor


logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 1408


def _parse_embedding(raw: Any) -> List[float]:
    """Parse the embedding column value into a list of floats.

    pgvector may return the column as a string ``"[1.0,2.0,...]"`` or, if
    a custom type adapter is registered, as a list-like object.

    Args:
        raw: The value of the embedding column from the database.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        ValueError: If the value cannot be parsed or has unexpected type.
    """
    if raw is None:
        raise ValueError("Embedding value is None.")

    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]

    if hasattr(raw, "tolist"):
        return raw.tolist()

    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            s = s[1:]
        if s.endswith("]"):
            s = s[:-1]
        if not s:
            raise ValueError("Embedding string is empty.")
        return [float(x.strip()) for x in s.split(",")]

    raise ValueError(f"Cannot parse embedding type: {type(raw)}")


def verify_db() -> None:
    """Connect to PostgreSQL, query multimodal_embeddings, and verify latest embedding.

    - Logs the total row count of ``multimodal_embeddings``.
    - Fetches the latest row (by ``id``) and logs ``id``, ``source_file``,
      ``timestamp``.
    - Parses the ``embedding`` column and checks that its length is 1408;
      if so, logs ``[SUCCESS] 向量维度正确: 1408``.

    Raises:
        DatabaseError: If connecting or querying the database fails.
        ValueError: If the table is empty or the embedding cannot be parsed.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM multimodal_embeddings;")
            row = cur.fetchone()
            total = int(row["cnt"])
            logger.info("multimodal_embeddings 表总行数: %d", total)

            if total == 0:
                logger.warning("表中无数据，无法验证最新一条记录。")
                return

            cur.execute(
                """
                SELECT id, source_file, timestamp, embedding
                FROM multimodal_embeddings
                ORDER BY id DESC
                LIMIT 1;
                """
            )
            latest = cur.fetchone()
            if not latest:
                logger.warning("未查询到最新记录。")
                return

            logger.info(
                "最新一条记录: id=%s, source_file=%s, timestamp=%s",
                latest["id"],
                latest["source_file"],
                latest["timestamp"],
            )

            raw_embedding = latest["embedding"]
            embedding = _parse_embedding(raw_embedding)
            dim = len(embedding)

            if dim == EXPECTED_EMBEDDING_DIM:
                logger.info("[SUCCESS] 向量维度正确: %d", EXPECTED_EMBEDDING_DIM)
            else:
                logger.warning(
                    "向量维度不符合预期: expected=%d, actual=%d",
                    EXPECTED_EMBEDDING_DIM,
                    dim,
                )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        verify_db()
    except DatabaseError:
        logger.exception("数据库连接或查询失败。")
        raise SystemExit(1)
    except ValueError as e:
        logger.exception("验证过程出错: %s", e)
        raise SystemExit(2)
