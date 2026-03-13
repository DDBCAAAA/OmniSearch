"""PDF ingestion utilities for extracting and storing text chunks."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.embedding.db_manager import DatabaseError, init_db, insert_embedding
from src.embedding.embedder import VertexClientConfig, VertexEmbeddingError, VertexMultimodalEmbedder


logger = logging.getLogger(__name__)


def _import_pdf_backend() -> Tuple[str, object]:
    """Import and return an available PDF parsing backend.

    Returns:
        A tuple of backend name and module object.

    Raises:
        ImportError: If neither ``PyMuPDF`` (fitz) nor ``pdfplumber`` is installed.
    """
    try:
        import fitz  # type: ignore[import-not-found]

        return "fitz", fitz
    except ImportError:
        pass

    try:
        import pdfplumber  # type: ignore[import-not-found]

        return "pdfplumber", pdfplumber
    except ImportError as exc:
        raise ImportError(
            "No PDF parser found. Install one of: `pip install pymupdf` or `pip install pdfplumber`."
        ) from exc


def _split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into semantically-friendly chunks near sentence boundaries.

    Args:
        text: Source text to split.
        chunk_size: Target maximum chunk size in characters.

    Returns:
        A list of non-empty text chunks.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    sentences = re.split(r"(?<=[.!?。！？])\s+", normalized)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Extremely long sentence fallback: split hard by character window.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            for i in range(0, len(sentence), chunk_size):
                piece = sentence[i : i + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            continue

        projected_len = current_len + len(sentence) + (1 if current else 0)
        if projected_len <= chunk_size:
            current.append(sentence)
            current_len = projected_len
        else:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def process_pdf(pdf_path: str, chunk_size: int = 500) -> List[Dict[str, object]]:
    """Extract text from a PDF and split into page-aware chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum characters per chunk.

    Returns:
        A list of chunk dictionaries in the form:
        ``[{"page": 1, "content": "..."}, ...]``.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        ValueError: If ``chunk_size`` is not positive.
        ImportError: If PDF parsing dependencies are missing.
        RuntimeError: If text extraction fails unexpectedly.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    backend_name, backend = _import_pdf_backend()
    logger.info("Using PDF backend: %s", backend_name)

    def _clean_page_text(text: str) -> str:
        """Clean extracted page text by removing noisy header/footer artifacts.

        Rules:
        - Drop pure numeric lines (common page numbers).
        - Drop very short lines (< 5 chars).
        """
        cleaned_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.isdigit():
                continue
            if len(line) < 5:
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    page_texts: List[Tuple[int, str]] = []
    try:
        if backend_name == "fitz":
            doc = backend.open(str(path))
            try:
                for page_idx, page in enumerate(doc, start=1):
                    rect = page.rect
                    clip_box = backend.Rect(
                        rect.x0,
                        rect.y0 + rect.height * 0.08,
                        rect.x1,
                        rect.y1 - rect.height * 0.10,
                    )
                    text = page.get_text("text", clip=clip_box) or ""
                    cleaned = _clean_page_text(text)
                    page_texts.append((page_idx, cleaned))
            finally:
                doc.close()
        else:
            with backend.open(str(path)) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    cleaned = _clean_page_text(text)
                    page_texts.append((page_idx, cleaned))
    except Exception as exc:
        logger.exception("Failed to extract text from PDF: %s", path)
        raise RuntimeError(f"Failed to extract text from PDF: {path}") from exc

    chunks: List[Dict[str, object]] = []
    for page_num, page_text in page_texts:
        page_chunks = _split_text_into_chunks(page_text, chunk_size=chunk_size)
        for chunk in page_chunks:
            chunks.append({"page": page_num, "content": chunk})

    logger.info("Extracted %d chunks from PDF: %s", len(chunks), path.name)
    return chunks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_root = Path(__file__).resolve().parents[2]
    raw_docs_dir = repo_root / "data" / "raw_docs"

    if not raw_docs_dir.exists():
        logger.error("PDF directory does not exist: %s", raw_docs_dir)
        raise SystemExit(1)

    pdf_candidates: Sequence[Path] = sorted(raw_docs_dir.glob("*.pdf"))
    if not pdf_candidates:
        logger.error("No PDF files found under: %s", raw_docs_dir)
        raise SystemExit(2)

    pdf_file = pdf_candidates[0]
    logger.info("Selected test PDF: %s", pdf_file)

    try:
        chunks = process_pdf(str(pdf_file), chunk_size=500)
    except ImportError as exc:
        logger.error(str(exc))
        raise SystemExit(3)
    except Exception:
        logger.exception("PDF processing failed.")
        raise SystemExit(4)

    if not chunks:
        logger.warning("No text chunks extracted from PDF: %s", pdf_file.name)
        raise SystemExit(0)

    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_REGION", "us-central1")
    if not project_id:
        logger.error("Environment variable GCP_PROJECT_ID is required.")
        raise SystemExit(5)

    init_db()
    embedder = VertexMultimodalEmbedder(
        config=VertexClientConfig(project_id=project_id, location=location)
    )

    total = len(chunks)
    inserted = 0
    for idx, chunk in enumerate(chunks, start=1):
        page_num = int(chunk["page"])
        content = str(chunk["content"])
        if not content.strip():
            continue

        try:
            embedding = embedder.get_text_embedding(content)
            row_id = insert_embedding(
                content_type="text",
                source_file=pdf_file.name,
                timestamp=str(page_num),
                content_payload=content,
                embedding=embedding,
            )
            inserted += 1
            logger.info(
                "Inserted chunk %d/%d | row_id=%d | page=%d | chars=%d",
                idx,
                total,
                row_id,
                page_num,
                len(content),
            )
        except (VertexEmbeddingError, DatabaseError):
            logger.exception("Failed to embed/insert chunk %d (page=%d).", idx, page_num)

    logger.info(
        "PDF ingestion completed. source=%s total_chunks=%d inserted=%d",
        pdf_file.name,
        total,
        inserted,
    )

