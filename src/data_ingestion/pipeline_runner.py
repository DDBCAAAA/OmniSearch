"""One-click offline ingestion pipeline runner.

This module orchestrates video frame extraction, audio transcription, and PDF
text chunk ingestion, then writes embeddings into PostgreSQL.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from src.data_ingestion.audio_processor import (
    AudioProcessingError,
    extract_audio,
    transcribe_audio,
)
from src.data_ingestion.pdf_processor import process_pdf
from src.data_ingestion.video_processor import FFmpegProcessingError, extract_frames
from src.embedding.db_manager import DatabaseError, init_db, insert_embedding
from src.embedding.embedder import VertexClientConfig, VertexEmbeddingError, VertexMultimodalEmbedder


logger = logging.getLogger(__name__)


def _resolve_input_files(
    directory: Path,
    pattern: str,
    explicit_path: Optional[str],
    all_raw: bool,
) -> List[Path]:
    """Resolve input files from explicit path, all files, or latest file.

    Args:
        directory: Directory used when explicit path is not provided.
        pattern: Glob pattern to search under directory.
        explicit_path: Optional explicit file path.
        all_raw: Whether to resolve all matching files under ``directory``.

    Returns:
        Resolved input file paths.

    Raises:
        FileNotFoundError: If no matching file can be resolved.
    """
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        return [path]

    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files found under {directory} matching pattern: {pattern}")
    if all_raw:
        return candidates
    return [max(candidates, key=lambda p: p.stat().st_mtime)]


def _insert_text_records(
    *,
    source_file: str,
    content_type: str,
    timestamps: Iterable[str],
    contents: Iterable[str],
    embedder: VertexMultimodalEmbedder,
) -> int:
    """Embed and insert text records into DB.

    Args:
        source_file: Source file identifier.
        content_type: Content type for DB records.
        timestamps: Timestamps/page markers.
        contents: Text payloads.
        embedder: Shared embedder instance.

    Returns:
        Number of successfully inserted records.
    """
    inserted = 0
    for idx, (ts, text) in enumerate(zip(timestamps, contents), start=1):
        payload = text.strip()
        if not payload:
            continue
        try:
            embedding = embedder.get_text_embedding(payload)
            row_id = insert_embedding(
                content_type=content_type,
                source_file=source_file,
                timestamp=ts,
                content_payload=payload,
                embedding=embedding,
            )
            inserted += 1
            logger.info(
                "Inserted %s record #%d as row_id=%d | source=%s | ts=%s",
                content_type,
                idx,
                row_id,
                source_file,
                ts,
            )
        except (VertexEmbeddingError, DatabaseError):
            logger.exception("Failed to embed/insert %s record #%d", content_type, idx)
    return inserted


def run_pipeline(
    *,
    video_path: Optional[str],
    pdf_path: Optional[str],
    fps: float,
    max_frames: int,
    whisper_model: str,
    pdf_chunk_size: int,
    all_raw: bool,
    skip_frames: bool,
    skip_audio: bool,
    skip_pdf: bool,
) -> None:
    """Execute the end-to-end offline ingestion pipeline.

    Args:
        video_path: Optional explicit test video path.
        pdf_path: Optional explicit test PDF path.
        fps: Frame extraction FPS.
        max_frames: Max frame count for extraction.
        whisper_model: Whisper model size.
        pdf_chunk_size: PDF chunk size in characters.
        all_raw: If True, ingest all matching raw files under data directories.
        skip_frames: Skip frame extraction + image embedding stage.
        skip_audio: Skip audio extraction + transcript embedding stage.
        skip_pdf: Skip PDF extraction + text embedding stage.
    """
    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_REGION", "us-central1")
    if not project_id:
        raise RuntimeError("Environment variable GCP_PROJECT_ID is required.")

    repo_root = Path(__file__).resolve().parents[2]
    raw_videos_dir = repo_root / "data" / "raw_videos"
    frames_dir = repo_root / "data" / "frames"
    raw_audio_dir = repo_root / "data" / "raw_audio"
    raw_docs_dir = repo_root / "data" / "raw_docs"

    init_db()
    embedder = VertexMultimodalEmbedder(
        config=VertexClientConfig(project_id=project_id, location=location)
    )
    logger.info("Ingestion pipeline started.")

    selected_videos: List[Path] = []
    if not skip_frames or not skip_audio:
        selected_videos = _resolve_input_files(raw_videos_dir, "*.mp4", video_path, all_raw)
        logger.info("Selected %d video(s) for ingestion.", len(selected_videos))

    if not skip_frames:
        logger.info("Stage 1/3: Frame extraction + image embedding")
        total_frames = 0
        inserted_images = 0
        for video_idx, selected_video in enumerate(selected_videos, start=1):
            logger.info("Frame ingestion for video %d/%d: %s", video_idx, len(selected_videos), selected_video)
            frames = extract_frames(
                input_video=selected_video,
                output_dir=frames_dir,
                fps=fps,
                max_frames=max_frames,
                image_format="jpg",
            )
            total_frames += len(frames)
            for idx, frame_path in enumerate(frames, start=1):
                ts = frame_path.stem.split("_")[-1]
                try:
                    embedding = embedder.get_image_embedding(str(frame_path))
                    row_id = insert_embedding(
                        content_type="image",
                        source_file=str(frame_path),
                        timestamp=ts,
                        content_payload=f"Frame extracted from {selected_video.name}",
                        embedding=embedding,
                    )
                    inserted_images += 1
                    logger.info(
                        "Inserted image frame %d/%d (video=%s) as row_id=%d",
                        idx,
                        len(frames),
                        selected_video.name,
                        row_id,
                    )
                except (VertexEmbeddingError, DatabaseError):
                    logger.exception("Failed to embed/insert frame %s", frame_path.name)
        logger.info("Frame stage complete. extracted=%d inserted=%d", total_frames, inserted_images)

    if not skip_audio:
        logger.info("Stage 2/3: Audio extraction + transcription embedding")
        total_segments = 0
        inserted_transcripts_total = 0
        for video_idx, selected_video in enumerate(selected_videos, start=1):
            logger.info(
                "Audio ingestion for video %d/%d: %s", video_idx, len(selected_videos), selected_video
            )
            wav_path = raw_audio_dir / f"{selected_video.stem}.wav"
            extracted_audio = extract_audio(str(selected_video), str(wav_path))
            segments = transcribe_audio(audio_path=extracted_audio, model_size=whisper_model)
            total_segments += len(segments)
            timestamps = [f"{float(seg['start']):.2f}-{float(seg['end']):.2f}" for seg in segments]
            contents = [str(seg["text"]) for seg in segments]
            inserted_transcripts = _insert_text_records(
                source_file=selected_video.name,
                content_type="transcript",
                timestamps=timestamps,
                contents=contents,
                embedder=embedder,
            )
            inserted_transcripts_total += inserted_transcripts
        logger.info(
            "Audio stage complete. segments=%d inserted=%d",
            total_segments,
            inserted_transcripts_total,
        )

    if not skip_pdf:
        logger.info("Stage 3/3: PDF text extraction + embedding")
        selected_pdfs = _resolve_input_files(raw_docs_dir, "*.pdf", pdf_path, all_raw)
        logger.info("Selected %d PDF file(s) for ingestion.", len(selected_pdfs))
        total_chunks = 0
        inserted_text_total = 0
        for pdf_idx, selected_pdf in enumerate(selected_pdfs, start=1):
            logger.info("PDF ingestion for doc %d/%d: %s", pdf_idx, len(selected_pdfs), selected_pdf)
            chunks = process_pdf(str(selected_pdf), chunk_size=pdf_chunk_size)
            total_chunks += len(chunks)
            timestamps = [str(int(chunk["page"])) for chunk in chunks]
            contents = [str(chunk["content"]) for chunk in chunks]
            inserted_text = _insert_text_records(
                source_file=selected_pdf.name,
                content_type="text",
                timestamps=timestamps,
                contents=contents,
                embedder=embedder,
            )
            inserted_text_total += inserted_text
        logger.info("PDF stage complete. chunks=%d inserted=%d", total_chunks, inserted_text_total)

    logger.info("Ingestion pipeline finished.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for one-click pipeline execution."""
    parser = argparse.ArgumentParser(description="Run one-click offline ingestion pipeline.")
    parser.add_argument("--video-path", type=str, default=None, help="Explicit path to input .mp4 video.")
    parser.add_argument("--pdf-path", type=str, default=None, help="Explicit path to input .pdf document.")
    parser.add_argument(
        "--all-raw",
        action="store_true",
        help="Ingest all files under raw directories (instead of latest only).",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS.")
    parser.add_argument("--max-frames", type=int, default=10, help="Max extracted frames per video.")
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription.",
    )
    parser.add_argument("--pdf-chunk-size", type=int, default=500, help="PDF chunk size in chars.")
    parser.add_argument("--skip-frames", action="store_true", help="Skip frame extraction stage.")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio transcription stage.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF ingestion stage.")
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _build_arg_parser().parse_args()
    try:
        run_pipeline(
            video_path=args.video_path,
            pdf_path=args.pdf_path,
            fps=args.fps,
            max_frames=args.max_frames,
            whisper_model=args.whisper_model,
            pdf_chunk_size=args.pdf_chunk_size,
            all_raw=args.all_raw,
            skip_frames=args.skip_frames,
            skip_audio=args.skip_audio,
            skip_pdf=args.skip_pdf,
        )
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        raise SystemExit(2)
    except (RuntimeError, FileNotFoundError, ValueError, FFmpegProcessingError, AudioProcessingError):
        logger.exception("Pipeline failed.")
        raise SystemExit(1)

