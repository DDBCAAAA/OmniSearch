"""Audio extraction and transcription pipeline for video ingestion."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from src.embedding.db_manager import DatabaseError, init_db, insert_embedding
from src.embedding.embedder import VertexClientConfig, VertexEmbeddingError, VertexMultimodalEmbedder


logger = logging.getLogger(__name__)


class AudioProcessingError(RuntimeError):
    """Raised when audio extraction or transcription fails."""


def extract_audio(video_path: str, output_audio_path: str) -> str:
    """Extract a 16kHz mono WAV audio track from a video file.

    Args:
        video_path: Path to the source video file.
        output_audio_path: Path to the output WAV file.

    Returns:
        The output audio path string.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        AudioProcessingError: If ffmpeg extraction fails.
    """
    input_path = Path(video_path)
    output_path = Path(output_audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]

    logger.info("Extracting audio with ffmpeg. input=%s output=%s", input_path, output_path)
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stderr:
            logger.debug("ffmpeg stderr: %s", result.stderr)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        logger.error("ffmpeg audio extraction failed. stderr=%s", stderr)
        raise AudioProcessingError(
            f"ffmpeg failed with exit code {exc.returncode} while extracting audio."
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during audio extraction.")
        raise AudioProcessingError("Unexpected error during audio extraction.") from exc

    logger.info("Audio extraction complete: %s", output_path)
    return str(output_path)


def transcribe_audio(audio_path: str, model_size: str = "base") -> List[Dict[str, Any]]:
    """Transcribe audio into timestamped text segments using Whisper.

    Args:
        audio_path: Path to a WAV audio file.
        model_size: Whisper model variant (e.g., ``base`` or ``small``).

    Returns:
        A list of dicts with keys ``start``, ``end``, and ``text``.

    Raises:
        FileNotFoundError: If ``audio_path`` does not exist.
        ImportError: If the whisper dependency is missing.
        AudioProcessingError: If transcription fails.
    """
    input_path = Path(audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Whisper is not installed. Run: `pip install openai-whisper`."
        ) from exc

    logger.info("Loading Whisper model: %s", model_size)
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(str(input_path))
    except Exception as exc:
        logger.exception("Whisper transcription failed.")
        raise AudioProcessingError("Whisper transcription failed.") from exc

    raw_segments = result.get("segments", []) if isinstance(result, dict) else []
    segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        segments.append({"start": start, "end": end, "text": text})

    logger.info("Transcription complete. segments=%d", len(segments))
    return segments


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_root = Path(__file__).resolve().parents[2]
    raw_videos_dir = repo_root / "data" / "raw_videos"
    raw_audio_dir = repo_root / "data" / "raw_audio"

    if not raw_videos_dir.exists():
        logger.error("Video directory does not exist: %s", raw_videos_dir)
        raise SystemExit(1)

    candidates = sorted(raw_videos_dir.glob("*.mp4"))
    if not candidates:
        logger.error("No .mp4 files found under: %s", raw_videos_dir)
        raise SystemExit(2)

    test_video = candidates[0]
    output_wav = raw_audio_dir / f"{test_video.stem}.wav"
    logger.info("Selected test video: %s", test_video)

    try:
        audio_path = extract_audio(str(test_video), str(output_wav))
        segments = transcribe_audio(audio_path=audio_path, model_size="base")
    except ImportError as exc:
        logger.error("%s", exc)
        raise SystemExit(3)
    except (AudioProcessingError, FileNotFoundError):
        logger.exception("Audio processing pipeline failed before DB ingest.")
        raise SystemExit(4)

    if not segments:
        logger.warning("No transcript segments extracted from audio: %s", audio_path)
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

    total = len(segments)
    inserted = 0
    for idx, seg in enumerate(segments, start=1):
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg["text"]).strip()
        if not text:
            continue
        timestamp = f"{start:.2f}-{end:.2f}"

        logger.info(
            "Embedding transcript segment %d/%d | ts=%s | chars=%d",
            idx,
            total,
            timestamp,
            len(text),
        )
        try:
            embedding = embedder.get_text_embedding(text)
            row_id = insert_embedding(
                content_type="transcript",
                source_file=test_video.name,
                timestamp=timestamp,
                content_payload=text,
                embedding=embedding,
            )
            inserted += 1
            logger.info("Inserted transcript segment %d/%d as row_id=%d", idx, total, row_id)
        except (VertexEmbeddingError, DatabaseError):
            logger.exception("Failed to embed/insert transcript segment %d/%d", idx, total)

    logger.info(
        "Audio ingestion complete. video=%s segments=%d inserted=%d",
        test_video.name,
        total,
        inserted,
    )

