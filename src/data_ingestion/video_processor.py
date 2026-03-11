"""Video data ingestion utilities for OmniSearch.

This module contains utilities for processing video inputs in the offline
data ingestion pipeline, including frame extraction via FFmpeg.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence


logger = logging.getLogger(__name__)


class FFmpegProcessingError(RuntimeError):
    """Raised when an FFmpeg subprocess fails."""


def _validate_input_video(input_video: Path) -> None:
    """Validate that the input video exists and is a file.

    Args:
        input_video: Path to the input video.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path exists but is not a file.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not input_video.is_file():
        raise ValueError(f"Input video is not a file: {input_video}")


def _ensure_output_dir(output_dir: Path) -> None:
    """Ensure output directory exists.

    Args:
        output_dir: Directory to create if needed.

    Raises:
        ValueError: If output_dir exists but is not a directory.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _run_ffmpeg_with_frame_progress(
    cmd: Sequence[str],
    output_dir: Path,
    output_glob: str,
    progress_log_interval_s: float = 2.0,
) -> None:
    """Run an FFmpeg command and log progress based on produced frames.

    FFmpeg prints progress to stderr, but that output is not stable across
    versions and flags. For extraction jobs, a robust proxy is the number of
    output files created.

    Args:
        cmd: Full ffmpeg command as a list of arguments.
        output_dir: Directory where outputs are written.
        output_glob: Glob pattern to count produced files, relative to output_dir.
        progress_log_interval_s: How often to log produced output counts.

    Raises:
        FFmpegProcessingError: If FFmpeg exits with a non-zero return code.
    """
    logger.info("Running ffmpeg command: %s", " ".join(cmd))
    start_time = time.time()
    produced_last = -1

    process = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        while True:
            return_code = process.poll()
            produced_now = len(list(output_dir.glob(output_glob)))
            if produced_now != produced_last:
                elapsed_s = time.time() - start_time
                logger.info(
                    "Frame extraction progress: %d frames produced (elapsed %.1fs)",
                    produced_now,
                    elapsed_s,
                )
                produced_last = produced_now

            if return_code is not None:
                break

            time.sleep(progress_log_interval_s)

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error("FFmpeg failed (exit %s). stderr: %s", process.returncode, stderr)
            raise FFmpegProcessingError(
                f"FFmpeg failed with exit code {process.returncode}. "
                f"Command: {' '.join(cmd)}"
            )
        if stderr:
            logger.debug("FFmpeg stderr: %s", stderr)
        if stdout:
            logger.debug("FFmpeg stdout: %s", stdout)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def extract_frames(
    input_video: Path,
    output_dir: Path,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    image_format: str = "jpg",
) -> List[Path]:
    """Extract frames from a video file using FFmpeg (via subprocess).

    This function is part of the Phase 1 ingestion pipeline and is responsible
    for converting input videos into a sequence of image frames that can later
    be embedded by the multimodal encoder.

    Args:
        input_video: Path to the input video file.
        output_dir: Directory where extracted frames will be written. The
            directory will be created if it does not already exist.
        fps: Target frame rate (frames per second) for extraction. For Phase 1,
            the default is 1 frame per second.
        max_frames: Optional upper bound on the total number of frames to
            extract. If ``None``, all frames matching the sampling strategy
            will be extracted.
        image_format: Image format/extension for output frames, such as
            ``"jpg"`` or ``"png"``.

    Returns:
        A list of paths to the extracted frame image files, ordered by time.
        Output filenames follow the convention: ``{video_stem}_{timestamp}.jpg``.
        In Phase 1 with ``fps=1``, the timestamp corresponds to a 1-based second
        index encoded as a zero-padded integer (e.g., ``video_000001.jpg``).

    Raises:
        FileNotFoundError: If ``input_video`` does not exist.
        ValueError: If any of the arguments are invalid (e.g., non-positive
            ``fps`` or unsupported ``image_format``).
        FFmpegProcessingError: If the underlying FFmpeg command fails.
    """
    _validate_input_video(input_video)
    _ensure_output_dir(output_dir)

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive when set, got {max_frames}")

    normalized_format = image_format.lower().lstrip(".")
    if normalized_format not in {"jpg", "jpeg", "png", "webp"}:
        raise ValueError(f"Unsupported image_format: {image_format}")

    video_stem = input_video.stem
    # We write to a numeric sequence first. For fps=1, the frame index maps
    # cleanly to seconds for a stable "{video}_{timestamp}" convention.
    pattern = str(output_dir / f"{video_stem}_%06d.{normalized_format}")

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        f"fps={fps}",
        "-vsync",
        "vfr",
        pattern,
    ]
    if max_frames is not None:
        cmd.insert(cmd.index("-vsync"), "-frames:v")
        cmd.insert(cmd.index("-vsync"), str(max_frames))

    logger.info("Starting frame extraction. input=%s output_dir=%s fps=%s", input_video, output_dir, fps)
    try:
        _run_ffmpeg_with_frame_progress(
            cmd=cmd,
            output_dir=output_dir,
            output_glob=f"{video_stem}_*.{normalized_format}",
        )
    except FFmpegProcessingError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during frame extraction.")
        raise FFmpegProcessingError("Unexpected error during frame extraction.") from exc

    frames = sorted(output_dir.glob(f"{video_stem}_*.{normalized_format}"))
    logger.info("Completed frame extraction. frames=%d output_dir=%s", len(frames), output_dir)
    return frames


def extract_audio(
    input_video: Path,
    output_dir: Path,
    audio_format: str = "wav",
    sample_rate_hz: int = 16_000,
    channels: int = 1,
) -> Path:
    """Extract audio stream from a video file using FFmpeg (via subprocess).

    This function prepares audio for downstream Speech-to-Text (STT) by
    converting to a consistent sample rate and channel count.

    Args:
        input_video: Path to the input video file.
        output_dir: Directory where the extracted audio will be written. The
            directory will be created if it does not already exist.
        audio_format: Output audio format. Supported values: ``"wav"``, ``"mp3"``.
        sample_rate_hz: Output sample rate in Hz (default: 16000).
        channels: Number of audio channels (default: 1 for mono).

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If ``input_video`` does not exist.
        ValueError: If any of the arguments are invalid or unsupported.
        FFmpegProcessingError: If the underlying FFmpeg command fails.
    """
    _validate_input_video(input_video)
    _ensure_output_dir(output_dir)

    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")

    normalized_format = audio_format.lower().lstrip(".")
    if normalized_format not in {"wav", "mp3"}:
        raise ValueError(f"Unsupported audio_format: {audio_format}")

    out_path = output_dir / f"{input_video.stem}.{normalized_format}"

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate_hz),
    ]

    if normalized_format == "mp3":
        cmd += ["-codec:a", "libmp3lame"]
    cmd += [str(out_path)]

    logger.info(
        "Starting audio extraction. input=%s output=%s sr=%dHz ch=%d",
        input_video,
        out_path,
        sample_rate_hz,
        channels,
    )
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stderr:
            logger.debug("FFmpeg stderr: %s", result.stderr)
        if result.stdout:
            logger.debug("FFmpeg stdout: %s", result.stdout)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        logger.error("FFmpeg failed extracting audio. stderr: %s", stderr)
        raise FFmpegProcessingError(
            f"FFmpeg failed extracting audio (exit code {exc.returncode}). Command: {' '.join(cmd)}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during audio extraction.")
        raise FFmpegProcessingError("Unexpected error during audio extraction.") from exc

    logger.info("Completed audio extraction. output=%s", out_path)
    return out_path


def extract_video_frames(
    input_video: Path,
    output_dir: Path,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    image_format: str = "jpg",
) -> List[Path]:
    """Backward-compatible alias for frame extraction.

    Args:
        input_video: Path to the input video file.
        output_dir: Directory where extracted frames will be written.
        fps: Target frame rate (frames per second). If ``None``, defaults to 1.
        max_frames: Optional upper bound on the total number of frames.
        image_format: Output image format.

    Returns:
        A list of extracted frame paths.
    """
    return extract_frames(
        input_video=input_video,
        output_dir=output_dir,
        fps=1.0 if fps is None else fps,
        max_frames=max_frames,
        image_format=image_format,
    )


__all__ = ["FFmpegProcessingError", "extract_audio", "extract_frames", "extract_video_frames"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_root = Path(__file__).resolve().parents[2]
    raw_videos_dir = repo_root / "data" / "raw_videos"
    frames_dir = repo_root / "data" / "frames"

    logger.info("Looking for mp4 videos under: %s", raw_videos_dir)
    if not raw_videos_dir.exists():
        logger.error("Directory not found: %s", raw_videos_dir)
        raise SystemExit(1)

    mp4_files = list(raw_videos_dir.glob("*.mp4"))
    if not mp4_files:
        logger.error("No .mp4 files found under: %s", raw_videos_dir)
        raise SystemExit(1)

    input_video_path = max(mp4_files, key=lambda p: p.stat().st_mtime)
    logger.info("Selected video for test: %s", input_video_path)

    try:
        extracted = extract_frames(
            input_video=input_video_path,
            output_dir=frames_dir,
            fps=1.0,
            max_frames=3,
            image_format="jpg",
        )
        logger.info("Test finished successfully. Extracted %d frames into %s", len(extracted), frames_dir)
    except FFmpegProcessingError:
        logger.exception("Test failed due to FFmpeg error.")
        raise SystemExit(2)
    except Exception:
        logger.exception("Test failed due to unexpected error.")
        raise SystemExit(3)

