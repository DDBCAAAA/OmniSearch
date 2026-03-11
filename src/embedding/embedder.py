"""Vertex AI multimodal embedding utilities.

This module provides a thin, explicit wrapper around Vertex AI's
``MultiModalEmbeddingModel`` to obtain text and image embeddings for the
OmniSearch retrieval pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel


logger = logging.getLogger(__name__)


class VertexEmbeddingError(RuntimeError):
    """Raised when a Vertex AI embedding call fails."""


@dataclass(frozen=True)
class VertexClientConfig:
    """Configuration for VertexMultimodalEmbedder.

    Attributes:
        project_id: Google Cloud project ID.
        location: Region where Vertex AI is hosted (e.g., "us-central1").
        model_name: Name of the multimodal embedding model.
    """

    project_id: str
    location: str
    model_name: str = "multimodalembedding@001"


class VertexMultimodalEmbedder:
    """Client wrapper around Vertex AI's MultiModalEmbeddingModel.

    This class is intentionally lightweight and explicit to keep the
    interaction with Vertex AI transparent and easy to debug in production.
    """

    def __init__(self, config: VertexClientConfig) -> None:
        """Initialize the embedder with the given configuration.

        Args:
            config: Vertex client configuration, including project, region,
                and model name.
        """
        self._config = config
        logger.info(
            "Initializing VertexMultimodalEmbedder with project=%s location=%s model=%s",
            config.project_id,
            config.location,
            config.model_name,
        )
        vertexai.init(project=config.project_id, location=config.location)
        self._model = MultiModalEmbeddingModel.from_pretrained(config.model_name)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(VertexEmbeddingError),
    )
    def get_image_embedding(self, image_path: str) -> List[float]:
        """Get an embedding vector for an image file.

        The underlying model returns a 1408-dimensional vector for the
        requested image.

        Args:
            image_path: Path to the image file on local disk.

        Returns:
            The image embedding vector as a list of floats.

        Raises:
            FileNotFoundError: If the image file does not exist.
            VertexEmbeddingError: If the Vertex AI API call fails or returns
                an invalid response. The call is retried with exponential
                backoff up to three attempts.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        logger.info("Requesting image embedding from Vertex AI. path=%s", path)

        try:
            image = Image.load_from_file(str(path))
            response = self._model.get_embeddings(image=image)
            # Newer SDK versions may return either:
            #   * an object with `.values` (recommended path), or
            #   * a plain Python list of floats.
            image_embedding = getattr(response, "image_embedding", None)
            if image_embedding is None:
                raise VertexEmbeddingError("Vertex AI response is missing image_embedding field.")

            if hasattr(image_embedding, "values"):
                vector = list(image_embedding.values)  # type: ignore[union-attr]
            elif isinstance(image_embedding, list):
                if not image_embedding:
                    raise VertexEmbeddingError("Received empty list of image embeddings from Vertex AI.")
                if isinstance(image_embedding[0], float):
                    vector = list(image_embedding)
                elif hasattr(image_embedding[0], "values"):
                    vector = list(image_embedding[0].values)  # type: ignore[union-attr]
                else:
                    raise VertexEmbeddingError(
                        f"Unsupported image_embedding element type: {type(image_embedding[0])}"
                    )
            else:
                raise VertexEmbeddingError(f"Unsupported image_embedding type: {type(image_embedding)}")
        except Exception as exc:
            # This will trigger tenacity retry if within retry budget.
            logger.warning("Failed to obtain image embedding from Vertex AI: %s", exc)
            raise VertexEmbeddingError("Vertex AI image embedding request failed.") from exc

        if not vector:
            raise VertexEmbeddingError("Received empty image embedding from Vertex AI.")

        logger.info("Successfully obtained image embedding. dim=%d", len(vector))
        return vector

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(VertexEmbeddingError),
    )
    def get_text_embedding(self, text: str) -> List[float]:
        """Get an embedding vector for a text string.

        The underlying model returns a 1408-dimensional vector for the
        requested text.

        Args:
            text: Input text to embed.

        Returns:
            The text embedding vector as a list of floats.

        Raises:
            ValueError: If the text is empty.
            VertexEmbeddingError: If the Vertex AI API call fails or returns
                an invalid response. The call is retried with exponential
                backoff up to three attempts.
        """
        if not text.strip():
            raise ValueError("Text for embedding must not be empty.")

        logger.info("Requesting text embedding from Vertex AI. length=%d", len(text))

        try:
            response = self._model.get_embeddings(text=text)
            text_embedding = getattr(response, "text_embedding", None)
            if text_embedding is None:
                raise VertexEmbeddingError("Vertex AI response is missing text_embedding field.")

            if hasattr(text_embedding, "values"):
                vector = list(text_embedding.values)  # type: ignore[union-attr]
            elif isinstance(text_embedding, list):
                if not text_embedding:
                    raise VertexEmbeddingError("Received empty list of text embeddings from Vertex AI.")
                if isinstance(text_embedding[0], float):
                    vector = list(text_embedding)
                elif hasattr(text_embedding[0], "values"):
                    vector = list(text_embedding[0].values)  # type: ignore[union-attr]
                else:
                    raise VertexEmbeddingError(
                        f"Unsupported text_embedding element type: {type(text_embedding[0])}"
                    )
            else:
                raise VertexEmbeddingError(f"Unsupported text_embedding type: {type(text_embedding)}")
        except Exception as exc:
            logger.warning("Failed to obtain text embedding from Vertex AI: %s", exc)
            raise VertexEmbeddingError("Vertex AI text embedding request failed.") from exc

        if not vector:
            raise VertexEmbeddingError("Received empty text embedding from Vertex AI.")

        logger.info("Successfully obtained text embedding. dim=%d", len(vector))
        return vector


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
    logger.info("Selected test image: %s", test_image)

    try:
        embedding = embedder.get_image_embedding(str(test_image))
    except VertexEmbeddingError:
        logger.exception("Failed to obtain embedding for test image.")
        raise SystemExit(4)
    except Exception:
        logger.exception("Unexpected error while obtaining embedding for test image.")
        raise SystemExit(5)

    dim = len(embedding)
    first_values = embedding[:5]
    logger.info("成功提取！向量维度为: %d，前 5 个数值为: %s", dim, first_values)

