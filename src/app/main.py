"""Streamlit chat app for OmniSearch multimodal RAG."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import streamlit as st
import vertexai
from google.api_core import exceptions as gax_exceptions
from vertexai.generative_models import GenerativeModel, Image as VertexImage, Part

from src.config.settings import load_settings
from src.retrieval.search_engine import SearchEngineError, VectorSearchEngine


logger = logging.getLogger(__name__)
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
DEFAULT_GEMINI_REGION = os.environ.get("GEMINI_REGION", "us-central1")
DEFAULT_TOP_K = int(os.environ.get("SEARCH_TOP_K", "6"))
MIN_SIMILARITY_THRESHOLD = float(os.environ.get("MIN_SIMILARITY_THRESHOLD", "0.20"))
ALLOWED_GEMINI_PREFIX = "gemini-2.0"

SYSTEM_PROMPT = (
    "你是一个专业的多模态 AI 助手。请严格基于我提供的视频截图和上下文来回答用户的问题。"
    "你可以基于画面中可见元素做保守描述，但不要扩展到不可见细节。"
    "如果提供的画面中没有相关信息，请直接回答‘在检索到的片段中未找到相关信息’，"
    "绝对不能编造。"
)


def _candidate_model_names() -> List[str]:
    """Return ordered Gemini model candidates for fallback attempts.

    Returns:
        Ordered unique model names.
    """
    configured = os.environ.get("GEMINI_MODEL_CANDIDATES", "")
    configured_list = [m.strip() for m in configured.split(",") if m.strip()]
    defaults = [
        DEFAULT_GEMINI_MODEL,
        "gemini-2.0-flash-001",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro",
        "gemini-2.0-pro-001",
    ]
    ordered: List[str] = []
    for name in [*configured_list, *defaults]:
        if not name:
            continue
        if not name.startswith(ALLOWED_GEMINI_PREFIX):
            logger.warning("Ignoring non-2.0 Gemini model in candidates: %s", name)
            continue
        if name not in ordered:
            ordered.append(name)
    return ordered


@st.cache_resource
def get_search_engine() -> VectorSearchEngine:
    """Create and cache VectorSearchEngine.

    Returns:
        A cached ``VectorSearchEngine`` instance.
    """
    return VectorSearchEngine()


@st.cache_resource
def get_generative_model() -> Tuple[str, GenerativeModel]:
    """Create and cache Vertex Gemini model.

    Returns:
        The selected model name and a cached ``GenerativeModel`` instance.
    """
    settings = load_settings()
    # Keep embedding/search region and Gemini generation region decoupled.
    # In some projects, embedding works in `us-central1` while Gemini models
    # are available via `global`.
    vertexai.init(project=settings.gcp_project_id, location=DEFAULT_GEMINI_REGION)
    selected = _candidate_model_names()[0]
    return selected, GenerativeModel(selected)


def _build_context(results: List[Dict[str, Any]]) -> str:
    """Build retrieval context text from search results.

    Args:
        results: Retrieval result list.

    Returns:
        A plain text context block for the model prompt.
    """
    if not results:
        return "未检索到可用片段。"

    lines: List[str] = []
    for idx, item in enumerate(results, start=1):
        lines.append(
            (
                f"[片段 {idx}] "
                f"id={item.get('id')} | "
                f"source_file={item.get('source_file')} | "
                f"timestamp_or_page={item.get('timestamp_or_page')} | "
                f"similarity={item.get('similarity'):.6f}"
            )
        )
        payload = item.get("content_payload")
        if payload:
            lines.append(f"[片段 {idx} 文本补充] {payload}")
    return "\n".join(lines)


def _format_result_line(item: Dict[str, Any]) -> str:
    """Format one retrieval result line for UI display.

    Args:
        item: One retrieval result record.

    Returns:
        A human-readable result summary line.
    """
    source_file = str(item.get("source_file", "N/A"))
    file_name = Path(source_file).name if source_file else "N/A"
    page_or_ts = str(item.get("timestamp_or_page", "N/A"))
    content_type = str(item.get("content_type", "N/A"))
    score = float(item.get("similarity", 0.0))
    return (
        f"文档: {file_name} | 页码/时间: {page_or_ts} | 类型: {content_type} | 相似度: {score:.4f}"
    )


def _render_retrieval_results(results: List[Dict[str, Any]]) -> None:
    """Render retrieval results list in chat UI.

    Args:
        results: Retrieval result list.
    """
    st.markdown("### 检索结果")
    if not results:
        st.info("暂未检索到相关片段，可以尝试换个问法或扩大数据范围。")
        return

    for idx, item in enumerate(results, start=1):
        st.markdown(f"**Top {idx}**  \n{_format_result_line(item)}")
        snippet = str(item.get("content_payload", "")).strip()
        if snippet:
            st.caption(snippet[:240] + ("..." if len(snippet) > 240 else ""))
        if idx < len(results):
            st.divider()


def _load_image_parts(results: List[Dict[str, Any]]) -> List[Part]:
    """Load local citation images and convert them to Gemini Parts.

    Args:
        results: Retrieval result list.

    Returns:
        A list of image ``Part`` objects. Invalid paths are skipped safely.
    """
    parts: List[Part] = []
    for item in results:
        if str(item.get("content_type", "")).lower() != "image":
            continue
        image_path = str(item.get("image_path", ""))
        if not image_path:
            continue
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            logger.warning("Citation image path does not exist or is not a file: %s", path)
            continue
        try:
            image = VertexImage.load_from_file(str(path))
            parts.append(Part.from_image(image))
        except Exception:
            logger.exception("Failed to load image for prompt part: %s", path)
    return parts


def _build_prompt_parts(query: str, results: List[Dict[str, Any]]) -> List[Any]:
    """Build robust multimodal prompt parts for Gemini.

    Args:
        query: User question.
        results: Retrieval results.

    Returns:
        Prompt parts combining system text, retrieval text, image parts, and question.
    """
    context_text = _build_context(results)
    image_parts = _load_image_parts(results)
    text_parts: List[str] = []
    for idx, item in enumerate(results, start=1):
        text_parts.append(f"[Result {idx}] {_format_result_line(item)}")
        payload = str(item.get("content_payload", "")).strip()
        if payload:
            text_parts.append(f"[Result {idx} Content] {payload}")

    return [
        f"System Prompt:\n{SYSTEM_PROMPT}",
        f"检索上下文摘要:\n{context_text}",
        *text_parts,
        *image_parts,
        (
            "请先判断提供的证据是否足够，再给出简洁准确回答。"
            "如果证据不足，必须回答：在检索到的片段中未找到相关信息。\n\n"
            f"用户问题: {query}"
        ),
    ]


def _stream_answer(
    model_name: str,
    model: GenerativeModel,
    query: str,
    results: List[Dict[str, Any]],
) -> Iterable[str]:
    """Stream model answer for the query using retrieved multimodal context.

    Args:
        model_name: Primary model name associated with ``model``.
        model: Initialized Gemini model.
        query: User question.
        results: Top-k retrieval results.

    Yields:
        Incremental text chunks from Gemini streaming response.

    Raises:
        RuntimeError: If Gemini API invocation fails.
    """
    prompt_parts = _build_prompt_parts(query=query, results=results)

    attempts = [model_name, *[m for m in _candidate_model_names() if m != model_name]]
    errors: List[str] = []

    for idx, candidate in enumerate(attempts):
        active_model = model if idx == 0 else GenerativeModel(candidate)
        try:
            responses = active_model.generate_content(prompt_parts, stream=True)
            for chunk in responses:
                chunk_text = getattr(chunk, "text", "")
                if chunk_text:
                    yield chunk_text
            return
        except (gax_exceptions.NotFound, gax_exceptions.PermissionDenied) as exc:
            logger.warning("Gemini model unavailable: %s (%s)", candidate, exc.__class__.__name__)
            errors.append(f"{candidate}: {exc.__class__.__name__}")
            continue
        except Exception as exc:
            logger.exception("Gemini generation failed on model: %s", candidate)
            raise RuntimeError(f"Gemini generation failed on model {candidate}.") from exc

    attempted = ", ".join(attempts)
    error_summary = "; ".join(errors) if errors else "unknown model access error"
    raise RuntimeError(
        "Gemini 模型不可用。"
        f"已尝试: {attempted}。"
        f"错误: {error_summary}。"
        "请检查模型名、项目权限或区域配置。"
    )


def _render_citations(results: List[Dict[str, Any]]) -> None:
    """Render retrieval citations in an expandable Streamlit section.

    Args:
        results: Top-k retrieval results.
    """
    with st.expander("查看参考画面 (Citations)"):
        if not results:
            st.info("无可展示的参考画面。")
            return

        for idx, item in enumerate(results, start=1):
            image_path = str(item.get("image_path", ""))
            caption = (
                f"Top {idx} | "
                f"similarity={float(item.get('similarity', 0.0)):.6f} | "
                f"timestamp_or_page={item.get('timestamp_or_page')} | "
                f"file={Path(image_path).name if image_path else 'N/A'}"
            )
            if image_path and Path(image_path).exists():
                st.image(image_path, caption=caption, width="stretch")
            else:
                st.warning(f"图片不存在: {image_path}")


def _max_similarity(results: List[Dict[str, Any]]) -> float:
    """Compute max similarity score from retrieval results.

    Args:
        results: Retrieval results.

    Returns:
        Maximum similarity score, or 0.0 when results are empty.
    """
    if not results:
        return 0.0
    return max(float(item.get("similarity", 0.0)) for item in results)


def main() -> None:
    """Run the Streamlit chat interface for OmniSearch."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    st.set_page_config(page_title="OmniSearch - 企业级多模态 RAG 检索", layout="wide")
    st.title("OmniSearch - 企业级多模态 RAG 检索")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                citations = message.get("citations", [])
                if citations:
                    _render_citations(citations)

    user_query = st.chat_input("请输入你的问题，例如：这个滑雪动作的重心控制有什么特点？")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            search_engine = get_search_engine()
            model_name, model = get_generative_model()
        except Exception:
            logger.exception("Failed to initialize search engine or Gemini model.")
            error_msg = "初始化检索或模型失败，请检查环境变量与 Vertex AI 配置。"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "citations": []}
            )
            return

        try:
            results = search_engine.search(
                user_query,
                top_k_image=max(1, DEFAULT_TOP_K // 2),
                top_k_text=max(1, DEFAULT_TOP_K - (DEFAULT_TOP_K // 2)),
            )
        except (SearchEngineError, ValueError) as exc:
            logger.exception("Search failed.")
            error_msg = f"检索失败：{exc}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "citations": []}
            )
            return

        if not results:
            no_info = "在检索到的片段中未找到相关信息。"
            st.markdown(no_info)
            st.session_state.messages.append(
                {"role": "assistant", "content": no_info, "citations": []}
            )
            return

        max_similarity = _max_similarity(results)
        st.caption(
            "检索质量："
            f"top_k={len(results)}，最高相似度={max_similarity:.4f}，"
            f"阈值={MIN_SIMILARITY_THRESHOLD:.2f}"
        )
        _render_retrieval_results(results)
        if max_similarity < MIN_SIMILARITY_THRESHOLD:
            no_info = "在检索到的片段中未找到相关信息。"
            st.markdown(no_info)
            _render_citations(results)
            st.session_state.messages.append(
                {"role": "assistant", "content": no_info, "citations": results}
            )
            return

        try:
            assistant_text = st.write_stream(_stream_answer(model_name, model, user_query, results))
            if not assistant_text:
                assistant_text = "在检索到的片段中未找到相关信息。"
                st.markdown(assistant_text)
        except Exception as exc:
            logger.exception("Generation failed.")
            assistant_text = f"生成回答失败：{exc}"
            st.error(assistant_text)

        _render_citations(results)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text, "citations": results}
        )


if __name__ == "__main__":
    main()

