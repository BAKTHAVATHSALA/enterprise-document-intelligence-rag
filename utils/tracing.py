"""LangSmith Tracing and Observability Utility Layer.

Provides non-intrusive tracing instrumentation for Hybrid RAG pipeline stages:
- Query workflow
- Semantic embedding
- Multi-source retrieval (Vector, BM25, Graph)
- Reciprocal Rank Fusion (RRF)
- Neural reranking
- Grounded LLM generation
- Source citation validation

Requirements:
- Observability only; does not modify retrieval, ranking, or generation logic.
- Credentials loaded strictly from environment/config (.env).
- Supports tracing being disabled when credentials or config flags are absent.
- Captures query_id, document_ids, retrieval_method, model_name, latency, and errors.
- Never logs API keys or sensitive document content in metadata.
"""

import os
import time
import functools
from typing import Any, Callable, Optional
from utils.config import get_config
from utils.logger import logger

FUNC_CONFIG_TRACING: str = "configure_tracing_environment"
FUNC_TRACE_STEP: str = "trace_step"

# Programmatic override for unit tests (None = follow config)
_TRACING_OVERRIDE: Optional[bool] = None

# Sensitive key substring patterns to strictly redact
SENSITIVE_KEY_PATTERNS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "auth",
    "credential",
)

SENSITIVE_VALUE_PREFIXES: tuple[str, ...] = (
    "sk-",
    "pcsk_",
    "lsv2_",
    "npg_",
    "neo4j",
)


def set_tracing_override(enabled: Optional[bool]) -> None:
    """Set explicit programmatic override for tracing (useful for unit tests).

    @param enabled: True to enable, False to disable, None to use environment config.
    """
    global _TRACING_OVERRIDE
    _TRACING_OVERRIDE = enabled
    configure_tracing_environment()


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently active and configured.

    @returns: True if tracing is enabled and API key is present, False otherwise.
    """
    if _TRACING_OVERRIDE is not None:
        return _TRACING_OVERRIDE

    config = get_config()
    api_key = config.langsmith_api_key or os.getenv("LANGSMITH_API_KEY", "")
    tracing_flag = config.langsmith_tracing or os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    return bool(tracing_flag and api_key.strip())


def configure_tracing_environment() -> None:
    """Synchronize environment variables required by LangChain / LangSmith SDK."""
    if is_tracing_enabled():
        config = get_config()
        api_key = config.langsmith_api_key or os.getenv("LANGSMITH_API_KEY", "")
        project = config.langsmith_project or os.getenv("LANGSMITH_PROJECT", "distributed-rag")

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Scrub sensitive keys and secret values from metadata dictionary.

    @param meta: Raw metadata dictionary.
    @returns: Sanitized dictionary safe for remote observability.
    """
    clean_meta: dict[str, Any] = {}
    for k, v in meta.items():
        k_lower = str(k).lower()
        if any(pat in k_lower for pat in SENSITIVE_KEY_PATTERNS):
            clean_meta[k] = "[REDACTED]"
            continue

        if isinstance(v, str):
            if any(v.startswith(pfx) for pfx in SENSITIVE_VALUE_PREFIXES):
                clean_meta[k] = "[REDACTED]"
                continue
            clean_meta[k] = v
        elif isinstance(v, (int, float, bool)) or v is None:
            clean_meta[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Redact if elements look sensitive, otherwise store sanitized representation
            clean_meta[k] = [
                "[REDACTED]" if isinstance(item, str) and any(item.startswith(pfx) for pfx in SENSITIVE_VALUE_PREFIXES)
                else item
                for item in v
            ]
        else:
            clean_meta[k] = str(v)

    return clean_meta


# In-memory record history for verification and testing
_TRACE_RECORDS: list[dict[str, Any]] = []


def get_trace_records() -> list[dict[str, Any]]:
    """Return list of recorded trace step events."""
    return list(_TRACE_RECORDS)


def clear_trace_records() -> None:
    """Clear recorded trace step events."""
    global _TRACE_RECORDS
    _TRACE_RECORDS.clear()


def trace_step(
    name: str,
    run_type: str = "chain",
    extract_metadata: Optional[Callable[[tuple, dict, Any, Optional[Exception]], dict[str, Any]]] = None,
) -> Callable:
    """Decorator to instrument a RAG pipeline function with LangSmith tracing.

    Executes without overhead or network calls when tracing is disabled.
    Catches and logs any tracing-specific errors without affecting pipeline execution.

    @param name: Descriptive span name.
    @param run_type: LangSmith run type ('chain', 'retriever', 'embedding', 'llm').
    @param extract_metadata: Optional callable taking (args, kwargs, result, error) returning metadata dict.
    @returns: Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        # Import LangSmith traceable lazily
        try:
            from langsmith.run_helpers import traceable, get_current_run_tree
        except ImportError:
            traceable = None
            get_current_run_tree = None

        def _record_execution(
            args: tuple,
            kwargs: dict,
            res: Any,
            error_obj: Optional[Exception],
            duration_ms: float,
        ) -> dict[str, Any]:
            step_meta: dict[str, Any] = {"latency_ms": duration_ms}
            
            # Link correlation, query, and document context from contextvars
            from utils.logger import get_correlation_id, get_document_id, get_query_id
            corr_id = get_correlation_id()
            if corr_id and "correlation_id" not in step_meta:
                step_meta["correlation_id"] = corr_id
            doc_id = get_document_id()
            if doc_id and "document_id" not in step_meta:
                step_meta["document_id"] = doc_id
            qid = get_query_id()
            if qid and "query_id" not in step_meta:
                step_meta["query_id"] = qid

            if error_obj is not None:
                step_meta["error"] = str(error_obj)
                step_meta["error_type"] = getattr(error_obj, "error_code", type(error_obj).__name__)

            if extract_metadata is not None:
                try:
                    custom_meta = extract_metadata(args, kwargs, res, error_obj)
                    if isinstance(custom_meta, dict):
                        step_meta.update(custom_meta)
                except Exception as meta_extract_exc:
                    logger.warning(FUNC_TRACE_STEP, f"Error extracting metadata for span '{name}': {meta_extract_exc}")

            clean = sanitize_metadata(step_meta)
            _TRACE_RECORDS.append({
                "name": name,
                "run_type": run_type,
                "latency_ms": duration_ms,
                "metadata": clean,
                "error": str(error_obj) if error_obj else None,
            })
            return clean

        if traceable is not None:
            @traceable(name=name, run_type=run_type)
            def _traced_impl(*a, **kw) -> Any:
                start_t = time.perf_counter()
                error_obj: Optional[Exception] = None
                res = None
                try:
                    res = func(*a, **kw)
                    return res
                except Exception as exc:
                    error_obj = exc
                    raise exc
                finally:
                    duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
                    clean_meta = _record_execution(a, kw, res, error_obj, duration_ms)
                    try:
                        if get_current_run_tree is not None:
                            run_tree = get_current_run_tree()
                            if run_tree is not None:
                                run_tree.add_metadata(clean_meta)
                    except Exception as meta_exc:
                        logger.warning(FUNC_TRACE_STEP, f"Non-critical tracing metadata error in span '{name}': {meta_exc}")
        else:
            _traced_impl = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            if _traced_impl is not None:
                return _traced_impl(*args, **kwargs)

            # Fallback if traceable is unavailable but tracing flag was set
            start_t = time.perf_counter()
            err = None
            out = None
            try:
                out = func(*args, **kwargs)
                return out
            except Exception as e:
                err = e
                raise e
            finally:
                duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
                _record_execution(args, kwargs, out, err, duration_ms)

        return wrapper
    return decorator

