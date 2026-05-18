from __future__ import annotations

import asyncio
import ast
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# --- 接入后端服务 ---
_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from llm_factory import get_llm_service
from customer_service import generate_customer_service_reply_as_dict
from insights import top_pain_points_from_results
from rag_utils import SimpleRAGIndex
from reporting import (
    build_business_insight_payload,
    build_report_html,
    build_report_markdown,
    build_report_snapshot_svg,
    export_records_csv_bytes,
    export_records_excel_bytes,
    read_recent_log_events,
)
from app_repository import AppRepository
from task_manager import BatchAnalysisTaskManager
from config import (
    get_app_access_password,
    get_app_admin_password,
    get_app_db_path,
    get_llm_concurrency,
    get_llm_log_path,
    get_llm_max_retries,
    get_llm_rate_limit_rps,
    get_llm_timeout_seconds,
)

# 导入前端工具函数
from utils.cleaning import clean_review_dataframe
from utils.loaders import load_dataframe

# Session State 键名
SS_DF = "workspace_df"
SS_NAME = "workspace_filename"
SS_BATCH_RESULTS = "batch_results"
SS_BATCH_FAILED = "batch_failed"
SS_BATCH_RUNNING = "batch_running"
SS_BATCH_TEXT_COL = "batch_text_col"
SS_BATCH_JOB_ID = "batch_job_id"
SS_CS_CHAT_HISTORY = "customer_service_chat_history"
SS_UPLOAD_SIGNATURE = "workspace_upload_signature"
SS_UPLOAD_INFO = "workspace_upload_info"
SS_LAST_UPLOAD_ID = "last_upload_id"
SS_DEMO_CONTEXT = "demo_context"
SS_CS_DEFAULT_RULES = "customer_service_default_rules"
SS_ACCESS_GRANTED = "access_granted"
SS_USER_ROLE = "user_role"


_EN_WORD = re.compile(r"[A-Za-z]+")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_KB_DIR = _ROOT / "data" / "kb"
_SENTIMENT_ORDER = ["positive", "neutral", "negative"]
_SENTIMENT_COLORS = ["#10b981", "#f59e0b", "#ef4444"]

_DEMO_POSITIVE_WORDS = [
    "好",
    "满意",
    "喜欢",
    "不错",
    "流畅",
    "推荐",
    "excellent",
    "great",
    "good",
    "love",
    "satisfied",
]
_DEMO_NEGATIVE_WORDS = [
    "差",
    "坏",
    "慢",
    "破",
    "退款",
    "退货",
    "失望",
    "问题",
    "broken",
    "bad",
    "slow",
    "poor",
    "refund",
    "disappointed",
]
_DEMO_PAIN_KEYWORDS = {
    "物流慢": ["物流", "快递", "配送", "送货", "shipping", "delivery", "slow"],
    "包装破损": ["包装", "破损", "压坏", "漏", "package", "packaging", "damaged"],
    "质量不稳定": ["质量", "做工", "瑕疵", "坏", "断", "裂", "quality", "defect", "broken"],
    "尺寸不符": ["尺码", "尺寸", "偏大", "偏小", "size", "fit"],
    "客服响应慢": ["客服", "没人", "回复", "售后", "service", "support", "reply"],
    "价格体验不佳": ["贵", "价格", "优惠", "price", "expensive"],
    "功能故障": ["卡顿", "闪退", "不能用", "故障", "bug", "crash", "stuck"],
}


def _should_reply_in_english(config_lang: str, text: str) -> bool:
    """Use English when config is en, or input text is likely English."""
    if (config_lang or "").strip().lower() == "en":
        return True
    src = (text or "").strip()
    if not src:
        return False

    en_words = _EN_WORD.findall(src)
    cjk_chars = _CJK_CHAR.findall(src)
    if len(en_words) >= 3 and len(cjk_chars) == 0:
        return True
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in src)
    if ascii_letters >= 12 and ascii_letters > len(cjk_chars) * 2:
        return True
    return False


def _inject_demo_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
        }
        [data-testid="stSidebar"] {
            background: #f8fafc;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stMetric"] label {
            color: #475569;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
        }
        .stButton > button {
            border-radius: 7px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _available_kb_files(lang: str) -> list[Path]:
    if not _KB_DIR.exists():
        return []
    files = sorted(_KB_DIR.glob("*.md"))
    if lang == "en":
        return sorted(files, key=lambda p: (not p.stem.endswith("_en"), p.name))
    return sorted(files, key=lambda p: (p.stem.endswith("_en"), p.name))


def _read_kb_files(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        try:
            chunks.append(f"# {path.stem}\n{path.read_text(encoding='utf-8')}")
        except OSError:
            continue
    return "\n\n".join(chunks).strip()


def _read_saved_kb_docs_text() -> str:
    chunks: list[str] = []
    for doc in _list_saved_kb_docs():
        title = str(doc.get("title", "") or "").strip()
        content = str(doc.get("content", "") or "").strip()
        if content:
            chunks.append(f"# {title}\n{content}" if title else content)
    return "\n\n".join(chunks).strip()


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return _coerce_text_list(parsed)
        except (SyntaxError, ValueError):
            pass

    for sep in ["，", ",", "；", ";", "|", "\n"]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]
    return [text]


def _results_to_records(res_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in res_df.to_dict("records"):
        row["sentiment"] = str(row.get("sentiment", "")).strip().lower()
        row["pain_points"] = _coerce_text_list(row.get("pain_points"))
        records.append(row)
    return records


def _records_for_export(
    res_df: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
    text_col: str | None = None,
) -> list[dict[str, Any]]:
    records = _results_to_records(res_df)
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        return records

    for row in records:
        try:
            idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if idx not in source_df.index:
            continue
        source_row = source_df.loc[idx]
        if isinstance(source_row, pd.DataFrame):
            source_row = source_row.iloc[0]
        for col, value in source_row.items():
            key = f"source_{col}"
            if key in row:
                continue
            if text_col and col == text_col and not row.get("raw_text"):
                row["raw_text"] = value
            elif col != text_col:
                row[key] = value
    return records


def _record_summary(row: dict[str, Any], lang: str) -> str:
    keys = ["summary_en", "summary", "summary_zh"] if lang == "en" else ["summary_zh", "summary", "summary_en"]
    for key in keys:
        val = str(row.get(key, "") or "").strip()
        if val:
            return val
    return ""


def _shorten(text: Any, limit: int = 220) -> str:
    val = re.sub(r"\s+", " ", str(text or "")).strip()
    return val if len(val) <= limit else f"{val[:limit]}..."


def _format_bytes(size: int | float | None) -> str:
    value = float(size or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    if seconds < 1:
        return "<1s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _format_timestamp(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "-"
    normalized = raw.replace("Z", "+00:00")
    try:
        dt = pd.to_datetime(normalized, utc=True)
    except Exception:
        return raw[:19]
    if pd.isna(dt):
        return raw[:19]
    return dt.tz_convert("Asia/Hong_Kong").strftime("%Y-%m-%d %H:%M")


def _get_repo() -> AppRepository:
    return AppRepository()


def _get_task_manager() -> BatchAnalysisTaskManager:
    return BatchAnalysisTaskManager(_get_repo())


def _cancel_batch_job(job_id: str, reason: str = "cancelled by user") -> None:
    _get_task_manager().mark_cancelled(job_id, reason)


def _set_batch_job_archived(job_id: str, archived: bool = True) -> None:
    _get_task_manager().set_archived(job_id, archived)


def _get_configured_access_password() -> str:
    secret_value = None
    try:
        secret_value = st.secrets.get("APP_ACCESS_PASSWORD")
    except Exception:
        secret_value = None
    return str(secret_value or get_app_access_password() or "").strip()


def _get_configured_admin_password() -> str:
    secret_value = None
    try:
        secret_value = st.secrets.get("APP_ADMIN_PASSWORD")
    except Exception:
        secret_value = None
    return str(secret_value or get_app_admin_password() or "").strip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower())
    return slug.strip("-") or "demo-merchant"


def _is_admin() -> bool:
    return st.session_state.get(SS_USER_ROLE, "operator") == "admin"


def _get_demo_context() -> dict[str, str]:
    if SS_DEMO_CONTEXT not in st.session_state:
        merchant_name = str(st.session_state.get("merchant_name", "Demo Merchant") or "Demo Merchant").strip()
        merchant_slug = _slugify(st.session_state.get("merchant_slug", merchant_name))
        operator_name = str(st.session_state.get("operator_name", "Demo User") or "Demo User").strip()
        ctx = _get_repo().ensure_context(
            merchant_slug=merchant_slug,
            merchant_name=merchant_name,
            user_name=operator_name,
        )
        st.session_state[SS_DEMO_CONTEXT] = {
            "merchant_id": ctx.merchant_id,
            "user_id": ctx.user_id,
            "merchant_name": ctx.merchant_name or merchant_name,
            "merchant_slug": ctx.merchant_slug or merchant_slug,
            "user_name": ctx.user_name or operator_name,
        }
    return st.session_state[SS_DEMO_CONTEXT]


def _persist_upload(upload_info: dict[str, Any], upload_signature: str) -> None:
    ctx = _get_demo_context()
    upload_id = _get_repo().record_upload(
        merchant_id=ctx["merchant_id"],
        user_id=ctx["user_id"],
        filename=str(upload_info.get("name", "-")),
        upload_signature=upload_signature,
        file_size=int(upload_info.get("size", 0) or 0),
        row_count=int(upload_info.get("rows", 0) or 0),
        col_count=int(upload_info.get("cols", 0) or 0),
    )
    st.session_state[SS_LAST_UPLOAD_ID] = upload_id


def _create_batch_job(
    *,
    filename: str,
    text_column: str,
    row_count: int,
    provider: str,
    model: str,
    summary_language: str,
    parent_job_id: str | None = None,
    rerun_scope: str | None = None,
):
    ctx = _get_demo_context()
    task = _get_task_manager().create_job(
        merchant_id=ctx["merchant_id"],
        user_id=ctx["user_id"],
        uploaded_file_id=st.session_state.get(SS_LAST_UPLOAD_ID),
        filename=filename,
        text_column=text_column,
        provider=provider,
        model=model,
        summary_language=summary_language,
        row_count=row_count,
        parent_job_id=parent_job_id,
        rerun_scope=rerun_scope,
    )
    st.session_state[SS_BATCH_JOB_ID] = task.job_id
    return task


def _execute_batch_task(
    *,
    task,
    rows: list[tuple[int, str]],
    i18n: dict[str, str],
    progress_text: str,
) -> list[dict[str, Any]]:
    progress = st.progress(0.0, text=i18n["progress_preparing"])
    status_placeholder = st.empty()
    task_manager = _get_task_manager()

    try:
        st.session_state[SS_BATCH_RUNNING] = True
        st.session_state[SS_BATCH_JOB_ID] = task.job_id
        task_manager.mark_running(task.job_id)

        def _progress_hook(processed_count: int, failed_count: int) -> None:
            task_manager.mark_progress(
                task.job_id,
                processed_count=processed_count,
                failed_count=failed_count,
            )

        if st.session_state.get("demo_mode", False):
            results = _run_demo_batch_analysis(
                rows,
                task.summary_language,
                progress,
                progress_text=progress_text,
                status_placeholder=status_placeholder,
                i18n=i18n,
                progress_hook=_progress_hook,
            )
        else:
            service = get_llm_service(
                provider=task.provider,
                model=task.model,
            )
            results = asyncio.run(
                _run_batch_analysis(
                    service,
                    rows,
                    task.summary_language,
                    progress,
                    progress_text=progress_text,
                    status_placeholder=status_placeholder,
                    i18n=i18n,
                    progress_hook=_progress_hook,
                )
            )

        task_manager.mark_completed(task.job_id, results)
        return results
    except Exception as exc:
        task_manager.mark_failed(task.job_id, str(exc))
        raise
    finally:
        st.session_state[SS_BATCH_RUNNING] = False


def _persist_customer_service_reply(
    *,
    review_text: str,
    merchant_rules: str,
    knowledge_base_used: bool,
    result: dict[str, Any],
) -> None:
    ctx = _get_demo_context()
    _get_repo().record_customer_service_reply(
        merchant_id=ctx["merchant_id"],
        user_id=ctx["user_id"],
        review_text=review_text,
        merchant_rules=merchant_rules,
        knowledge_base_used=knowledge_base_used,
        result=result,
    )


def _get_default_rules() -> str:
    ctx = _get_demo_context()
    settings = _get_repo().get_merchant_settings(ctx["merchant_id"])
    rules = str(settings.get("default_rules", "") or "")
    st.session_state[SS_CS_DEFAULT_RULES] = rules
    return rules


def _save_default_rules(rules_text: str) -> None:
    ctx = _get_demo_context()
    _get_repo().save_merchant_rules(ctx["merchant_id"], rules_text)
    st.session_state[SS_CS_DEFAULT_RULES] = rules_text


def _list_saved_kb_docs() -> list[dict[str, Any]]:
    ctx = _get_demo_context()
    return _get_repo().list_knowledge_base_docs(ctx["merchant_id"], limit=100)


def _save_kb_doc(title: str, content: str, doc_id: str | None = None) -> str:
    ctx = _get_demo_context()
    return _get_repo().upsert_knowledge_base_doc(
        merchant_id=ctx["merchant_id"],
        user_id=ctx["user_id"],
        title=title,
        content=content,
        doc_id=doc_id,
    )


def _delete_kb_doc(doc_id: str) -> bool:
    ctx = _get_demo_context()
    return _get_repo().delete_knowledge_base_doc(ctx["merchant_id"], doc_id)


def _uploaded_signature(uploaded: Any) -> str:
    return f"{getattr(uploaded, 'name', '')}:{getattr(uploaded, 'size', 0)}"


def _estimate_parse_seconds(uploaded: Any) -> float:
    size_mb = float(getattr(uploaded, "size", 0) or 0) / (1024 * 1024)
    name = str(getattr(uploaded, "name", "") or "").lower()
    factor = 0.9 if name.endswith((".xlsx", ".xls")) else 0.35
    return max(0.5, size_mb * factor)


def _load_dataframe_with_status(uploaded: Any, i18n: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    progress = st.progress(0.05, text=i18n["upload_progress_receiving"])
    status_placeholder = st.empty()
    started = time.perf_counter()
    estimated = _estimate_parse_seconds(uploaded)
    status_placeholder.caption(
        i18n["upload_meta"].format(
            name=getattr(uploaded, "name", "-"),
            size=_format_bytes(getattr(uploaded, "size", 0)),
            elapsed=_format_duration(0),
            eta=_format_duration(estimated),
        )
    )

    progress.progress(0.3, text=i18n["upload_progress_parsing"].format(eta=_format_duration(estimated)))
    df_new = load_dataframe(uploaded)
    elapsed = time.perf_counter() - started

    progress.progress(0.82, text=i18n["upload_progress_validating"])
    row_count = int(len(df_new))
    col_count = int(len(df_new.columns))
    progress.progress(
        1.0,
        text=i18n["upload_progress_ready"].format(rows=f"{row_count:,}", cols=col_count),
    )
    status_placeholder.caption(
        i18n["upload_meta"].format(
            name=getattr(uploaded, "name", "-"),
            size=_format_bytes(getattr(uploaded, "size", 0)),
            elapsed=_format_duration(elapsed),
            eta=_format_duration(0),
        )
    )
    info = {
        "name": getattr(uploaded, "name", "-"),
        "size": int(getattr(uploaded, "size", 0) or 0),
        "rows": row_count,
        "cols": col_count,
        "elapsed_seconds": round(elapsed, 3),
    }
    return df_new, info


def _render_recent_activity(i18n: dict[str, str]) -> None:
    ctx = _get_demo_context()
    repo = _get_repo()
    uploads = repo.list_recent_uploads(ctx["merchant_id"], limit=3)
    jobs = repo.list_recent_jobs(ctx["merchant_id"], limit=3)
    replies = repo.list_recent_replies(ctx["merchant_id"], limit=3)
    with st.expander(i18n["activity_title"], expanded=False):
        st.caption(i18n["activity_uploads"])
        if uploads:
            for item in uploads:
                st.caption(
                    i18n["activity_upload_item"].format(
                        name=item.get("filename", "-"),
                        rows=int(item.get("row_count", 0) or 0),
                        cols=int(item.get("col_count", 0) or 0),
                    )
                )
        else:
            st.caption(i18n["activity_empty"])

        st.caption(i18n["activity_jobs"])
        if jobs:
            for item in jobs:
                st.caption(
                    i18n["activity_job_item"].format(
                        status=item.get("status", "-"),
                        name=item.get("filename", "-"),
                        rows=int(item.get("row_count", 0) or 0),
                    )
                )
        else:
            st.caption(i18n["activity_empty"])

        st.caption(i18n["activity_replies"])
        if replies:
            for item in replies:
                st.caption(
                    i18n["activity_reply_item"].format(
                        lang=item.get("reply_language", "-"),
                        guard=item.get("guardrail_action", "-"),
                    )
                )
        else:
            st.caption(i18n["activity_empty"])


def _dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _summarize_result_errors(result_rows: list[dict[str, Any]]) -> tuple[int, list[tuple[str, int]]]:
    counts: dict[str, int] = {}
    for row in result_rows:
        message = str(row.get("error_message", "") or "").strip()
        if message:
            counts[message] = counts.get(message, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return sum(counts.values()), ranked


def _format_job_event(event: dict[str, Any], i18n: dict[str, str]) -> str:
    timestamp = _format_timestamp(event.get("created_at"))
    event_type = str(event.get("event_type", "") or "").strip()
    message = str(event.get("message", "") or "").strip()
    meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}

    if event_type == "rerun_created" and meta:
        scope = meta.get("rerun_scope", "all")
        parent = meta.get("parent_job_id", "-")
        message = i18n["history_event_rerun_created"].format(parent=parent, scope=scope)
    elif event_type == "progress" and meta:
        message = i18n["history_event_progress"].format(
            processed=int(meta.get("processed_count", 0) or 0),
            failed=int(meta.get("failed_count", 0) or 0),
        )
    elif not message:
        message = i18n["history_event_generic"].format(event_type=event_type or "info")
    return f"{timestamp} · {message}"


def _load_recent_log_events(limit: int = 200) -> list[dict[str, Any]]:
    log_path = get_llm_log_path()
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    events: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)) :]:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _render_admin_center(i18n: dict[str, str]) -> None:
    st.subheader(i18n["admin_title"])
    if not _is_admin():
        st.info(i18n["admin_only"])
        return

    ctx = _get_demo_context()
    events = _load_recent_log_events(limit=300)
    ok_count = sum(1 for row in events if row.get("status") == "ok")
    error_count = sum(1 for row in events if row.get("status") == "error")
    avg_latency = (
        round(sum(float(row.get("latency_ms", 0) or 0) for row in events if row.get("latency_ms") is not None) / max(1, len([row for row in events if row.get("latency_ms") is not None])), 1)
        if events
        else 0.0
    )

    top_cols = st.columns(4)
    top_cols[0].metric(i18n["admin_metric_logs"], len(events))
    top_cols[1].metric(i18n["admin_metric_ok"], ok_count)
    top_cols[2].metric(i18n["admin_metric_error"], error_count)
    top_cols[3].metric(i18n["admin_metric_latency"], f"{avg_latency} ms")

    config_rows = [
        {i18n["admin_config_key"]: "merchant", i18n["admin_config_value"]: ctx.get("merchant_name", "-")},
        {i18n["admin_config_key"]: "merchant_slug", i18n["admin_config_value"]: ctx.get("merchant_slug", "-")},
        {i18n["admin_config_key"]: "operator", i18n["admin_config_value"]: ctx.get("user_name", "-")},
        {i18n["admin_config_key"]: "llm_provider", i18n["admin_config_value"]: st.session_state.get("llm_provider", "-")},
        {i18n["admin_config_key"]: "llm_model", i18n["admin_config_value"]: st.session_state.get("llm_model", "-")},
        {i18n["admin_config_key"]: "timeout_seconds", i18n["admin_config_value"]: get_llm_timeout_seconds()},
        {i18n["admin_config_key"]: "max_retries", i18n["admin_config_value"]: get_llm_max_retries()},
        {i18n["admin_config_key"]: "concurrency", i18n["admin_config_value"]: get_llm_concurrency()},
        {i18n["admin_config_key"]: "rate_limit_rps", i18n["admin_config_value"]: get_llm_rate_limit_rps()},
        {i18n["admin_config_key"]: "app_db_path", i18n["admin_config_value"]: str(get_app_db_path())},
        {i18n["admin_config_key"]: "llm_log_path", i18n["admin_config_value"]: str(get_llm_log_path())},
    ]
    st.markdown(f"**{i18n['admin_config_title']}**")
    st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)

    if events:
        error_types: dict[str, int] = {}
        guard_actions: dict[str, int] = {}
        for row in events:
            error_type = str(row.get("error_type", "") or "").strip()
            guard_action = str(row.get("guardrail_action", "") or "").strip()
            if error_type:
                error_types[error_type] = error_types.get(error_type, 0) + 1
            if guard_action:
                guard_actions[guard_action] = guard_actions.get(guard_action, 0) + 1

        summary_cols = st.columns(2)
        with summary_cols[0]:
            st.markdown(f"**{i18n['admin_error_types_title']}**")
            if error_types:
                error_df = pd.DataFrame(
                    [{i18n["admin_error_type_col"]: k, i18n["admin_count_col"]: v} for k, v in sorted(error_types.items(), key=lambda item: item[1], reverse=True)]
                )
                st.dataframe(error_df, use_container_width=True, hide_index=True)
            else:
                st.info(i18n["admin_empty"])
        with summary_cols[1]:
            st.markdown(f"**{i18n['admin_guard_title']}**")
            if guard_actions:
                guard_df = pd.DataFrame(
                    [{i18n["admin_guard_col"]: k, i18n["admin_count_col"]: v} for k, v in sorted(guard_actions.items(), key=lambda item: item[1], reverse=True)]
                )
                st.dataframe(guard_df, use_container_width=True, hide_index=True)
            else:
                st.info(i18n["admin_empty"])

        st.markdown(f"**{i18n['admin_logs_title']}**")
        recent_df = pd.DataFrame(
            [
                {
                    i18n["admin_log_time"]: _format_timestamp(row.get("ts")),
                    i18n["admin_log_operation"]: row.get("operation", "-"),
                    i18n["admin_log_status"]: row.get("status", "-"),
                    i18n["admin_log_provider"]: row.get("provider", "-"),
                    i18n["admin_log_model"]: row.get("model", "-"),
                    i18n["admin_log_latency"]: row.get("latency_ms", "-"),
                    i18n["admin_log_error"]: row.get("error_type", ""),
                    i18n["admin_log_request"]: row.get("request_id", ""),
                    i18n["admin_log_guard"]: row.get("guardrail_action", ""),
                }
                for row in reversed(events[-100:])
            ]
        )
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        st.download_button(
            i18n["admin_download_logs"],
            data=_dataframe_to_csv_bytes(recent_df),
            file_name="admin_recent_logs.csv",
            mime="text/csv",
            key="download_admin_logs",
        )
    else:
        st.info(i18n["admin_no_logs"])


def _render_history_center(i18n: dict[str, str]) -> None:
    ctx = _get_demo_context()
    repo = _get_repo()
    uploads = repo.list_recent_uploads(ctx["merchant_id"], limit=20)
    jobs = repo.list_recent_jobs(ctx["merchant_id"], limit=20)
    replies = repo.list_customer_service_replies(ctx["merchant_id"], limit=20)
    can_export = _is_admin()

    st.subheader(i18n["history_title"])
    metric_cols = st.columns(3)
    metric_cols[0].metric(i18n["history_metric_uploads"], len(uploads))
    metric_cols[1].metric(i18n["history_metric_jobs"], len(jobs))
    metric_cols[2].metric(i18n["history_metric_replies"], len(replies))

    st.caption(i18n["history_hint"])

    st.markdown(f"**{i18n['history_uploads_title']}**")
    if uploads:
        uploads_df = pd.DataFrame(
            [
                {
                    i18n["history_col_time"]: _format_timestamp(row.get("created_at")),
                    i18n["history_col_file"]: row.get("filename", "-"),
                    i18n["history_col_rows"]: int(row.get("row_count", 0) or 0),
                    i18n["history_col_cols"]: int(row.get("col_count", 0) or 0),
                    i18n["history_col_size"]: _format_bytes(int(row.get("file_size", 0) or 0)),
                }
                for row in uploads
            ]
        )
        if can_export:
            st.download_button(
                i18n["history_download_uploads"],
                data=_dataframe_to_csv_bytes(uploads_df),
                file_name="uploads_history.csv",
                mime="text/csv",
                key="download_uploads_history",
            )
        st.dataframe(uploads_df, use_container_width=True, hide_index=True)
    else:
        st.info(i18n["history_empty"])

    st.markdown(f"**{i18n['history_jobs_title']}**")
    if jobs:
        filter_cols = st.columns([1, 1, 2, 1])
        with filter_cols[0]:
            status_filter = st.selectbox(
                i18n["history_filter_status"],
                options=[i18n["history_filter_all"], "queued", "running", "completed", "failed", "cancelled"],
                key="history_filter_status",
            )
        with filter_cols[1]:
            provider_values = sorted({str(row.get("provider", "-")) for row in jobs})
            provider_filter = st.selectbox(
                i18n["history_filter_provider"],
                options=[i18n["history_filter_all"], *provider_values],
                key="history_filter_provider",
            )
        with filter_cols[2]:
            keyword = st.text_input(i18n["history_filter_keyword"], key="history_filter_keyword").strip().lower()
        with filter_cols[3]:
            include_archived = st.checkbox(i18n["history_filter_archived"], value=False, key="history_filter_archived")

        filtered_jobs = []
        for row in jobs:
            status_ok = status_filter == i18n["history_filter_all"] or row.get("status") == status_filter
            provider_ok = provider_filter == i18n["history_filter_all"] or str(row.get("provider", "-")) == provider_filter
            keyword_ok = not keyword or keyword in str(row.get("filename", "")).lower()
            archive_ok = include_archived or not bool(row.get("archived", False))
            if status_ok and provider_ok and keyword_ok and archive_ok:
                filtered_jobs.append(row)

        if not filtered_jobs:
            st.info(i18n["history_filter_empty"])
        else:
            job_options = {
                (
                    f"{_format_timestamp(row.get('created_at'))} | "
                    f"{row.get('status', '-')} | {row.get('filename', '-')}"
                ): row["id"]
                for row in filtered_jobs
            }
            jobs_df = pd.DataFrame(
                [
                    {
                        i18n["history_col_time"]: _format_timestamp(row.get("created_at")),
                        i18n["history_col_status"]: row.get("status", "-"),
                        i18n["history_col_file"]: row.get("filename", "-"),
                        i18n["history_col_rows"]: int(row.get("row_count", 0) or 0),
                        i18n["history_col_progress"]: f"{int(row.get('processed_count', 0) or 0)}/{int(row.get('row_count', 0) or 0)}",
                        i18n["history_col_provider"]: row.get("provider", "-"),
                        i18n["history_col_model"]: row.get("model", "-"),
                        i18n["history_col_lang"]: row.get("summary_language", "-"),
                        i18n["history_col_archived"]: i18n["history_yes"] if row.get("archived", False) else i18n["history_no"],
                    }
                    for row in filtered_jobs
                ]
            )
            if can_export:
                st.download_button(
                    i18n["history_download_jobs"],
                    data=_dataframe_to_csv_bytes(jobs_df),
                    file_name="analysis_jobs.csv",
                    mime="text/csv",
                    key="download_jobs_history",
                )
            st.dataframe(jobs_df, use_container_width=True, hide_index=True)

            selected_label = st.selectbox(
                i18n["history_job_picker"],
                options=list(job_options.keys()),
                key="history_job_picker",
            )
            selected_job = repo.get_analysis_job(job_options[selected_label])
            if selected_job:
                detail_cols = st.columns(4)
                detail_cols[0].metric(i18n["history_detail_status"], selected_job.get("status", "-"))
                detail_cols[1].metric(i18n["history_detail_rows"], int(selected_job.get("row_count", 0) or 0))
                detail_cols[2].metric(i18n["history_detail_provider"], str(selected_job.get("provider", "-")).upper())
                detail_cols[3].metric(
                    i18n["history_detail_progress"],
                    f"{int(selected_job.get('processed_count', 0) or 0)}/{int(selected_job.get('row_count', 0) or 0)}",
                )
                st.caption(
                    i18n["history_job_meta"].format(
                        filename=selected_job.get("filename", "-"),
                        text_column=selected_job.get("text_column", "-"),
                        model=selected_job.get("model", "-"),
                        created_at=_format_timestamp(selected_job.get("created_at")),
                        completed_at=_format_timestamp(selected_job.get("completed_at")),
                    )
                )
                if selected_job.get("error_message"):
                    st.warning(i18n["history_job_error"].format(error=selected_job.get("error_message", "")))

                result_rows = repo.list_analysis_results(str(selected_job.get("id")), limit=200)
                job_events = repo.list_analysis_job_events(str(selected_job.get("id")), limit=30)
                failed_total, failure_breakdown = _summarize_result_errors(result_rows)

                if failed_total or job_events:
                    insight_cols = st.columns(2)
                    with insight_cols[0]:
                        st.markdown(f"**{i18n['history_failure_title']}**")
                        if failed_total:
                            st.caption(
                                i18n["history_failure_summary"].format(
                                    failed=failed_total,
                                    unique=len(failure_breakdown),
                                )
                            )
                            failure_df = pd.DataFrame(
                                [
                                    {
                                        i18n["history_failure_error"]: _shorten(message, 120),
                                        i18n["history_failure_count"]: count,
                                    }
                                    for message, count in failure_breakdown[:8]
                                ]
                            )
                            st.dataframe(failure_df, use_container_width=True, hide_index=True)
                        else:
                            st.caption(i18n["history_failure_empty"])
                    with insight_cols[1]:
                        st.markdown(f"**{i18n['history_timeline_title']}**")
                        if job_events:
                            for event in job_events:
                                st.caption(_format_job_event(event, i18n))
                        else:
                            st.caption(i18n["history_timeline_empty"])

                admin_job_cols = st.columns([1, 1])
                with admin_job_cols[0]:
                    can_cancel = str(selected_job.get("status", "")) in {"queued", "running"}
                    if st.button(
                        i18n["history_cancel_job"],
                        key=f"history_cancel_{selected_job.get('id')}",
                        use_container_width=True,
                        disabled=not can_cancel,
                    ):
                        _cancel_batch_job(str(selected_job.get("id")))
                        st.success(i18n["history_cancel_success"])
                        st.rerun()
                with admin_job_cols[1]:
                    archive_label = i18n["history_unarchive_job"] if selected_job.get("archived", False) else i18n["history_archive_job"]
                    if st.button(
                        archive_label,
                        key=f"history_archive_{selected_job.get('id')}",
                        use_container_width=True,
                    ):
                        _set_batch_job_archived(str(selected_job.get("id")), not bool(selected_job.get("archived", False)))
                        st.success(i18n["history_archive_success"])
                        st.rerun()
                rerun_cols = st.columns([1, 1])
                with rerun_cols[0]:
                    rerun_all = st.button(
                        i18n["history_rerun_all"],
                        key=f"history_rerun_all_{selected_job.get('id')}",
                        use_container_width=True,
                    )
                with rerun_cols[1]:
                    rerun_failed = st.button(
                        i18n["history_rerun_failed"],
                        key=f"history_rerun_failed_{selected_job.get('id')}",
                        use_container_width=True,
                        disabled=not any(str(row.get("error_message", "") or "").strip() for row in result_rows),
                    )

                if rerun_all or rerun_failed:
                    rerun_source = result_rows
                    if rerun_failed:
                        rerun_source = [
                            row for row in result_rows if str(row.get("error_message", "") or "").strip()
                        ]
                    rerun_rows = [
                        (int(row.get("row_index", 0) or 0), str(row.get("raw_text", "") or row.get("preview", "") or ""))
                        for row in rerun_source
                        if str(row.get("raw_text", "") or row.get("preview", "") or "").strip()
                    ]
                    if not rerun_rows:
                        st.warning(i18n["history_rerun_empty"])
                    else:
                        rerun_task = _create_batch_job(
                            filename=str(selected_job.get("filename", "rerun.csv")),
                            text_column=str(selected_job.get("text_column", "review_text")),
                            row_count=len(rerun_rows),
                            provider=str(selected_job.get("provider", st.session_state.get("llm_provider", "unknown"))),
                            model=str(selected_job.get("model", st.session_state.get("llm_model", ""))),
                            summary_language=str(selected_job.get("summary_language", st.session_state.get("summary_language", "zh"))),
                            parent_job_id=str(selected_job.get("id", "")),
                            rerun_scope="failed" if rerun_failed else "all",
                        )
                        try:
                            rerun_results = _execute_batch_task(
                                task=rerun_task,
                                rows=rerun_rows,
                                i18n=i18n,
                                progress_text=i18n["history_rerun_progress"],
                            )
                            rerun_df = pd.DataFrame(rerun_results)
                            st.session_state[SS_BATCH_RESULTS] = rerun_df
                            st.session_state[SS_BATCH_TEXT_COL] = str(selected_job.get("text_column", "review_text"))
                            if "error" in rerun_df.columns:
                                st.session_state[SS_BATCH_FAILED] = rerun_df[rerun_df["error"].notna()].copy()
                            else:
                                st.session_state[SS_BATCH_FAILED] = pd.DataFrame()
                            st.success(i18n["history_rerun_success"])
                            st.rerun()
                        except Exception as exc:
                            st.error(i18n["history_rerun_error"].format(error=exc))

                if result_rows:
                    result_df = pd.DataFrame(
                        [
                            {
                                i18n["history_result_row"]: int(row.get("row_index", 0) or 0),
                                i18n["history_result_preview"]: _shorten(row.get("preview", ""), 80),
                                i18n["history_result_sentiment"]: row.get("sentiment", "-"),
                                i18n["history_result_confidence"]: round(float(row.get("confidence", 0.0) or 0.0), 3),
                                i18n["history_result_pain_points"]: ", ".join(row.get("pain_points", []) or []),
                                i18n["history_result_summary"]: row.get("summary_text", ""),
                                i18n["history_result_error"]: row.get("error_message", ""),
                            }
                            for row in result_rows
                        ]
                    )
                    if can_export:
                        st.download_button(
                            i18n["history_download_results"],
                            data=_dataframe_to_csv_bytes(result_df),
                            file_name=f"analysis_results_{selected_job.get('id', 'job')}.csv",
                            mime="text/csv",
                            key=f"download_job_results_{selected_job.get('id', 'job')}",
                        )
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                else:
                    st.info(i18n["history_job_no_results"])
    else:
        st.info(i18n["history_empty"])

    st.markdown(f"**{i18n['history_replies_title']}**")
    if replies:
        reply_cols = st.columns([1, 1, 2])
        with reply_cols[0]:
            reply_lang_filter = st.selectbox(
                i18n["history_filter_reply_lang"],
                options=[i18n["history_filter_all"], "zh", "en"],
                key="history_filter_reply_lang",
            )
        with reply_cols[1]:
            guard_values = sorted({str(row.get("guardrail_action", "-")) for row in replies})
            guard_filter = st.selectbox(
                i18n["history_filter_guard"],
                options=[i18n["history_filter_all"], *guard_values],
                key="history_filter_guard",
            )
        with reply_cols[2]:
            reply_keyword = st.text_input(i18n["history_filter_reply_keyword"], key="history_filter_reply_keyword").strip().lower()

        filtered_replies = []
        for row in replies:
            lang_ok = reply_lang_filter == i18n["history_filter_all"] or str(row.get("reply_language", "-")) == reply_lang_filter
            guard_ok = guard_filter == i18n["history_filter_all"] or str(row.get("guardrail_action", "-")) == guard_filter
            review_text = str(row.get("review_text", "") or "")
            reply_text = str(row.get("reply_text", "") or "")
            keyword_ok = not reply_keyword or reply_keyword in review_text.lower() or reply_keyword in reply_text.lower()
            if lang_ok and guard_ok and keyword_ok:
                filtered_replies.append(row)

        if not filtered_replies:
            st.info(i18n["history_filter_empty"])
        else:
            replies_df = pd.DataFrame(
                [
                    {
                        i18n["history_col_time"]: _format_timestamp(row.get("created_at")),
                        i18n["history_col_lang"]: row.get("reply_language", "-"),
                        i18n["history_col_provider"]: row.get("provider", "-"),
                        i18n["history_col_guard"]: row.get("guardrail_action", "-"),
                        i18n["history_col_review"]: _shorten(row.get("review_text", ""), 70),
                        i18n["history_col_reply"]: _shorten(row.get("reply_text", ""), 90),
                    }
                    for row in filtered_replies
                ]
            )
            if can_export:
                st.download_button(
                    i18n["history_download_replies"],
                    data=_dataframe_to_csv_bytes(replies_df),
                    file_name="customer_service_replies.csv",
                    mime="text/csv",
                    key="download_replies_history",
                )
            st.dataframe(replies_df, use_container_width=True, hide_index=True)
    else:
        st.info(i18n["history_empty"])

    if not can_export:
        st.caption(i18n["history_export_admin_only"])


def _render_rules_center(i18n: dict[str, str]) -> None:
    is_admin = _is_admin()
    st.subheader(i18n["rules_title"])
    st.caption(i18n["rules_hint"])
    if not is_admin:
        st.info(i18n["rules_admin_only"])

    default_rules = st.text_area(
        i18n["rules_default_label"],
        value=st.session_state.get(SS_CS_DEFAULT_RULES, _get_default_rules()),
        height=180,
        key="rules_default_text",
        disabled=not is_admin,
    )
    save_col, refresh_col = st.columns([1, 1])
    with save_col:
        if st.button(i18n["rules_save_btn"], type="primary", use_container_width=True, disabled=not is_admin):
            _save_default_rules(default_rules.strip())
            st.success(i18n["rules_save_success"])
    with refresh_col:
        if st.button(i18n["rules_reload_btn"], use_container_width=True):
            st.session_state[SS_CS_DEFAULT_RULES] = _get_default_rules()
            st.rerun()

    st.markdown(f"**{i18n['rules_kb_title']}**")
    kb_docs = _list_saved_kb_docs()
    if kb_docs:
        kb_df = pd.DataFrame(
            [
                {
                    i18n["rules_kb_col_title"]: row.get("title", "-"),
                    i18n["history_col_time"]: _format_timestamp(row.get("updated_at") or row.get("created_at")),
                    i18n["rules_kb_col_chars"]: len(str(row.get("content", "") or "")),
                }
                for row in kb_docs
            ]
        )
        st.dataframe(kb_df, use_container_width=True, hide_index=True)
    else:
        st.info(i18n["rules_kb_empty"])

    selected_label_map = {
        f"{row.get('title', '-') } | {_format_timestamp(row.get('updated_at') or row.get('created_at'))}": row
        for row in kb_docs
    }
    selected_label = st.selectbox(
        i18n["rules_kb_picker"],
        options=[i18n["rules_kb_new_option"], *selected_label_map.keys()],
        key="rules_kb_picker",
    )
    selected_doc = selected_label_map.get(selected_label)

    kb_title = st.text_input(
        i18n["rules_kb_title_input"],
        value=str(selected_doc.get("title", "") if selected_doc else ""),
        key=f"rules_kb_title_input_{selected_doc.get('id', 'new') if selected_doc else 'new'}",
        disabled=not is_admin,
    )
    kb_content = st.text_area(
        i18n["rules_kb_content_input"],
        value=str(selected_doc.get("content", "") if selected_doc else ""),
        height=220,
        key=f"rules_kb_content_input_{selected_doc.get('id', 'new') if selected_doc else 'new'}",
        disabled=not is_admin,
    )

    kb_action_cols = st.columns([1, 1])
    with kb_action_cols[0]:
        if st.button(i18n["rules_kb_save_btn"], use_container_width=True, disabled=not is_admin):
            if not kb_title.strip() or not kb_content.strip():
                st.warning(i18n["rules_kb_warn_empty"])
            else:
                _save_kb_doc(kb_title.strip(), kb_content.strip(), selected_doc.get("id") if selected_doc else None)
                st.success(i18n["rules_kb_save_success"])
                st.rerun()
    with kb_action_cols[1]:
        if selected_doc and st.button(i18n["rules_kb_delete_btn"], use_container_width=True, disabled=not is_admin):
            _delete_kb_doc(str(selected_doc.get("id")))
            st.success(i18n["rules_kb_delete_success"])
            st.rerun()


def _progress_stats(started: float, finished: int, total: int) -> tuple[float, float, float | None]:
    elapsed = max(time.perf_counter() - started, 0.001)
    rate = finished / elapsed if finished else 0.0
    eta = ((total - finished) / rate) if rate > 0 else None
    return elapsed, rate, eta


def _update_batch_progress(
    progress,
    status_placeholder,
    i18n: dict[str, str],
    label: str,
    finished: int,
    total: int,
    failed: int,
    started: float,
) -> None:
    total_safe = max(1, total)
    elapsed, rate, eta = _progress_stats(started, finished, total_safe)
    progress.progress(
        min(1.0, finished / total_safe),
        text=f"{label} {finished}/{total_safe} ({finished / total_safe:.0%})",
    )
    status_placeholder.caption(
        i18n["progress_status"].format(
            finished=finished,
            total=total_safe,
            failed=failed,
            elapsed=_format_duration(elapsed),
            eta=_format_duration(eta),
            rate=f"{rate:.2f}",
        )
    )


def _local_analyze_review(review_text: str, summary_language: str = "zh") -> dict[str, Any]:
    text = (review_text or "").strip()
    lower_text = text.lower()
    pos_score = sum(1 for word in _DEMO_POSITIVE_WORDS if word in lower_text or word in text)
    neg_score = sum(1 for word in _DEMO_NEGATIVE_WORDS if word in lower_text or word in text)

    if neg_score > pos_score:
        sentiment = "negative"
    elif pos_score > neg_score:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    pain_points: list[str] = []
    if sentiment in {"negative", "neutral"}:
        for label, keywords in _DEMO_PAIN_KEYWORDS.items():
            if any(keyword in lower_text or keyword in text for keyword in keywords):
                pain_points.append(label)
    if sentiment == "negative" and not pain_points:
        pain_points.append("综合体验不佳")

    confidence = min(0.96, 0.62 + 0.08 * abs(pos_score - neg_score) + 0.04 * len(pain_points))
    summary_zh = f"该评论整体偏{ {'positive': '正向', 'neutral': '中性', 'negative': '负向'}[sentiment] }，核心关注点为{('、'.join(pain_points) if pain_points else '常规体验反馈')}。"
    summary_en = (
        f"The review is {sentiment}; key issue: "
        f"{(', '.join(pain_points) if pain_points else 'general customer experience')}."
    )
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "pain_points": pain_points,
        "summary_zh": summary_zh,
        "summary_en": summary_en,
        "demo_mode": True,
    }


def _run_demo_batch_analysis(
    rows: list[tuple[int, str]],
    summary_language: str,
    progress,
    progress_text: str,
    status_placeholder,
    i18n: dict[str, str],
    progress_hook=None,
) -> list[dict[str, Any]]:
    total = max(1, len(rows))
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    _update_batch_progress(progress, status_placeholder, i18n, progress_text, 0, total, 0, started)
    for finished_count, (idx, raw_text) in enumerate(rows, start=1):
        res = _local_analyze_review(raw_text, summary_language=summary_language)
        results.append(
            {
                "index": idx,
                "preview": str(raw_text)[:80] + ("..." if len(str(raw_text)) > 80 else ""),
                "raw_text": str(raw_text),
                **res,
            }
        )
        if progress_hook:
            progress_hook(finished_count, 0)
        _update_batch_progress(progress, status_placeholder, i18n, progress_text, finished_count, total, 0, started)
    return results


def _generate_demo_reply_as_dict(
    *,
    review_text: str,
    merchant_rules: str,
    sentiment: str | None,
    pain_points: list[str] | None,
    style_hint: str | None,
    reply_language: str,
    knowledge_base_text: str,
    kb_top_k: int,
) -> dict[str, Any]:
    kb_source = (knowledge_base_text or "").strip() or (merchant_rules or "").strip()
    retrieved_chunks: list[str] = []
    if kb_source:
        hits = SimpleRAGIndex.from_text(kb_source, chunk_size=300, overlap=60).retrieve(
            review_text,
            top_k=kb_top_k,
        )
        retrieved_chunks = [hit.text for hit in hits]

    points = pain_points or _local_analyze_review(review_text).get("pain_points", [])
    if reply_language == "en":
        reply = (
            "Hi, thanks for sharing this with us. "
            f"We are sorry about {', '.join(points) if points else 'the experience you mentioned'}. "
            "We will check the order details and offer a practical next step based on the store policy."
        )
    else:
        reply = (
            "您好，感谢您把这个情况告诉我们。"
            f"关于{('、'.join(points) if points else '您反馈的问题')}，我们已经记录并会结合订单情况核实。"
            "我们会按店铺规则尽快给出补发、换货或售后处理方案。"
        )
    if style_hint:
        reply = f"{reply}\n\n语气备注：{style_hint}"

    return {
        "reply_text": reply,
        "provider": "demo",
        "model": "local-template",
        "reply_language": reply_language,
        "used_rules": bool(kb_source),
        "retrieved_chunks": retrieved_chunks,
        "request_id": "demo-local",
        "edge_case_flags": [],
        "guardrail_action": "normal",
    }


def _detect_review_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None

    columns = list(df.columns)
    if not columns:
        return None

    # 1) Name-based priority (strict -> fuzzy), explicitly exclude id-like columns
    names = {col: str(col).strip().lower() for col in columns}
    exclude_tokens = ["_id", "id_", " id", "编号", "序号", "index", "索引"]

    exact_priority = [
        "review_text",
        "reviewtext",
        "comment_text",
        "comments_text",
        "评论内容",
        "评价内容",
        "评论文本",
        "内容",
    ]
    for key in exact_priority:
        for col, name in names.items():
            if name == key:
                return col

    fuzzy_priority = ["review_text", "comment_text", "评论", "评价", "内容", "text", "comment", "review"]
    for col, name in names.items():
        if any(tok in name for tok in exclude_tokens):
            continue
        if any(token in name for token in fuzzy_priority):
            return col

    # 2) Content-based fallback
    best_col = None
    best_score = -1.0
    for col in columns:
        s = df[col]
        if not (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        ):
            continue

        name = str(col).strip().lower()
        sample = s.dropna().astype("string").str.strip().head(200)
        if sample.empty:
            continue
        non_empty_ratio = (sample.str.len() > 0).mean()
        avg_len = sample.str.len().mean()
        unique_ratio = sample.nunique() / max(len(sample), 1)
        score = non_empty_ratio * 0.45 + min(avg_len / 60.0, 1.0) * 0.35 + unique_ratio * 0.2
        if any(tok in name for tok in exclude_tokens):
            score -= 0.7
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _find_time_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        s = df[col]
        col_name = str(col).strip().lower()
        name_hint = any(token in col_name for token in ["time", "date", "日期", "时间"])
        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append(col)
            continue
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        ):
            cleaned = s.astype("string").str.strip()
            parsed = pd.to_datetime(cleaned, errors="coerce")
            # 列名命中时间关键词时，降低识别阈值；否则保持较高阈值
            threshold = 0.4 if name_hint else 0.7
            if len(parsed) > 0 and parsed.notna().mean() >= threshold:
                candidates.append(col)
    return candidates


def _render_sentiment_charts(res_df: pd.DataFrame, source_df: pd.DataFrame, i18n: dict[str, str]) -> None:
    if "sentiment" not in res_df.columns:
        return

    chart_df = res_df[res_df["sentiment"].notna()].copy()
    if chart_df.empty:
        return

    sentiment_counts = chart_df["sentiment"].value_counts().reindex(_SENTIMENT_ORDER, fill_value=0).reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    st.divider()
    st.subheader(i18n["chart_title"])

    c1, c2 = st.columns(2)
    with c1:
        bar_chart = alt.Chart(sentiment_counts).mark_bar(size=36).encode(
            x=alt.X("sentiment:N", sort=_SENTIMENT_ORDER, title="Sentiment", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=_SENTIMENT_ORDER,
                    range=_SENTIMENT_COLORS,
                ),
                legend=None,
            ),
            tooltip=["sentiment", "count"],
        ).properties(height=320)
        st.altair_chart(bar_chart, use_container_width=True)

    with c2:
        pie_chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=_SENTIMENT_ORDER,
                    range=_SENTIMENT_COLORS,
                ),
                title="Sentiment",
            ),
            tooltip=["sentiment", "count"],
        ).properties(height=320)
        st.altair_chart(pie_chart, use_container_width=True)

    st.subheader(i18n["trend_title"])
    time_cols = _find_time_columns(source_df)
    if not time_cols:
        st.info(i18n["trend_no_time"])
        return

    selected_time_col = st.selectbox(i18n["trend_pick_col"], options=time_cols, key="trend_time_col")
    mapped = source_df[[selected_time_col]].copy()
    mapped["index"] = mapped.index
    merged = chart_df.merge(mapped, on="index", how="left")
    merged["_ts"] = pd.to_datetime(merged[selected_time_col], errors="coerce")
    merged = merged[merged["_ts"].notna()]

    if merged.empty:
        st.info(i18n["trend_no_data"])
        return

    trend_df = (
        merged.assign(day=merged["_ts"].dt.date)
        .groupby(["day", "sentiment"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x=alt.X("day:T", title="Date"),
        y=alt.Y("count:Q", title="Count"),
        color=alt.Color(
            "sentiment:N",
            scale=alt.Scale(
                domain=_SENTIMENT_ORDER,
                range=_SENTIMENT_COLORS,
            ),
            title="Sentiment",
        ),
        tooltip=["day:T", "sentiment", "count"],
    ).properties(height=360)
    st.altair_chart(line_chart, use_container_width=True)


def _render_pain_word_cloud(pain_df: pd.DataFrame) -> None:
    if pain_df.empty:
        return

    cloud_df = pain_df.copy().reset_index(drop=True)
    max_count = max(int(cloud_df["count"].max()), 1)
    cloud_df["x"] = [12 + ((i * 37) % 76) for i in range(len(cloud_df))]
    cloud_df["y"] = [18 + ((i * 29) % 64) for i in range(len(cloud_df))]
    cloud_df["size"] = cloud_df["count"].map(lambda x: 18 + 34 * (int(x) / max_count))

    chart = alt.Chart(cloud_df).mark_text(
        align="center",
        baseline="middle",
        font="sans-serif",
        fontWeight="bold",
    ).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0, 100])),
        text="pain_point:N",
        size=alt.Size("size:Q", legend=None),
        color=alt.Color(
            "count:Q",
            scale=alt.Scale(range=["#0f766e", "#2563eb", "#dc2626"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("pain_point:N", title="Pain Point"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)


def _render_pain_point_insights(res_df: pd.DataFrame | None, i18n: dict[str, str], lang: str) -> None:
    st.subheader(i18n["insights_title"])

    if not isinstance(res_df, pd.DataFrame) or res_df.empty:
        st.info(i18n["insights_empty"])
        return

    col_a, col_b = st.columns([1, 1])
    with col_a:
        top_k = st.slider(i18n["insights_top_k"], min_value=3, max_value=12, value=6, step=1)
    with col_b:
        include_neutral = st.checkbox(i18n["insights_include_neutral"], value=False)

    records = _results_to_records(res_df)
    top_points = top_pain_points_from_results(records, top_k=int(top_k), include_neutral=include_neutral)
    analyzed_count = len(records)
    negative_count = sum(1 for row in records if row.get("sentiment") == "negative")
    covered_count = sum(1 for row in records if row.get("pain_points"))
    unique_count = len({pp for row in records for pp in row.get("pain_points", [])})

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric(i18n["insights_analyzed"], f"{analyzed_count:,}")
    metric_b.metric(i18n["insights_negative"], f"{negative_count:,}")
    metric_c.metric(i18n["insights_coverage"], f"{covered_count:,}")
    metric_d.metric(i18n["insights_unique"], f"{unique_count:,}")

    if not top_points:
        st.info(i18n["insights_no_pain"])
        return

    pain_df = pd.DataFrame(top_points)
    chart_left, chart_right = st.columns([1, 1])
    with chart_left:
        st.markdown(f"**{i18n['insights_word_cloud']}**")
        _render_pain_word_cloud(pain_df)
    with chart_right:
        st.markdown(f"**{i18n['insights_top_chart']}**")
        bar = alt.Chart(pain_df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("pain_point:N", sort="-x", title=None),
            color=alt.Color("count:Q", scale=alt.Scale(range=["#0f766e", "#2563eb"]), legend=None),
            tooltip=["pain_point", "count"],
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)

    st.markdown(f"**{i18n['insights_defect_list']}**")
    for rank, item in enumerate(top_points, start=1):
        pain_point = str(item["pain_point"])
        related = [
            row
            for row in records
            if pain_point in row.get("pain_points", [])
            and (include_neutral or row.get("sentiment") == "negative")
        ]
        with st.expander(f"{rank}. {pain_point} · {item['count']}", expanded=rank <= 3):
            for sample_idx, row in enumerate(related[:4], start=1):
                text = row.get("raw_text") or row.get("preview") or ""
                st.write(f"{sample_idx}. {_shorten(text)}")
                summary = _record_summary(row, lang)
                if summary:
                    st.caption(f"{i18n['insights_summary']}: {summary}")


def _render_result_exports(res_df: pd.DataFrame, source_df: pd.DataFrame | None, i18n: dict[str, str]) -> None:
    st.subheader(i18n["export_title"])
    records = _records_for_export(
        res_df,
        source_df,
        st.session_state.get(SS_BATCH_TEXT_COL),
    )
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            i18n["export_csv"],
            data=export_records_csv_bytes(records),
            file_name="ai_analysis_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_analysis_csv",
        )
    with col_xlsx:
        try:
            excel_bytes = export_records_excel_bytes(records)
            st.download_button(
                i18n["export_excel"],
                data=excel_bytes,
                file_name="ai_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_analysis_excel",
            )
        except Exception as exc:
            st.warning(i18n["export_excel_unavailable"].format(e=exc))


def _render_report_snapshot(
    res_df: pd.DataFrame | None,
    source_df: pd.DataFrame | None,
    i18n: dict[str, str],
    lang: str,
) -> None:
    st.subheader(i18n["report_title"])
    if not isinstance(res_df, pd.DataFrame) or res_df.empty:
        st.info(i18n["report_empty"])
        return

    config_a, config_b, config_c, config_d = st.columns(4)
    with config_a:
        top_k = st.slider(i18n["report_top_k"], min_value=3, max_value=10, value=5, step=1, key="report_top_k")
    with config_b:
        orders = st.number_input(
            i18n["report_orders"],
            min_value=0,
            value=1000,
            step=100,
            key="report_orders",
        )
    with config_c:
        average_order_value = st.number_input(
            i18n["report_aov"],
            min_value=0.0,
            value=99.0,
            step=10.0,
            key="report_aov",
        )
    with config_d:
        return_loss_rate = st.slider(
            i18n["report_loss_rate"],
            min_value=0.0,
            max_value=0.8,
            value=0.15,
            step=0.01,
            key="report_loss_rate",
        )

    records = _records_for_export(
        res_df,
        source_df,
        st.session_state.get(SS_BATCH_TEXT_COL),
    )
    payload = build_business_insight_payload(
        records,
        source_name=st.session_state.get(SS_NAME, ""),
        top_k=int(top_k),
        estimated_orders_per_month=int(orders),
        average_order_value=float(average_order_value),
        return_loss_rate=float(return_loss_rate),
        language=lang,
    )

    st.markdown(f"**{i18n['report_headline']}**")
    st.info(payload["headline"])

    metrics = payload["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(i18n["insights_analyzed"], f"{int(metrics['analyzed_count']):,}")
    m2.metric(i18n["insights_negative"], f"{metrics['negative_rate']:.1%}")
    m3.metric(i18n["insights_coverage"], f"{metrics['pain_coverage']:.1%}")
    m4.metric(i18n["insights_unique"], f"{int(metrics['unique_pain_points']):,}")

    top_points = payload.get("top_pain_points", [])
    if top_points:
        rows = []
        for item in top_points:
            rec = item.get("recommendation", {})
            rows.append(
                {
                    i18n["report_col_pain"]: item.get("pain_point", ""),
                    i18n["report_col_count"]: item.get("count", 0),
                    i18n["report_col_share"]: f"{item.get('share_of_negative', 0):.1%}",
                    i18n["report_col_orders"]: item.get("estimated_affected_orders", 0),
                    i18n["report_col_loss"]: item.get("estimated_monthly_loss", 0),
                    i18n["report_col_action"]: rec.get("action", ""),
                }
            )
        st.markdown(f"**{i18n['report_recommendations']}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    md = build_report_markdown(payload)
    html_doc = build_report_html(payload)
    svg = build_report_snapshot_svg(payload)
    down_a, down_b, down_c = st.columns(3)
    with down_a:
        st.download_button(
            i18n["report_download_md"],
            data=md.encode("utf-8"),
            file_name="business_insight_snapshot.md",
            mime="text/markdown",
            use_container_width=True,
            key="download_report_md",
        )
    with down_b:
        st.download_button(
            i18n["report_download_html"],
            data=html_doc.encode("utf-8"),
            file_name="business_insight_snapshot.html",
            mime="text/html",
            use_container_width=True,
            key="download_report_html",
        )
    with down_c:
        st.download_button(
            i18n["report_download_svg"],
            data=svg.encode("utf-8"),
            file_name="business_insight_snapshot.svg",
            mime="image/svg+xml",
            use_container_width=True,
            key="download_report_svg",
        )


def _render_log_panel(i18n: dict[str, str]) -> None:
    st.subheader(i18n["log_panel_title"])
    col_a, col_b = st.columns([1, 1])
    with col_a:
        limit = st.slider(i18n["log_limit"], min_value=20, max_value=500, value=100, step=20)
    with col_b:
        status_choice = st.selectbox(i18n["log_status_filter"], options=["all", "ok", "error", "guarded"])

    try:
        events = read_recent_log_events(
            limit=int(limit),
            status=None if status_choice == "all" else status_choice,
        )
    except Exception as exc:
        st.error(i18n["log_error"].format(e=exc))
        return

    if not events:
        st.info(i18n["log_empty"])
        return

    events_df = pd.DataFrame(events)
    counts = events_df["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    st.markdown(f"**{i18n['log_status_counts']}**")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    display_cols = [
        "ts",
        "status",
        "operation",
        "provider",
        "model",
        "latency_ms",
        "attempts",
        "request_id",
        "guardrail_action",
        "error_type",
        "error_message",
    ]
    st.dataframe(
        events_df[[col for col in display_cols if col in events_df.columns]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        i18n["log_download"],
        data=export_records_csv_bytes(events),
        file_name="backend_recent_logs.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_recent_logs",
    )


async def _run_batch_analysis(
    service,
    rows: list[tuple[int, str]],
    summary_language: str,
    progress,
    progress_text,
    status_placeholder,
    i18n: dict[str, str],
    progress_hook=None,
) -> list[dict]:
    total = len(rows)
    finished_count = 0
    failed_count = 0
    started = time.perf_counter()
    _update_batch_progress(progress, status_placeholder, i18n, progress_text, 0, total, 0, started)

    async def one_call(current_idx: int, raw_text: str) -> dict:
        nonlocal finished_count, failed_count
        try:
            res = await service.async_analyze_review_as_dict(
                str(raw_text),
                summary_language=summary_language,
            )
            return {
                "index": current_idx,
                "preview": str(raw_text)[:80] + ("..." if len(str(raw_text)) > 80 else ""),
                "raw_text": str(raw_text),
                **res,
            }
        except Exception as exc:
            failed_count += 1
            return {
                "index": current_idx,
                "preview": str(raw_text)[:80] + ("..." if len(str(raw_text)) > 80 else ""),
                "raw_text": str(raw_text),
                "error": str(exc),
            }
        finally:
            finished_count += 1
            if progress_hook:
                progress_hook(finished_count, failed_count)
            _update_batch_progress(
                progress,
                status_placeholder,
                i18n,
                progress_text,
                finished_count,
                total,
                failed_count,
                started,
            )

    tasks = [one_call(idx, text) for idx, text in rows]
    return await asyncio.gather(*tasks)


# 配置 Streamlit 页面属性
st.set_page_config(
    page_title="评论情感分析 · MVP",
    page_icon="📊",
    layout="wide",
)

# --- 侧边栏配置 ---
with st.sidebar:
    lang_choice = st.radio("Language / 语言", options=["中文", "English"], index=0)
    lang = "zh" if lang_choice == "中文" else "en"

    access_password = _get_configured_access_password()
    admin_password = _get_configured_admin_password()
    auth_title = "访问控制" if lang == "zh" else "Access"
    auth_pw_label = "访问密码" if lang == "zh" else "Access password"
    auth_admin_pw_label = "管理员密码（可选）" if lang == "zh" else "Admin password (optional)"
    auth_merchant_label = "商家名称" if lang == "zh" else "Merchant name"
    auth_slug_label = "商家标识" if lang == "zh" else "Merchant slug"
    auth_user_label = "操作员名称" if lang == "zh" else "Operator name"
    auth_btn = "进入应用" if lang == "zh" else "Enter app"
    auth_ok = "已通过访问校验" if lang == "zh" else "Access granted"
    auth_need = "请输入正确密码后继续。" if lang == "zh" else "Enter the correct password to continue."
    auth_open = "当前未配置访问密码，应用处于开放试用模式。" if lang == "zh" else "No access password configured. App is in open trial mode."
    auth_role_admin = "管理员" if lang == "zh" else "Admin"
    auth_role_operator = "操作员" if lang == "zh" else "Operator"
    auth_role_now = "当前角色：{role}" if lang == "zh" else "Current role: {role}"

    with st.expander(auth_title, expanded=True):
        merchant_name_input = st.text_input(auth_merchant_label, value=st.session_state.get("merchant_name", "Demo Merchant"))
        merchant_slug_input = st.text_input(
            auth_slug_label,
            value=st.session_state.get("merchant_slug", _slugify(merchant_name_input)),
        )
        operator_name_input = st.text_input(auth_user_label, value=st.session_state.get("operator_name", "Demo User"))
        password_input = ""
        admin_password_input = ""
        if access_password:
            password_input = st.text_input(auth_pw_label, type="password", value="", key="access_password_input")
        else:
            st.caption(auth_open)
            st.session_state[SS_ACCESS_GRANTED] = True
            st.session_state[SS_USER_ROLE] = "admin"

        if admin_password:
            admin_password_input = st.text_input(auth_admin_pw_label, type="password", value="", key="admin_password_input")

        if st.button(auth_btn, use_container_width=True, key="auth_submit"):
            if access_password and password_input.strip() != access_password:
                st.session_state[SS_ACCESS_GRANTED] = False
            else:
                old_slug = st.session_state.get("merchant_slug")
                old_user = st.session_state.get("operator_name")
                st.session_state["merchant_name"] = merchant_name_input.strip() or "Demo Merchant"
                st.session_state["merchant_slug"] = _slugify(merchant_slug_input or merchant_name_input)
                st.session_state["operator_name"] = operator_name_input.strip() or "Demo User"
                if old_slug != st.session_state["merchant_slug"] or old_user != st.session_state["operator_name"]:
                    st.session_state.pop(SS_DEMO_CONTEXT, None)
                    st.session_state.pop(SS_CS_DEFAULT_RULES, None)
                st.session_state[SS_ACCESS_GRANTED] = True
                if admin_password and admin_password_input.strip() == admin_password:
                    st.session_state[SS_USER_ROLE] = "admin"
                else:
                    st.session_state[SS_USER_ROLE] = "operator" if access_password or admin_password else "admin"
                st.rerun()

        if st.session_state.get(SS_ACCESS_GRANTED):
            st.success(auth_ok)
            role_label = auth_role_admin if _is_admin() else auth_role_operator
            st.caption(auth_role_now.format(role=role_label))
        elif access_password:
            st.warning(auth_need)

    I18N = {
        "zh": {
            "sidebar_header": "评论分析系统",
            "sidebar_config": "模型配置 (LLM Config)",
            "config_provider": "提供商 (Provider)",
            "config_model": "模型型号 (Model)",
            "config_summary_lang": "摘要语言 (Summary)",
            "config_demo_mode": "演示备用模式",
            "page_title": "多模型评论情感分析系统",
            "page_subtitle": "基于大语言模型（LLM）的评论内容自动化情感分析与汇总。",
            "tab_batch": "批量分析",
            "tab_insights": "痛点洞察",
            "tab_report": "报告快照",
            "tab_single": "单条评论",
            "tab_cs_chat": "模拟客服聊天",
            "tab_history": "历史记录",
            "tab_rules": "规则与知识库",
            "tab_admin": "系统状态",
            "expander_upload": "① 上传文件",
            "file_uploader_label": "选择 CSV 或 Excel",
            "upload_progress_receiving": "正在接收文件...",
            "upload_progress_parsing": "正在解析表格，预计剩余 {eta}",
            "upload_progress_validating": "正在校验列与数据规模...",
            "upload_progress_ready": "文件已就绪：{rows} 行，{cols} 列",
            "upload_meta": "{name} · {size} · 已用 {elapsed} · 预计剩余 {eta}",
            "upload_cached": "已载入：{name} · {rows} 行 · {cols} 列",
            "upload_error": "文件解析失败：{e}",
            "btn_clear": "清空数据",
            "subheader_preview": "数据预览",
            "metric_rows": "总行数",
            "metric_cols": "总列数",
            "subheader_sample": "分析",
            "selectbox_col": "请选择内容列",
            "auto_col_hint": "已自动识别评论列：`{col}`。如不准确可手动调整。",
            "manual_col": "手动选择评论列",
            "slider_n": "分析数量 N",
            "btn_start": "开始分析",
            "progress_preparing": "准备中...",
            "progress_done": "分析完成",
            "subheader_single": "单条交互分析",
            "label_text": "输入内容",
            "btn_single": "立即分析",
            "warn_empty_text": "文本不能为空",
            "spinner_calling": "AI 正在分析中...",
            "error_runtime": "分析失败：{e}",
            "success_done": "分析已完成",
            "btn_clean": "数据清洗",
            "subheader_stats": "清洗结果报告",
            "chart_title": "情感分布可视化",
            "trend_title": "情感时间趋势",
            "trend_pick_col": "选择时间列",
            "trend_no_time": "未检测到可用时间列，无法绘制趋势图。",
            "trend_no_data": "时间列存在，但当前分析结果无法映射出有效时间数据。",
            "failed_title": "失败项明细",
            "failed_retry": "重试失败项",
            "failed_none": "当前批次无失败项。",
            "running_tip": "任务执行中，请勿重复提交。",
            "progress_status": "已完成 {finished}/{total} · 失败 {failed} · 已用 {elapsed} · 预计剩余 {eta} · {rate} 条/秒",
            "batch_wait_upload": "等待上传数据文件。",
            "result_table": "分析结果",
            "insights_title": "深度痛点分析",
            "insights_empty": "暂无批量分析结果。",
            "insights_top_k": "Top K 痛点",
            "insights_include_neutral": "包含中性评论",
            "insights_analyzed": "已分析",
            "insights_negative": "差评数",
            "insights_coverage": "含痛点评论",
            "insights_unique": "唯一痛点",
            "insights_no_pain": "当前结果没有可聚合的痛点。",
            "insights_word_cloud": "高频负面词云",
            "insights_top_chart": "痛点排行",
            "insights_defect_list": "核心缺陷明细",
            "insights_summary": "AI 摘要",
            "export_title": "结果导出",
            "export_csv": "下载 CSV",
            "export_excel": "下载 Excel",
            "export_excel_unavailable": "Excel 导出不可用：{e}",
            "report_title": "商业洞察报告快照",
            "report_empty": "请先完成批量分析，再生成报告快照。",
            "report_top_k": "报告痛点数量",
            "report_orders": "预估月订单数",
            "report_aov": "客单价",
            "report_loss_rate": "退货损失率",
            "report_headline": "核心结论",
            "report_recommendations": "落地建议",
            "report_col_pain": "痛点",
            "report_col_count": "次数",
            "report_col_share": "差评占比",
            "report_col_orders": "预估影响订单",
            "report_col_loss": "预估月损失",
            "report_col_action": "建议动作",
            "report_download_md": "下载 Markdown",
            "report_download_html": "下载 HTML",
            "report_download_svg": "下载长图 SVG",
            "log_panel_title": "后台运行日志",
            "log_limit": "最近日志条数",
            "log_status_filter": "状态筛选",
            "log_status_counts": "状态统计",
            "log_empty": "暂无日志记录。",
            "log_error": "读取日志失败：{e}",
            "log_download": "下载日志 CSV",
            "cs_intro": "输入用户评论/提问，结合商家规则实时生成客服回复。",
            "cs_review": "用户评论 / 提问",
            "cs_rules": "商家规则（可留空）",
            "cs_use_kb": "使用本地知识库",
            "cs_kb_docs": "知识库文档",
            "cs_kb_empty": "未发现本地知识库文档。",
            "cs_retrieved_chunks": "RAG 命中文档片段",
            "cs_style": "语气偏好（可选）",
            "cs_sentiment": "情感提示（可选）",
            "cs_pain_points": "痛点关键词（可选，用逗号分隔）",
            "cs_btn_reply": "生成回复",
            "cs_btn_clear": "清空聊天记录",
            "cs_thinking": "客服 AI 正在生成回复...",
            "cs_error": "生成失败：{e}",
            "cs_warn_empty": "请输入用户评论或提问。",
            "cs_meta": "Provider: {provider} | Model: {model} | Rules used: {used_rules} | Request: {request_id} | Guard: {guardrail}",
            "activity_title": "最近活动",
            "activity_uploads": "上传记录",
            "activity_jobs": "分析任务",
            "activity_replies": "客服回复",
            "activity_empty": "暂无记录",
            "activity_upload_item": "文件：{name} · {rows} 行 · {cols} 列",
            "activity_job_item": "任务：{status} · {name} · {rows} 行",
            "activity_reply_item": "回复：语言 {lang} · Guard {guard}",
            "history_title": "历史/任务中心",
            "history_hint": "这里会保留最近的上传、批量分析任务和客服回复，方便回看试用记录。",
            "history_metric_uploads": "上传数",
            "history_metric_jobs": "任务数",
            "history_metric_replies": "回复数",
            "history_uploads_title": "最近上传",
            "history_jobs_title": "分析任务",
            "history_replies_title": "客服回复记录",
            "history_empty": "当前还没有历史记录。",
            "history_col_time": "时间",
            "history_col_file": "文件名",
            "history_col_rows": "行数",
            "history_col_cols": "列数",
            "history_col_size": "大小",
            "history_col_status": "状态",
            "history_col_progress": "进度",
            "history_col_provider": "Provider",
            "history_col_model": "模型",
            "history_col_lang": "语言",
            "history_col_guard": "Guard",
            "history_col_review": "用户内容",
            "history_col_reply": "客服回复",
            "history_job_picker": "选择一个任务查看详情",
            "history_detail_status": "任务状态",
            "history_detail_rows": "分析行数",
            "history_detail_provider": "Provider",
            "history_detail_lang": "摘要语言",
            "history_detail_progress": "任务进度",
            "history_job_meta": "文件：{filename} | 文本列：{text_column} | 模型：{model} | 创建：{created_at} | 完成：{completed_at}",
            "history_job_error": "任务失败原因：{error}",
            "history_job_no_results": "这个任务目前没有保存到结果明细。",
            "history_failure_title": "失败概览",
            "history_failure_summary": "失败行数：{failed} · 错误类型：{unique}",
            "history_failure_error": "失败原因",
            "history_failure_count": "次数",
            "history_failure_empty": "当前任务没有失败明细。",
            "history_timeline_title": "任务时间线",
            "history_timeline_empty": "当前任务还没有可展示的事件记录。",
            "history_event_progress": "任务进度更新：已处理 {processed} 行，失败 {failed} 行",
            "history_event_rerun_created": "由任务 {parent} 创建重跑，范围：{scope}",
            "history_event_generic": "任务事件：{event_type}",
            "history_rerun_all": "重跑整批",
            "history_rerun_failed": "仅重跑失败项",
            "history_rerun_progress": "历史任务重跑中",
            "history_rerun_success": "历史任务已重新执行，结果已刷新到当前会话。",
            "history_rerun_error": "重跑失败：{error}",
            "history_rerun_empty": "该任务没有可重跑的原始文本。",
            "history_result_row": "行号",
            "history_result_preview": "评论预览",
            "history_result_sentiment": "情感",
            "history_result_confidence": "置信度",
            "history_result_pain_points": "痛点",
            "history_result_summary": "摘要",
            "history_result_error": "错误",
            "history_filter_all": "全部",
            "history_filter_status": "按状态筛选",
            "history_filter_provider": "按 Provider 筛选",
            "history_filter_keyword": "按文件名搜索",
            "history_filter_reply_lang": "按回复语言筛选",
            "history_filter_guard": "按 Guard 筛选",
            "history_filter_reply_keyword": "按内容搜索",
            "history_filter_empty": "当前筛选条件下没有记录。",
            "history_download_uploads": "导出上传记录 CSV",
            "history_download_jobs": "导出任务列表 CSV",
            "history_download_results": "导出当前任务结果 CSV",
            "history_download_replies": "导出客服回复 CSV",
            "history_export_admin_only": "导出功能仅对管理员开放。",
            "rules_title": "规则与知识库",
            "rules_hint": "把默认商家规则和常用知识库文档保存在这里，客服回复页可以直接复用。",
            "rules_admin_only": "当前为操作员视角，可查看但不能修改规则与知识库。",
            "rules_default_label": "默认商家规则",
            "rules_save_btn": "保存规则",
            "rules_reload_btn": "重新加载",
            "rules_save_success": "商家规则已保存。",
            "rules_kb_title": "知识库文档",
            "rules_kb_empty": "还没有保存的知识库文档。",
            "rules_kb_col_title": "标题",
            "rules_kb_col_chars": "字数",
            "rules_kb_picker": "选择一个文档进行编辑",
            "rules_kb_new_option": "新建文档",
            "rules_kb_title_input": "文档标题",
            "rules_kb_content_input": "文档内容",
            "rules_kb_save_btn": "保存文档",
            "rules_kb_delete_btn": "删除文档",
            "rules_kb_warn_empty": "标题和内容都不能为空。",
            "rules_kb_save_success": "知识库文档已保存。",
            "rules_kb_delete_success": "知识库文档已删除。",
            "admin_title": "系统状态 / 日志",
            "admin_only": "仅管理员可查看此页面。",
            "admin_metric_logs": "日志条数",
            "admin_metric_ok": "成功调用",
            "admin_metric_error": "失败调用",
            "admin_metric_latency": "平均耗时",
            "admin_config_title": "当前运行配置",
            "admin_config_key": "配置项",
            "admin_config_value": "当前值",
            "admin_error_types_title": "错误类型统计",
            "admin_guard_title": "Guard 动作统计",
            "admin_error_type_col": "错误类型",
            "admin_guard_col": "Guard 动作",
            "admin_count_col": "次数",
            "admin_logs_title": "最近调用日志",
            "admin_log_time": "时间",
            "admin_log_operation": "操作",
            "admin_log_status": "状态",
            "admin_log_provider": "Provider",
            "admin_log_model": "模型",
            "admin_log_latency": "耗时(ms)",
            "admin_log_error": "错误类型",
            "admin_log_request": "Request ID",
            "admin_log_guard": "Guard",
            "admin_download_logs": "导出最近日志 CSV",
            "admin_no_logs": "当前还没有可展示的日志。",
            "admin_empty": "当前没有相关统计数据。",
            "batch_job_status": "当前任务：{job_id} · 状态 {status} · 进度 {processed}/{total} · 失败 {failed}",
            "lang_zh": "中文",
            "lang_en": "English",
        },
        "en": {
            "sidebar_header": "Sentiment Analysis",
            "sidebar_config": "LLM Config",
            "config_provider": "Provider",
            "config_model": "Model",
            "config_summary_lang": "Summary Language",
            "config_demo_mode": "Demo fallback mode",
            "page_title": "Multi-LLM Sentiment Analysis",
            "page_subtitle": "Automated review analysis dashboard powered by Large Language Models.",
            "tab_batch": "Batch Analysis",
            "tab_insights": "Pain Insights",
            "tab_report": "Report Snapshot",
            "tab_single": "Single Review",
            "tab_cs_chat": "Simulated CS Chat",
            "tab_history": "History Center",
            "tab_rules": "Rules & KB",
            "tab_admin": "System Status",
            "expander_upload": "① Upload file",
            "file_uploader_label": "Select CSV or Excel",
            "upload_progress_receiving": "Receiving file...",
            "upload_progress_parsing": "Parsing table, ETA {eta}",
            "upload_progress_validating": "Validating columns and size...",
            "upload_progress_ready": "File ready: {rows} rows, {cols} cols",
            "upload_meta": "{name} · {size} · elapsed {elapsed} · ETA {eta}",
            "upload_cached": "Loaded: {name} · {rows} rows · {cols} cols",
            "upload_error": "File parsing failed: {e}",
            "btn_clear": "Clear",
            "subheader_preview": "Preview",
            "metric_rows": "Rows",
            "metric_cols": "Cols",
            "subheader_sample": "Analysis",
            "selectbox_col": "Select Column",
            "auto_col_hint": "Auto-detected review column: `{col}`. You can switch manually if needed.",
            "manual_col": "Pick column manually",
            "slider_n": "Analysis Count N",
            "btn_start": "Run Analysis",
            "progress_preparing": "Preparing...",
            "progress_done": "Done",
            "subheader_single": "Single Interaction",
            "label_text": "Enter Text",
            "btn_single": "Analyze",
            "warn_empty_text": "Text cannot be empty",
            "spinner_calling": "AI analyzing...",
            "error_runtime": "Analysis failed: {e}",
            "success_done": "Completed",
            "btn_clean": "Clean Data",
            "subheader_stats": "Cleaning Stats",
            "chart_title": "Sentiment Visuals",
            "trend_title": "Sentiment Trend Over Time",
            "trend_pick_col": "Pick a time column",
            "trend_no_time": "No usable time column detected, trend chart unavailable.",
            "trend_no_data": "Time column exists but no valid mapped dates in current results.",
            "failed_title": "Failed Items",
            "failed_retry": "Retry Failed Items",
            "failed_none": "No failed items in this batch.",
            "running_tip": "A batch is running. Please avoid duplicate submissions.",
            "progress_status": "Completed {finished}/{total} · Failed {failed} · Elapsed {elapsed} · ETA {eta} · {rate} rows/s",
            "batch_wait_upload": "Waiting for an uploaded data file.",
            "result_table": "Analysis Results",
            "insights_title": "Deep Pain Point Analysis",
            "insights_empty": "No batch analysis results yet.",
            "insights_top_k": "Top K Pain Points",
            "insights_include_neutral": "Include neutral reviews",
            "insights_analyzed": "Analyzed",
            "insights_negative": "Negative",
            "insights_coverage": "With Pain Points",
            "insights_unique": "Unique Points",
            "insights_no_pain": "No aggregate pain points in current results.",
            "insights_word_cloud": "High-Frequency Negative Word Cloud",
            "insights_top_chart": "Pain Point Ranking",
            "insights_defect_list": "Core Defect Details",
            "insights_summary": "AI Summary",
            "export_title": "Result Export",
            "export_csv": "Download CSV",
            "export_excel": "Download Excel",
            "export_excel_unavailable": "Excel export unavailable: {e}",
            "report_title": "Business Insight Report Snapshot",
            "report_empty": "Run batch analysis first to generate a report snapshot.",
            "report_top_k": "Report pain points",
            "report_orders": "Estimated monthly orders",
            "report_aov": "Average order value",
            "report_loss_rate": "Return loss rate",
            "report_headline": "Headline",
            "report_recommendations": "Actionable Recommendations",
            "report_col_pain": "Pain Point",
            "report_col_count": "Count",
            "report_col_share": "Negative Share",
            "report_col_orders": "Affected Orders",
            "report_col_loss": "Monthly Loss",
            "report_col_action": "Action",
            "report_download_md": "Download Markdown",
            "report_download_html": "Download HTML",
            "report_download_svg": "Download Long SVG",
            "log_panel_title": "Backend Runtime Logs",
            "log_limit": "Recent log rows",
            "log_status_filter": "Status filter",
            "log_status_counts": "Status Counts",
            "log_empty": "No log records yet.",
            "log_error": "Failed to read logs: {e}",
            "log_download": "Download Log CSV",
            "cs_intro": "Enter a customer review/question and generate an AI customer-service reply with merchant rules context.",
            "cs_review": "Customer review / question",
            "cs_rules": "Merchant rules (optional)",
            "cs_use_kb": "Use local knowledge base",
            "cs_kb_docs": "Knowledge base docs",
            "cs_kb_empty": "No local knowledge-base docs found.",
            "cs_retrieved_chunks": "RAG retrieved chunks",
            "cs_style": "Style hint (optional)",
            "cs_sentiment": "Sentiment hint (optional)",
            "cs_pain_points": "Pain points (optional, comma-separated)",
            "cs_btn_reply": "Generate reply",
            "cs_btn_clear": "Clear chat history",
            "cs_thinking": "Generating customer-service reply...",
            "cs_error": "Generation failed: {e}",
            "cs_warn_empty": "Please enter a review or question.",
            "cs_meta": "Provider: {provider} | Model: {model} | Rules used: {used_rules} | Request: {request_id} | Guard: {guardrail}",
            "activity_title": "Recent Activity",
            "activity_uploads": "Uploads",
            "activity_jobs": "Analysis jobs",
            "activity_replies": "CS replies",
            "activity_empty": "No records yet.",
            "activity_upload_item": "File: {name} · {rows} rows · {cols} cols",
            "activity_job_item": "Job: {status} · {name} · {rows} rows",
            "activity_reply_item": "Reply: lang {lang} · guard {guard}",
            "history_title": "History / Task Center",
            "history_hint": "Recent uploads, batch-analysis jobs, and CS replies are stored here for trial replay and review.",
            "history_metric_uploads": "Uploads",
            "history_metric_jobs": "Jobs",
            "history_metric_replies": "Replies",
            "history_uploads_title": "Recent Uploads",
            "history_jobs_title": "Analysis Jobs",
            "history_replies_title": "Customer-Service Replies",
            "history_empty": "No history records yet.",
            "history_col_time": "Time",
            "history_col_file": "File",
            "history_col_rows": "Rows",
            "history_col_cols": "Cols",
            "history_col_size": "Size",
            "history_col_status": "Status",
            "history_col_progress": "Progress",
            "history_col_provider": "Provider",
            "history_col_model": "Model",
            "history_col_lang": "Language",
            "history_col_archived": "Archived",
            "history_col_guard": "Guard",
            "history_col_review": "Customer input",
            "history_col_reply": "Reply",
            "history_job_picker": "Select a job to inspect",
            "history_detail_status": "Job status",
            "history_detail_rows": "Rows analyzed",
            "history_detail_provider": "Provider",
            "history_detail_lang": "Summary language",
            "history_detail_progress": "Progress",
            "history_job_meta": "File: {filename} | Text column: {text_column} | Model: {model} | Created: {created_at} | Completed: {completed_at}",
            "history_job_error": "Job failed because: {error}",
            "history_job_no_results": "No saved row-level results for this job yet.",
            "history_failure_title": "Failure summary",
            "history_failure_summary": "Failed rows: {failed} · Unique errors: {unique}",
            "history_failure_error": "Failure reason",
            "history_failure_count": "Count",
            "history_failure_empty": "No failed rows in this job.",
            "history_timeline_title": "Job timeline",
            "history_timeline_empty": "No timeline events recorded for this job yet.",
            "history_event_progress": "Progress updated: {processed} rows processed, {failed} failed",
            "history_event_rerun_created": "Rerun created from job {parent}, scope: {scope}",
            "history_event_generic": "Job event: {event_type}",
            "history_rerun_all": "Rerun full job",
            "history_rerun_failed": "Rerun failed only",
            "history_rerun_progress": "Re-running historical job",
            "history_rerun_success": "Historical job rerun completed and current-session results were refreshed.",
            "history_rerun_error": "Rerun failed: {error}",
            "history_rerun_empty": "This job has no reusable raw text.",
            "history_cancel_job": "Cancel job",
            "history_cancel_success": "Job cancelled.",
            "history_archive_job": "Archive job",
            "history_unarchive_job": "Unarchive job",
            "history_archive_success": "Job archive state updated.",
            "history_result_row": "Row",
            "history_result_preview": "Preview",
            "history_result_sentiment": "Sentiment",
            "history_result_confidence": "Confidence",
            "history_result_pain_points": "Pain points",
            "history_result_summary": "Summary",
            "history_result_error": "Error",
            "history_filter_all": "All",
            "history_filter_status": "Filter by status",
            "history_filter_provider": "Filter by provider",
            "history_filter_keyword": "Search by filename",
            "history_filter_archived": "Include archived",
            "history_filter_reply_lang": "Filter by reply language",
            "history_filter_guard": "Filter by guard",
            "history_filter_reply_keyword": "Search by content",
            "history_filter_empty": "No records match the current filters.",
            "history_yes": "Yes",
            "history_no": "No",
            "history_download_uploads": "Export uploads CSV",
            "history_download_jobs": "Export jobs CSV",
            "history_download_results": "Export selected results CSV",
            "history_download_replies": "Export replies CSV",
            "history_export_admin_only": "Exports are available to admins only.",
            "rules_title": "Rules & Knowledge Base",
            "rules_hint": "Save reusable merchant rules and knowledge-base notes here so the CS workflow can reuse them directly.",
            "rules_admin_only": "You are in operator view. Rules and KB are read-only.",
            "rules_default_label": "Default merchant rules",
            "rules_save_btn": "Save rules",
            "rules_reload_btn": "Reload saved rules",
            "rules_save_success": "Merchant rules saved.",
            "rules_kb_title": "Knowledge-base documents",
            "rules_kb_empty": "No saved knowledge-base docs yet.",
            "rules_kb_col_title": "Title",
            "rules_kb_col_chars": "Chars",
            "rules_kb_picker": "Select a document to edit",
            "rules_kb_new_option": "Create new document",
            "rules_kb_title_input": "Document title",
            "rules_kb_content_input": "Document content",
            "rules_kb_save_btn": "Save document",
            "rules_kb_delete_btn": "Delete document",
            "rules_kb_warn_empty": "Title and content cannot be empty.",
            "rules_kb_save_success": "Knowledge-base document saved.",
            "rules_kb_delete_success": "Knowledge-base document deleted.",
            "admin_title": "System Status / Logs",
            "admin_only": "This page is available to admins only.",
            "admin_metric_logs": "Log rows",
            "admin_metric_ok": "Successful calls",
            "admin_metric_error": "Failed calls",
            "admin_metric_latency": "Avg latency",
            "admin_config_title": "Runtime Config",
            "admin_config_key": "Key",
            "admin_config_value": "Value",
            "admin_error_types_title": "Error Type Stats",
            "admin_guard_title": "Guard Action Stats",
            "admin_error_type_col": "Error type",
            "admin_guard_col": "Guard action",
            "admin_count_col": "Count",
            "admin_logs_title": "Recent Call Logs",
            "admin_log_time": "Time",
            "admin_log_operation": "Operation",
            "admin_log_status": "Status",
            "admin_log_provider": "Provider",
            "admin_log_model": "Model",
            "admin_log_latency": "Latency(ms)",
            "admin_log_error": "Error type",
            "admin_log_request": "Request ID",
            "admin_log_guard": "Guard",
            "admin_download_logs": "Export recent logs CSV",
            "admin_no_logs": "No logs available yet.",
            "admin_empty": "No stats available.",
            "batch_job_status": "Current job: {job_id} · status {status} · progress {processed}/{total} · failed {failed}",
            "lang_zh": "Chinese",
            "lang_en": "English",
        },
    }
    d = I18N[lang]

    st.header(d["sidebar_header"])
    st.divider()

    with st.expander(d["sidebar_config"], expanded=True):
        p_choice = st.selectbox(d["config_provider"], ["Gemini", "DeepSeek"], index=1)
        default_models = {"Gemini": "gemini-2.5-flash", "DeepSeek": "deepseek-chat"}
        selected_model = st.text_input(d["config_model"], value=default_models.get(p_choice, ""))
        summary_lang_label = st.selectbox(
            d["config_summary_lang"],
            options=[d["lang_zh"], d["lang_en"]],
            index=0,
        )
        demo_mode = st.checkbox(d["config_demo_mode"], value=False)
        summary_lang = "zh" if summary_lang_label == d["lang_zh"] else "en"
        st.session_state["llm_provider"] = p_choice.lower()
        st.session_state["llm_model"] = selected_model
        st.session_state["summary_language"] = summary_lang
        st.session_state["demo_mode"] = demo_mode

    if access_password and not st.session_state.get(SS_ACCESS_GRANTED):
        st.stop()

    _render_recent_activity(d)

st.title(d["page_title"])
st.markdown(d["page_subtitle"])

_inject_demo_css()

tab_batch, tab_insights, tab_report, tab_single, tab_cs, tab_history, tab_rules, tab_admin = st.tabs(
    [
        d["tab_batch"],
        d["tab_insights"],
        d["tab_report"],
        d["tab_single"],
        d["tab_cs_chat"],
        d["tab_history"],
        d["tab_rules"],
        d["tab_admin"],
    ]
)

with st.expander(d["expander_upload"], expanded=True):
    uploaded = st.file_uploader(d["file_uploader_label"], type=["csv", "xlsx", "xls"])
    if uploaded:
        upload_sig = _uploaded_signature(uploaded)
        if st.session_state.get(SS_UPLOAD_SIGNATURE) != upload_sig:
            try:
                for k in [SS_DF, SS_NAME, SS_BATCH_RESULTS, SS_BATCH_FAILED, SS_BATCH_TEXT_COL]:
                    st.session_state.pop(k, None)
                df_new, upload_info = _load_dataframe_with_status(uploaded, d)
                st.session_state[SS_DF] = df_new
                st.session_state[SS_NAME] = uploaded.name
                st.session_state[SS_UPLOAD_SIGNATURE] = upload_sig
                st.session_state[SS_UPLOAD_INFO] = upload_info
                _persist_upload(upload_info, upload_sig)
            except Exception as exc:
                st.error(d["upload_error"].format(e=exc))
        else:
            upload_info = st.session_state.get(SS_UPLOAD_INFO, {})
            if upload_info:
                st.caption(
                    d["upload_cached"].format(
                        name=upload_info.get("name", uploaded.name),
                        rows=f"{int(upload_info.get('rows', 0)):,}",
                        cols=int(upload_info.get("cols", 0)),
                    )
                )
    if st.session_state.get(SS_DF) is not None:
        if st.button(d["btn_clear"], key="btn_clear_data"):
            for k in [
                SS_DF,
                SS_NAME,
                SS_BATCH_RESULTS,
                SS_BATCH_FAILED,
                SS_BATCH_TEXT_COL,
                SS_UPLOAD_SIGNATURE,
                SS_UPLOAD_INFO,
                SS_LAST_UPLOAD_ID,
            ]:
                st.session_state.pop(k, None)
            st.rerun()

df = st.session_state.get(SS_DF)

with tab_batch:
    if st.session_state.get(SS_BATCH_RUNNING):
        st.info(d["running_tip"])

    if df is None:
        st.info(d["batch_wait_upload"])
    else:
        st.subheader(d["subheader_preview"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(d["metric_rows"], f"{len(df):,}")
        with c2:
            st.metric(d["metric_cols"], len(df.columns))
        with c3:
            st.metric("Provider", st.session_state.llm_provider.upper())

        st.dataframe(df.head(100), use_container_width=True)

        st.subheader(d["subheader_sample"])
        auto_col = _detect_review_column(df)
        all_cols = list(df.columns)
        default_idx = all_cols.index(auto_col) if auto_col in all_cols else 0
        col_text = all_cols[default_idx]
        st.caption(d["auto_col_hint"].format(col=col_text))
        if st.checkbox(d["manual_col"], value=False, key="manual_col_toggle"):
            col_text = st.selectbox(d["selectbox_col"], all_cols, index=default_idx, key="select_col_text")

        with st.expander(d["btn_clean"]):
            c1, c2 = st.columns(2)
            with c1:
                do_dup_text = st.checkbox("去重 (Text)", value=True)
                do_empty = st.checkbox("删空 (Empty)", value=True)
            with c2:
                do_dup_row = st.checkbox("整行去重 (Row)", value=False)
                min_len = st.number_input("最小长度 (Min Len)", value=5, min_value=1)

            if st.button(d["btn_clean"], use_container_width=True, key="btn_clean_data"):
                cleaned_df, stats = clean_review_dataframe(
                    df,
                    col_text,
                    drop_duplicate_text=do_dup_text,
                    drop_empty_text=do_empty,
                    drop_full_row_duplicates=do_dup_row,
                    min_length=min_len,
                )
                st.session_state[SS_DF] = cleaned_df
                st.info(
                    f"""
                **{d['subheader_stats']}**
                - {d['metric_rows']} IN: {stats['rows_in']}
                - {d['metric_rows']} OUT: {stats['rows_out']}
                - 空文字过滤: {stats['empty_dropped']}
                - 长度过滤 (<{min_len}): {stats['too_short_dropped']}
                - 文本去重: {stats['dup_text_dropped']}
                """
                )
                st.rerun()

        n_rows = st.number_input(
            d["slider_n"],
            min_value=1,
            max_value=max(1, len(df)),
            value=min(max(1, len(df)), 5),
            step=1,
            key="number_batch_rows",
        )

        run_disabled = bool(st.session_state.get(SS_BATCH_RUNNING, False))
        current_job_id = st.session_state.get(SS_BATCH_JOB_ID)
        if current_job_id:
            current_job = _get_repo().get_analysis_job(current_job_id)
            if current_job:
                st.caption(
                    d["batch_job_status"].format(
                        job_id=current_job.get("id", "-"),
                        status=current_job.get("status", "-"),
                        processed=int(current_job.get("processed_count", 0) or 0),
                        total=int(current_job.get("row_count", 0) or 0),
                        failed=int(current_job.get("failed_count", 0) or 0),
                    )
                )
        if st.button(d["btn_start"], type="primary", disabled=run_disabled, key="btn_run_batch"):
            sample = df.head(int(n_rows))
            rows = [(int(idx), str(raw)) for idx, raw in sample[col_text].items()]
            job = _create_batch_job(
                filename=st.session_state.get(SS_NAME, "uploaded_file"),
                text_column=col_text,
                row_count=len(rows),
                provider=st.session_state.get("llm_provider", "unknown"),
                model=st.session_state.get("llm_model", ""),
                summary_language=st.session_state.get("summary_language", "zh"),
            )

            try:
                results = _execute_batch_task(
                    task=job,
                    rows=rows,
                    i18n=d,
                    progress_text="Demo analyzing" if st.session_state.get("demo_mode", False) else "Analyzing",
                )
                res_df = pd.DataFrame(results)
                st.session_state[SS_BATCH_RESULTS] = res_df
                st.session_state[SS_BATCH_TEXT_COL] = col_text
                if "error" in res_df.columns:
                    st.session_state[SS_BATCH_FAILED] = res_df[res_df["error"].notna()].copy()
                else:
                    st.session_state[SS_BATCH_FAILED] = pd.DataFrame()
                st.success(d["progress_done"])
            except Exception as exc:
                st.error(f"Failed to run batch: {exc}")

        res_df = st.session_state.get(SS_BATCH_RESULTS)
        if isinstance(res_df, pd.DataFrame) and not res_df.empty:
            st.subheader(d["result_table"])
            st.dataframe(res_df.drop(columns=["raw_text"], errors="ignore"), use_container_width=True)
            _render_result_exports(res_df, df, d)

            failed_df = st.session_state.get(SS_BATCH_FAILED)
            st.subheader(d["failed_title"])
            if isinstance(failed_df, pd.DataFrame) and not failed_df.empty:
                st.dataframe(
                    failed_df[["index", "preview", "error"]],
                    use_container_width=True,
                )

                if st.button(d["failed_retry"], key="btn_retry_failed"):
                    retry_rows = [(int(r["index"]), str(r["raw_text"])) for _, r in failed_df.iterrows()]
                    retry_job = _create_batch_job(
                        filename=st.session_state.get(SS_NAME, "uploaded_file"),
                        text_column=st.session_state.get(SS_BATCH_TEXT_COL, col_text),
                        row_count=len(retry_rows),
                        provider=st.session_state.get("llm_provider", "unknown"),
                        model=st.session_state.get("llm_model", ""),
                        summary_language=st.session_state.get("summary_language", "zh"),
                    )
                    try:
                        retried = _execute_batch_task(
                            task=retry_job,
                            rows=retry_rows,
                            i18n=d,
                            progress_text="Retrying",
                        )
                        retry_df = pd.DataFrame(retried)

                        merged = pd.concat(
                            [res_df[~res_df["index"].isin(retry_df["index"])], retry_df],
                            ignore_index=True,
                        ).sort_values("index")

                        st.session_state[SS_BATCH_RESULTS] = merged
                        if "error" in merged.columns:
                            st.session_state[SS_BATCH_FAILED] = merged[merged["error"].notna()].copy()
                        else:
                            st.session_state[SS_BATCH_FAILED] = pd.DataFrame()
                        st.success("Retry completed")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Retry failed: {exc}")
            else:
                st.info(d["failed_none"])

            _render_sentiment_charts(res_df, df, d)

with tab_insights:
    _render_pain_point_insights(st.session_state.get(SS_BATCH_RESULTS), d, lang)

with tab_report:
    _render_report_snapshot(st.session_state.get(SS_BATCH_RESULTS), df, d, lang)

with tab_single:
    st.subheader(d["subheader_single"])
    text_input = st.text_area(d["label_text"], height=150)
    if st.button(d["btn_single"], type="primary", key="btn_single_analyze"):
        if not text_input.strip():
            st.warning(d["warn_empty_text"])
        else:
            with st.spinner(d["spinner_calling"]):
                try:
                    single_lang = (
                        "en"
                        if _should_reply_in_english(
                            st.session_state.get("summary_language", "zh"),
                            text_input,
                        )
                        else "zh"
                    )
                    if st.session_state.get("demo_mode", False):
                        res = _local_analyze_review(text_input.strip(), summary_language=single_lang)
                    else:
                        service = get_llm_service(
                            provider=st.session_state.llm_provider,
                            model=st.session_state.llm_model,
                        )
                        res = service.analyze_review_as_dict(
                            text_input.strip(),
                            summary_language=single_lang,
                        )
                    st.success(d["success_done"])
                    st.json(res)
                except Exception as exc:
                    st.error(d["error_runtime"].format(e=exc))

with tab_history:
    _render_history_center(d)

with tab_rules:
    _render_rules_center(d)

with tab_admin:
    _render_admin_center(d)

with tab_cs:
    st.subheader(d["tab_cs_chat"])
    st.caption(d["cs_intro"])

    if SS_CS_CHAT_HISTORY not in st.session_state:
        st.session_state[SS_CS_CHAT_HISTORY] = []

    with st.expander("Context / ???", expanded=True):
        review_text = st.text_area(d["cs_review"], height=120, key="cs_review_text")
        merchant_rules = st.text_area(
            d["cs_rules"],
            height=120,
            key="cs_rules_text",
            value=st.session_state.get(SS_CS_DEFAULT_RULES, _get_default_rules()),
        )
        saved_kb_text = _read_saved_kb_docs_text()
        kb_files = _available_kb_files(lang)
        has_any_kb = bool(kb_files or saved_kb_text.strip())
        use_kb = st.checkbox(d["cs_use_kb"], value=has_any_kb, disabled=not has_any_kb)
        local_kb_text = ""
        if kb_files:
            default_kb = [p.name for p in kb_files[:2]]
            selected_kb_names = st.multiselect(
                d["cs_kb_docs"],
                options=[p.name for p in kb_files],
                default=default_kb,
                disabled=not use_kb,
                key="cs_kb_docs",
            )
            selected_kb_paths = [p for p in kb_files if p.name in selected_kb_names]
            local_kb_text = _read_kb_files(selected_kb_paths) if use_kb else ""
        elif not saved_kb_text.strip():
            st.caption(d["cs_kb_empty"])
        knowledge_base_text = "\n\n".join(x for x in [saved_kb_text, local_kb_text] if x.strip()) if use_kb else ""
        col_a, col_b = st.columns(2)
        with col_a:
            style_hint = st.text_input(d["cs_style"], key="cs_style_hint")
        with col_b:
            sentiment_hint = st.selectbox(
                d["cs_sentiment"],
                options=["", "positive", "neutral", "negative"],
                index=0,
                key="cs_sentiment_hint",
            )
        pain_points_raw = st.text_input(d["cs_pain_points"], key="cs_pain_points")

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        do_reply = st.button(d["cs_btn_reply"], type="primary", use_container_width=True)
    with btn_col2:
        do_clear = st.button(d["cs_btn_clear"], use_container_width=True)

    if do_clear:
        st.session_state[SS_CS_CHAT_HISTORY] = []
        st.rerun()

    if do_reply:
        if not review_text.strip():
            st.warning(d["cs_warn_empty"])
        else:
            pain_points = [x.strip() for x in pain_points_raw.split(",") if x.strip()] if pain_points_raw else None
            with st.spinner(d["cs_thinking"]):
                try:
                    reply_language = (
                        "en"
                        if _should_reply_in_english(
                            st.session_state.get("summary_language", "zh"),
                            review_text,
                        )
                        else "zh"
                    )
                    if st.session_state.get("demo_mode", False):
                        res = _generate_demo_reply_as_dict(
                            review_text=review_text.strip(),
                            merchant_rules=merchant_rules.strip(),
                            sentiment=sentiment_hint or None,
                            pain_points=pain_points,
                            style_hint=style_hint.strip() or None,
                            reply_language=reply_language,
                            knowledge_base_text=knowledge_base_text,
                            kb_top_k=3,
                        )
                    else:
                        res = generate_customer_service_reply_as_dict(
                            review_text=review_text.strip(),
                            merchant_rules=merchant_rules.strip(),
                            provider=st.session_state.get("llm_provider", "deepseek"),
                            model=st.session_state.get("llm_model") or None,
                            sentiment=sentiment_hint or None,
                            pain_points=pain_points,
                            style_hint=style_hint.strip() or None,
                            reply_language=reply_language,
                            knowledge_base_text=knowledge_base_text,
                            kb_top_k=3,
                    )
                    used_rules = bool(res.get("used_rules") or knowledge_base_text.strip())
                    _persist_customer_service_reply(
                        review_text=review_text.strip(),
                        merchant_rules=merchant_rules.strip(),
                        knowledge_base_used=used_rules,
                        result=res,
                    )
                    st.session_state[SS_CS_CHAT_HISTORY].append(
                        {
                            "review_text": review_text.strip(),
                            "reply_text": res.get("reply_text", ""),
                            "meta": d["cs_meta"].format(
                                provider=res.get("provider", "-"),
                                model=res.get("model", "-"),
                                used_rules=used_rules,
                                request_id=res.get("request_id", "-"),
                                guardrail=res.get("guardrail_action", "normal"),
                            ),
                            "retrieved_chunks": res.get("retrieved_chunks", []),
                            "edge_case_flags": res.get("edge_case_flags", []),
                        }
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(d["cs_error"].format(e=exc))

    for item in st.session_state.get(SS_CS_CHAT_HISTORY, []):
        with st.chat_message("user"):
            st.write(item["review_text"])
        with st.chat_message("assistant"):
            st.write(item["reply_text"])
            st.caption(item["meta"])
            if item.get("edge_case_flags"):
                st.caption("Edge flags: " + ", ".join(item["edge_case_flags"]))
            chunks = item.get("retrieved_chunks") or []
            if chunks:
                with st.expander(d["cs_retrieved_chunks"], expanded=False):
                    for idx, chunk in enumerate(chunks, start=1):
                        st.write(f"{idx}. {_shorten(chunk, 320)}")

with tab_admin:
    _render_log_panel(d)
