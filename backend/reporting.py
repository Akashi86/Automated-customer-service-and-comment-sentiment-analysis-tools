from __future__ import annotations

import csv
import html
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from config import get_llm_log_path
except ModuleNotFoundError:  # pragma: no cover
    from .config import get_llm_log_path


SENTIMENTS = ("positive", "neutral", "negative")


def coerce_pain_points(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and str(value) == "nan":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            return coerce_pain_points(parsed)
        except json.JSONDecodeError:
            pass

    for sep in ("，", ",", "；", ";", "|", "\n"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def normalize_analysis_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in records:
        item = dict(row)
        sentiment = str(item.get("sentiment", "") or "").strip().lower()
        item["sentiment"] = sentiment if sentiment in SENTIMENTS else ""
        item["pain_points"] = coerce_pain_points(item.get("pain_points"))
        normalized.append(item)
    return normalized


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percent(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _recommend_for_pain_point(pain_point: str, language: str) -> dict[str, str]:
    text = pain_point.lower()
    zh = language == "zh"

    rules: list[tuple[tuple[str, ...], dict[str, str], dict[str, str]]] = [
        (
            ("包装", "破损", "压坏", "漏", "packag", "damaged"),
            {
                "category": "包装与仓储防护",
                "action": "优先复查出库包装标准，针对易损商品增加缓冲材料和封箱抽检，并记录破损订单批次。",
                "owner": "供应链 / 仓储",
                "priority": "高",
            },
            {
                "category": "Packaging and warehouse protection",
                "action": "Review outbound packaging standards, add cushioning for fragile SKUs, and track damaged-order batches.",
                "owner": "Supply chain / warehouse",
                "priority": "High",
            },
        ),
        (
            ("物流", "快递", "配送", "送货", "shipping", "delivery", "slow"),
            {
                "category": "物流履约",
                "action": "按地区统计超时订单，替换高延迟线路或增加备选承运商，并在客服回复中主动同步物流节点。",
                "owner": "运营 / 物流",
                "priority": "高",
            },
            {
                "category": "Fulfillment logistics",
                "action": "Track delayed orders by region, replace slow routes or carriers, and proactively share shipment milestones.",
                "owner": "Operations / logistics",
                "priority": "High",
            },
        ),
        (
            ("质量", "做工", "瑕疵", "断", "裂", "坏", "quality", "defect", "broken"),
            {
                "category": "产品质量",
                "action": "把高频缺陷反馈给供应商，增加到货抽检比例，并对问题批次做售后补偿和复盘。",
                "owner": "产品 / 供应商管理",
                "priority": "高",
            },
            {
                "category": "Product quality",
                "action": "Escalate frequent defects to suppliers, increase inbound QC sampling, and review affected batches.",
                "owner": "Product / vendor management",
                "priority": "High",
            },
        ),
        (
            ("尺寸", "尺码", "偏大", "偏小", "size", "fit"),
            {
                "category": "尺码与商品信息",
                "action": "重做尺码表和商品详情页提示，增加真实测量图，降低因预期不一致带来的退换货。",
                "owner": "商品运营",
                "priority": "中",
            },
            {
                "category": "Sizing and product information",
                "action": "Improve size charts and PDP guidance with real measurements to reduce expectation gaps.",
                "owner": "Merchandising",
                "priority": "Medium",
            },
        ),
        (
            ("客服", "售后", "回复", "没人", "service", "support", "reply"),
            {
                "category": "客服响应",
                "action": "建立高风险评论的人工兜底队列，设置首响 SLA，并把常见问题沉淀进知识库。",
                "owner": "客服主管",
                "priority": "中",
            },
            {
                "category": "Customer-service response",
                "action": "Create a human handoff queue for high-risk reviews, set first-response SLAs, and update the knowledge base.",
                "owner": "CS lead",
                "priority": "Medium",
            },
        ),
    ]

    for keywords, zh_rec, en_rec in rules:
        if any(keyword in text or keyword in pain_point for keyword in keywords):
            return zh_rec if zh else en_rec

    if zh:
        return {
            "category": "综合体验",
            "action": "抽样回看相关原始评论，确认是否能合并为更具体的问题标签，再分派给对应负责人处理。",
            "owner": "项目经理 / 运营",
            "priority": "中",
        }
    return {
        "category": "Overall experience",
        "action": "Sample the related raw reviews, refine the issue label if possible, and assign it to the right owner.",
        "owner": "PM / operations",
        "priority": "Medium",
    }


def build_business_insight_payload(
    records: list[dict[str, Any]],
    *,
    source_name: str = "",
    top_k: int = 5,
    include_neutral: bool = False,
    estimated_orders_per_month: int = 1000,
    average_order_value: float = 99.0,
    return_loss_rate: float = 0.15,
    language: str = "zh",
) -> dict[str, Any]:
    normalized = normalize_analysis_records(records)
    valid = [row for row in normalized if row.get("sentiment") in SENTIMENTS and not row.get("error")]
    sentiment_counts = Counter(str(row.get("sentiment", "")) for row in valid)
    analyzed_count = len(valid)
    negative_count = sentiment_counts.get("negative", 0)

    pain_counter: Counter[str] = Counter()
    pain_record_count = 0
    for row in valid:
        sentiment = str(row.get("sentiment", ""))
        if sentiment != "negative" and not (include_neutral and sentiment == "neutral"):
            continue
        pain_points = row.get("pain_points") or []
        if pain_points:
            pain_record_count += 1
        for pain_point in pain_points:
            pain_counter[str(pain_point).strip()] += 1

    confidence_values = [
        _as_float(row.get("confidence"), default=-1.0)
        for row in valid
        if _as_float(row.get("confidence"), default=-1.0) >= 0
    ]
    avg_confidence = round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0

    top_points: list[dict[str, Any]] = []
    for pain_point, count in pain_counter.most_common(max(1, top_k)):
        share_total = _percent(count, analyzed_count)
        share_negative = _percent(count, max(negative_count, 1))
        affected_orders = int(round(max(0, estimated_orders_per_month) * share_total))
        estimated_loss = round(affected_orders * max(0.0, average_order_value) * max(0.0, return_loss_rate), 2)
        top_points.append(
            {
                "pain_point": pain_point,
                "count": count,
                "share_of_total": share_total,
                "share_of_negative": share_negative,
                "estimated_affected_orders": affected_orders,
                "estimated_monthly_loss": estimated_loss,
                "recommendation": _recommend_for_pain_point(pain_point, language),
            }
        )

    negative_rate = _percent(negative_count, analyzed_count)
    pain_coverage = _percent(pain_record_count, analyzed_count)
    if language == "zh":
        if top_points:
            headline = (
                f"当前批次共分析 {analyzed_count} 条评论，差评率 {negative_rate:.1%}。"
                f"首要痛点为「{top_points[0]['pain_point']}」，出现 {top_points[0]['count']} 次，"
                f"约占全部已分析评论的 {top_points[0]['share_of_total']:.1%}。"
            )
        else:
            headline = f"当前批次共分析 {analyzed_count} 条评论，暂未发现可聚合的高频负面痛点。"
    else:
        if top_points:
            headline = (
                f"This batch analyzed {analyzed_count} reviews with a {negative_rate:.1%} negative rate. "
                f"The top pain point is '{top_points[0]['pain_point']}', appearing {top_points[0]['count']} times "
                f"({top_points[0]['share_of_total']:.1%} of analyzed reviews)."
            )
        else:
            headline = f"This batch analyzed {analyzed_count} reviews, with no aggregate negative pain point found."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "language": "zh" if language == "zh" else "en",
        "assumptions": {
            "estimated_orders_per_month": int(max(0, estimated_orders_per_month)),
            "average_order_value": round(max(0.0, average_order_value), 2),
            "return_loss_rate": round(max(0.0, return_loss_rate), 4),
        },
        "metrics": {
            "records_received": len(normalized),
            "analyzed_count": analyzed_count,
            "failed_count": len([row for row in normalized if row.get("error")]),
            "sentiment_counts": {sentiment: sentiment_counts.get(sentiment, 0) for sentiment in SENTIMENTS},
            "negative_rate": negative_rate,
            "pain_coverage": pain_coverage,
            "unique_pain_points": len(pain_counter),
            "average_confidence": avg_confidence,
        },
        "headline": headline,
        "top_pain_points": top_points,
    }


def _flatten_for_export(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in normalize_analysis_records(records):
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple, set)):
                item[key] = "; ".join(str(part) for part in value)
            elif isinstance(value, dict):
                item[key] = json.dumps(value, ensure_ascii=False)
            else:
                item[key] = value
        flattened.append(item)
    return flattened


def export_records_csv_bytes(records: list[dict[str, Any]]) -> bytes:
    flattened = _flatten_for_export(records)
    preferred = [
        "index",
        "preview",
        "raw_text",
        "sentiment",
        "confidence",
        "pain_points",
        "summary_zh",
        "summary_en",
        "error",
    ]
    extra_keys = sorted({key for row in flattened for key in row.keys()} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in flattened)] + extra_keys

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames or ["empty"])
    writer.writeheader()
    for row in flattened:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buf.getvalue().encode("utf-8-sig")


def export_records_excel_bytes(records: list[dict[str, Any]]) -> bytes:
    import pandas as pd

    flattened = _flatten_for_export(records)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(flattened).to_excel(writer, index=False, sheet_name="analysis_results")
    return buf.getvalue()


def build_report_markdown(payload: dict[str, Any]) -> str:
    zh = payload.get("language") == "zh"
    metrics = payload.get("metrics", {})
    title = "商业洞察分析报告快照" if zh else "Business Insight Report Snapshot"
    lines = [
        f"# {title}",
        "",
        str(payload.get("headline", "")),
        "",
        "## 核心指标" if zh else "## Core Metrics",
        f"- {'分析评论数' if zh else 'Analyzed reviews'}: {metrics.get('analyzed_count', 0)}",
        f"- {'差评率' if zh else 'Negative rate'}: {metrics.get('negative_rate', 0):.1%}",
        f"- {'痛点覆盖率' if zh else 'Pain-point coverage'}: {metrics.get('pain_coverage', 0):.1%}",
        f"- {'唯一痛点数' if zh else 'Unique pain points'}: {metrics.get('unique_pain_points', 0)}",
        "",
        "## 高频痛点与建议" if zh else "## Top Pain Points and Actions",
    ]
    for item in payload.get("top_pain_points", []):
        rec = item.get("recommendation", {})
        lines.extend(
            [
                f"### {item.get('pain_point', '')}",
                f"- {'出现次数' if zh else 'Count'}: {item.get('count', 0)}",
                f"- {'差评占比' if zh else 'Share of negative'}: {item.get('share_of_negative', 0):.1%}",
                f"- {'预估月影响订单' if zh else 'Estimated affected monthly orders'}: {item.get('estimated_affected_orders', 0)}",
                f"- {'预估月损失' if zh else 'Estimated monthly loss'}: {item.get('estimated_monthly_loss', 0)}",
                f"- {'建议' if zh else 'Action'}: {rec.get('action', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_report_html(payload: dict[str, Any]) -> str:
    markdown = html.escape(build_report_markdown(payload))
    title = "商业洞察分析报告快照" if payload.get("language") == "zh" else "Business Insight Report Snapshot"
    return f"""<!doctype html>
<html lang="{html.escape(payload.get('language', 'zh'))}">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 40px; color: #111827; line-height: 1.65; }}
    pre {{ white-space: pre-wrap; font-family: inherit; }}
  </style>
</head>
<body>
  <pre>{markdown}</pre>
</body>
</html>
"""


def _wrap_svg_text(text: str, max_chars: int = 36) -> list[str]:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return [clean]
    lines: list[str] = []
    current = ""
    for char in clean:
        current += char
        if len(current) >= max_chars:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines[:4]


def build_report_snapshot_svg(payload: dict[str, Any]) -> str:
    zh = payload.get("language") == "zh"
    metrics = payload.get("metrics", {})
    top_points = payload.get("top_pain_points", [])
    width = 1080
    height = 700 + max(1, len(top_points)) * 118
    max_pain_count = max([int(item.get("count", 0)) for item in top_points] + [1])
    sentiment_counts = metrics.get("sentiment_counts", {})
    max_sentiment = max([int(sentiment_counts.get(s, 0)) for s in SENTIMENTS] + [1])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<rect x="44" y="36" width="992" height="112" rx="8" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<text x="72" y="82" font-size="32" font-weight="700" fill="#111827">{"商业洞察分析报告快照" if zh else "Business Insight Report Snapshot"}</text>',
    ]
    y = 116
    for line in _wrap_svg_text(str(payload.get("headline", "")), max_chars=62):
        parts.append(f'<text x="72" y="{y}" font-size="18" fill="#475569">{html.escape(line)}</text>')
        y += 26

    metric_items = [
        ("已分析" if zh else "Analyzed", metrics.get("analyzed_count", 0)),
        ("差评率" if zh else "Negative Rate", f"{metrics.get('negative_rate', 0):.1%}"),
        ("痛点覆盖" if zh else "Pain Coverage", f"{metrics.get('pain_coverage', 0):.1%}"),
        ("唯一痛点" if zh else "Unique Points", metrics.get("unique_pain_points", 0)),
    ]
    for idx, (label, value) in enumerate(metric_items):
        x = 44 + idx * 254
        parts.append(f'<rect x="{x}" y="174" width="232" height="96" rx="8" fill="#ffffff" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x + 24}" y="214" font-size="18" fill="#64748b">{html.escape(str(label))}</text>')
        parts.append(f'<text x="{x + 24}" y="248" font-size="28" font-weight="700" fill="#0f172a">{html.escape(str(value))}</text>')

    parts.append(f'<text x="52" y="326" font-size="24" font-weight="700" fill="#111827">{"情感分布" if zh else "Sentiment Distribution"}</text>')
    color_map = {"positive": "#10b981", "neutral": "#f59e0b", "negative": "#ef4444"}
    labels = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    bar_x = 230
    for idx, sentiment in enumerate(SENTIMENTS):
        count = int(sentiment_counts.get(sentiment, 0))
        bar_y = 352 + idx * 42
        bar_w = int(620 * count / max_sentiment)
        parts.append(f'<text x="72" y="{bar_y + 24}" font-size="18" fill="#334155">{labels[sentiment]}</text>')
        parts.append(f'<rect x="{bar_x}" y="{bar_y}" width="620" height="24" rx="4" fill="#e2e8f0"/>')
        parts.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="24" rx="4" fill="{color_map[sentiment]}"/>')
        parts.append(f'<text x="{bar_x + 640}" y="{bar_y + 20}" font-size="18" fill="#334155">{count}</text>')

    start_y = 520
    parts.append(f'<text x="52" y="{start_y}" font-size="24" font-weight="700" fill="#111827">{"高频痛点与行动建议" if zh else "Top Pain Points and Actions"}</text>')
    y = start_y + 34
    if not top_points:
        parts.append(f'<text x="72" y="{y + 28}" font-size="18" fill="#64748b">{"暂无可聚合痛点" if zh else "No aggregate pain points."}</text>')
    for rank, item in enumerate(top_points, start=1):
        rec = item.get("recommendation", {})
        parts.append(f'<rect x="44" y="{y}" width="992" height="96" rx="8" fill="#ffffff" stroke="#e5e7eb"/>')
        parts.append(f'<text x="72" y="{y + 34}" font-size="20" font-weight="700" fill="#0f172a">{rank}. {html.escape(str(item.get("pain_point", "")))}</text>')
        bar_w = int(340 * int(item.get("count", 0)) / max_pain_count)
        parts.append(f'<rect x="440" y="{y + 20}" width="340" height="18" rx="4" fill="#e2e8f0"/>')
        parts.append(f'<rect x="440" y="{y + 20}" width="{bar_w}" height="18" rx="4" fill="#2563eb"/>')
        parts.append(f'<text x="798" y="{y + 36}" font-size="16" fill="#334155">{item.get("count", 0)} · {item.get("share_of_negative", 0):.1%}</text>')
        detail = rec.get("action", "")
        for line_idx, line in enumerate(_wrap_svg_text(detail, max_chars=58)):
            parts.append(f'<text x="72" y="{y + 64 + line_idx * 20}" font-size="16" fill="#475569">{html.escape(line)}</text>')
        y += 118

    parts.append("</svg>")
    return "\n".join(parts)


def read_recent_log_events(
    *,
    limit: int = 200,
    status: str | None = None,
    log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = Path(log_path) if log_path else get_llm_log_path()
    if not path.is_file():
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[dict[str, Any]] = []
    for line in reversed(lines):
        if len(events) >= max(1, limit):
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if status and str(payload.get("status", "")).lower() != status.lower():
            continue
        events.append(payload)
    return list(reversed(events))
