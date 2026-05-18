from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_app_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    clean = "".join(ch for ch in prefix.lower() if ch.isalnum() or ch in {"_", "-"}).strip("_-") or "id"
    return f"{clean}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class DemoContext:
    merchant_id: str
    user_id: str
    merchant_name: str | None = None
    merchant_slug: str | None = None
    user_name: str | None = None


class AppRepository:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else get_app_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        if self.db_path.exists() and self.db_path.stat().st_size > 0:
            return
        self._write(
            {
                "merchants": [],
                "users": [],
                "merchant_memberships": [],
                "merchant_settings": [],
                "uploaded_files": [],
                "analysis_jobs": [],
                "analysis_results": [],
                "analysis_job_events": [],
                "customer_service_replies": [],
                "knowledge_base_docs": [],
            }
        )

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            self.init_db()
        return json.loads(self.db_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        self.db_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_context(
        self,
        *,
        merchant_slug: str,
        merchant_name: str,
        user_name: str,
        user_email: str | None = None,
    ) -> DemoContext:
        payload = self._read()

        merchants = payload["merchants"]
        users = payload["users"]
        memberships = payload["merchant_memberships"]

        normalized_slug = (
            "".join(ch.lower() if ch.isalnum() else "-" for ch in str(merchant_slug or "").strip()).strip("-")
            or "demo-merchant"
        )
        normalized_merchant_name = str(merchant_name or "").strip() or "Demo Merchant"
        normalized_user_name = str(user_name or "").strip() or "Demo User"
        normalized_email = str(user_email or "").strip() or None

        merchant = next((m for m in merchants if m["slug"] == normalized_slug), None)
        if merchant is None:
            merchant = {
                "id": _new_id("merchant"),
                "name": normalized_merchant_name,
                "slug": normalized_slug,
                "created_at": _utc_now(),
            }
            merchants.append(merchant)
        else:
            merchant["name"] = normalized_merchant_name

        user = next(
            (
                u
                for u in users
                if u["display_name"] == normalized_user_name
                and (normalized_email is None or u.get("email") == normalized_email)
            ),
            None,
        )
        if user is None:
            user = {
                "id": _new_id("user"),
                "display_name": normalized_user_name,
                "email": normalized_email,
                "created_at": _utc_now(),
            }
            users.append(user)
        else:
            user["display_name"] = normalized_user_name
            user["email"] = normalized_email

        member = next(
            (
                row
                for row in memberships
                if row["merchant_id"] == merchant["id"] and row["user_id"] == user["id"]
            ),
            None,
        )
        if member is None:
            memberships.append(
                {
                    "merchant_id": merchant["id"],
                    "user_id": user["id"],
                    "role": "owner",
                    "created_at": _utc_now(),
                }
            )

        self._write(payload)
        return DemoContext(
            merchant_id=str(merchant["id"]),
            user_id=str(user["id"]),
            merchant_name=str(merchant["name"]),
            merchant_slug=str(merchant["slug"]),
            user_name=str(user["display_name"]),
        )

    def ensure_demo_context(self) -> DemoContext:
        return self.ensure_context(
            merchant_slug="demo-merchant",
            merchant_name="Demo Merchant",
            user_name="Demo User",
        )

    def get_merchant_settings(self, merchant_id: str) -> dict[str, Any]:
        payload = self._read()
        settings = next(
            (row for row in payload["merchant_settings"] if row.get("merchant_id") == merchant_id),
            None,
        )
        if settings is None:
            settings = {
                "merchant_id": merchant_id,
                "default_rules": "",
                "updated_at": _utc_now(),
            }
            payload["merchant_settings"].append(settings)
            self._write(payload)
        return settings

    def save_merchant_rules(self, merchant_id: str, rules_text: str) -> None:
        payload = self._read()
        settings = next(
            (row for row in payload["merchant_settings"] if row.get("merchant_id") == merchant_id),
            None,
        )
        if settings is None:
            settings = {"merchant_id": merchant_id}
            payload["merchant_settings"].append(settings)
        settings["default_rules"] = str(rules_text or "")
        settings["updated_at"] = _utc_now()
        self._write(payload)

    def list_knowledge_base_docs(self, merchant_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._read()
        rows = [row for row in payload["knowledge_base_docs"] if row.get("merchant_id") == merchant_id]
        rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
        return rows[: max(1, int(limit))]

    def upsert_knowledge_base_doc(
        self,
        *,
        merchant_id: str,
        user_id: str,
        title: str,
        content: str,
        doc_id: str | None = None,
    ) -> str:
        payload = self._read()
        now = _utc_now()
        docs = payload["knowledge_base_docs"]
        target = None
        if doc_id:
            target = next((row for row in docs if row.get("id") == doc_id and row.get("merchant_id") == merchant_id), None)

        if target is None:
            target = {
                "id": _new_id("kbdoc"),
                "merchant_id": merchant_id,
                "user_id": user_id,
                "created_at": now,
            }
            docs.append(target)

        target["title"] = str(title or "").strip() or "Untitled KB Doc"
        target["content"] = str(content or "")
        target["updated_at"] = now
        self._write(payload)
        return str(target["id"])

    def delete_knowledge_base_doc(self, merchant_id: str, doc_id: str) -> bool:
        payload = self._read()
        docs = payload["knowledge_base_docs"]
        before = len(docs)
        payload["knowledge_base_docs"] = [
            row for row in docs if not (row.get("merchant_id") == merchant_id and row.get("id") == doc_id)
        ]
        changed = len(payload["knowledge_base_docs"]) != before
        if changed:
            self._write(payload)
        return changed

    def record_upload(
        self,
        *,
        merchant_id: str,
        user_id: str,
        filename: str,
        upload_signature: str,
        file_size: int,
        row_count: int,
        col_count: int,
        source: str = "streamlit",
    ) -> str:
        payload = self._read()
        upload_id = _new_id("upload")
        payload["uploaded_files"].append(
            {
                "id": upload_id,
                "merchant_id": merchant_id,
                "user_id": user_id,
                "filename": filename,
                "upload_signature": upload_signature,
                "file_size": int(file_size),
                "row_count": int(row_count),
                "col_count": int(col_count),
                "source": source,
                "created_at": _utc_now(),
            }
        )
        self._write(payload)
        return upload_id

    def create_analysis_job(
        self,
        *,
        merchant_id: str,
        user_id: str,
        uploaded_file_id: str | None,
        filename: str,
        text_column: str,
        provider: str,
        model: str,
        summary_language: str,
        row_count: int,
    ) -> str:
        payload = self._read()
        job_id = _new_id("job")
        payload["analysis_jobs"].append(
            {
                "id": job_id,
                "merchant_id": merchant_id,
                "user_id": user_id,
                "uploaded_file_id": uploaded_file_id,
                "filename": filename,
                "text_column": text_column,
                "provider": provider,
                "model": model,
                "summary_language": summary_language,
                "row_count": int(row_count),
                "status": "queued",
                "processed_count": 0,
                "failed_count": 0,
                "archived": False,
                "error_message": None,
                "created_at": _utc_now(),
                "started_at": None,
                "completed_at": None,
            }
        )
        self._append_event_in_payload(
            payload,
            job_id,
            event_type="created",
            message=f"Job created for {filename}",
            meta={
                "text_column": text_column,
                "provider": provider,
                "model": model,
                "summary_language": summary_language,
                "row_count": int(row_count),
            },
        )
        self._write(payload)
        return job_id

    def start_analysis_job(self, job_id: str) -> None:
        payload = self._read()
        for job in payload["analysis_jobs"]:
            if job["id"] == job_id:
                job["status"] = "running"
                job["started_at"] = _utc_now()
                job["error_message"] = None
                break
        self._append_event_in_payload(payload, job_id, event_type="running", message="Job started")
        self._write(payload)

    def update_analysis_job_progress(self, job_id: str, *, processed_count: int, failed_count: int = 0) -> None:
        payload = self._read()
        should_log_progress = False
        for job in payload["analysis_jobs"]:
            if job["id"] == job_id:
                previous_processed = int(job.get("processed_count", 0) or 0)
                previous_failed = int(job.get("failed_count", 0) or 0)
                job["processed_count"] = max(0, int(processed_count))
                job["failed_count"] = max(0, int(failed_count))
                if job.get("status") == "queued":
                    job["status"] = "running"
                    job["started_at"] = job.get("started_at") or _utc_now()
                total = max(1, int(job.get("row_count", 0) or 0))
                current_processed = int(job["processed_count"])
                should_log_progress = (
                    current_processed == total
                    or current_processed in {1, total // 2}
                    or current_processed - previous_processed >= 10
                    or int(job["failed_count"]) != previous_failed
                )
                break
        if should_log_progress:
            self._append_event_in_payload(
                payload,
                job_id,
                event_type="progress",
                message=f"Processed {processed_count} rows",
                meta={"processed_count": int(processed_count), "failed_count": int(failed_count)},
            )
        self._write(payload)

    def complete_analysis_job(self, job_id: str, results: list[dict[str, Any]]) -> None:
        payload = self._read()
        payload["analysis_results"] = [row for row in payload["analysis_results"] if row.get("job_id") != job_id]
        for row in results:
            payload["analysis_results"].append(
                {
                    "id": _new_id("result"),
                    "job_id": job_id,
                    "row_index": int(row.get("index", 0)),
                    "preview": str(row.get("preview", "") or ""),
                    "raw_text": str(row.get("raw_text", "") or ""),
                    "sentiment": str(row.get("sentiment", "") or ""),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "pain_points": list(row.get("pain_points", []) or []),
                    "summary_text": str(row.get("summary_zh", "") or row.get("summary", "") or ""),
                    "error_message": str(row.get("error", "") or ""),
                    "created_at": _utc_now(),
                }
            )

        for job in payload["analysis_jobs"]:
            if job["id"] == job_id:
                job["status"] = "completed"
                job["processed_count"] = int(job.get("row_count", 0) or 0)
                job["failed_count"] = sum(1 for row in results if str(row.get("error", "") or "").strip())
                job["started_at"] = job.get("started_at") or _utc_now()
                job["completed_at"] = _utc_now()
                job["error_message"] = None
                break
        failed_count = sum(1 for row in results if str(row.get("error", "") or "").strip())
        self._append_event_in_payload(
            payload,
            job_id,
            event_type="completed",
            message=f"Job completed with {failed_count} failed rows",
            meta={"processed_count": len(results), "failed_count": failed_count},
        )
        self._write(payload)

    def fail_analysis_job(self, job_id: str, error_message: str) -> None:
        payload = self._read()
        for job in payload["analysis_jobs"]:
            if job["id"] == job_id:
                job["status"] = "failed"
                job["started_at"] = job.get("started_at") or _utc_now()
                job["completed_at"] = _utc_now()
                job["error_message"] = error_message[:500]
                break
        self._append_event_in_payload(
            payload,
            job_id,
            event_type="failed",
            message=error_message[:500],
        )
        self._write(payload)

    def cancel_analysis_job(self, job_id: str, reason: str = "cancelled by user") -> None:
        payload = self._read()
        for job in payload["analysis_jobs"]:
            if job["id"] == job_id and job.get("status") in {"queued", "running"}:
                job["status"] = "cancelled"
                job["completed_at"] = _utc_now()
                job["error_message"] = reason[:500]
                break
        self._append_event_in_payload(
            payload,
            job_id,
            event_type="cancelled",
            message=reason[:500],
        )
        self._write(payload)

    def set_analysis_job_archived(self, job_id: str, archived: bool = True) -> None:
        payload = self._read()
        for job in payload["analysis_jobs"]:
            if job["id"] == job_id:
                job["archived"] = bool(archived)
                break
        self._append_event_in_payload(
            payload,
            job_id,
            event_type="archived" if archived else "unarchived",
            message="Job archived" if archived else "Job restored from archive",
        )
        self._write(payload)

    def append_analysis_job_event(
        self,
        job_id: str,
        *,
        event_type: str,
        message: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        payload = self._read()
        self._append_event_in_payload(payload, job_id, event_type=event_type, message=message, meta=meta)
        self._write(payload)

    def list_analysis_job_events(self, job_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        payload = self._read()
        rows = [row for row in payload.get("analysis_job_events", []) if row.get("job_id") == job_id]
        rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        return rows[: max(1, int(limit))]

    def record_customer_service_reply(
        self,
        *,
        merchant_id: str,
        user_id: str,
        review_text: str,
        merchant_rules: str,
        knowledge_base_used: bool,
        result: dict[str, Any],
    ) -> str:
        payload = self._read()
        reply_id = _new_id("reply")
        payload["customer_service_replies"].append(
            {
                "id": reply_id,
                "merchant_id": merchant_id,
                "user_id": user_id,
                "review_text": review_text,
                "merchant_rules": merchant_rules,
                "knowledge_base_used": bool(knowledge_base_used),
                "reply_text": str(result.get("reply_text", "") or ""),
                "provider": str(result.get("provider", "") or ""),
                "model": str(result.get("model", "") or ""),
                "reply_language": str(result.get("reply_language", "") or ""),
                "guardrail_action": str(result.get("guardrail_action", "") or ""),
                "edge_case_flags": list(result.get("edge_case_flags", []) or []),
                "request_id": str(result.get("request_id", "") or ""),
                "created_at": _utc_now(),
            }
        )
        self._write(payload)
        return reply_id

    def _recent_items(
        self,
        table: str,
        merchant_id: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        payload = self._read()
        rows = [row for row in payload[table] if row.get("merchant_id") == merchant_id]
        rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        return rows[: max(1, int(limit))]

    def list_recent_uploads(self, merchant_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self._recent_items("uploaded_files", merchant_id, limit=limit)

    def list_recent_jobs(self, merchant_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self._recent_items("analysis_jobs", merchant_id, limit=limit)

    def list_recent_replies(self, merchant_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self._recent_items("customer_service_replies", merchant_id, limit=limit)

    def get_analysis_job(self, job_id: str) -> dict[str, Any] | None:
        payload = self._read()
        return next((row for row in payload["analysis_jobs"] if row.get("id") == job_id), None)

    def list_analysis_results(self, job_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        payload = self._read()
        rows = [row for row in payload["analysis_results"] if row.get("job_id") == job_id]
        rows.sort(key=lambda row: int(row.get("row_index", 0)))
        return rows[: max(1, int(limit))]

    def list_customer_service_replies(
        self,
        merchant_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self._recent_items("customer_service_replies", merchant_id, limit=limit)

    def _append_event_in_payload(
        self,
        payload: dict[str, list[dict[str, Any]]],
        job_id: str,
        *,
        event_type: str,
        message: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        payload.setdefault("analysis_job_events", [])
        payload["analysis_job_events"].append(
            {
                "id": _new_id("jobevent"),
                "job_id": job_id,
                "event_type": str(event_type or "").strip() or "info",
                "message": str(message or "").strip(),
                "meta": dict(meta or {}),
                "created_at": _utc_now(),
            }
        )
