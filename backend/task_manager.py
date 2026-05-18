from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app_repository import AppRepository


@dataclass(frozen=True)
class BatchTaskContext:
    job_id: str
    merchant_id: str
    user_id: str
    filename: str
    text_column: str
    provider: str
    model: str
    summary_language: str
    row_count: int


class BatchAnalysisTaskManager:
    def __init__(self, repo: AppRepository | None = None) -> None:
        self.repo = repo or AppRepository()

    def create_job(
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
        parent_job_id: str | None = None,
        rerun_scope: str | None = None,
    ) -> BatchTaskContext:
        job_id = self.repo.create_analysis_job(
            merchant_id=merchant_id,
            user_id=user_id,
            uploaded_file_id=uploaded_file_id,
            filename=filename,
            text_column=text_column,
            provider=provider,
            model=model,
            summary_language=summary_language,
            row_count=row_count,
        )
        if parent_job_id:
            scope = str(rerun_scope or "all")
            self.repo.append_analysis_job_event(
                job_id,
                event_type="rerun_created",
                message=f"Rerun created from job {parent_job_id}",
                meta={"parent_job_id": parent_job_id, "rerun_scope": scope},
            )
        return BatchTaskContext(
            job_id=job_id,
            merchant_id=merchant_id,
            user_id=user_id,
            filename=filename,
            text_column=text_column,
            provider=provider,
            model=model,
            summary_language=summary_language,
            row_count=row_count,
        )

    def mark_running(self, job_id: str) -> None:
        self.repo.start_analysis_job(job_id)

    def mark_progress(self, job_id: str, *, processed_count: int, failed_count: int = 0) -> None:
        self.repo.update_analysis_job_progress(
            job_id,
            processed_count=processed_count,
            failed_count=failed_count,
        )

    def mark_completed(self, job_id: str, results: list[dict[str, Any]]) -> None:
        self.repo.complete_analysis_job(job_id, results)

    def mark_failed(self, job_id: str, error_message: str) -> None:
        self.repo.fail_analysis_job(job_id, error_message)

    def mark_cancelled(self, job_id: str, reason: str = "cancelled by user") -> None:
        self.repo.cancel_analysis_job(job_id, reason)

    def set_archived(self, job_id: str, archived: bool = True) -> None:
        self.repo.set_analysis_job_archived(job_id, archived)
