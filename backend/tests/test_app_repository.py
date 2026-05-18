from __future__ import annotations

from app_repository import AppRepository


def test_repository_creates_demo_context_and_persists_records(tmp_path) -> None:
    repo = AppRepository(tmp_path / "app_state.json")
    ctx = repo.ensure_context(
        merchant_slug="merchant-a",
        merchant_name="Merchant A",
        user_name="Operator A",
    )

    upload_id = repo.record_upload(
        merchant_id=ctx.merchant_id,
        user_id=ctx.user_id,
        filename="reviews.csv",
        upload_signature="reviews.csv:1024",
        file_size=1024,
        row_count=20,
        col_count=4,
    )
    assert upload_id

    job_id = repo.create_analysis_job(
        merchant_id=ctx.merchant_id,
        user_id=ctx.user_id,
        uploaded_file_id=upload_id,
        filename="reviews.csv",
        text_column="review_text",
        provider="deepseek",
        model="deepseek-chat",
        summary_language="zh",
        row_count=2,
    )
    queued_job = repo.get_analysis_job(job_id)
    assert queued_job and queued_job["status"] == "queued"

    repo.start_analysis_job(job_id)
    repo.update_analysis_job_progress(job_id, processed_count=1, failed_count=0)

    repo.complete_analysis_job(
        job_id,
        [
            {
                "index": 0,
                "preview": "包装破损",
                "raw_text": "包装破损，客服回复慢",
                "sentiment": "negative",
                "confidence": 0.91,
                "pain_points": ["包装破损", "客服响应慢"],
                "summary_zh": "用户反映包装破损且客服回复慢。",
            }
        ],
    )

    reply_id = repo.record_customer_service_reply(
        merchant_id=ctx.merchant_id,
        user_id=ctx.user_id,
        review_text="包装破损",
        merchant_rules="先致歉后补发",
        knowledge_base_used=True,
        result={
            "reply_text": "非常抱歉给您带来不便，我们会尽快为您处理补发。",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "reply_language": "zh",
            "guardrail_action": "normal",
            "edge_case_flags": [],
            "request_id": "reply-123",
        },
    )
    assert reply_id

    uploads = repo.list_recent_uploads(ctx.merchant_id, limit=5)
    jobs = repo.list_recent_jobs(ctx.merchant_id, limit=5)
    replies = repo.list_recent_replies(ctx.merchant_id, limit=5)
    job = repo.get_analysis_job(job_id)
    results = repo.list_analysis_results(job_id, limit=10)
    events = repo.list_analysis_job_events(job_id, limit=20)
    reply_history = repo.list_customer_service_replies(ctx.merchant_id, limit=10)
    repo.save_merchant_rules(ctx.merchant_id, "先致歉，再处理售后")
    kb_doc_id = repo.upsert_knowledge_base_doc(
        merchant_id=ctx.merchant_id,
        user_id=ctx.user_id,
        title="售后规则",
        content="签收后 7 天内支持退换。",
    )
    settings = repo.get_merchant_settings(ctx.merchant_id)
    kb_docs = repo.list_knowledge_base_docs(ctx.merchant_id, limit=10)

    assert uploads and uploads[0]["filename"] == "reviews.csv"
    assert jobs and jobs[0]["status"] == "completed"
    assert replies and replies[0]["reply_language"] == "zh"
    assert job and job["text_column"] == "review_text"
    assert job["processed_count"] == 2
    assert job["failed_count"] == 0
    assert job["started_at"] is not None
    assert results and results[0]["sentiment"] == "negative"
    assert events and events[0]["event_type"] == "completed"
    assert reply_history and reply_history[0]["request_id"] == "reply-123"
    assert settings["default_rules"] == "先致歉，再处理售后"
    assert kb_docs and kb_docs[0]["id"] == kb_doc_id


def test_repository_isolates_merchants_by_slug(tmp_path) -> None:
    repo = AppRepository(tmp_path / "app_state.json")
    ctx_a = repo.ensure_context(merchant_slug="merchant-a", merchant_name="Merchant A", user_name="Alice")
    ctx_b = repo.ensure_context(merchant_slug="merchant-b", merchant_name="Merchant B", user_name="Bob")

    repo.record_upload(
        merchant_id=ctx_a.merchant_id,
        user_id=ctx_a.user_id,
        filename="a.csv",
        upload_signature="a",
        file_size=1,
        row_count=1,
        col_count=1,
    )
    repo.record_upload(
        merchant_id=ctx_b.merchant_id,
        user_id=ctx_b.user_id,
        filename="b.csv",
        upload_signature="b",
        file_size=1,
        row_count=1,
        col_count=1,
    )

    uploads_a = repo.list_recent_uploads(ctx_a.merchant_id, limit=10)
    uploads_b = repo.list_recent_uploads(ctx_b.merchant_id, limit=10)

    assert len(uploads_a) == 1 and uploads_a[0]["filename"] == "a.csv"
    assert len(uploads_b) == 1 and uploads_b[0]["filename"] == "b.csv"


def test_repository_can_cancel_and_archive_jobs(tmp_path) -> None:
    repo = AppRepository(tmp_path / "app_state.json")
    ctx = repo.ensure_context(merchant_slug="merchant-a", merchant_name="Merchant A", user_name="Alice")

    job_id = repo.create_analysis_job(
        merchant_id=ctx.merchant_id,
        user_id=ctx.user_id,
        uploaded_file_id=None,
        filename="cancel.csv",
        text_column="review_text",
        provider="deepseek",
        model="deepseek-chat",
        summary_language="zh",
        row_count=3,
    )
    repo.start_analysis_job(job_id)
    repo.cancel_analysis_job(job_id, "manual cancel")
    repo.set_analysis_job_archived(job_id, True)

    job = repo.get_analysis_job(job_id)
    events = repo.list_analysis_job_events(job_id, limit=10)
    assert job and job["status"] == "cancelled"
    assert job["archived"] is True
    assert job["error_message"] == "manual cancel"
    assert any(event["event_type"] == "cancelled" for event in events)
    assert any(event["event_type"] == "archived" for event in events)
