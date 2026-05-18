# Week 10 Persistence Delivery

## Goal

Move the project one step closer to a trial-ready product by adding a lightweight persistence layer instead of relying only on Streamlit session state.

## What Was Added

### 1. JSON repository layer

New file:

- `backend/app_repository.py`

This module adds a minimal product-oriented storage layer using a JSON state file.

It initializes and manages these tables:

- `merchants`
- `users`
- `merchant_memberships`
- `merchant_settings`
- `uploaded_files`
- `analysis_jobs`
- `analysis_results`
- `customer_service_replies`
- `knowledge_base_docs`

### 2. Configurable application database path

Updated:

- `backend/config.py`

New config entry:

- `APP_DB_PATH`

Default location:

- `backend/data/app_state.json`

### 3. Frontend persistence integration

Updated:

- `frontend/app.py`

Integrated persistence into three key flows:

- upload success -> save upload record
- batch analysis completion/failure -> save analysis job and results
- customer-service reply generation -> save reply history

### 4. Recent activity panel

Added a small Streamlit sidebar activity panel that reads persisted records and shows:

- recent uploads
- recent analysis jobs
- recent customer-service replies

### 5. History and rules center

The frontend now also includes:

- a history/task center for uploads, jobs, row-level results, and reply history
- a rules/knowledge-base center for default merchant rules and saved KB docs
- customer-service context prefill from saved rules and KB docs

### 6. Repository test

New test:

- `backend/tests/test_app_repository.py`

This verifies that the repository can:

- create demo merchant/user context
- save upload records
- save analysis jobs and results
- save customer-service reply history
- save merchant default rules
- save and list KB documents

## Why This Matters

Before this change, uploads and results mostly lived in session state, which made the app behave like a demo.

After this change, the app starts to behave more like a trialable product:

- activity can survive reruns
- uploads and jobs can be traced
- reply history can be retained
- later user/account isolation has a storage foundation

## Current Limitation

This is still a lightweight MVP persistence layer:

- no real login system yet
- currently uses a demo merchant/user context
- no background queue yet
- no file-object blob storage yet

## Recommended Next Step

Use this repository layer as the base for:

1. real merchant/user login and isolation
2. async job status polling
3. rules and knowledge-base version management
4. exportable history and reporting
