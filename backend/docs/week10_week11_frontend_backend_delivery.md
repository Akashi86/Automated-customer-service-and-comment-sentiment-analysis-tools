# Week 10-11 Frontend/Backend Delivery

This document summarizes the completed frontend/backend work through Week 11.

## Week 10 Backend

Delivered:
- Added `backend/reporting.py` as a typed backend service layer for:
  - analysis result normalization
  - CSV/Excel export helpers
  - recent JSONL runtime-log reading
  - reusable business-insight aggregation
- Extended `backend/main.py` with `--task report` so exported analysis results can be converted into business insight payloads from the CLI.

Example:

```powershell
python backend/main.py --task report `
  --results-file ai_analysis_results.csv `
  --summary-language zh `
  --output business_insight_snapshot.svg
```

## Week 10 Frontend

Delivered:
- Added one-click CSV export for AI-labeled analysis results.
- Added one-click Excel export for AI-labeled analysis results.
- Added a backend runtime-log admin tab that reads `backend/logs/llm_calls.jsonl`, shows status counts, recent request metadata, errors, latency, and guardrail fields.
- Added log CSV export for debugging and weekly reporting.

## Week 11 Backend

Delivered:
- Added custom business-report aggregation through `build_business_insight_payload(...)`.
- Aggregation output includes:
  - analyzed count
  - failed count
  - sentiment distribution
  - negative rate
  - pain-point coverage
  - unique pain-point count
  - top pain points
  - estimated affected monthly orders
  - estimated monthly loss
  - actionable recommendations and owner categories
- Added report artifact builders:
  - Markdown report
  - HTML report
  - long SVG snapshot

## Week 11 Frontend

Delivered:
- Added a dedicated report snapshot tab.
- Users can tune business assumptions:
  - estimated monthly order count
  - average order value
  - return-loss rate
  - number of pain points included in the report
- The UI now displays the generated business headline, core metrics, and actionable recommendations.
- The current analysis can be exported as:
  - Markdown report
  - HTML report
  - long SVG report snapshot

## Test Coverage

Added:
- `backend/tests/test_reporting.py`

Covered:
- pain-point aggregation and business metrics
- CSV export bytes
- long SVG snapshot generation
- filtered backend log reading
