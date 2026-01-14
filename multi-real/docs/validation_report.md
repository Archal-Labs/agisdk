# Multi-REAL Verifier Validation Report

*Generated: 2026-01-14 19:19:08 UTC*

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 72 |
| Ground Truth Coverage | 8.3% |
| Query Validity | 62.7% |
| Expected Values Complete | 14.8% |
| GT Validation Pass Rate | 50.0% |
| Action Items | 4 |

## Coverage

- Tasks with ground truth: **6** / 72
- Apps with schema knowledge: **10**
- Apps missing schema: `networkin, omnizon`

## Query Health

- Total queries: 263
- Valid (schema-correct): 165 (62.7%)
- Invalid (schema issues): 98

## Expected Values

- Complete: 39 / 263
- TODO (need filling): 224

## Ground Truth Validation

- Queries tested: 54
- Passed: 27
- Failed: 27
- Pass rate: 50.0%

## Action Items

### [HIGH] Collect ground truth for apps: networkin, omnizon
*Impact: 2 apps have no schema knowledge*

### [MEDIUM] Fill 224 TODO expected values
*Impact: Required for automated evaluation*

### [MEDIUM] Fix 98 invalid queries
*Impact: Queries may fail during evaluation*

### [HIGH] Investigate 27 queries failing against ground truth
*Impact: These queries won't work during evaluation*

---

*Re-run `tools/gen_validation_report.py` after making changes to see updated metrics.*