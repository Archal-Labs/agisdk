# Multi-REAL Verifier Validation Report

*Generated: 2026-01-14 22:47:57 UTC*

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 69 |
| Ground Truth Coverage | 13.0% |
| Query Validity | 67.5% |
| Expected Values Complete | 100.0% |
| GT Validation Pass Rate | 100.0% |
| Action Items | 1 |

## Coverage

- Tasks with ground truth: **9** / 69
- Apps with schema knowledge: **12**

## Query Health

- Total queries: 252
- Valid (schema-correct): 170 (67.5%)
- Invalid (schema issues): 82

## Expected Values

- Complete: 252 / 252
- TODO (need filling): 0

## Ground Truth Validation

- Queries tested: 30
- Passed: 30
- Failed: 0
- Pass rate: 100.0%

## Action Items

### [MEDIUM] Fix 82 invalid queries
*Impact: Queries may fail during evaluation*

---

*Re-run `tools/gen_validation_report.py` after making changes to see updated metrics.*