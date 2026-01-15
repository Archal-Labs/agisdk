# Multi-REAL Verifier Validation Report

*Generated: 2026-01-15 01:03:49 UTC*

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 64 |
| Ground Truth Coverage | 14.1% |
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

- Complete: 233 / 233
- TODO (need filling): 0

## Ground Truth Validation

- Queries tested: 49
- Passed: 49
- Failed: 0
- Pass rate: 100.0%

## Action Items

### [MEDIUM] Fix 82 invalid queries
*Impact: Queries may fail during evaluation*

---

*Re-run `tools/gen_validation_report.py` after making changes to see updated metrics.*