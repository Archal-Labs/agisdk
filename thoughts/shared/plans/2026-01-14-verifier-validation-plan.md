# Multi-REAL Verifier Validation Plan

## Overview

Ensure the validity and correctness of JMESPath evaluation queries for all 71 tasks in `multi-real/tasks/`. The current state has significant issues: 198 TODO expected values, schema mismatches between queries and actual ground truth, and only 6 ground truth files covering a fraction of tasks.

## Current State Analysis

### Key Issues Identified

| Issue | Count | Severity |
|-------|-------|----------|
| TODO expected values | 198 | Critical |
| Ground truth files | 6/71 tasks | Critical |
| Schema mismatches | Unknown (likely 30%+) | Critical |
| `expected_value: false` (possibly wrong) | 20 | Medium |

### Ground Truth Coverage

**Available Ground Truth Files (6):**
- `dashdish-gomail-1.json` (7.0MB)
- `flyunified-gocalendar-staynb-1.json` (2.7MB)
- `gomail-marrsuite-1.json` (5.0MB)
- `gomail-topwork-3.json` (0.3MB)
- `gomail-zilloft-1.json` (15MB)
- `opendining-udriver-1.json` (0.4MB)

**Missing Coverage:**
- NetworkIn: 0/6 tasks
- GoCalendar combinations: 1/24+ tasks
- Omnizon: 0/4 tasks
- Most app combinations have 0-1 coverage

### Schema Mismatch Examples

**Problem 1: Query uses non-existent path**
```
Query: marrisuite.differences.bookings.added[?...]
Actual: marrisuite.bookingDetailsDiff.added (empty array)
        marrisuite.initialfinaldiff.added.booking (dict with bookingDetails)
```

**Problem 2: Different app-specific schemas**
- GoMail: Has `differences.emails.added/deleted/updated` (correct)
- Marrisuite: Has `bookingDetailsDiff`, `cancellationDetailsDiff` (not `differences`)
- DashDish: Has `differences.foodOrders.added` (dict, not array)
- FlyUnified: Has `differences.bookedFlights.added` (correct)

**Problem 3: initialfinaldiff vs differences**
- Some queries expect `differences.*.added[]` (array)
- Actual data often in `initialfinaldiff.added.*` (dict)

## Desired End State

1. **100% Query Validity**: Every JMESPath query compiles and returns non-null when run against a successful task completion
2. **Correct Expected Values**: All `expected_value` fields contain accurate values (not TODO)
3. **Schema Documentation**: Clear documentation of each app's finish JSON structure
4. **Validation Tools**: Scripts that verify query correctness against ground truth
5. **Ground Truth Pipeline**: Automated way to generate more ground truth from model runs

### Verification Criteria

- [ ] All 71 tasks have at least one working query set
- [ ] No JMESPath syntax errors
- [ ] Queries match actual finish JSON schema for each app
- [ ] Expected values validated against ground truth where available
- [ ] Tasks without ground truth have queries validated for syntax and schema correctness

## What We're NOT Doing

- NOT running all 71 tasks to generate ground truth (too expensive/time-consuming for planning)
- NOT changing task goals or evaluation criteria semantics
- NOT adding new evaluation types (staying with JMESPath + LLM judge)
- NOT creating synthetic ground truth automatically (needs model runs)

## Implementation Phases

---

## Phase 1: Schema Discovery and Documentation

### Overview
Document the actual finish JSON schema for each web application. This is foundational - we can't fix queries without knowing the correct paths.

### Changes Required:

#### 1. Create Schema Documentation
**File**: `multi-real/docs/finish_schemas.md`

Document each app's structure:
- Top-level keys (actionhistory, initialstate, finalstate, initialfinaldiff, differences, state)
- Domain-specific diff keys (bookingDetailsDiff, foodOrders, etc.)
- Array vs dict structures
- Nested paths for common query targets (emails, bookings, events, etc.)

#### 2. Schema Discovery Script
**File**: `multi-real/tools/discover_schemas.py`

```python
# Analyze ground truth files to extract schema patterns
# Output: JSON file mapping app_id -> schema structure
# Include:
#   - Available diff paths
#   - Whether arrays or dicts
#   - Common queryable fields
```

### Success Criteria:

#### Automated Verification:
- [ ] Schema discovery script runs: `uv run python multi-real/tools/discover_schemas.py`
- [ ] Outputs schema JSON covering all 6 ground truth files

#### Manual Verification:
- [ ] Schema documentation covers all web apps with available ground truth
- [ ] Each app's queryable paths are clearly documented

---

## Phase 2: Query Pattern Analysis

### Overview
Analyze all existing queries to identify common patterns and mismatches with documented schema.

### Changes Required:

#### 1. Query Analyzer
**File**: `multi-real/tools/analyze_queries.py`

```python
# Extract all JMESPath queries from tasks
# Classify by:
#   - App prefix (gomail., marrisuite., etc.)
#   - Path pattern (differences vs initialfinaldiff)
#   - Query type (boolean, field extraction, filter)
# Cross-reference with discovered schema
# Output: Report of valid vs invalid query patterns
```

#### 2. Update fix_query_patterns.py
Add new patterns based on schema discovery:
- `marrisuite.differences.*` → `marrisuite.*Diff.*` mappings
- `app.differences.*.added[]` → `app.initialfinaldiff.added.*` where needed
- App-specific corrections

### Success Criteria:

#### Automated Verification:
- [ ] Query analyzer runs: `uv run python multi-real/tools/analyze_queries.py`
- [ ] Reports mismatches between queries and schema

#### Manual Verification:
- [ ] Review output to identify systemic pattern issues
- [ ] Validate proposed fixes are semantically correct

---

## Phase 3: Query Correction

### Overview
Fix all queries to match actual schema. Must preserve semantic meaning while fixing structural issues.

### Changes Required:

#### 1. Fix Marrisuite Queries
Transform:
```
marrisuite.differences.bookings.added[?condition]
```
To:
```
marrisuite.bookingDetailsDiff.added[?condition]
# OR
values(marrisuite.initialfinaldiff.added.booking.bookingDetails)[?condition]
```

#### 2. Fix GoMail Queries
Verify `gomail.differences.emails.added` works (it does per schema)

#### 3. Fix Other App-Specific Queries
Based on Phase 1-2 findings.

### Success Criteria:

#### Automated Verification:
- [ ] Run fix script: `uv run python multi-real/tools/fix_query_patterns.py`
- [ ] Query validation passes: `uv run python multi-real/evaluation/validate_queries.py --all -d multi-real/final_states/manual`

#### Manual Verification:
- [ ] Spot-check 5 fixed queries against ground truth
- [ ] Verify semantic meaning preserved

---

## Phase 4: Expected Value Population

### Overview
Fill in TODO expected values using ground truth. Only 6 tasks have ground truth, so ~65 tasks will remain with TODO values.

### Changes Required:

#### 1. Run fill_expected_values.py
```bash
uv run python multi-real/tools/fill_expected_values.py --verbose
```

#### 2. Review and Fix Incorrect Expected Values
Check tasks with `expected_value: false` - verify these are intentional:
- `dashdish-gomail-1.json` (2 occurrences)
- `opendining-udriver-1.json` (3 occurrences)
- Others (15 occurrences)

### Success Criteria:

#### Automated Verification:
- [ ] Fill script completes without errors
- [ ] Provenance report generated

#### Manual Verification:
- [ ] Review provenance_report.json
- [ ] Spot-check 5 populated expected values against ground truth

---

## Phase 5: Syntax-Only Validation for Remaining Tasks

### Overview
For the ~65 tasks without ground truth, validate that queries are at least syntactically valid and use plausible schema paths.

### Changes Required:

#### 1. Create Syntax Validator
**File**: `multi-real/tools/validate_syntax.py`

```python
# For each task:
# 1. Compile JMESPath query (catch syntax errors)
# 2. Check app prefix matches task's websites
# 3. Check path pattern matches known schema patterns
# 4. Flag queries using unknown paths
```

#### 2. Create Schema Stub Generator
Generate minimal stub data for each app based on discovered schema:
```python
# For apps without ground truth, create stubs like:
# {"networkin": {"differences": {"posts": {"added": []}}}}
# This allows queries to execute without returning null
```

### Success Criteria:

#### Automated Verification:
- [ ] All 71 tasks pass syntax validation
- [ ] No JMESPath compile errors

#### Manual Verification:
- [ ] Review flagged queries for plausibility

---

## Phase 6: Ground Truth Generation Pipeline

### Overview
Create infrastructure to generate new ground truth by running models on tasks. This is for future use - not running full benchmark now.

### Changes Required:

#### 1. Document Ground Truth Collection Process
**File**: `multi-real/docs/collecting_ground_truth.md`

Steps:
1. Run model on task with `--capture-finish-state` flag
2. Validate finish state with LLM judge
3. If high confidence, save to `final_states/synthetic/`
4. Update task expected values using fill script

#### 2. Enhance capture_pipeline.py
If not already present, add:
- Automatic LLM validation of collected states
- Confidence scoring
- Integration with fill_expected_values.py

### Success Criteria:

#### Automated Verification:
- [ ] Documentation is complete
- [ ] Pipeline script exists and has help text

#### Manual Verification:
- [ ] Run pipeline on 1 task manually
- [ ] Verify ground truth is collected and validated

---

## Phase 7: Validation Report Generation

### Overview
Create a comprehensive report of verifier validity status for all tasks.

### Changes Required:

#### 1. Create Validation Report Generator
**File**: `multi-real/tools/gen_validation_report.py`

Output:
```markdown
# Multi-REAL Verifier Validation Report

## Summary
- Tasks validated against ground truth: 6/71
- Tasks with TODO expected values: 65/71
- Tasks with schema-validated queries: 71/71
- Queries with syntax errors: 0

## Per-Task Status
| Task | GT Available | Queries Valid | Expected Values |
|------|--------------|---------------|-----------------|
| dashdish-gomail-1 | ✅ | ✅ | ✅ |
| flyunified-gocalendar-1 | ❌ | ✅ | TODO |
...
```

### Success Criteria:

#### Automated Verification:
- [ ] Report generates: `uv run python multi-real/tools/gen_validation_report.py`

#### Manual Verification:
- [ ] Report accurately reflects validation status
- [ ] Clear action items for remaining TODO tasks

---

## Testing Strategy

### Unit Tests:
- JMESPath query compilation for all tasks
- Schema stub generation produces valid JSON
- Fill script handles edge cases (null results, type mismatches)

### Integration Tests:
- Full validation pipeline: discover → analyze → fix → fill → validate → report
- Test against all 6 ground truth files

### Manual Testing Steps:
1. Run schema discovery on all ground truth files
2. Review schema documentation for completeness
3. Run query analysis and review mismatches
4. Apply query fixes and verify with validation script
5. Run fill script and review provenance report
6. Generate final validation report
7. Spot-check 10 random tasks for overall correctness

---

## Appendix: Existing Tools

| Tool | Purpose | Status |
|------|---------|--------|
| `evaluation/validate_queries.py` | Test queries against ground truth | Working |
| `tools/fill_expected_values.py` | Populate TODO expected values | Working |
| `tools/fix_query_patterns.py` | Fix common query pattern errors | Working, needs expansion |
| `evaluation/validate_ground_truth.py` | Validate GT quality | Exists |
| `tools/discover_schemas.py` | Document finish JSON schema | **TO CREATE** |
| `tools/analyze_queries.py` | Cross-reference queries with schema | **TO CREATE** |
| `tools/validate_syntax.py` | Syntax-only validation | **TO CREATE** |
| `tools/gen_validation_report.py` | Generate validation summary | **TO CREATE** |

---

## References

- Existing plan: `thoughts/shared/plans/2026-01-12-multi-real-benchmark-harness.md`
- Task files: `multi-real/tasks/*.json`
- Ground truth: `multi-real/final_states/manual/`
- Evaluator: `src/agisdk/REAL/browsergym/webclones/evaluate.py`
