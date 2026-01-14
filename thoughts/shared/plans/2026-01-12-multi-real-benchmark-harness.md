# Multi-REAL Benchmark Harness Implementation Plan

## Overview

Build a standalone benchmark harness for the Multi-REAL multi-app task suite that enables fair comparison of frontier models (GPT-5.2, Claude Opus 4.5, Claude Computer Use, Gemini 2.5 Pro) on 68 cross-application browser tasks. The harness uses hybrid evaluation (JMESPath primary + LLM judge fallback) and operates independently from the main REAL registry while leveraging existing infrastructure.

## Current State Analysis

### What Exists
- **68 multi-app tasks** in `multi-real/tasks/*.json` covering 12 web applications
- **Core infrastructure** in `src/agisdk/REAL/` with built-in multi-app support:
  - `browsergym/webclones/base.py`: Multi-app browser management (`_get_multi_app_finish_json()`)
  - `browsergym/webclones/evaluate.py`: JMESPath evaluation with website prefixes
  - `demo_agent/basic_agent.py`: Model abstraction for OpenAI, Anthropic, OpenRouter
- **7 ground truth final states** in `multi-real/final_states/manual/`
- **Validation infrastructure**: `validate_query_structure.py`, `llm_judge.py`, `hybrid_validator.py`

### Ground Truth Coverage Analysis

**Current 7 ground truth states cover:**
| Task ID | Apps | Coverage |
|---------|------|----------|
| dashdish-gomail-1 | DashDish, GoMail | 2-app |
| flyunified-gocalendar-staynb-1 | FlyUnified, GoCalendar, StaynB | 3-app |
| gomail-marrsuite-1 | GoMail, Marrisuite | 2-app |
| gomail-omnizon-1 | GoMail, Omnizon | 2-app |
| gomail-topwork-3 | GoMail, TopWork | 2-app |
| gomail-zilloft-3 | GoMail, Zilloft | 2-app |
| opendining-udriver-1 | OpenDining, Udriver | 2-app |

**App coverage gaps:**
- NetworkIn: **0 ground truth** (critical gap - 6 tasks use NetworkIn)
- GoCalendar: Only 1 ground truth (24 tasks use it)
- Most apps have only 1-2 ground truth samples

### Key Discoveries
- Multi-app support is **already built** into the core harness (`base.py:284-317`)
- Task naming uses `multi.{task-id}` prefix convention
- JMESPath queries use `{website_id}.path.to.field` prefixes for multi-app
- Hybrid evaluation exists but needs integration with main runner

## Desired End State

A complete benchmark pipeline that:
1. Runs any frontier model on all 68 multi-app tasks with consistent conditions
2. Captures `/finish` state from all websites involved in each task
3. Evaluates results using hybrid JMESPath + LLM judge with confidence scoring
4. Produces comparable results across models with clear methodology
5. Generates publication-ready reports with appropriate caveats for limited ground truth

**Verification:**
- All 68 tasks complete without infrastructure errors
- Results match ground truth on the 7 validated tasks
- Hybrid evaluation produces consistent scores across runs
- Comparison reports accurately reflect model performance

## What We're NOT Doing

- NOT integrating into main REAL task registry (keeping separate)
- NOT collecting additional ground truth before initial model runs
- NOT building a new leaderboard (reusing existing infrastructure if available). The leaderboard indeed is irrelevant for this project. I only want results.
- NOT supporting models beyond the 4 specified frontier models initially
- NOT implementing real-time streaming results (batch processing only)

## Ground Truth Coverage Efficacy Analysis

### Scenario Comparison

| Coverage | Tasks | Manual Effort | Query Validation | Result Confidence | Publication Readiness |
|----------|-------|---------------|------------------|-------------------|----------------------|
| **100% (68)** | All | ~15-20 hours | Complete | High (95%+) | Definitive claims |
| **50% (34)** | Strategic | ~8-10 hours | Statistical | Medium (80-90%) | Results with caveats |
| **10% (7)** | Current | 0 hours | Pattern-based | Lower (60-70%) | Preliminary results |

### Why 7 Ground Truth Can Work (With Hybrid Evaluation)

The hybrid evaluation strategy fundamentally changes the ground truth requirements:

**Without LLM Judge:**
- JMESPath query bugs → false negatives → unreliable results
- Need ground truth to validate every query pattern
- 7/68 coverage is insufficient

**With Hybrid Evaluation:**
- JMESPath pass → **High confidence** (deterministic match)
- JMESPath fail + LLM pass → **Medium confidence** (likely query bug, flag for review)
- Both fail → **High confidence failure** (task genuinely failed)
- JMESPath pass + LLM fail → **Should not happen** (investigate LLM judge)

**Statistical Argument:**
- 7 ground truth validates JMESPath for ~10% of tasks
- If JMESPath accuracy is 85%+ on validated tasks, extrapolate to others
- LLM judge catches the 15% of query bugs
- Report results with confidence intervals based on validation sample

**Recommendation:**
Proceed with 7 ground truth + hybrid evaluation. Report results as:
- "X% success rate (95% CI: Y-Z%)" for validated tasks
- "X% success rate (with LLM judge fallback)" for all tasks
- Flag tasks where JMESPath and LLM judge disagree

## Implementation Approach

### Architecture Overview

```
multi-real/
├── run_benchmark.py          # Main entry point
├── multi_harness.py          # Multi-app harness wrapper
├── model_configs/            # Per-model configurations
│   ├── gpt52.yaml
│   ├── claude_opus45.yaml
│   ├── claude_cua.yaml
│   └── gemini25pro.yaml
├── results/                  # Benchmark results
│   ├── raw/                  # Per-task JSON results
│   ├── reports/              # Comparison reports
│   └── aggregated/           # Summary statistics
├── hybrid_validator.py       # Existing - enhance
├── llm_judge.py             # Existing - enhance
└── tasks/                    # Task definitions
```

### Model Configuration

Each model gets a YAML config specifying:
- API provider and credentials
- Model identifier
- Token limits and pricing
- Agent-specific settings (e.g., computer use for Claude CUA)

---

## Phase 1: Task Registry and Configuration [IMPLEMENTED]

### Overview
Create a standalone task registry for multi-real tasks that loads task definitions and provides the `multi.{task-id}` naming convention.

### Files Created:
- `multi-real/task_registry.py` - Task registry with `MultiRealTask` dataclass and `MultiRealRegistry` class
- `multi-real/model_configs/schema.py` - `ModelConfig` dataclass and `ModelProvider` enum
- `multi-real/model_configs/__init__.py` - Module exports
- `multi-real/model_configs/gpt52.yaml` - GPT-5.2 configuration
- `multi-real/model_configs/claude_opus45.yaml` - Claude Opus 4.5 configuration
- `multi-real/model_configs/claude_cua.yaml` - Claude Computer Use configuration
- `multi-real/model_configs/gemini25pro.yaml` - Gemini 2.5 Pro configuration

### Verification:
- [x] Task registry loads 64 tasks (some excluded as examples)
- [x] Model configs parse correctly
- [x] No syntax errors

---

## Phase 2: Multi-Real Harness Wrapper [IMPLEMENTED]

### Overview
Create a harness wrapper that adapts the existing REAL infrastructure for multi-app tasks, handling task loading, browser setup, and result capture.

### Files Created:
- `multi-real/multi_harness.py` - Main harness with `MultiRealHarness` class and `MultiRealResult` dataclass

### Key Design Decisions:

#### 1. **Disk-based Result Loading (Critical)**
The existing `ExpArgs.run()` doesn't return results—it saves them to disk as `summary_info.json` and `finish_state.json`. Our harness:
- Calls `exp_args.prepare()` to create experiment directory
- Calls `exp_args.run()` which executes the task and saves results
- Loads results from `summary_info.json` after completion
- Extracts `finish_state`, token counts, timing, and errors from saved JSON

**Why**: The core REAL infrastructure is designed for batch processing with results persisted to disk. Trying to intercept return values would require modifying core code.

#### 2. **Multi-App Task Configuration**
Multi-app tasks require special environment setup:
- Pass full task config (id, goal, websites, evals) via `task_kwargs` parameter
- Use `browsergym/webclones.multi` as the task name to trigger multi-app handling
- The core infrastructure's `AbstractWebCloneTask` already supports multi-app via `_get_multi_app_finish_json()`

**Why**: The existing task loading expects tasks from JSON files in VERSION_DIRS. We bypass this by passing config directly via task_kwargs.

#### 3. **Evaluation Strategy: JMESPath First, LLM Fallback**
Evaluation happens in two phases:
1. **JMESPath evaluation** (deterministic, fast, free)
   - Use `WebCloneEvaluator` to run JMESPath queries against finish_state
   - If all pass → high confidence success
   - If any fail → proceed to phase 2 (if hybrid enabled)

2. **LLM Judge evaluation** (semantic, slower, costs money)
   - Only runs if JMESPath fails AND hybrid_eval=True
   - Uses Claude to evaluate if task semantically succeeded
   - If LLM passes → medium confidence (likely JMESPath query bug)
   - If LLM fails too → high confidence failure

**Why**: JMESPath is deterministic and free, so we prefer it. LLM judge catches query bugs but has API costs, so it's a fallback.

#### 4. **Confidence Scoring**
Results include confidence levels to indicate evaluation reliability:
- **High**: JMESPath passed, or both JMESPath and LLM agree (pass or fail)
- **Medium**: JMESPath failed but LLM passed (suggests query bug), or JMESPath failed with no LLM judge
- **Low**: Missing finish_state or other evaluation errors

**Why**: With only 7 ground truth tasks, we need to signal which results are trustworthy. High-confidence results can be trusted; medium-confidence may need manual review.

#### 5. **API Key Handling**
The harness supports two API key methods:
- Environment variable (standard): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- Direct key in YAML (convenience): If `api_key_env` starts with `sk-`, use it directly as the key

**Why**: During testing, it's easier to put keys directly in YAML. For production, use environment variables for security.

#### 6. **Fixed Seed for Reproducibility**
All tasks run with `task_seed=42` (fixed):
- Ensures deterministic behavior across runs
- Makes results comparable between models
- Enables debugging by reproducing exact same scenario

**Why**: Benchmark fairness requires identical conditions for all models. Random seeds would introduce variance.

#### 7. **Cost Tracking**
Extract token usage from `summary_info.json` and calculate costs:
- `stats.cum_input_token` × `input_price_per_1k` / 1000
- `stats.cum_output_token` × `output_price_per_1k` / 1000

**Why**: Running 68 tasks across 4 models can be expensive. Cost tracking helps budget and compare model efficiency.

#### 8. **Result Storage Structure**
Results saved to:
```
results/raw/
├── task-id/
│   ├── gpt_52.json
│   ├── claude_opus_45.json
│   └── ...
```

Each result JSON contains:
- Success/score/confidence
- JMESPath results (per-eval details)
- LLM judge results (if triggered)
- finish_state (full browser state)
- Token usage and costs
- exp_dir (for debugging)

**Why**: Flat structure makes it easy to aggregate across models. Including exp_dir enables debugging failed tasks.

### Verification:

#### Automated Verification:
- [x] Harness imports without errors
- [ ] Can instantiate with model config
- [ ] Run with a test task to verify integration

#### Manual Verification (Pending):
- [ ] Run a single task with a real model and verify finish_state is captured
- [ ] Verify hybrid evaluation triggers LLM judge when JMESPath fails
- [ ] Check results are saved in correct directory structure
- [ ] Verify cost calculation is accurate

---

## Phase 3: Hybrid Validator Enhancement [IMPLEMENTED]

### Overview
Added `HybridValidator` class to the existing `hybrid_validator.py` for LLM-as-judge fallback evaluation with confidence scoring.

### Files Modified:
- `multi-real/hybrid_validator.py` - Added `HybridValidator` class while preserving existing CLI functionality

### Key Implementation Details:

**1. HybridValidator Class:**
- `evaluate()` method takes task_goal, finish_state, and evals
- Calls Claude API (default: claude-sonnet-4-20250514) for semantic evaluation
- Returns dict with overall_pass, criteria_results, reasoning, and confidence (0.0-1.0)
- Handles API failures gracefully (returns low-confidence failure)

**2. Prompt Engineering:**
- Truncates large finish_state to 10k chars to stay within token limits
- Asks for JSON response with structured format
- Requests confidence scoring (0.9+ very confident, 0.7-0.9 confident, 0.5-0.7 uncertain)
- Emphasizes semantic correctness over exact string matching

**3. Ground Truth Validation:**
- `validate_against_ground_truth()` compares finish_state against known-good state
- `_deep_compare()` recursively compares data structures
- Returns match_ratio, passes_threshold, and detailed differences list
- Used for measuring JMESPath query accuracy and LLM judge reliability

**4. Dual Interface:**
- Class interface for programmatic use (imported by multi_harness.py)
- CLI interface preserved for standalone validation scripts
- Existing CLI functions still work (load_json, extract_concrete_evidence, etc.)

### Verification:

#### Automated Verification:
- [x] HybridValidator class imports successfully
- [ ] Can instantiate and call evaluate() method
- [ ] Returns expected dict structure

#### Manual Verification (Pending):
- [ ] LLM judge correctly evaluates a known-good finish state
- [ ] Ground truth comparison identifies differences accurately
- [ ] Hybrid evaluation correctly falls back to LLM when JMESPath fails

---

## Phase 4: Benchmark Runner and CLI

### Overview
Create the main benchmark runner with CLI interface for running models on tasks and generating results.

### Changes Required:

#### 1. Main Benchmark Runner
**File**: `multi-real/run_benchmark.py`

```python
#!/usr/bin/env python3
"""
Multi-REAL Benchmark Runner

Usage:
    # Run all tasks with a model
    python run_benchmark.py --model gpt52

    # Run specific tasks
    python run_benchmark.py --model claude_opus45 --tasks multi.gocalendar-gomail-1 multi.dashdish-gomail-1

    # Run with filters
    python run_benchmark.py --model gemini25pro --websites gomail gocalendar --max-apps 2

    # Dry run (list tasks without running)
    python run_benchmark.py --model gpt52 --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from task_registry import registry, MultiRealTask
from model_configs.schema import ModelConfig
from multi_harness import MultiRealHarness, MultiRealResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "model_configs"
RESULTS_DIR = Path(__file__).parent / "results"


def load_model_config(model_name: str) -> ModelConfig:
    """Load model configuration by name."""
    config_path = CONFIGS_DIR / f"{model_name}.yaml"
    if not config_path.exists():
        available = [f.stem for f in CONFIGS_DIR.glob("*.yaml") if f.stem != "schema"]
        raise ValueError(
            f"Unknown model: {model_name}. Available: {', '.join(available)}"
        )
    return ModelConfig.from_yaml(config_path)


def filter_tasks(
    tasks: list[str] | None = None,
    websites: list[str] | None = None,
    min_apps: int | None = None,
    max_apps: int | None = None,
) -> list[MultiRealTask]:
    """Filter tasks based on criteria."""
    if tasks:
        # Specific task IDs
        result = []
        for task_id in tasks:
            task = registry.get(task_id)
            if task:
                result.append(task)
            else:
                logger.warning(f"Task not found: {task_id}")
        return result

    # Filter by criteria
    return list(registry.filter(
        websites=websites,
        min_apps=min_apps,
        max_apps=max_apps,
    ))


def save_run_summary(
    results: list[MultiRealResult],
    model_config: ModelConfig,
    output_dir: Path,
) -> Path:
    """Save summary of benchmark run."""
    summary = {
        "model": model_config.name,
        "model_id": model_config.model_id,
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(results),
        "passed": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
        "total_time": sum(r.elapsed_time for r in results),
        "total_cost": sum(r.total_cost for r in results),
        "by_confidence": {
            "high": sum(1 for r in results if r.confidence == "high" and r.success),
            "medium": sum(1 for r in results if r.confidence == "medium" and r.success),
            "low": sum(1 for r in results if r.confidence == "low" and r.success),
        },
        "by_eval_method": {
            "jmespath": sum(1 for r in results if r.eval_method == "jmespath"),
            "llm_judge": sum(1 for r in results if r.eval_method == "llm_judge"),
            "hybrid": sum(1 for r in results if r.eval_method == "hybrid"),
        },
        "tasks": [r.to_dict() for r in results],
    }

    output_file = output_dir / f"summary_{model_config.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run Multi-REAL benchmark")
    parser.add_argument("--model", required=True, help="Model config name (e.g., gpt52, claude_opus45)")
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to run")
    parser.add_argument("--websites", nargs="+", help="Filter by websites")
    parser.add_argument("--min-apps", type=int, help="Minimum number of apps")
    parser.add_argument("--max-apps", type=int, help="Maximum number of apps")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser headless")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Show browser")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid evaluation")
    parser.add_argument("--dry-run", action="store_true", help="List tasks without running")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory")

    args = parser.parse_args()

    # Load model config
    model_config = load_model_config(args.model)
    logger.info(f"Loaded model config: {model_config.name}")

    # Filter tasks
    tasks = filter_tasks(
        tasks=args.tasks,
        websites=args.websites,
        min_apps=args.min_apps,
        max_apps=args.max_apps,
    )
    logger.info(f"Selected {len(tasks)} tasks")

    if args.dry_run:
        print(f"\nDry run - would execute {len(tasks)} tasks with {model_config.name}:\n")
        for task in tasks:
            print(f"  {task.prefixed_id}: {task.goal[:60]}...")
        return

    # Run benchmark
    harness = MultiRealHarness(
        model_config=model_config,
        results_dir=args.output_dir / "raw",
        headless=args.headless,
        use_hybrid_eval=not args.no_hybrid,
    )

    results = harness.run_all(tasks)

    # Save summary
    summary_path = save_run_summary(results, model_config, args.output_dir / "aggregated")

    # Print results
    passed = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE: {model_config.name}")
    print(f"{'='*60}")
    print(f"Tasks: {len(results)}")
    print(f"Passed: {passed} ({100*passed/len(results):.1f}%)")
    print(f"Failed: {len(results) - passed}")
    print(f"Total time: {sum(r.elapsed_time for r in results):.1f}s")
    print(f"Total cost: ${sum(r.total_cost for r in results):.2f}")
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] CLI help works: `python run_benchmark.py --help`
- [ ] Dry run lists tasks: `python run_benchmark.py --model gpt52 --dry-run`
- [ ] Task filtering works: `python run_benchmark.py --model gpt52 --websites gomail --dry-run`

#### Manual Verification:
- [ ] Run a single task with real model and verify results are saved
- [ ] Run full benchmark on one model and check summary output
- [ ] Verify costs are calculated correctly

---

## Phase 5: Results Aggregation and Reporting

### Overview
Create tools for aggregating results across models and generating comparison reports.

### Changes Required:

#### 1. Comparison Report Generator
**File**: `multi-real/generate_comparison_report.py` (update existing)

```python
#!/usr/bin/env python3
"""
Generate comparison reports across models for Multi-REAL benchmark.

Usage:
    python generate_comparison_report.py --output-dir results/reports/
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_summaries(results_dir: Path) -> list[dict]:
    """Load all summary files from results directory."""
    summaries = []
    aggregated_dir = results_dir / "aggregated"

    if not aggregated_dir.exists():
        return summaries

    for summary_file in aggregated_dir.glob("summary_*.json"):
        with open(summary_file) as f:
            summaries.append(json.load(f))

    return summaries


def generate_markdown_report(summaries: list[dict]) -> str:
    """Generate markdown comparison report."""
    if not summaries:
        return "# Multi-REAL Benchmark Report\n\nNo results available."

    # Sort by success rate descending
    summaries.sort(key=lambda x: x.get("success_rate", 0), reverse=True)

    lines = [
        "# Multi-REAL Benchmark Comparison Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary",
        "",
        "| Model | Tasks | Passed | Success Rate | Time | Cost |",
        "|-------|-------|--------|--------------|------|------|",
    ]

    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['total_tasks']} | {s['passed']} | "
            f"{100*s['success_rate']:.1f}% | {s['total_time']:.0f}s | ${s['total_cost']:.2f} |"
        )

    lines.extend([
        "",
        "## Evaluation Method Distribution",
        "",
        "| Model | JMESPath | LLM Judge | Hybrid |",
        "|-------|----------|-----------|--------|",
    ])

    for s in summaries:
        by_method = s.get("by_eval_method", {})
        lines.append(
            f"| {s['model']} | {by_method.get('jmespath', 0)} | "
            f"{by_method.get('llm_judge', 0)} | {by_method.get('hybrid', 0)} |"
        )

    lines.extend([
        "",
        "## Confidence Distribution (Passed Tasks)",
        "",
        "| Model | High | Medium | Low |",
        "|-------|------|--------|-----|",
    ])

    for s in summaries:
        by_conf = s.get("by_confidence", {})
        lines.append(
            f"| {s['model']} | {by_conf.get('high', 0)} | "
            f"{by_conf.get('medium', 0)} | {by_conf.get('low', 0)} |"
        )

    # Per-task comparison
    if len(summaries) > 1:
        lines.extend([
            "",
            "## Per-Task Results",
            "",
        ])

        # Build task -> model -> result mapping
        task_results: dict[str, dict[str, dict]] = defaultdict(dict)
        for s in summaries:
            for task in s.get("tasks", []):
                task_results[task["task_id"]][s["model"]] = task

        # Header
        model_names = [s["model"] for s in summaries]
        lines.append("| Task | " + " | ".join(model_names) + " |")
        lines.append("|------|" + "|".join(["------"] * len(model_names)) + "|")

        # Rows
        for task_id in sorted(task_results.keys()):
            row = [task_id.replace("multi.", "")]
            for model in model_names:
                result = task_results[task_id].get(model, {})
                if result:
                    status = "PASS" if result.get("success") else "FAIL"
                    conf = result.get("confidence", "?")[0].upper()
                    row.append(f"{status} ({conf})")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Methodology Notes",
        "",
        "- **JMESPath**: Deterministic query evaluation (high confidence)",
        "- **LLM Judge**: Claude-based semantic evaluation (medium confidence when JMESPath fails)",
        "- **Hybrid**: Both JMESPath and LLM judge evaluated (used for discrepancy detection)",
        "",
        "### Confidence Levels",
        "- **High**: JMESPath pass OR both methods agree",
        "- **Medium**: LLM judge pass when JMESPath fails (possible query bug)",
        "- **Low**: Ambiguous result requiring manual review",
        "",
        "### Ground Truth Coverage",
        f"- 7 tasks have validated ground truth states (~10% coverage)",
        "- Results on non-validated tasks should be interpreted with appropriate caution",
        "",
    ])

    return "\n".join(lines)


def generate_json_report(summaries: list[dict]) -> dict:
    """Generate structured JSON report."""
    return {
        "generated_at": datetime.now().isoformat(),
        "models": [
            {
                "name": s["model"],
                "model_id": s["model_id"],
                "total_tasks": s["total_tasks"],
                "passed": s["passed"],
                "failed": s["failed"],
                "success_rate": s["success_rate"],
                "total_time_seconds": s["total_time"],
                "total_cost_usd": s["total_cost"],
                "by_eval_method": s.get("by_eval_method", {}),
                "by_confidence": s.get("by_confidence", {}),
            }
            for s in summaries
        ],
        "metadata": {
            "total_tasks_in_benchmark": 68,
            "ground_truth_coverage": 7 / 68,
            "evaluation_method": "hybrid (JMESPath + LLM judge)",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate comparison reports")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Results directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/reports"), help="Output directory")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(args.results_dir)

    if not summaries:
        print("No results found. Run benchmark first.")
        return

    # Generate markdown report
    md_report = generate_markdown_report(summaries)
    md_path = args.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d')}.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Markdown report: {md_path}")

    # Generate JSON report
    json_report = generate_json_report(summaries)
    json_path = args.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d')}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report: {json_path}")

    # Print summary
    print(f"\nLoaded {len(summaries)} model results:")
    for s in summaries:
        print(f"  - {s['model']}: {s['passed']}/{s['total_tasks']} ({100*s['success_rate']:.1f}%)")


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] Report generator runs: `python generate_comparison_report.py --help`
- [ ] Generates empty report when no results: `python generate_comparison_report.py`

#### Manual Verification:
- [ ] After running benchmarks, reports show correct statistics
- [ ] Markdown renders correctly with tables
- [ ] JSON report has all expected fields

---

## Phase 6: Ground Truth Validation Pipeline [IMPLEMENTED]

### Overview
Create a pipeline to validate JMESPath queries against ground truth states (both manual and synthetic), measuring eval reliability.

### Changes Required:

#### 1. Ground Truth Validator
**File**: `multi-real/validate_ground_truth.py` ✅ Created

**Key Features:**
- Loads both manual and synthetic ground truth states
- Manual GT takes precedence (won't be overwritten by synthetic)
- Validates JMESPath queries against finish states
- Identifies: passed, failed, errors, and TODO placeholders
- Supports `--verbose` for detailed output, `--json` for machine-readable output
- `--no-synthetic` flag to only use manually verified ground truth

**Initial Validation Results (manual GT only):**
- 7 ground truth states loaded
- 25% pass rate (5/20 evals)
- Issues identified: task naming mismatches, query bugs
- TODO: Fix ground truth file naming to match task registry

### Success Criteria:

#### Automated Verification:
- [ ] Validator runs: `python validate_ground_truth.py`
- [ ] Verbose mode shows details: `python validate_ground_truth.py --verbose`

#### Manual Verification:
- [ ] Manual ground truth tasks validate (or issues are identified and logged)
- [ ] Failed queries are clearly reported with expected vs actual values
- [ ] Unverified (TODO) evals are flagged appropriately
- [ ] Synthetic ground truth states load correctly alongside manual ones

---

## Phase 7: Synthetic Ground Truth Generation ⭐ RECOMMENDED FIRST [IMPLEMENTED]

### Overview
Automatically generate validated ground truth states by running a model on all tasks, collecting successful finish states, and validating them with LLM judge. This provides 30-40 ground truth states with minimal manual work.

### Strategy: Use Model Results as Ground Truth

**The Insight:** When a model successfully completes a task, the finish state IS ground truth (if validated by LLM judge).

**Process:**
1. Run a capable model (GPT-4o, Claude Opus, or Gemini) on all 68 tasks
2. Collect finish states from runs that appear successful
3. Use LLM judge to validate: "Does this state indicate task success?"
4. Keep high-confidence states (LLM confidence ≥ 0.8) as synthetic ground truth
5. Save to `final_states/synthetic/`

**Expected Outcome:**
- ~40-50 tasks will complete successfully
- ~30-40 will pass LLM judge validation
- Combined with 7 manual states = **~35-45 total ground truth (50-65% coverage)**
- 0 hours manual work

### Changes Required:

#### 1. Synthetic Ground Truth Generator
**File**: `multi-real/generate_synthetic_ground_truth.py` ✅ Created

**Key Design Decisions:**
- **Finish State Format**: Captured via `_get_multi_app_finish_json()` from /finish endpoints. Format: `{"website_id": {"actionhistory": [...], "emails": [...], ...}, ...}`
- **Size Handling**: Finish states can be hundreds of thousands of lines. HybridValidator automatically truncates to 10k chars before LLM validation (hybrid_validator.py:139)
- **Validation Flow**:
  1. Run model on all tasks without hybrid eval (faster)
  2. Collect finish states from summary_info.json
  3. Validate each with LLM judge separately
  4. Save only high-confidence (≥0.8) states
- **Incremental Updates**: Skip tasks that already have synthetic GT unless --force-rerun
- **Storage**: Same format as manual ground truth in final_states/manual/

### Success Criteria:

#### Automated Verification:
- [ ] Script imports and runs: `uv run python generate_synthetic_ground_truth.py --help`
- [ ] Can load model config and tasks
- [ ] LLM validator initializes correctly

#### Manual Verification:
- [ ] Run on test subset (5 tasks) and verify states are collected
- [ ] Check LLM validation catches invalid states
- [ ] Verify synthetic GT files are saved correctly
- [ ] Run full generation and achieve ~30-40 validated states

### Expected Timeline:
- **Setup**: 30 minutes (implement script)
- **Execution**: 1-2 hours (run model on 68 tasks)
- **Validation**: Automatic (LLM judge during run)
- **Total**: ~2.5 hours, 0 manual work

### Next Steps After Phase 7:
1. Run Phase 6 validation with combined manual + synthetic GT (~35-45 states)
2. Measure JMESPath query accuracy
3. If accuracy is <80%, manually review flagged queries
4. Run full benchmark with validated queries

---

## Testing Strategy

### Unit Tests
- Task registry loads all 68 tasks correctly
- Model configs parse without errors
- JMESPath queries compile successfully
- Hybrid validator comparison logic works

### Integration Tests
- Run single task end-to-end with mock model
- Verify finish state capture from all websites
- Test hybrid evaluation fallback behavior
- Validate results serialization

### Manual Testing Steps
1. Run `validate_ground_truth.py` and fix any failing queries
2. Execute single task with each model config (headless=false) to verify browser behavior
3. Run full benchmark on one model and verify results directory structure
4. Generate comparison report and verify markdown renders correctly
5. Compare results between two models to ensure fairness

---

## Performance Considerations

- **Parallelization**: Consider adding `--workers N` flag for concurrent task execution (via Ray or multiprocessing)
- **Caching**: Cache browser initialization to reduce startup time between tasks
- **Cost tracking**: Monitor API costs per model to stay within budget
- **Timeouts**: Set reasonable timeouts (5 min per task) to handle stuck agents

---

## References

- Core harness: `src/agisdk/REAL/harness.py`
- Multi-app support: `src/agisdk/REAL/browsergym/webclones/base.py:284-317`
- Evaluation: `src/agisdk/REAL/browsergym/webclones/evaluate.py`
- Task config: `src/agisdk/REAL/browsergym/webclones/task_config.py`
- Agent abstraction: `src/agisdk/REAL/demo_agent/basic_agent.py`
