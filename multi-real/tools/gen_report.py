#!/usr/bin/env python3
"""
Generate comparison reports across models for Multi-REAL benchmark.

Loads summary files from results/aggregated/ and generates:
- Markdown comparison report with tables
- JSON structured report for programmatic use

Usage:
    # Generate reports from all available summaries
    uv run python multi-real/generate_comparison_report.py

    # Specify custom directories
    uv run python multi-real/generate_comparison_report.py \
        --results-dir results \
        --output-dir results/reports
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


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
        "- 7 tasks have validated ground truth states (~10% coverage)",
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


def analyze_disagreements(summaries: list[dict]) -> list[dict]:
    """Find tasks where models disagree on success."""
    if len(summaries) < 2:
        return []

    # Build task -> model -> success mapping
    task_results: dict[str, dict[str, bool]] = defaultdict(dict)
    for s in summaries:
        for task in s.get("tasks", []):
            task_results[task["task_id"]][s["model"]] = task.get("success", False)

    disagreements = []
    for task_id, results in task_results.items():
        successes = list(results.values())
        if len(set(successes)) > 1:  # Models disagree
            disagreements.append({
                "task_id": task_id,
                "results": results,
                "passed_by": [m for m, s in results.items() if s],
                "failed_by": [m for m, s in results.items() if not s],
            })

    return disagreements


def main():
    parser = argparse.ArgumentParser(description="Generate comparison reports")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Results directory (default: multi-real/results)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "reports",
        help="Output directory (default: multi-real/results/reports)"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(args.results_dir)

    if not summaries:
        print(f"No results found in {args.results_dir / 'aggregated'}.")
        print("Run the benchmark first with: uv run python run_benchmark.py --model <model>")
        return 1

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

    # Analyze disagreements if multiple models
    if len(summaries) > 1:
        disagreements = analyze_disagreements(summaries)
        if disagreements:
            disagreements_path = args.output_dir / f"disagreements_{datetime.now().strftime('%Y%m%d')}.json"
            with open(disagreements_path, "w") as f:
                json.dump(disagreements, f, indent=2)
            print(f"Disagreements report: {disagreements_path}")

    # Print summary
    print(f"\nLoaded {len(summaries)} model results:")
    for s in sorted(summaries, key=lambda x: x.get("success_rate", 0), reverse=True):
        print(f"  - {s['model']}: {s['passed']}/{s['total_tasks']} ({100*s['success_rate']:.1f}%)")

    if len(summaries) > 1:
        disagreements = analyze_disagreements(summaries)
        print(f"\nTasks with disagreement: {len(disagreements)}")

    return 0


if __name__ == "__main__":
    exit(main())
