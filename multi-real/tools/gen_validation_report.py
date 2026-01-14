#!/usr/bin/env python3
"""
Generate a comprehensive validation report for Multi-REAL verifiers.

Combines data from:
- Schema discovery (docs/schemas.json)
- Query analysis (docs/query_analysis.json)
- Fill expected values (provenance_report.json)
- Direct validation against ground truth

INCREMENTAL DESIGN:
- Re-run as you add more ground truth or fix queries
- Shows progress over time if previous reports exist
- Outputs both markdown (human) and JSON (machine) formats

Usage:
    uv run python multi-real/tools/gen_validation_report.py
    uv run python multi-real/tools/gen_validation_report.py --output-dir reports/
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jmespath

BASE_DIR = Path(__file__).parent.parent  # multi-real/
TASKS_DIR = BASE_DIR / "tasks"
GT_DIR = BASE_DIR / "final_states" / "manual"
DOCS_DIR = BASE_DIR / "docs"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_todos(tasks_dir: Path) -> dict:
    """Count TODO expected values across all tasks."""
    total_evals = 0
    todo_evals = 0
    complete_evals = 0
    by_task = {}

    for task_file in sorted(tasks_dir.glob("*.json")):
        with open(task_file, encoding="utf-8") as f:
            task = json.load(f)

        task_total = 0
        task_todos = 0
        for e in task.get("evals", []):
            if e.get("type") == "jmespath":
                task_total += 1
                total_evals += 1
                ev = e.get("expected_value")
                if ev is None or (isinstance(ev, str) and "TODO" in ev):
                    task_todos += 1
                    todo_evals += 1
                else:
                    complete_evals += 1

        by_task[task["id"]] = {"total": task_total, "todos": task_todos}

    return {
        "total_evals": total_evals,
        "todo_evals": todo_evals,
        "complete_evals": complete_evals,
        "by_task": by_task,
    }


def validate_against_gt(tasks_dir: Path, gt_dir: Path) -> dict:
    """Validate queries against available ground truth."""
    results = {"passed": 0, "failed": 0, "no_gt": 0, "by_task": {}}

    gt_files = {f.stem: f for f in gt_dir.glob("*.json")}

    for task_file in sorted(tasks_dir.glob("*.json")):
        with open(task_file, encoding="utf-8") as f:
            task = json.load(f)

        task_id = task["id"]

        # Find matching GT - require exact task_id match only
        gt_file = gt_files.get(task_id)

        if not gt_file:
            results["no_gt"] += 1
            results["by_task"][task_id] = {"status": "no_gt", "passed": 0, "failed": 0}
            continue

        with open(gt_file, encoding="utf-8") as f:
            gt_data = json.load(f)

        task_passed = 0
        task_failed = 0
        for e in task.get("evals", []):
            if e.get("type") != "jmespath":
                continue
            query = e.get("query", "")
            try:
                result = jmespath.search(query, gt_data)
                if result is not None:
                    task_passed += 1
                    results["passed"] += 1
                else:
                    task_failed += 1
                    results["failed"] += 1
            except Exception:
                task_failed += 1
                results["failed"] += 1

        results["by_task"][task_id] = {
            "status": "validated",
            "gt_file": gt_file.name,
            "passed": task_passed,
            "failed": task_failed,
        }

    return results


def generate_report(output_dir: Path) -> dict:
    """Generate the full validation report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {},
        "coverage": {},
        "query_health": {},
        "expected_values": {},
        "ground_truth_validation": {},
        "action_items": [],
    }

    # Load existing analysis files
    schemas = load_json(DOCS_DIR / "schemas.json")
    query_analysis = load_json(DOCS_DIR / "query_analysis.json")
    provenance = load_json(BASE_DIR / "provenance_report.json")

    # Count tasks
    task_files = list(TASKS_DIR.glob("*.json"))
    report["summary"]["total_tasks"] = len(task_files)

    # Coverage from schemas
    if schemas and "coverage" in schemas:
        cov = schemas["coverage"]
        report["coverage"] = {
            "tasks_with_gt": cov.get("tasks_with_gt", 0),
            "tasks_without_gt": cov.get("tasks_without_gt", 0),
            "gt_percentage": round(100 * cov.get("tasks_with_gt", 0) / len(task_files), 1) if task_files else 0,
            "apps_with_schema": cov.get("apps_with_schema", 0),
            "apps_missing_schema": cov.get("apps_missing_schema", []),
        }

    # Query health from analysis
    if query_analysis and "summary" in query_analysis:
        qa = query_analysis["summary"]
        report["query_health"] = {
            "total_queries": qa.get("total_queries", 0),
            "valid_queries": qa.get("valid_queries", 0),
            "invalid_queries": qa.get("invalid_queries", 0),
            "validity_percentage": round(100 * qa.get("valid_queries", 0) / qa.get("total_queries", 1), 1),
        }

    # Expected values
    ev_stats = count_todos(TASKS_DIR)
    report["expected_values"] = {
        "total_evals": ev_stats["total_evals"],
        "complete": ev_stats["complete_evals"],
        "todo": ev_stats["todo_evals"],
        "completion_percentage": round(100 * ev_stats["complete_evals"] / ev_stats["total_evals"], 1) if ev_stats["total_evals"] else 0,
    }

    # Ground truth validation
    gt_results = validate_against_gt(TASKS_DIR, GT_DIR)
    total_validated = gt_results["passed"] + gt_results["failed"]
    report["ground_truth_validation"] = {
        "queries_tested": total_validated,
        "passed": gt_results["passed"],
        "failed": gt_results["failed"],
        "pass_rate": round(100 * gt_results["passed"] / total_validated, 1) if total_validated else 0,
        "tasks_without_gt": gt_results["no_gt"],
    }

    # Generate action items
    if report["coverage"].get("apps_missing_schema"):
        report["action_items"].append({
            "priority": "high",
            "action": f"Collect ground truth for apps: {', '.join(report['coverage']['apps_missing_schema'])}",
            "impact": f"{len(report['coverage']['apps_missing_schema'])} apps have no schema knowledge",
        })

    if report["expected_values"]["todo"] > 0:
        report["action_items"].append({
            "priority": "medium",
            "action": f"Fill {report['expected_values']['todo']} TODO expected values",
            "impact": "Required for automated evaluation",
        })

    if report["query_health"].get("invalid_queries", 0) > 0:
        report["action_items"].append({
            "priority": "medium",
            "action": f"Fix {report['query_health']['invalid_queries']} invalid queries",
            "impact": "Queries may fail during evaluation",
        })

    if gt_results["failed"] > 0:
        report["action_items"].append({
            "priority": "high",
            "action": f"Investigate {gt_results['failed']} queries failing against ground truth",
            "impact": "These queries won't work during evaluation",
        })

    # Summary
    report["summary"].update({
        "gt_coverage": f"{report['coverage'].get('gt_percentage', 0)}%",
        "query_validity": f"{report['query_health'].get('validity_percentage', 0)}%",
        "expected_values_complete": f"{report['expected_values'].get('completion_percentage', 0)}%",
        "gt_pass_rate": f"{report['ground_truth_validation'].get('pass_rate', 0)}%",
        "action_items": len(report["action_items"]),
    })

    return report


def generate_markdown(report: dict) -> str:
    """Generate markdown report."""
    lines = [
        "# Multi-REAL Verifier Validation Report",
        "",
        f"*Generated: {report['generated_at'][:19].replace('T', ' ')} UTC*",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Tasks | {report['summary'].get('total_tasks', 0)} |",
        f"| Ground Truth Coverage | {report['summary'].get('gt_coverage', '0%')} |",
        f"| Query Validity | {report['summary'].get('query_validity', '0%')} |",
        f"| Expected Values Complete | {report['summary'].get('expected_values_complete', '0%')} |",
        f"| GT Validation Pass Rate | {report['summary'].get('gt_pass_rate', '0%')} |",
        f"| Action Items | {report['summary'].get('action_items', 0)} |",
        "",
    ]

    # Coverage details
    cov = report.get("coverage", {})
    lines.extend([
        "## Coverage",
        "",
        f"- Tasks with ground truth: **{cov.get('tasks_with_gt', 0)}** / {cov.get('tasks_with_gt', 0) + cov.get('tasks_without_gt', 0)}",
        f"- Apps with schema knowledge: **{cov.get('apps_with_schema', 0)}**",
    ])
    if cov.get("apps_missing_schema"):
        lines.append(f"- Apps missing schema: `{', '.join(cov['apps_missing_schema'])}`")
    lines.append("")

    # Query health
    qh = report.get("query_health", {})
    lines.extend([
        "## Query Health",
        "",
        f"- Total queries: {qh.get('total_queries', 0)}",
        f"- Valid (schema-correct): {qh.get('valid_queries', 0)} ({qh.get('validity_percentage', 0)}%)",
        f"- Invalid (schema issues): {qh.get('invalid_queries', 0)}",
        "",
    ])

    # Expected values
    ev = report.get("expected_values", {})
    lines.extend([
        "## Expected Values",
        "",
        f"- Complete: {ev.get('complete', 0)} / {ev.get('total_evals', 0)}",
        f"- TODO (need filling): {ev.get('todo', 0)}",
        "",
    ])

    # GT validation
    gt = report.get("ground_truth_validation", {})
    lines.extend([
        "## Ground Truth Validation",
        "",
        f"- Queries tested: {gt.get('queries_tested', 0)}",
        f"- Passed: {gt.get('passed', 0)}",
        f"- Failed: {gt.get('failed', 0)}",
        f"- Pass rate: {gt.get('pass_rate', 0)}%",
        "",
    ])

    # Action items
    if report.get("action_items"):
        lines.extend([
            "## Action Items",
            "",
        ])
        for item in report["action_items"]:
            priority = item.get("priority", "medium").upper()
            lines.append(f"### [{priority}] {item.get('action', '')}")
            lines.append(f"*Impact: {item.get('impact', '')}*")
            lines.append("")

    lines.extend([
        "---",
        "",
        "*Re-run `tools/gen_validation_report.py` after making changes to see updated metrics.*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate verifier validation report")
    parser.add_argument("--output-dir", type=Path, default=DOCS_DIR, help="Output directory")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating validation report...")

    report = generate_report(args.output_dir)

    # Save JSON
    json_path = args.output_dir / "validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report: {json_path}")

    # Save markdown
    md_path = args.output_dir / "validation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown(report))
    print(f"Markdown report: {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 60)
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")

    if report.get("action_items"):
        print(f"\n{len(report['action_items'])} action items identified.")

    return 0


if __name__ == "__main__":
    exit(main())
