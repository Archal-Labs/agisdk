#!/usr/bin/env python3
"""
Re-evaluate benchmark results with updated queries.

This script re-runs evaluation on existing task results after queries have been fixed.
It's much faster than re-running the entire benchmark since it only re-evaluates
the existing env_state data.

Usage:
    # Re-evaluate all tasks
    uv run python multi-real/re_evaluate_results.py results/openai_cua_20260111_123456

    # Re-evaluate specific tasks
    uv run python multi-real/re_evaluate_results.py results/openai_cua_20260111_123456 --filter gomail-topwork

    # Save to new directory
    uv run python multi-real/re_evaluate_results.py results/openai_cua_20260111_123456 --output results/openai_cua_20260111_123456_reeval

The script will:
1. Load existing task results (env_state + model_response)
2. Load current task definitions (with updated queries)
3. Re-run WebCloneEvaluator on each task
4. Save updated results with new success/reward/eval_message
5. Generate comparison report showing changes
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig
from agisdk.REAL.logging import logger as rich_logger
from agisdk.REAL.tasks import all_tasks as tasks


def re_evaluate_task(task_result: dict[str, Any], task_config: TaskConfig) -> dict[str, Any]:
    """
    Re-evaluate a single task result with current task definition.

    Args:
        task_result: Original task result with env_state and model_response
        task_config: Current task configuration (may have updated queries)

    Returns:
        Updated task result with new evaluation
    """
    task_id = task_result["task_id"]
    env_state = task_result.get("env_state", {})
    model_response = task_result.get("model_response", "")

    # Run evaluation with current task config
    evaluator = WebCloneEvaluator(task_config)
    reward, _, eval_message, eval_details = evaluator.evaluate(
        env_state=env_state,
        model_response=model_response
    )

    # Create updated result
    updated = task_result.copy()
    updated["reward"] = reward
    updated["success"] = reward > 0
    updated["eval_message"] = eval_message
    updated["eval_details"] = eval_details
    updated["re_evaluated_at"] = datetime.now(timezone.utc).isoformat()

    return updated


def load_task_config(task_id: str) -> TaskConfig | None:
    """Load task config from current task definitions."""
    for task in tasks:
        if task["id"] == task_id:
            return TaskConfig(task_id, task.get("version", "v1"))
    return None


def re_evaluate_results(
    results_dir: Path,
    output_dir: Path | None = None,
    task_filter: str | None = None
) -> dict:
    """
    Re-evaluate all tasks in a results directory.

    Args:
        results_dir: Directory containing original benchmark results
        output_dir: Where to save re-evaluated results (default: results_dir + "_reeval")
        task_filter: Optional filter for task IDs (e.g., "gomail-topwork" matches all gomail-topwork-*)

    Returns:
        Dict with statistics and comparison
    """
    # Validate results directory
    tasks_dir = results_dir / "tasks"
    if not tasks_dir.exists():
        raise ValueError(f"Tasks directory not found: {tasks_dir}")

    # Set output directory
    if output_dir is None:
        output_dir = Path(str(results_dir) + "_reeval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tasks_dir = output_dir / "tasks"
    output_tasks_dir.mkdir(exist_ok=True)

    # Load original summary
    summary_file = results_dir / "summary.json"
    original_summary = {}
    if summary_file.exists():
        with open(summary_file) as f:
            original_summary = json.load(f)

    rich_logger.info(f"📁 Loading results from: {results_dir}")
    rich_logger.info(f"📁 Saving to: {output_dir}")
    print()

    # Find task files
    task_files = list(tasks_dir.glob("task_*.json"))
    if task_filter:
        task_files = [f for f in task_files if task_filter in f.stem]

    rich_logger.info(f"🔄 Re-evaluating {len(task_files)} tasks...")
    print()

    # Track statistics
    stats = {
        'total': 0,
        'original_success': 0,
        'new_success': 0,
        'improved': 0,
        'degraded': 0,
        'unchanged': 0,
        'tasks': []
    }

    re_evaluated_results = []

    for task_file in sorted(task_files):
        stats['total'] += 1

        # Load original result
        with open(task_file) as f:
            original_result = json.load(f)

        task_id = original_result["task_id"]
        original_success = original_result.get("success", False)

        # Load current task config
        task_config = load_task_config(task_id)
        if task_config is None:
            rich_logger.warning(f"⚠ {task_id}: Task definition not found, skipping")
            continue

        # Re-evaluate
        try:
            updated_result = re_evaluate_task(original_result, task_config)
            new_success = updated_result["success"]

            # Track changes
            if original_success:
                stats['original_success'] += 1
            if new_success:
                stats['new_success'] += 1

            if original_success and not new_success:
                status = "degraded"
                stats['degraded'] += 1
                icon = "⬇"
            elif not original_success and new_success:
                status = "improved"
                stats['improved'] += 1
                icon = "⬆"
            else:
                status = "unchanged"
                stats['unchanged'] += 1
                icon = "=" if new_success else "✗"

            rich_logger.info(
                f"{icon} {task_id}: "
                f"{'SUCCESS' if original_success else 'FAIL'} → "
                f"{'SUCCESS' if new_success else 'FAIL'} "
                f"(reward: {original_result.get('reward', 0)} → {updated_result['reward']})"
            )

            stats['tasks'].append({
                'task_id': task_id,
                'status': status,
                'original_success': original_success,
                'new_success': new_success,
                'original_reward': original_result.get('reward', 0),
                'new_reward': updated_result['reward'],
                'original_eval_message': original_result.get('eval_message', ''),
                'new_eval_message': updated_result['eval_message']
            })

            # Save updated result
            output_file = output_tasks_dir / f"task_{task_id}.json"
            with open(output_file, "w") as f:
                json.dump(updated_result, f, indent=2)

            re_evaluated_results.append(updated_result)

        except Exception as e:
            rich_logger.error(f"✗ {task_id}: Re-evaluation failed: {e}")
            stats['tasks'].append({
                'task_id': task_id,
                'status': 'error',
                'error': str(e)
            })

    # Generate summary
    print()
    print("=" * 80)
    print("RE-EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total tasks: {stats['total']}")
    print()
    print("Original Results:")
    print(f"  Success: {stats['original_success']}")
    print(f"  Fail: {stats['total'] - stats['original_success']}")
    print(f"  Success rate: {stats['original_success'] / stats['total'] * 100:.1f}%")
    print()
    print("New Results:")
    print(f"  Success: {stats['new_success']}")
    print(f"  Fail: {stats['total'] - stats['new_success']}")
    print(f"  Success rate: {stats['new_success'] / stats['total'] * 100:.1f}%")
    print()
    print("Changes:")
    print(f"  Improved: {stats['improved']} (FAIL → SUCCESS)")
    print(f"  Degraded: {stats['degraded']} (SUCCESS → FAIL)")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Net change: {stats['new_success'] - stats['original_success']:+d} tasks")
    print()

    # Show improved tasks
    if stats['improved'] > 0:
        print("Improved Tasks (query fixes worked!):")
        for task in stats['tasks']:
            if task.get('status') == 'improved':
                print(f"  ⬆ {task['task_id']}: {task['original_reward']} → {task['new_reward']}")
        print()

    # Show degraded tasks
    if stats['degraded'] > 0:
        print("Degraded Tasks (need investigation):")
        for task in stats['tasks']:
            if task.get('status') == 'degraded':
                print(f"  ⬇ {task['task_id']}: {task['original_reward']} → {task['new_reward']}")
        print()

    # Save statistics
    stats_file = output_dir / "re_evaluation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Statistics saved to: {stats_file}")

    # Save updated summary
    new_summary = {
        "run_name": original_summary.get("run_name", "unknown") + "_reeval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": original_summary.get("model", "unknown"),
        "original_results_dir": str(results_dir),
        "total_tasks": stats['total'],
        "successful_tasks": stats['new_success'],
        "failed_tasks": stats['total'] - stats['new_success'],
        "success_rate": stats['new_success'] / stats['total'] * 100 if stats['total'] > 0 else 0,
        "original_success_rate": stats['original_success'] / stats['total'] * 100 if stats['total'] > 0 else 0,
        "improved_tasks": stats['improved'],
        "degraded_tasks": stats['degraded'],
        "avg_time": original_summary.get("avg_time", 0),
        "total_time": original_summary.get("total_time", 0),
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(new_summary, f, indent=2)
    print(f"📄 Summary saved to: {summary_file}")

    # Save full results
    results_file = output_dir / "results.json"
    full_results = new_summary.copy()
    full_results["tasks"] = re_evaluated_results
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"📄 Full results saved to: {results_file}")

    return stats


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Re-evaluate benchmark results with updated queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-evaluate all tasks
  uv run python multi-real/re_evaluate_results.py results/openai_cua_20260111_123456

  # Re-evaluate specific tasks
  uv run python multi-real/re_evaluate_results.py results/openai_cua_20260111_123456 --filter gomail-topwork

  # Save to custom directory
  uv run python multi-real/re_evaluate_results.py results/openai_cua_20260111_123456 --output results/custom_reeval
        """
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing original benchmark results"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: results_dir + '_reeval')"
    )
    parser.add_argument(
        "-f", "--filter",
        default=None,
        help="Filter tasks by ID substring (e.g., 'gomail-topwork')"
    )

    args = parser.parse_args()

    # Validate results directory exists
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    try:
        re_evaluate_results(args.results_dir, args.output, args.filter)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
