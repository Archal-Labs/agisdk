#!/usr/bin/env python3
"""
Agent-Assisted Capture Pipeline for Multi-Real Benchmark Validation.

This pipeline:
1. Loads all task configs from multi-real/tasks/
2. Runs agent(s) on each task (supports multi-model with agreement checking)
3. Captures the finish JSON after each run
4. Compares finish JSONs across models and flags disagreements
5. Saves verified finish JSONs to final_states/
6. Logs results for human review

Usage:
    # Single model
    uv run python multi-real/capture_pipeline.py --model gpt-4o --max-steps 30

    # Multi-model with agreement checking
    uv run python multi-real/capture_pipeline.py \
        --models claude-sonnet-3.7,gpt-4o,gemini-2.0 \
        --agreement-threshold 2 \
        --max-steps 30

    # Single task
    uv run python multi-real/capture_pipeline.py --task dashdish-gomail-1 --headless=false
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agisdk import REAL
from agisdk.REAL.browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from agisdk.REAL.demo_agent.basic_agent import DemoAgentArgs


def extract_key_entities(finish_json: dict) -> dict:
    """
    Extract key entities from finish JSON for comparison.
    Returns counts of major entities (bookings, emails, events, etc).
    """
    from collections import defaultdict

    entities = defaultdict(lambda: defaultdict(int))

    for app_name, app_data in finish_json.items():
        if not isinstance(app_data, dict):
            continue

        differences = app_data.get("differences", {})

        # Count bookings
        bookings = differences.get("bookings", {}).get("added", [])
        if bookings:
            entities[app_name]["bookings"] = len(bookings)

        # Count emails
        emails_added = differences.get("emails", {}).get("added", [])
        emails_sent = differences.get("emails", {}).get("sent", [])
        if emails_added:
            entities[app_name]["emails_added"] = len(emails_added)
        if emails_sent:
            entities[app_name]["emails_sent"] = len(emails_sent)

        # Count events
        events = differences.get("events", {}).get("added")
        if events:
            if isinstance(events, dict):
                entities[app_name]["events"] = len(events)
            elif isinstance(events, list):
                entities[app_name]["events"] = len(events)

        # Count flights
        flights = differences.get("bookedFlights", [])
        if flights:
            entities[app_name]["flights"] = len(flights)

        # Count food orders
        food_orders = differences.get("foodOrders", {}).get("added", [])
        if food_orders:
            entities[app_name]["food_orders"] = len(food_orders)

    return dict(entities)


def compare_finish_jsons(finish_jsons: list[dict]) -> tuple[bool, float]:
    """
    Compare multiple finish JSONs to check agreement.

    Returns:
        (has_agreement, agreement_score)
        has_agreement: True if JSONs are sufficiently similar
        agreement_score: 0-1 score indicating similarity
    """
    if len(finish_jsons) < 2:
        return True, 1.0

    # Extract entities from each
    entity_lists = [extract_key_entities(fj) for fj in finish_jsons]

    # Compare entity counts
    # Two finish JSONs agree if they have the same entity counts
    agreements = []

    for i in range(len(entity_lists)):
        for j in range(i + 1, len(entity_lists)):
            e1 = entity_lists[i]
            e2 = entity_lists[j]

            # Get all apps from both
            all_apps = set(e1.keys()) | set(e2.keys())

            matches = 0
            total = 0

            for app in all_apps:
                app1 = e1.get(app, {})
                app2 = e2.get(app, {})

                all_types = set(app1.keys()) | set(app2.keys())

                for entity_type in all_types:
                    count1 = app1.get(entity_type, 0)
                    count2 = app2.get(entity_type, 0)

                    total += 1
                    if count1 == count2:
                        matches += 1

            agreement = matches / total if total > 0 else 1.0
            agreements.append(agreement)

    avg_agreement = sum(agreements) / len(agreements) if agreements else 1.0
    has_agreement = avg_agreement >= 0.8  # 80% similarity threshold

    return has_agreement, avg_agreement


def run_multi_model_task(
    task_config: dict,
    models: list[str],
    agreement_threshold: int = 2,
    max_steps: int = 30,
    headless: bool = True,
    results_dir: str = "./multi-real/results",
) -> dict:
    """
    Run task with multiple models and check agreement.

    Args:
        task_config: Task configuration
        models: List of model names to try
        agreement_threshold: Minimum number of models that must agree
        max_steps: Max steps per run
        headless: Run headless
        results_dir: Results directory

    Returns:
        dict with keys: task_id, success, model_results, agreed_finish_json,
                       has_agreement, agreement_score, needs_review
    """
    task_id = task_config["id"]

    result = {
        "task_id": task_id,
        "model_results": [],
        "agreed_finish_json": None,
        "has_agreement": False,
        "agreement_score": 0.0,
        "needs_review": False,
        "timestamp": datetime.now().isoformat(),
    }

    # Run with each model
    print(f"  Running with {len(models)} model(s): {', '.join(models)}")

    for model in models:
        print(f"    → {model}...", end=" ")
        model_result = run_single_task(
            task_config=task_config,
            model=model,
            max_steps=max_steps,
            headless=headless,
            results_dir=results_dir,
        )
        model_result["model"] = model
        result["model_results"].append(model_result)

        status = "✓" if model_result["success"] else "✗"
        print(f"{status} (reward={model_result['reward']})")

    # Check agreement among successful runs
    successful_runs = [r for r in result["model_results"] if r.get("finish_json")]

    if len(successful_runs) == 0:
        result["needs_review"] = True
        print("  ⚠ No successful runs - needs review")
        return result

    if len(successful_runs) == 1:
        # Only one model succeeded - accept but flag for review
        result["agreed_finish_json"] = successful_runs[0]["finish_json"]
        result["has_agreement"] = False
        result["agreement_score"] = 1.0
        result["needs_review"] = True
        print("  ⚠ Only one model succeeded - needs review")
        return result

    # Multiple successful runs - check agreement
    finish_jsons = [r["finish_json"] for r in successful_runs]
    has_agreement, agreement_score = compare_finish_jsons(finish_jsons)

    result["has_agreement"] = has_agreement
    result["agreement_score"] = agreement_score

    if has_agreement and len(successful_runs) >= agreement_threshold:
        # Accept the first successful run's finish JSON
        result["agreed_finish_json"] = successful_runs[0]["finish_json"]
        result["needs_review"] = False
        print(f"  ✓ Agreement: {agreement_score:.1%} ({len(successful_runs)}/{len(models)} models)")
    else:
        result["needs_review"] = True
        print(f"  ⚠ Disagreement: {agreement_score:.1%} - needs review")

    return result


def load_task_configs(tasks_dir: str) -> list[dict]:
    """Load all task configurations from the tasks directory."""
    tasks = []
    tasks_path = Path(tasks_dir)

    for task_file in sorted(tasks_path.glob("*.json")):
        if task_file.name.endswith(".backup"):
            continue
        try:
            with open(task_file) as f:
                task = json.load(f)
                task["_file_path"] = str(task_file)
                tasks.append(task)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to load {task_file}: {e}")

    return tasks


def run_single_task(
    task_config: dict,
    model: str = "gpt-4o",
    max_steps: int = 30,
    headless: bool = True,
    results_dir: str = "./multi-real/results",
) -> dict:
    """
    Run an agent on a single task and capture results.

    Returns:
        dict with keys: task_id, success, reward, finish_json, exp_dir, error
    """
    task_id = task_config["id"]
    task_file = task_config["_file_path"]

    result = {
        "task_id": task_id,
        "task_file": task_file,
        "success": False,
        "reward": 0,
        "finish_json": None,
        "exp_dir": None,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Create agent args
        agent_args = DemoAgentArgs(
            model_name=model,
            chat_mode=False,
            demo_mode="default",
            use_html=False,
            use_axtree=True,
            use_screenshot=True,
        )

        # Create env args for multi-app task
        # Multi-app tasks need special handling - use the task file directly
        env_args = EnvArgs(
            task_name=f"multi-real.{task_id}",
            task_seed=None,
            max_steps=max_steps,
            headless=headless,
        )

        # Create and run experiment
        exp_args = ExpArgs(env_args=env_args, agent_args=agent_args)
        exp_args.prepare(results_dir)

        # Store exp_dir for later
        result["exp_dir"] = str(exp_args.exp_dir)

        # Run the experiment
        exp_args.run()

        # Get results
        exp_result = get_exp_result(exp_args.exp_dir)
        exp_record = exp_result.get_exp_record()

        result["reward"] = exp_record.get("cum_reward", 0)
        result["success"] = result["reward"] == 1

        # Try to load finish JSON from the experiment
        finish_json_path = exp_args.exp_dir / "finish_state.json"
        if finish_json_path.exists():
            with open(finish_json_path) as f:
                result["finish_json"] = json.load(f)

    except Exception as e:
        result["error"] = str(e)
        print(f"Error running task {task_id}: {e}")

    return result


def save_finish_json(result: dict, output_dir: str) -> bool:
    """Save finish JSON to the output directory if available."""
    if not result.get("finish_json"):
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    task_id = result["task_id"]
    finish_path = output_path / f"{task_id}.json"

    with open(finish_path, "w") as f:
        json.dump(result["finish_json"], f, indent=2)

    print(f"Saved finish JSON to {finish_path}")
    return True


def write_review_log(results: list[dict], log_path: str):
    """Write results to a CSV file for human review."""
    fieldnames = [
        "task_id",
        "success",
        "reward",
        "has_finish_json",
        "models_used",
        "has_agreement",
        "agreement_score",
        "error",
        "exp_dir",
        "timestamp",
        "needs_review",
    ]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Handle both single-model and multi-model results
            if "model_results" in result:
                # Multi-model result
                models_used = ",".join([r.get("model", "unknown") for r in result.get("model_results", [])])
                success = any(r.get("success") for r in result.get("model_results", []))
                rewards = [r.get("reward", 0) for r in result.get("model_results", [])]
                reward = max(rewards) if rewards else 0
                has_finish_json = bool(result.get("agreed_finish_json"))
                has_agreement = result.get("has_agreement", False)
                agreement_score = result.get("agreement_score", 0.0)
                errors = [r.get("error", "") for r in result.get("model_results", []) if r.get("error")]
                error = "; ".join(errors) if errors else ""
                exp_dirs = [r.get("exp_dir", "") for r in result.get("model_results", [])]
                exp_dir = "; ".join(exp_dirs) if exp_dirs else ""
            else:
                # Single-model result (legacy)
                models_used = result.get("model", "unknown")
                success = result.get("success", False)
                reward = result.get("reward", 0)
                has_finish_json = bool(result.get("finish_json"))
                has_agreement = True
                agreement_score = 1.0
                error = result.get("error", "")
                exp_dir = result.get("exp_dir", "")

            row = {
                "task_id": result["task_id"],
                "success": success,
                "reward": reward,
                "has_finish_json": has_finish_json,
                "models_used": models_used,
                "has_agreement": has_agreement,
                "agreement_score": f"{agreement_score:.2f}",
                "error": error,
                "exp_dir": exp_dir,
                "timestamp": result.get("timestamp", ""),
                "needs_review": result.get("needs_review", not success or not has_finish_json),
            }
            writer.writerow(row)

    print(f"Review log written to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Agent-Assisted Capture Pipeline for Multi-Real Benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model to use (e.g., gpt-4o). Use --models for multi-model capture.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models for multi-model capture (e.g., claude-sonnet-3.7,gpt-4o,gemini-2.0)",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=int,
        default=2,
        help="Minimum number of models that must agree (default: 2)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum steps per task (default: 30)",
    )
    parser.add_argument(
        "--headless",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Run browser in headless mode (default: true)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a specific task by ID (e.g., dashdish-gomail-1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="multi-real/tasks",
        help="Directory containing task JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="multi-real/final_states",
        help="Directory to save finish JSONs",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="multi-real/results",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--review-log",
        type=str,
        default="multi-real/capture_review.csv",
        help="Path for the review log CSV",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have finish JSONs",
    )

    args = parser.parse_args()

    # Determine models to use
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        print(f"Using multi-model capture with: {', '.join(models)}")
        print(f"Agreement threshold: {args.agreement_threshold}")
        use_multi_model = True
    elif args.model:
        models = [args.model]
        use_multi_model = False
    else:
        # Default to single model
        models = ["gpt-4o"]
        use_multi_model = False
        print("No model specified, using default: gpt-4o")

    # Load task configs
    print(f"Loading tasks from {args.tasks_dir}...")
    tasks = load_task_configs(args.tasks_dir)
    print(f"Found {len(tasks)} tasks")

    # Filter to specific task if requested
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"Task {args.task} not found")
            return
        print(f"Running single task: {args.task}")
    elif not args.all:
        print("Use --task <task_id> to run a specific task, or --all to run all tasks")
        print("\nAvailable tasks:")
        for t in tasks[:10]:
            print(f"  - {t['id']}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    # Skip existing if requested
    if args.skip_existing:
        existing = set()
        output_path = Path(args.output_dir)
        if output_path.exists():
            for f in output_path.glob("*.json"):
                existing.add(f.stem)

        original_count = len(tasks)
        tasks = [t for t in tasks if t["id"] not in existing]
        print(f"Skipping {original_count - len(tasks)} tasks with existing finish JSONs")

    # Run tasks
    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Running task: {task['id']}")

        if use_multi_model:
            # Run with multiple models
            result = run_multi_model_task(
                task_config=task,
                models=models,
                agreement_threshold=args.agreement_threshold,
                max_steps=args.max_steps,
                headless=args.headless,
                results_dir=args.results_dir,
            )

            results.append(result)

            # Save agreed finish JSON if available
            if result.get("agreed_finish_json"):
                # Create a result-like dict for save_finish_json
                save_result = {
                    "task_id": result["task_id"],
                    "finish_json": result["agreed_finish_json"]
                }
                save_finish_json(save_result, args.output_dir)

        else:
            # Run with single model
            result = run_single_task(
                task_config=task,
                model=models[0],
                max_steps=args.max_steps,
                headless=args.headless,
                results_dir=args.results_dir,
            )

            results.append(result)

            # Save finish JSON if successful
            if result.get("finish_json"):
                save_finish_json(result, args.output_dir)

            # Print status
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"  Result: {status} (reward={result['reward']})")
            if result.get("error"):
                print(f"  Error: {result['error']}")

    # Write review log
    write_review_log(results, args.review_log)

    # Print summary
    print("\n" + "=" * 60)
    print("CAPTURE PIPELINE SUMMARY")
    print("=" * 60)

    if use_multi_model:
        # Multi-model summary
        with_finish = sum(1 for r in results if r.get("agreed_finish_json"))
        needs_review = sum(1 for r in results if r.get("needs_review"))
        has_agreement = sum(1 for r in results if r.get("has_agreement"))

        print(f"Tasks run: {len(results)}")
        print(f"With agreed finish JSON: {with_finish}/{len(results)}")
        print(f"With model agreement: {has_agreement}/{len(results)}")
        print(f"Needs human review: {needs_review}/{len(results)}")

        if len(results) > 0:
            avg_agreement = sum(r.get("agreement_score", 0) for r in results) / len(results)
            print(f"Average agreement score: {avg_agreement:.1%}")

    else:
        # Single-model summary
        successful = sum(1 for r in results if r.get("success"))
        with_finish = sum(1 for r in results if r.get("finish_json"))
        print(f"Tasks run: {len(results)}")
        print(f"Successful: {successful}/{len(results)}")
        print(f"With finish JSON: {with_finish}/{len(results)}")

    print(f"Review log: {args.review_log}")
    print(f"Finish JSONs: {args.output_dir}/")


if __name__ == "__main__":
    main()
