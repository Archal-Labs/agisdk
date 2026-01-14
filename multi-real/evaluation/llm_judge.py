#!/usr/bin/env python3
"""
LLM-as-Judge Validation Tool

Reviews finish JSONs to determine if they satisfy the task goal.
Uses Claude Opus or GPT-4o for validation.

Usage:
    # Single task validation
    uv run python multi-real/llm_judge.py \
        --task multi-real/tasks/dashdish-gomail-1.json \
        --finish-json multi-real/final_states/dashdish-gomail-1.json \
        --model claude-opus-4

    # Batch validation
    uv run python multi-real/llm_judge.py \
        --finish-jsons "multi-real/final_states/automated/*.json" \
        --model claude-opus-4 \
        --output llm_judge_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import glob


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_task_goal(task_config: Dict[str, Any]) -> str:
    """Extract the human-readable goal from task config."""
    # Task configs have a 'goal' field
    return task_config.get("goal", "")


def create_validation_prompt(task_goal: str, finish_json: Dict[str, Any]) -> str:
    """Create prompt for LLM judge to validate finish state."""

    # Extract key state changes
    state_summary = []

    for app_name, app_data in finish_json.items():
        if not isinstance(app_data, dict):
            continue

        # Check differences
        differences = app_data.get("differences", {})
        if differences:
            state_summary.append(f"\n**{app_name.upper()} Changes:**")

            # Bookings
            bookings = differences.get("bookings", {}).get("added", [])
            if bookings:
                state_summary.append(f"  - Added {len(bookings)} booking(s)")

            # Emails
            emails_added = differences.get("emails", {}).get("added", [])
            emails_sent = differences.get("emails", {}).get("sent", [])
            if emails_added:
                state_summary.append(f"  - Added {len(emails_added)} email(s)")
            if emails_sent:
                state_summary.append(f"  - Sent {len(emails_sent)} email(s)")

            # Events
            events = differences.get("events", {}).get("added")
            if events:
                if isinstance(events, dict):
                    state_summary.append(f"  - Added {len(events)} event(s)")
                elif isinstance(events, list):
                    state_summary.append(f"  - Added {len(events)} event(s)")

            # Flights
            flights = differences.get("bookedFlights", [])
            if flights:
                state_summary.append(f"  - Booked {len(flights)} flight(s)")

            # Food orders
            food_orders = differences.get("foodOrders", {}).get("added", [])
            if food_orders:
                state_summary.append(f"  - Ordered {len(food_orders)} food order(s)")

            # Jobs
            jobs = differences.get("jobs", {})
            if jobs:
                state_summary.append(f"  - Job-related changes detected")

    state_summary_text = "\n".join(state_summary) if state_summary else "No significant state changes detected."

    prompt = f"""You are validating whether a computer-use agent successfully completed a task.

**Task Goal:**
{task_goal}

**Observed State Changes:**
{state_summary_text}

**Your Task:**
Determine if the observed state changes indicate successful completion of the task goal.

**Evaluation Criteria:**
1. Are the required actions completed? (e.g., if task requires booking a flight, is there a booked flight?)
2. Do the changes align with the task requirements? (e.g., correct restaurant, correct dates)
3. Are all multi-step requirements satisfied? (e.g., if task requires "book flight AND send email", both must be done)

**Response Format:**
You must respond with EXACTLY one of these verdicts on the first line:
- SUCCESS: The task goal is clearly satisfied by the state changes
- FAILURE: The task goal is not satisfied or only partially satisfied
- UNCERTAIN: Insufficient information to determine success (edge case)

Then on subsequent lines, provide a brief explanation (2-3 sentences) of your reasoning.

**Example Response:**
SUCCESS
The agent successfully booked a restaurant at OpenDining and sent a confirmation email via GoMail containing the booking details. Both required actions from the task goal are completed.
"""

    return prompt


def call_llm_judge(prompt: str, model: str = "claude-opus-4") -> tuple[str, str]:
    """
    Call LLM to judge the finish state.

    Returns:
        (verdict, explanation) where verdict is SUCCESS, FAILURE, or UNCERTAIN
    """

    # Try to import anthropic
    try:
        import anthropic

        if "claude" in model.lower():
            client = anthropic.Anthropic()

            message = client.messages.create(
                model=model if "claude" in model else "claude-opus-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Parse verdict from first line
            lines = response_text.strip().split("\n")
            verdict = lines[0].strip().upper()
            explanation = "\n".join(lines[1:]).strip()

            # Normalize verdict
            if "SUCCESS" in verdict:
                verdict = "SUCCESS"
            elif "FAILURE" in verdict or "FAIL" in verdict:
                verdict = "FAILURE"
            else:
                verdict = "UNCERTAIN"

            return verdict, explanation

    except ImportError:
        pass

    # Try OpenAI
    try:
        import openai

        if "gpt" in model.lower():
            client = openai.OpenAI()

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )

            response_text = response.choices[0].message.content

            # Parse verdict
            lines = response_text.strip().split("\n")
            verdict = lines[0].strip().upper()
            explanation = "\n".join(lines[1:]).strip()

            # Normalize
            if "SUCCESS" in verdict:
                verdict = "SUCCESS"
            elif "FAILURE" in verdict or "FAIL" in verdict:
                verdict = "FAILURE"
            else:
                verdict = "UNCERTAIN"

            return verdict, explanation

    except ImportError:
        pass

    # Fallback: structural check only
    print("Warning: No LLM API available. Using structural checks only.", file=sys.stderr)
    return "UNCERTAIN", "LLM API not available. Manual review required."


def validate_single_task(
    task_path: Path,
    finish_json_path: Path,
    model: str = "claude-opus-4"
) -> Dict[str, Any]:
    """
    Validate a single task's finish JSON.

    Returns:
        {
            "task": task_id,
            "verdict": "SUCCESS" | "FAILURE" | "UNCERTAIN",
            "explanation": str,
            "model": str
        }
    """

    task_config = load_json(task_path)
    finish_json = load_json(finish_json_path)

    task_id = task_config.get("id", task_path.stem)
    task_goal = extract_task_goal(task_config)

    if not task_goal:
        return {
            "task": task_id,
            "verdict": "UNCERTAIN",
            "explanation": "Task goal not found in config",
            "model": model
        }

    prompt = create_validation_prompt(task_goal, finish_json)
    verdict, explanation = call_llm_judge(prompt, model)

    return {
        "task": task_id,
        "verdict": verdict,
        "explanation": explanation,
        "model": model,
        "goal": task_goal
    }


def validate_batch(
    finish_json_pattern: str,
    tasks_dir: Path,
    model: str = "claude-opus-4"
) -> list[Dict[str, Any]]:
    """
    Validate multiple finish JSONs.

    Args:
        finish_json_pattern: Glob pattern for finish JSONs
        tasks_dir: Directory containing task configs
        model: LLM model to use

    Returns:
        List of validation results
    """

    results = []
    finish_json_paths = sorted(glob.glob(finish_json_pattern))

    print(f"Found {len(finish_json_paths)} finish JSONs to validate")

    for i, finish_json_path in enumerate(finish_json_paths, 1):
        finish_json_path = Path(finish_json_path)
        task_id = finish_json_path.stem

        # Find corresponding task config
        task_path = tasks_dir / f"{task_id}.json"
        if not task_path.exists():
            print(f"Warning: Task config not found for {task_id}", file=sys.stderr)
            continue

        print(f"[{i}/{len(finish_json_paths)}] Validating {task_id}...", end=" ")

        result = validate_single_task(task_path, finish_json_path, model)
        results.append(result)

        print(f"{result['verdict']}")

    return results


def print_summary(results: list[Dict[str, Any]]):
    """Print validation summary."""

    total = len(results)
    success = sum(1 for r in results if r["verdict"] == "SUCCESS")
    failure = sum(1 for r in results if r["verdict"] == "FAILURE")
    uncertain = sum(1 for r in results if r["verdict"] == "UNCERTAIN")

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total tasks: {total}")
    print(f"✓ SUCCESS:   {success} ({success/total*100:.1f}%)")
    print(f"✗ FAILURE:   {failure} ({failure/total*100:.1f}%)")
    print(f"? UNCERTAIN: {uncertain} ({uncertain/total*100:.1f}%)")
    print("="*60)

    if failure > 0:
        print("\nFailed tasks:")
        for r in results:
            if r["verdict"] == "FAILURE":
                print(f"  - {r['task']}: {r['explanation'][:80]}...")

    if uncertain > 0:
        print("\nUncertain tasks (manual review recommended):")
        for r in results:
            if r["verdict"] == "UNCERTAIN":
                print(f"  - {r['task']}: {r['explanation'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge validation for finish JSONs")
    parser.add_argument("--task", type=Path, help="Single task config path")
    parser.add_argument("--finish-json", type=Path, help="Single finish JSON path")
    parser.add_argument("--finish-jsons", type=str, help="Glob pattern for batch validation")
    parser.add_argument("--tasks-dir", type=Path, default=Path("multi-real/tasks"), help="Directory with task configs")
    parser.add_argument("--model", default="claude-opus-4", help="LLM model (claude-opus-4 or gpt-4o)")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")

    args = parser.parse_args()

    if args.task and args.finish_json:
        # Single task validation
        result = validate_single_task(args.task, args.finish_json, args.model)

        print(f"\nTask: {result['task']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Explanation: {result['explanation']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump([result], f, indent=2)

        sys.exit(0 if result["verdict"] == "SUCCESS" else 1)

    elif args.finish_jsons:
        # Batch validation
        results = validate_batch(args.finish_jsons, args.tasks_dir, args.model)

        print_summary(results)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        # Exit with failure if any tasks failed
        has_failures = any(r["verdict"] == "FAILURE" for r in results)
        sys.exit(1 if has_failures else 0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
