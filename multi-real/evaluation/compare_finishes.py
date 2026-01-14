#!/usr/bin/env python3
"""
Finish JSON Comparison Tool

Compares agent-generated finish JSONs to manual ground truth.
Used to validate automation reliability before scaling to all 64 tasks.

Usage:
    # Compare single task
    uv run python multi-real/compare_finishes.py \
        --manual multi-real/final_states/manual/dashdish-gomail-1.json \
        --agent multi-real/final_states/automated/dashdish-gomail-1.json

    # Batch comparison
    uv run python multi-real/compare_finishes.py \
        --manual-dir multi-real/final_states/manual \
        --agent-dir multi-real/final_states/automated \
        --output comparison_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def normalize_value(value: Any) -> Any:
    """Normalize values for comparison (handle floats, whitespace, etc)."""
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, float):
        return round(value, 2)
    elif isinstance(value, list):
        return sorted([normalize_value(v) for v in value], key=lambda x: str(x))
    elif isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    return value


def extract_key_entities(finish_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key entities from finish JSON for comparison.

    Returns structured data about what was actually done in the task.
    """

    entities = defaultdict(lambda: defaultdict(list))

    for app_name, app_data in finish_json.items():
        if not isinstance(app_data, dict):
            continue

        # Extract from differences
        differences = app_data.get("differences", {})

        # Bookings
        bookings = differences.get("bookings", {}).get("added", [])
        if bookings:
            entities[app_name]["bookings"] = [
                {
                    "restaurant": b.get("restaurant", b.get("restaurantName")),
                    "date": b.get("date"),
                    "time": b.get("time"),
                    "guests": b.get("guests", b.get("partySize"))
                }
                for b in bookings
            ]

        # Emails
        emails_added = differences.get("emails", {}).get("added", [])
        emails_sent = differences.get("emails", {}).get("sent", [])

        if emails_added:
            entities[app_name]["emails_added"] = [
                {
                    "to": e.get("to"),
                    "subject": e.get("subject"),
                    "has_content": bool(e.get("content") or e.get("body"))
                }
                for e in emails_added
            ]

        if emails_sent:
            entities[app_name]["emails_sent"] = [
                {
                    "to": e.get("to"),
                    "subject": e.get("subject"),
                    "has_content": bool(e.get("content") or e.get("body"))
                }
                for e in emails_sent
            ]

        # Calendar events
        events = differences.get("events", {}).get("added")
        if events:
            if isinstance(events, dict):
                events = list(events.values())

            entities[app_name]["events"] = [
                {
                    "title": e.get("title", e.get("summary")),
                    "start": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
                    "end": e.get("end", {}).get("dateTime") or e.get("end", {}).get("date")
                }
                for e in events
            ]

        # Flights
        flights = differences.get("bookedFlights", [])
        if flights:
            entities[app_name]["flights"] = [
                {
                    "from": f.get("from", f.get("origin")),
                    "to": f.get("to", f.get("destination")),
                    "date": f.get("date", f.get("departureDate"))
                }
                for f in flights
            ]

        # Food orders
        food_orders = differences.get("foodOrders", {}).get("added", [])
        if food_orders:
            entities[app_name]["food_orders"] = [
                {
                    "restaurant": o.get("restaurant"),
                    "items_count": len(o.get("items", []))
                }
                for o in food_orders
            ]

        # Rides
        if "udriver" in app_name.lower():
            rides = app_data.get("initialfinaldiff", {}).get("added", {}).get("ride", {})
            booked_trips = rides.get("bookedTrips", {})
            if booked_trips:
                if isinstance(booked_trips, dict):
                    booked_trips = list(booked_trips.values())

                entities[app_name]["rides"] = [
                    {
                        "pickup": r.get("pickup", r.get("pickupLocation")),
                        "dropoff": r.get("dropoff", r.get("dropoffLocation"))
                    }
                    for r in booked_trips
                ]

    return dict(entities)


def compare_entities(manual: Dict[str, Any], agent: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Compare extracted entities between manual and agent finish JSONs.

    Returns:
        (is_match, list_of_differences)
    """

    differences = []
    is_match = True

    # Get all apps from both
    all_apps = set(manual.keys()) | set(agent.keys())

    for app in all_apps:
        manual_app = manual.get(app, {})
        agent_app = agent.get(app, {})

        # Get all entity types
        all_entity_types = set(manual_app.keys()) | set(agent_app.keys())

        for entity_type in all_entity_types:
            manual_entities = manual_app.get(entity_type, [])
            agent_entities = agent_app.get(entity_type, [])

            # Normalize
            manual_entities = normalize_value(manual_entities)
            agent_entities = normalize_value(agent_entities)

            if manual_entities != agent_entities:
                is_match = False

                # Count-level comparison
                manual_count = len(manual_entities) if isinstance(manual_entities, list) else 0
                agent_count = len(agent_entities) if isinstance(agent_entities, list) else 0

                if manual_count != agent_count:
                    differences.append(
                        f"{app}.{entity_type}: count mismatch (manual={manual_count}, agent={agent_count})"
                    )
                else:
                    differences.append(
                        f"{app}.{entity_type}: content differs despite same count"
                    )

    return is_match, differences


def calculate_agreement_score(manual: Dict[str, Any], agent: Dict[str, Any]) -> float:
    """
    Calculate agreement score between 0 and 1.

    Score is based on matching entity counts and types.
    """

    manual_entities = extract_key_entities(manual)
    agent_entities = extract_key_entities(agent)

    # Count total entities in manual
    total_manual = 0
    total_matching = 0

    all_apps = set(manual_entities.keys()) | set(agent_entities.keys())

    for app in all_apps:
        manual_app = manual_entities.get(app, {})
        agent_app = agent_entities.get(app, {})

        all_types = set(manual_app.keys()) | set(agent_app.keys())

        for entity_type in all_types:
            manual_list = manual_app.get(entity_type, [])
            agent_list = agent_app.get(entity_type, [])

            manual_count = len(manual_list) if isinstance(manual_list, list) else 0
            agent_count = len(agent_list) if isinstance(agent_list, list) else 0

            total_manual += manual_count

            # Give credit for matching counts
            if manual_count == agent_count:
                total_matching += manual_count

    if total_manual == 0:
        return 1.0  # Both empty = perfect match

    return total_matching / total_manual


def compare_single_task(
    manual_path: Path,
    agent_path: Path
) -> Dict[str, Any]:
    """
    Compare single task.

    Returns comparison result with agreement score and differences.
    """

    manual = load_json(manual_path)
    agent = load_json(agent_path)

    manual_entities = extract_key_entities(manual)
    agent_entities = extract_key_entities(agent)

    is_match, differences = compare_entities(manual_entities, agent_entities)
    agreement_score = calculate_agreement_score(manual, agent)

    return {
        "task": manual_path.stem,
        "is_match": is_match,
        "agreement_score": agreement_score,
        "differences": differences,
        "manual_entities": manual_entities,
        "agent_entities": agent_entities
    }


def compare_batch(
    manual_dir: Path,
    agent_dir: Path
) -> List[Dict[str, Any]]:
    """
    Compare all tasks in manual and agent directories.

    Returns list of comparison results.
    """

    results = []
    manual_jsons = sorted(manual_dir.glob("*.json"))

    print(f"Found {len(manual_jsons)} manual finish JSONs")

    for manual_path in manual_jsons:
        task_id = manual_path.stem
        agent_path = agent_dir / f"{task_id}.json"

        if not agent_path.exists():
            print(f"Warning: No agent finish JSON for {task_id}", file=sys.stderr)
            continue

        print(f"Comparing {task_id}...", end=" ")

        result = compare_single_task(manual_path, agent_path)
        results.append(result)

        status = "✓ MATCH" if result["is_match"] else f"✗ DIFFER ({result['agreement_score']:.1%})"
        print(status)

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print comparison summary."""

    total = len(results)
    matches = sum(1 for r in results if r["is_match"])
    avg_score = sum(r["agreement_score"] for r in results) / total if total > 0 else 0

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Total tasks compared: {total}")
    print(f"Perfect matches: {matches}/{total} ({matches/total*100:.1f}%)")
    print(f"Average agreement score: {avg_score:.1%}")
    print("="*60)

    if avg_score >= 0.90:
        print("\n✓ VALIDATION PASSED: Agreement rate ≥90%")
        print("  Automation is reliable. Safe to proceed with remaining 64 tasks.")
    else:
        print("\n✗ VALIDATION FAILED: Agreement rate <90%")
        print("  Investigate systematic issues before scaling automation.")

    # Show tasks with low agreement
    low_agreement = [r for r in results if r["agreement_score"] < 0.90]
    if low_agreement:
        print(f"\nTasks with <90% agreement ({len(low_agreement)}):")
        for r in sorted(low_agreement, key=lambda x: x["agreement_score"]):
            print(f"  - {r['task']}: {r['agreement_score']:.1%}")
            for diff in r["differences"][:3]:  # Show first 3 diffs
                print(f"      • {diff}")


def main():
    parser = argparse.ArgumentParser(description="Compare manual vs agent finish JSONs")
    parser.add_argument("--manual", type=Path, help="Manual finish JSON path (single)")
    parser.add_argument("--agent", type=Path, help="Agent finish JSON path (single)")
    parser.add_argument("--manual-dir", type=Path, help="Directory with manual finish JSONs (batch)")
    parser.add_argument("--agent-dir", type=Path, help="Directory with agent finish JSONs (batch)")
    parser.add_argument("--output", type=Path, help="Output JSON file for detailed results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed entity comparisons")

    args = parser.parse_args()

    if args.manual and args.agent:
        # Single comparison
        result = compare_single_task(args.manual, args.agent)

        print(f"\nTask: {result['task']}")
        print(f"Agreement Score: {result['agreement_score']:.1%}")
        print(f"Match: {'✓ YES' if result['is_match'] else '✗ NO'}")

        if result['differences']:
            print(f"\nDifferences ({len(result['differences'])}):")
            for diff in result['differences']:
                print(f"  • {diff}")

        if args.verbose:
            print("\nManual entities:")
            print(json.dumps(result['manual_entities'], indent=2))
            print("\nAgent entities:")
            print(json.dumps(result['agent_entities'], indent=2))

        if args.output:
            with open(args.output, "w") as f:
                json.dump([result], f, indent=2)

        sys.exit(0 if result['is_match'] else 1)

    elif args.manual_dir and args.agent_dir:
        # Batch comparison
        results = compare_batch(args.manual_dir, args.agent_dir)

        print_summary(results)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to {args.output}")

        # Exit with success if average agreement ≥90%
        avg_score = sum(r["agreement_score"] for r in results) / len(results)
        sys.exit(0 if avg_score >= 0.90 else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
