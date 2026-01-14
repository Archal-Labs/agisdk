#!/usr/bin/env python3
"""
Template for Multi-Real Evaluation Scripts.

Evaluation scripts receive the finish JSON path as sys.argv[1].
They must print "SUCCESS" or "FAILURE" to stdout.

Copy this template and modify the verify() function for your task.
"""

import json
import sys
from typing import Any


def safe_get(d: dict, *keys) -> Any:
    """Safely navigate nested dict structure."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def find_in_tree(node: Any, key: str, results: list | None = None) -> list:
    """Recursively find all values for a key in a nested structure."""
    if results is None:
        results = []

    if isinstance(node, dict):
        if key in node:
            results.append(node[key])
        for v in node.values():
            find_in_tree(v, key, results)
    elif isinstance(node, list):
        for item in node:
            find_in_tree(item, key, results)

    return results


def to_number(x: Any) -> float | None:
    """Safely convert a value to a number."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            # Remove currency symbols and commas
            cleaned = x.replace("$", "").replace(",", "").strip()
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_str(s: Any) -> str:
    """Normalize a string for comparison."""
    if s is None:
        return ""
    return str(s).strip().lower()


def verify(data: dict) -> bool:
    """
    Main verification logic. Modify this function for your task.

    Args:
        data: The finish JSON data (nested by website_id for multi-app tasks)

    Returns:
        True if the task was completed successfully, False otherwise
    """
    # Example: Check if dashdish cart has items
    # dashdish = data.get("dashdish", {})
    # cart_items = safe_get(dashdish, "initialfinaldiff", "added", "cart", "cartItems")
    # if not cart_items or len(cart_items) < 1:
    #     return False

    # Example: Check if gomail sent an email
    # gomail = data.get("gomail", {})
    # emails = safe_get(gomail, "initialfinaldiff", "added", "email", "emails")
    # if not emails or len(emails) < 1:
    #     return False

    # Example: Cross-app check - email mentions order details
    # cart_total = safe_get(dashdish, "initialfinaldiff", "added", "cart", "checkoutDetails", "charges", "totalAmount")
    # email_content = list(emails.values())[0].get("content", "") if emails else ""
    # if str(cart_total) not in email_content:
    #     return False

    # TODO: Implement your verification logic here
    return False


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("FAILURE: No JSON path provided")
            sys.exit(1)

        with open(sys.argv[1]) as f:
            data = json.load(f)

        if verify(data):
            print("SUCCESS")
        else:
            print("FAILURE")

    except Exception as e:
        print(f"FAILURE: {e}")
        sys.exit(1)
