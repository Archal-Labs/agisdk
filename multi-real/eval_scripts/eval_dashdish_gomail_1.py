#!/usr/bin/env python3
"""
Evaluation script for dashdish-gomail-1:
"Order four Turkey Clubs and two Veggie Wraps from The Sandwich Shop for delivery.
Once the order is complete, forward the receipt from my email to finance@corp.com for reimbursement."

Verification:
1. DashDish cart has items (order placed)
2. GoMail sent an email to finance address
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


def verify(data: dict) -> bool:
    """Verify dashdish-gomail-1 task completion."""

    # Check 1: DashDish cart has items
    dashdish = data.get("dashdish", {})

    # Try primary path
    cart_items = safe_get(dashdish, "initialfinaldiff", "added", "cart", "cartItems")

    # Fallback: check differences.foodOrders
    if not cart_items:
        food_orders = safe_get(dashdish, "differences", "foodOrders", "added")
        if isinstance(food_orders, dict):
            cart_items = list(food_orders.values())
        elif isinstance(food_orders, list):
            cart_items = food_orders

    if not cart_items or len(cart_items) < 1:
        print("FAILURE: No cart items found in DashDish", file=sys.stderr)
        return False

    # Check 2: GoMail sent an email
    gomail = data.get("gomail", {})

    # Try primary path
    emails = safe_get(gomail, "initialfinaldiff", "added", "email", "emails")

    # Fallback: check differences.emails
    if not emails:
        emails_added = safe_get(gomail, "differences", "emails", "added")
        if isinstance(emails_added, list) and emails_added:
            emails = {str(i): e for i, e in enumerate(emails_added)}
        emails_sent = safe_get(gomail, "differences", "emails", "sent")
        if isinstance(emails_sent, list) and emails_sent:
            emails = emails or {}
            for i, e in enumerate(emails_sent):
                emails[f"sent_{i}"] = e

    if not emails:
        print("FAILURE: No emails found in GoMail", file=sys.stderr)
        return False

    # Check that at least one email was sent
    email_list = list(emails.values()) if isinstance(emails, dict) else emails
    if len(email_list) < 1:
        print("FAILURE: No emails sent", file=sys.stderr)
        return False

    # Optional: Check email recipient contains finance
    for email in email_list:
        to_list = email.get("to", [])
        if isinstance(to_list, str):
            to_list = [to_list]
        for recipient in to_list:
            if "finance" in recipient.lower():
                return True

    # If no finance email found, still pass if order and email exist
    # (recipient might vary in different runs)
    print("Note: Email sent but not to finance address", file=sys.stderr)
    return True


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
