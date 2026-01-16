#!/usr/bin/env python3
"""
Hybrid Validation: JMESPath + LLM Judge

This module provides both a HybridValidator class (for programmatic use) and
a CLI script for standalone validation.

The HybridValidator combines deterministic JMESPath queries (primary) with semantic
LLM-based validation (fallback). It provides the best of both approaches:
- Fast, reproducible validation via JMESPath
- Semantic understanding for edge cases via LLM judge

Class Usage:
    from hybrid_validator import HybridValidator

    validator = HybridValidator()  # Uses Gemini 2.5 Pro via Vertex AI
    result = validator.evaluate(
        task_goal="Book a restaurant...",
        finish_state={...},
        evals=[...]
    )

CLI Usage:
    # Validate all tasks with finish JSONs
    uv run python multi-real/hybrid_validator.py --finish-jsons-dir multi-real/final_states/openai

    # Validate specific task
    uv run python multi-real/hybrid_validator.py --task gomail-topwork-1 --finish-json multi-real/final_states/openai/gomail-topwork-1.json

    # Use LLM fallback for uncertain cases
    uv run python multi-real/hybrid_validator.py --finish-jsons-dir multi-real/final_states/openai --use-llm-fallback
"""

import argparse
import base64
import jmespath
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys


# =============================================================================
# HybridValidator Class (for programmatic use)
# =============================================================================

class HybridValidator:
    """
    Hybrid evaluation combining deterministic JMESPath with LLM-as-judge.

    Uses Gemini 2.5 Pro via Vertex AI for LLM evaluation.
    Authentication is handled via Application Default Credentials (gcloud auth).

    Strategy:
    1. JMESPath passes → High confidence success
    2. JMESPath fails + LLM passes → Medium confidence success (likely query bug)
    3. Both fail → High confidence failure
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "us-central1",
        llm_model: str = "gemini-2.5-pro-preview-05-06",
    ):
        """
        Initialize the HybridValidator with Gemini via Vertex AI.

        Args:
            project_id: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var
                        or attempts to infer from gcloud config.
            location: GCP region for Vertex AI (default: us-central1).
            llm_model: Gemini model ID (default: gemini-2.5-pro-preview-05-06).
        """
        self.llm_model = llm_model
        self.location = location
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")

        # Lazy import vertexai
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform"
            )

        # Initialize Vertex AI - uses Application Default Credentials from gcloud auth
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.llm_model)

    def evaluate(
        self,
        task_goal: str,
        finish_state: dict,
        evals: list[dict],
    ) -> dict:
        """
        Evaluate task completion using LLM judge.

        Args:
            task_goal: The task's goal string
            finish_state: The final state JSON from /finish endpoint
            evals: List of evaluation criteria (JMESPath queries)

        Returns:
            dict with:
            - overall_pass: bool
            - criteria_results: list of per-criterion results
            - reasoning: str
            - confidence: float (0.0-1.0)
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(task_goal, finish_state, evals)

        # Call Gemini via Vertex AI
        try:
            from vertexai.generative_models import GenerationConfig

            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.0,
                ),
            )

            response_text = response.text
        except Exception as e:
            # If LLM call fails, return failure with low confidence
            return {
                "overall_pass": False,
                "criteria_results": [],
                "reasoning": f"LLM call failed: {str(e)}",
                "confidence": 0.0,
                "raw_response": None,
            }

        # Parse response
        return self._parse_llm_response(response_text, evals)

    def _build_evaluation_prompt(
        self,
        task_goal: str,
        finish_state: dict,
        evals: list[dict],
    ) -> str:
        """Build the LLM judge prompt."""
        criteria_text = "\n".join(
            f"{i+1}. {e.get('description', 'No description')}"
            for i, e in enumerate(evals)
        )

        # Truncate finish_state if too large (keep first 10k chars)
        finish_state_str = json.dumps(finish_state, indent=2)
        if len(finish_state_str) > 10000:
            finish_state_str = finish_state_str[:10000] + "\n...(truncated)"

        return f"""You are evaluating whether an AI agent successfully completed a multi-app browser task.

TASK GOAL:
{task_goal}

SUCCESS CRITERIA:
{criteria_text}

FINAL APPLICATION STATE (JSON):
```json
{finish_state_str}
```

For each criterion, determine if it was satisfied based on the final state.
Be lenient with exact string matching - focus on semantic correctness.

Respond in this exact JSON format:
{{
  "overall_pass": true/false,
  "criteria_results": [
    {{"criterion": 1, "pass": true/false, "reason": "brief explanation"}},
    ...
  ],
  "reasoning": "Overall assessment of task completion",
  "confidence": 0.85
}}

IMPORTANT:
- Only output the JSON, no other text.
- confidence should be 0.0-1.0 (0.9+ = very confident, 0.7-0.9 = confident, 0.5-0.7 = uncertain)
- Be specific in reasoning - reference actual data from the state"""

    def evaluate_with_vision(
        self,
        task_goal: str,
        screenshots: list[tuple[str, str]],  # List of (app_name, base64_image)
        axtree_txt: str | None,
        evals: list[dict],
        finish_state: dict | None = None,
    ) -> dict:
        """
        Evaluate task completion using screenshots and accessibility tree.

        This method provides more reliable validation than truncated JSON by
        using visual evidence (screenshots) and semantic UI state (a-tree).

        Args:
            task_goal: The task's goal string
            screenshots: List of (app_name, base64_data_url) tuples for each app
            axtree_txt: Formatted accessibility tree text from final step
            evals: List of evaluation criteria
            finish_state: Optional finish state for extracting concrete evidence

        Returns:
            dict with:
            - overall_pass: bool
            - criteria_results: list of per-criterion results
            - reasoning: str
            - confidence: float (0.0-1.0)
        """
        # Build evaluation prompt with vision content for Gemini
        prompt_content = self._build_vision_evaluation_prompt_gemini(
            task_goal, screenshots, axtree_txt, evals, finish_state
        )

        # Call Gemini with vision
        try:
            from vertexai.generative_models import GenerationConfig

            response = self.model.generate_content(
                prompt_content,
                generation_config=GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.0,
                ),
            )

            response_text = response.text
        except Exception as e:
            return {
                "overall_pass": False,
                "criteria_results": [],
                "reasoning": f"LLM vision call failed: {str(e)}",
                "confidence": 0.0,
                "raw_response": None,
            }

        return self._parse_llm_response(response_text, evals)

    def _build_vision_evaluation_prompt(
        self,
        task_goal: str,
        screenshots: list[tuple[str, str]],
        axtree_txt: str | None,
        evals: list[dict],
        finish_state: dict | None = None,
    ) -> list[dict]:
        """Build multimodal prompt with screenshots and a-tree text."""
        criteria_text = "\n".join(
            f"{i+1}. {e.get('description', 'No description')}"
            for i, e in enumerate(evals)
        )

        # Build text context
        text_parts = [
            f"""You are evaluating whether an AI agent successfully completed a multi-app browser task.

TASK GOAL:
{task_goal}

SUCCESS CRITERIA:
{criteria_text}
"""
        ]

        # Add concrete evidence from finish_state if available
        if finish_state:
            evidence_text = []
            for app_name in finish_state.keys():
                if isinstance(finish_state[app_name], dict):
                    evidence = extract_concrete_evidence(finish_state, app_name)
                    if evidence:
                        evidence_text.append(f"\n**{app_name.upper()} State Changes:**")
                        evidence_text.append(json.dumps(evidence, indent=2))
            if evidence_text:
                text_parts.append("\nCONCRETE EVIDENCE FROM APPLICATION STATE:")
                text_parts.extend(evidence_text)

        # Add a-tree if available (truncate if very long)
        if axtree_txt:
            # Keep a reasonable amount of a-tree context (20k chars)
            if len(axtree_txt) > 20000:
                axtree_txt = axtree_txt[:20000] + "\n...(truncated)"
            text_parts.append(f"""
ACCESSIBILITY TREE (UI Structure):
```
{axtree_txt}
```
""")

        text_parts.append("""
SCREENSHOTS:
Below are screenshots showing the final state of each application involved in this task.
Examine them carefully to verify whether the required actions were completed.

For each criterion, determine if it was satisfied based on:
1. Visual evidence in the screenshots (forms filled, emails sent, etc.)
2. The accessibility tree structure (confirms UI state)
3. The concrete evidence extracted from application state

Respond in this exact JSON format:
{
  "overall_pass": true/false,
  "criteria_results": [
    {"criterion": 1, "pass": true/false, "reason": "brief explanation referencing visual evidence"},
    ...
  ],
  "reasoning": "Overall assessment of task completion based on screenshots and UI state",
  "confidence": 0.85
}

IMPORTANT:
- Only output the JSON, no other text.
- confidence should be 0.0-1.0 (0.9+ = very confident, 0.7-0.9 = confident, 0.5-0.7 = uncertain)
- Reference specific visual elements you see in the screenshots
- If screenshots show completed actions (sent emails, calendar events, bookings), that's strong evidence
""")

        # Build multimodal content
        content = [{"type": "text", "text": "\n".join(text_parts)}]

        # Add screenshots as image content
        for app_name, base64_url in screenshots:
            # Add label for the screenshot
            content.append({
                "type": "text",
                "text": f"\n--- Screenshot: {app_name} ---"
            })
            # Add the image
            if base64_url.startswith("data:"):
                # Extract media type and base64 data
                parts = base64_url.split(",", 1)
                media_type = parts[0].split(":")[1].split(";")[0]
                base64_data = parts[1] if len(parts) > 1 else ""
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    }
                })
            else:
                # Assume it's raw base64, default to jpeg
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_url,
                    }
                })

        return content

    def _build_vision_evaluation_prompt_gemini(
        self,
        task_goal: str,
        screenshots: list[tuple[str, str]],
        axtree_txt: str | None,
        evals: list[dict],
        finish_state: dict | None = None,
    ) -> list:
        """Build multimodal prompt for Gemini with screenshots and a-tree text."""
        from vertexai.generative_models import Part, Image

        criteria_text = "\n".join(
            f"{i+1}. {e.get('description', 'No description')}"
            for i, e in enumerate(evals)
        )

        # Build text context
        text_parts = [
            f"""You are evaluating whether an AI agent successfully completed a multi-app browser task.

TASK GOAL:
{task_goal}

SUCCESS CRITERIA:
{criteria_text}
"""
        ]

        # Add concrete evidence from finish_state if available
        if finish_state:
            evidence_text = []
            for app_name in finish_state.keys():
                if isinstance(finish_state[app_name], dict):
                    evidence = extract_concrete_evidence(finish_state, app_name)
                    if evidence:
                        evidence_text.append(f"\n**{app_name.upper()} State Changes:**")
                        evidence_text.append(json.dumps(evidence, indent=2))
            if evidence_text:
                text_parts.append("\nCONCRETE EVIDENCE FROM APPLICATION STATE:")
                text_parts.extend(evidence_text)

        # Add a-tree if available (truncate if very long)
        if axtree_txt:
            # Keep a reasonable amount of a-tree context (20k chars)
            if len(axtree_txt) > 20000:
                axtree_txt = axtree_txt[:20000] + "\n...(truncated)"
            text_parts.append(f"""
ACCESSIBILITY TREE (UI Structure):
```
{axtree_txt}
```
""")

        text_parts.append("""
SCREENSHOTS:
Below are screenshots showing the final state of each application involved in this task.
Examine them carefully to verify whether the required actions were completed.

For each criterion, determine if it was satisfied based on:
1. Visual evidence in the screenshots (forms filled, emails sent, etc.)
2. The accessibility tree structure (confirms UI state)
3. The concrete evidence extracted from application state

Respond in this exact JSON format:
{
  "overall_pass": true/false,
  "criteria_results": [
    {"criterion": 1, "pass": true/false, "reason": "brief explanation referencing visual evidence"},
    ...
  ],
  "reasoning": "Overall assessment of task completion based on screenshots and UI state",
  "confidence": 0.85
}

IMPORTANT:
- Only output the JSON, no other text.
- confidence should be 0.0-1.0 (0.9+ = very confident, 0.7-0.9 = confident, 0.5-0.7 = uncertain)
- Reference specific visual elements you see in the screenshots
- If screenshots show completed actions (sent emails, calendar events, bookings), that's strong evidence
""")

        # Build multimodal content for Gemini
        content = [Part.from_text("\n".join(text_parts))]

        # Add screenshots as image parts
        for app_name, base64_url in screenshots:
            # Add label for the screenshot
            content.append(Part.from_text(f"\n--- Screenshot: {app_name} ---"))

            # Extract base64 data and convert to Gemini Image format
            if base64_url.startswith("data:"):
                # Extract media type and base64 data from data URL
                parts = base64_url.split(",", 1)
                media_type = parts[0].split(":")[1].split(";")[0]
                base64_data = parts[1] if len(parts) > 1 else ""
            else:
                # Assume raw base64, default to jpeg
                media_type = "image/jpeg"
                base64_data = base64_url

            # Create image part from base64 data
            image_bytes = base64.b64decode(base64_data)
            content.append(Part.from_image(Image.from_bytes(image_bytes)))

        return content

    def _parse_llm_response(self, response_text: str, evals: list[dict]) -> dict:
        """Parse LLM response into structured result."""
        try:
            # Extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])

                # Ensure all required fields
                if "overall_pass" not in result:
                    result["overall_pass"] = False
                if "confidence" not in result:
                    result["confidence"] = 0.5
                if "reasoning" not in result:
                    result["reasoning"] = "No reasoning provided"
                if "criteria_results" not in result:
                    result["criteria_results"] = []

                return result
        except json.JSONDecodeError as e:
            pass

        # Fallback: assume failure if can't parse
        return {
            "overall_pass": False,
            "criteria_results": [],
            "reasoning": "Failed to parse LLM response",
            "confidence": 0.0,
            "raw_response": response_text,
        }

    def validate_against_ground_truth(
        self,
        finish_state: dict,
        ground_truth_state: dict,
        tolerance: float = 0.9,
    ) -> dict:
        """
        Validate a finish state against known ground truth.

        Used for:
        1. Validating JMESPath queries are correct
        2. Measuring LLM judge accuracy

        Returns similarity metrics and detailed comparison.
        """
        # Deep comparison of state structures
        comparison = self._deep_compare(finish_state, ground_truth_state)

        return {
            "match_ratio": comparison["match_ratio"],
            "passes_threshold": comparison["match_ratio"] >= tolerance,
            "differences": comparison["differences"],
        }

    def _deep_compare(self, actual: Any, expected: Any, path: str = "") -> dict:
        """Recursively compare two data structures."""
        differences = []
        matches = 0
        total = 0

        if type(actual) != type(expected):
            differences.append({
                "path": path,
                "type": "type_mismatch",
                "actual": type(actual).__name__,
                "expected": type(expected).__name__,
            })
            return {"match_ratio": 0.0, "differences": differences}

        if isinstance(expected, dict):
            for key in set(list(actual.keys()) + list(expected.keys())):
                total += 1
                child_path = f"{path}.{key}" if path else key

                if key not in actual:
                    differences.append({"path": child_path, "type": "missing_key"})
                elif key not in expected:
                    differences.append({"path": child_path, "type": "extra_key"})
                else:
                    child_result = self._deep_compare(
                        actual[key], expected[key], child_path
                    )
                    matches += child_result["match_ratio"]
                    differences.extend(child_result["differences"])

        elif isinstance(expected, list):
            total = max(len(actual), len(expected))
            for i in range(total):
                child_path = f"{path}[{i}]"
                if i >= len(actual):
                    differences.append({"path": child_path, "type": "missing_item"})
                elif i >= len(expected):
                    differences.append({"path": child_path, "type": "extra_item"})
                else:
                    child_result = self._deep_compare(
                        actual[i], expected[i], child_path
                    )
                    matches += child_result["match_ratio"]
                    differences.extend(child_result["differences"])

        else:
            total = 1
            if actual == expected:
                matches = 1
            else:
                differences.append({
                    "path": path,
                    "type": "value_mismatch",
                    "actual": actual,
                    "expected": expected,
                })

        match_ratio = matches / total if total > 0 else 1.0
        return {"match_ratio": match_ratio, "differences": differences}


# =============================================================================
# CLI Helper Functions
# =============================================================================

def load_json(path: Path) -> dict:
    """Load JSON from file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_concrete_evidence(finish_json: Dict[str, Any], app_name: str) -> Dict[str, Any]:
    """
    Extract concrete evidence from finish JSON for LLM judge.

    Instead of just counts, extract actual data like:
    - Email subjects, recipients, content snippets
    - Event titles, dates, attendees
    - Booking details (restaurant names, dates, times)
    - Flight details (destinations, dates)

    Args:
        finish_json: The finish JSON data
        app_name: The app name (e.g., "gomail", "gocalendar")

    Returns:
        Dict with concrete evidence for this app
    """
    evidence = {}

    app_data = finish_json.get(app_name, {})
    if not isinstance(app_data, dict):
        return evidence

    differences = app_data.get("differences", {})

    # Extract emails
    emails = differences.get("emails", {})
    if emails:
        sent = emails.get("sent", [])
        if sent:
            evidence["emails_sent"] = [
                {
                    "subject": email.get("subject", ""),
                    "recipient": email.get("recipient", ""),
                    "content_snippet": email.get("content", "")[:200] + "..." if len(email.get("content", "")) > 200 else email.get("content", "")
                }
                for email in sent[:3]  # Limit to first 3
            ]

        added = emails.get("added", [])
        if added:
            evidence["emails_received"] = [
                {
                    "subject": email.get("subject", ""),
                    "sender": email.get("sender", ""),
                    "content_snippet": email.get("content", "")[:200] + "..."
                }
                for email in added[:3]
            ]

    # Extract events
    events = differences.get("events", {}).get("added")
    if events:
        if isinstance(events, dict):
            event_list = list(events.values())
        else:
            event_list = events

        evidence["events"] = [
            {
                "title": event.get("title", ""),
                "start": event.get("start", {}).get("date") or event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("date") or event.get("end", {}).get("dateTime"),
                "description": event.get("description", "")[:200] + "..." if len(event.get("description", "")) > 200 else event.get("description", "")
            }
            for event in event_list[:3]
        ]

    # Extract bookings
    bookings = differences.get("bookings", {}).get("added", [])
    if bookings:
        evidence["bookings"] = [
            {
                "restaurant": booking.get("restaurantName", ""),
                "date": booking.get("date", ""),
                "time": booking.get("time", ""),
                "party_size": booking.get("partySize", 0)
            }
            for booking in bookings[:3]
        ]

    # Extract flights
    flights = differences.get("bookedFlights", [])
    if flights:
        evidence["flights"] = [
            {
                "destination": flight.get("destination", ""),
                "departure_date": flight.get("departureDate", ""),
                "origin": flight.get("origin", "")
            }
            for flight in flights[:3]
        ]

    # Extract food orders
    food_orders = differences.get("foodOrders", {}).get("added", [])
    if food_orders:
        evidence["food_orders"] = [
            {
                "restaurant": order.get("restaurant", ""),
                "items": order.get("items", [])[:3],
                "total": order.get("total", 0)
            }
            for order in food_orders[:3]
        ]

    return evidence


def validate_with_jmespath(
    task_config: Dict[str, Any],
    finish_json: Dict[str, Any]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate task using JMESPath queries.

    Returns:
        (success, results) where results is list of query results
    """
    evals = task_config.get("evals", [])
    results = []

    all_pass = True
    for i, eval_item in enumerate(evals):
        if eval_item.get("type") != "jmespath":
            continue

        query = eval_item.get("query", "")
        description = eval_item.get("description", f"Query {i+1}")

        try:
            result = jmespath.search(query, finish_json)
            passed = result is not None and result is not False
            results.append({
                "query_index": i,
                "description": description,
                "query": query,
                "passed": passed,
                "result": result,
                "error": None
            })

            if not passed:
                all_pass = False

        except Exception as e:
            results.append({
                "query_index": i,
                "description": description,
                "query": query,
                "passed": False,
                "result": None,
                "error": str(e)
            })
            all_pass = False

    return all_pass, results


def create_enhanced_validation_prompt(
    task_goal: str,
    finish_json: Dict[str, Any],
    jmespath_results: List[Dict[str, Any]]
) -> str:
    """
    Create enhanced prompt with concrete evidence for LLM judge.

    Args:
        task_goal: The task goal
        finish_json: The finish JSON
        jmespath_results: Results from JMESPath validation

    Returns:
        Prompt string for LLM
    """
    # Extract concrete evidence from each app
    evidence_by_app = {}
    for app_name in finish_json.keys():
        if isinstance(finish_json[app_name], dict):
            evidence = extract_concrete_evidence(finish_json, app_name)
            if evidence:
                evidence_by_app[app_name] = evidence

    # Build evidence section
    evidence_text = []
    for app_name, evidence in evidence_by_app.items():
        evidence_text.append(f"\n**{app_name.upper()}:**")

        if "emails_sent" in evidence:
            evidence_text.append(f"  📧 Sent {len(evidence['emails_sent'])} email(s):")
            for email in evidence["emails_sent"]:
                evidence_text.append(f"    - To: {email['recipient']}")
                evidence_text.append(f"      Subject: {email['subject']}")
                evidence_text.append(f"      Content: {email['content_snippet']}")

        if "emails_received" in evidence:
            evidence_text.append(f"  📨 Received {len(evidence['emails_received'])} email(s):")
            for email in evidence["emails_received"]:
                evidence_text.append(f"    - From: {email['sender']}")
                evidence_text.append(f"      Subject: {email['subject']}")

        if "events" in evidence:
            evidence_text.append(f"  📅 Created {len(evidence['events'])} event(s):")
            for event in evidence["events"]:
                evidence_text.append(f"    - {event['title']}")
                evidence_text.append(f"      When: {event['start']} to {event['end']}")
                if event.get("description"):
                    evidence_text.append(f"      Description: {event['description']}")

        if "bookings" in evidence:
            evidence_text.append(f"  🍽️ Made {len(evidence['bookings'])} booking(s):")
            for booking in evidence["bookings"]:
                evidence_text.append(f"    - {booking['restaurant']}")
                evidence_text.append(f"      Date: {booking['date']} at {booking['time']}")
                evidence_text.append(f"      Party size: {booking['party_size']}")

        if "flights" in evidence:
            evidence_text.append(f"  ✈️ Booked {len(evidence['flights'])} flight(s):")
            for flight in evidence["flights"]:
                evidence_text.append(f"    - {flight['origin']} → {flight['destination']}")
                evidence_text.append(f"      Departure: {flight['departure_date']}")

        if "food_orders" in evidence:
            evidence_text.append(f"  🍔 Ordered {len(evidence['food_orders'])} food order(s):")
            for order in evidence["food_orders"]:
                evidence_text.append(f"    - Restaurant: {order['restaurant']}")
                evidence_text.append(f"      Items: {', '.join(order['items'])}")
                evidence_text.append(f"      Total: ${order['total']}")

    evidence_section = "\n".join(evidence_text) if evidence_text else "No significant state changes detected."

    # Build JMESPath results section
    jmespath_section = []
    if jmespath_results:
        jmespath_section.append("\n**Automated Query Results:**")
        for result in jmespath_results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            jmespath_section.append(f"  {status}: {result['description']}")
            if not result["passed"] and result.get("error"):
                jmespath_section.append(f"       Error: {result['error']}")

    jmespath_text = "\n".join(jmespath_section) if jmespath_section else ""

    prompt = f"""You are validating whether a computer-use agent successfully completed a task.

**Task Goal:**
{task_goal}

**Concrete Evidence:**{evidence_section}

{jmespath_text}

**Your Task:**
Based on the concrete evidence above, determine if the task goal is satisfied.

**Evaluation Criteria:**
1. Are the required actions completed with correct details?
2. Do the state changes align with the task requirements?
3. If automated queries failed, could this be a false negative (query bug) or genuine failure?

**Confidence Scoring:**
Along with your verdict, provide a confidence score (0.0 to 1.0) where:
- 1.0 = Completely certain
- 0.7-0.9 = High confidence but minor uncertainty
- 0.5-0.6 = Moderate confidence, edge case
- 0.0-0.4 = Low confidence, unclear

**Response Format:**
Line 1: VERDICT (SUCCESS, FAILURE, or UNCERTAIN)
Line 2: CONFIDENCE: <0.0-1.0>
Line 3+: Brief explanation (2-3 sentences) of your reasoning

**Example Response:**
SUCCESS
CONFIDENCE: 0.95
The agent successfully booked a restaurant at OpenDining for the correct date and sent a confirmation email via GoMail containing the booking details. Both required actions from the task goal are completed with accurate information.
"""

    return prompt


def call_llm_judge_with_confidence(
    prompt: str,
    model: str = "gemini-2.5-pro-preview-05-06",
    project_id: str | None = None,
    location: str = "us-central1",
) -> Tuple[str, float, str]:
    """
    Call LLM judge and extract verdict, confidence, and explanation.

    Uses Gemini 2.5 Pro via Vertex AI with Application Default Credentials.

    Args:
        prompt: The evaluation prompt
        model: Gemini model ID (default: gemini-2.5-pro-preview-05-06)
        project_id: GCP project ID (uses env var or gcloud config if not specified)
        location: GCP region for Vertex AI

    Returns:
        (verdict, confidence, explanation)
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        # Initialize Vertex AI
        vertexai.init(
            project=project_id or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location
        )

        gemini_model = GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                max_output_tokens=1500,
                temperature=0.0,
            ),
        )

        response_text = response.text
        lines = response_text.strip().split("\n")

        # Parse verdict
        verdict = lines[0].strip().upper()
        if "SUCCESS" in verdict:
            verdict = "SUCCESS"
        elif "FAILURE" in verdict or "FAIL" in verdict:
            verdict = "FAILURE"
        else:
            verdict = "UNCERTAIN"

        # Parse confidence
        confidence = 0.5  # default
        for line in lines[1:4]:  # Check first few lines
            if "CONFIDENCE:" in line.upper():
                try:
                    conf_str = line.split(":")[-1].strip()
                    confidence = float(conf_str)
                    break
                except ValueError:
                    pass

        # Parse explanation
        explanation_lines = []
        found_explanation = False
        for line in lines[1:]:
            if "CONFIDENCE:" in line.upper():
                found_explanation = True
                continue
            if found_explanation or (not any(x in line.upper() for x in ["CONFIDENCE", "VERDICT", verdict])):
                if line.strip():
                    explanation_lines.append(line)

        explanation = "\n".join(explanation_lines).strip()
        if not explanation:
            explanation = "\n".join(lines[2:]).strip()

        return verdict, confidence, explanation

    except ImportError:
        return "UNCERTAIN", 0.0, "google-cloud-aiplatform not installed"
    except Exception as e:
        return "UNCERTAIN", 0.0, f"Gemini API error: {str(e)}"


def hybrid_validate(
    task_config: Dict[str, Any],
    finish_json: Dict[str, Any],
    use_llm_fallback: bool = True,
    llm_model: str = "gemini-2.5-pro-preview-05-06"
) -> Dict[str, Any]:
    """
    Perform hybrid validation: JMESPath primary, LLM fallback.

    Args:
        task_config: Task configuration
        finish_json: Finish JSON data
        use_llm_fallback: Whether to use LLM for uncertain cases
        llm_model: LLM model to use

    Returns:
        Validation result dict
    """
    task_id = task_config.get("id", "unknown")
    task_goal = task_config.get("goal", "")

    # Step 1: JMESPath validation
    jmespath_success, jmespath_results = validate_with_jmespath(task_config, finish_json)

    # Determine if we need LLM fallback
    needs_llm = False
    if not jmespath_success and use_llm_fallback:
        # Check if failures might be false negatives
        # Use LLM if there are failed queries but some evidence of state changes
        has_state_changes = False
        for app_data in finish_json.values():
            if isinstance(app_data, dict):
                differences = app_data.get("differences", {})
                if any(differences.values()):
                    has_state_changes = True
                    break

        needs_llm = has_state_changes

    # Step 2: LLM validation (if needed)
    llm_verdict = None
    llm_confidence = None
    llm_explanation = None

    if needs_llm:
        prompt = create_enhanced_validation_prompt(task_goal, finish_json, jmespath_results)
        llm_verdict, llm_confidence, llm_explanation = call_llm_judge_with_confidence(prompt, llm_model)

    # Step 3: Determine final verdict
    if jmespath_success:
        # JMESPath passed - high confidence success
        final_verdict = "SUCCESS"
        final_confidence = 1.0
        final_method = "jmespath"
    elif llm_verdict == "SUCCESS" and llm_confidence >= 0.7:
        # LLM overrode JMESPath failure with high confidence
        final_verdict = "SUCCESS"
        final_confidence = llm_confidence
        final_method = "llm_override"
    elif llm_verdict == "FAILURE":
        # LLM confirmed failure
        final_verdict = "FAILURE"
        final_confidence = llm_confidence if llm_confidence else 0.9
        final_method = "llm_confirmed" if llm_verdict else "jmespath"
    else:
        # Uncertain or conflicting results
        final_verdict = "UNCERTAIN"
        final_confidence = llm_confidence if llm_confidence else 0.5
        final_method = "uncertain"

    return {
        "task_id": task_id,
        "goal": task_goal,
        "verdict": final_verdict,
        "confidence": final_confidence,
        "method": final_method,
        "jmespath": {
            "success": jmespath_success,
            "results": jmespath_results
        },
        "llm": {
            "used": needs_llm,
            "verdict": llm_verdict,
            "confidence": llm_confidence,
            "explanation": llm_explanation
        } if needs_llm else None
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Hybrid validation: JMESPath + LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Single task ID to validate"
    )
    parser.add_argument(
        "--finish-json",
        type=Path,
        help="Finish JSON file for single task"
    )
    parser.add_argument(
        "--finish-jsons-dir",
        type=Path,
        help="Directory containing finish JSONs"
    )
    parser.add_argument(
        "--use-llm-fallback",
        action="store_true",
        help="Use LLM judge for uncertain cases"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro-preview-05-06",
        help="Gemini model to use (default: gemini-2.5-pro-preview-05-06)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hybrid_validation_results.json"),
        help="Output file for results"
    )

    args = parser.parse_args()

    # Load tasks
    from agisdk.REAL.tasks import all_tasks

    results = []

    if args.task and args.finish_json:
        # Single task validation
        task_config = next((t for t in all_tasks if t["id"] == args.task), None)
        if not task_config:
            print(f"Error: Task {args.task} not found")
            return 1

        finish_json = load_json(args.finish_json)
        result = hybrid_validate(task_config, finish_json, args.use_llm_fallback, args.model)
        results.append(result)

        print(f"\n{result['task_id']}: {result['verdict']} (confidence: {result['confidence']:.2f}, method: {result['method']})")
        if result.get("llm") and result["llm"]["explanation"]:
            print(f"  Explanation: {result['llm']['explanation']}")

    elif args.finish_jsons_dir:
        # Batch validation
        finish_jsons = list(args.finish_jsons_dir.glob("*.json"))
        print(f"Found {len(finish_jsons)} finish JSONs")

        for finish_path in sorted(finish_jsons):
            task_id = finish_path.stem
            task_config = next((t for t in all_tasks if t["id"] == task_id), None)

            if not task_config:
                print(f"⚠ {task_id}: Task definition not found, skipping")
                continue

            finish_json = load_json(finish_path)
            result = hybrid_validate(task_config, finish_json, args.use_llm_fallback, args.model)
            results.append(result)

            icon = "✅" if result["verdict"] == "SUCCESS" else "❌" if result["verdict"] == "FAILURE" else "❓"
            print(f"{icon} {result['task_id']}: {result['verdict']} (conf: {result['confidence']:.2f}, method: {result['method']})")

    else:
        print("Error: Must specify either --task + --finish-json or --finish-jsons-dir")
        return 1

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(results)}")
    print(f"Success: {sum(1 for r in results if r['verdict'] == 'SUCCESS')}")
    print(f"Failure: {sum(1 for r in results if r['verdict'] == 'FAILURE')}")
    print(f"Uncertain: {sum(1 for r in results if r['verdict'] == 'UNCERTAIN')}")
    print(f"\nMethods:")
    print(f"  JMESPath only: {sum(1 for r in results if r['method'] == 'jmespath')}")
    print(f"  LLM override: {sum(1 for r in results if r['method'] == 'llm_override')}")
    print(f"  LLM confirmed: {sum(1 for r in results if r['method'] == 'llm_confirmed')}")
    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
