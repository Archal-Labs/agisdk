"""
Generate task JSON files from tasks.csv with placeholder evals.

IMPORTANT: The generated JMESPath queries are placeholders and must be manually refined.
After running this script, for each task:
1. Open the website(s) in a browser and complete the task manually
2. Navigate to {website_url}/finish and copy the JSON
3. Use the Eval Genie tool (https://eval-genie-checks-maker.lovable.app/) to generate proper JMESPath queries
4. Replace the placeholder evals in the generated JSON file

Usage:
    export VERTEX_PROJECT="datagen-483517"
    export VERTEX_LOCATION="us-central1"  # optional, defaults to us-central1
    python gen_jsons.py
"""

import csv
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Website Metadata Mapping
# =============================================================================

WEBSITE_METADATA: dict[str, dict[str, str]] = {
    "gomail": {
        "name": "GoMail",
        "similarTo": "Gmail",
        "previewImage": "/websitePreviews/gomail_preview.jpg",
        "url": "https://real-gomail.vercel.app",
    },
    "omnizon": {
        "name": "Omnizon",
        "similarTo": "Amazon",
        "previewImage": "/websitePreviews/omnizon_preview.jpg",
        "url": "https://real-omnizon.vercel.app",
    },
    "zilloft": {
        "name": "Zilloft",
        "similarTo": "Zillow",
        "previewImage": "/websitePreviews/zilloft_preview.jpg",
        "url": "https://real-zilloft.vercel.app",
    },
    "gocalendar": {
        "name": "GoCalendar",
        "similarTo": "Google Calendar",
        "previewImage": "/websitePreviews/gocalendar_preview.jpg",
        "url": "https://real-gocalendar.vercel.app",
    },
    "opendining": {
        "name": "OpenDining",
        "similarTo": "OpenTable",
        "previewImage": "/websitePreviews/opendining_preview.jpg",
        "url": "https://real-opendining.vercel.app",
    },
    "udriver": {
        "name": "Udriver",
        "similarTo": "Uber",
        "previewImage": "/websitePreviews/udriver_preview.jpg",
        "url": "https://real-udriver.vercel.app",
    },
    "dashdish": {
        "name": "DashDish",
        "similarTo": "DoorDash",
        "previewImage": "/websitePreviews/dashdish_preview.jpg",
        "url": "https://real-dashdish.vercel.app",
    },
    "topwork": {
        "name": "TopWork",
        "similarTo": "Indeed",
        "previewImage": "/websitePreviews/topwork_preview.jpg",
        "url": "https://real-topwork.vercel.app",
    },
    "staynb": {
        "name": "StaynB",
        "similarTo": "Airbnb",
        "previewImage": "/websitePreviews/staynb_preview.jpg",
        "url": "https://real-staynb.vercel.app",
    },
    "networkin": {
        "name": "NetworkIn",
        "similarTo": "LinkedIn",
        "previewImage": "/websitePreviews/networkin_preview.jpg",
        "url": "https://real-networkin.vercel.app",
    },
    "marrisuite": {
        "name": "Marrisuite",
        "similarTo": "Marriott",
        "previewImage": "/websitePreviews/marrisuite_preview.jpg",
        "url": "https://real-marrisuite.vercel.app",
    },
    "flyunified": {
        "name": "FlyUnified",
        "similarTo": "United Airlines",
        "previewImage": "/websitePreviews/flyunified_preview.jpg",
        "url": "https://real-flyunified.vercel.app",
    },
}

# Map CSV app names to JSON IDs
APP_NAME_TO_ID = {
    "DashDish": "dashdish",
    "GoMail": "gomail",
    "FlyUnified": "flyunified",
    "GoCalendar": "gocalendar",
    "StaynB": "staynb",
    "StayNB": "staynb",
    "OpenDining": "opendining",
    "Zilloft": "zilloft",
    "Marrisuite": "marrisuite",
    "TopWork": "topwork",
    "Omnizon": "omnizon",
    "NetworkIn": "networkin",
    "Udriver": "udriver",
    "UDriver": "udriver",
}


# =============================================================================
# URL Parsing Functions
# =============================================================================

def extract_website_id_from_url(url: str) -> str | None:
    """
    Extract website ID from URL pattern like 'https://real-{website_id}.vercel.app/'.
    
    Args:
        url: Full URL string
        
    Returns:
        Website ID (e.g., 'gomail', 'omnizon') or None if not found
    """
    url = url.strip()
    # Pattern: real-{website_id}.vercel.app or real-{website_id}-*.vercel.app
    match = re.search(r"real-([a-zA-Z]+)(?:-[a-zA-Z0-9]+)?\.vercel\.app", url)
    if match:
        return match.group(1).lower()
    
    # Also handle evals- prefix pattern
    match = re.search(r"evals-([a-zA-Z]+)(?:-[a-zA-Z0-9]+)?\.vercel\.app", url)
    if match:
        return match.group(1).lower()
    
    return None


def normalize_url(url: str) -> str:
    """
    Normalize URL by stripping trailing slashes and path components.
    
    Args:
        url: Full URL string
        
    Returns:
        Normalized base URL
    """
    url = url.strip()
    # Remove trailing path components (e.g., /calendar, /platform/dashboard/)
    # Keep just the base domain
    match = re.match(r"(https?://[^/]+)", url)
    if match:
        return match.group(1)
    return url.rstrip("/")


def parse_websites_from_names(websites_str: str) -> list[dict[str, str]]:
    """
    Parse comma-separated website names into list of website configurations.
    
    Args:
        websites_str: Comma-separated website names from CSV (e.g., "DashDish, GoMail")
        
    Returns:
        List of website dicts with id, name, similarTo, previewImage, url
    """
    websites = []
    website_names = [w.strip() for w in websites_str.split(",") if w.strip()]
    
    for name in website_names:
        website_id = APP_NAME_TO_ID.get(name)
        if not website_id:
            logger.warning(f"Unknown website name '{name}', skipping")
            continue
            
        if website_id not in WEBSITE_METADATA:
            logger.warning(f"Website ID '{website_id}' not in metadata, skipping")
            continue
        
        metadata = WEBSITE_METADATA[website_id]
        
        websites.append({
            "id": website_id,
            "name": metadata["name"],
            "similarTo": metadata["similarTo"],
            "previewImage": metadata["previewImage"],
            "url": metadata["url"],
        })
    
    return websites


# =============================================================================
# CSV Reading and Parsing
# =============================================================================

def read_tasks_csv(csv_path: Path) -> list[dict[str, Any]]:
    """
    Read tasks from CSV file.
    
    Args:
        csv_path: Path to tasks.csv
        
    Returns:
        List of task dicts with keys: prompt, possible, websites
    """
    assert csv_path.exists(), f"CSV file not found: {csv_path}"
    
    tasks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if not prompt:
                continue
            
            possible_str = row.get("possible", "True").strip()
            possible = possible_str.lower() == "true"
            websites = row.get("websites", "").strip()
            
            tasks.append({
                "prompt": prompt,
                "possible": possible,
                "websites": websites,
            })
    
    logger.info(f"Read {len(tasks)} tasks from {csv_path}")
    return tasks


# =============================================================================
# Task ID Generation
# =============================================================================

class TaskIdGenerator:
    """
    Generate unique task IDs based on website combinations.
    
    For multi-app tasks: {website1}-{website2}-{number}
    For single-app tasks: {website1}-{number}
    """
    
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
    
    def generate_id(self, website_ids: list[str]) -> str:
        """
        Generate a unique task ID for the given website combination.
        
        Args:
            website_ids: List of website IDs (e.g., ['gomail', 'omnizon'])
            
        Returns:
            Unique task ID (e.g., 'gomail-omnizon-1')
        """
        assert len(website_ids) > 0, "Must have at least one website"
        
        # Sort website IDs alphabetically for consistent naming
        sorted_ids = sorted(website_ids)
        combo_key = "-".join(sorted_ids)
        
        self._counters[combo_key] += 1
        count = self._counters[combo_key]
        
        return f"{combo_key}-{count}"


# =============================================================================
# Vertex AI Initialization and Eval Generation
# =============================================================================

def init_vertex_ai() -> None:
    """Initialize Vertex AI with project and location from environment."""
    project = os.environ.get("VERTEX_PROJECT")
    assert project, "VERTEX_PROJECT environment variable must be set"
    
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    aiplatform.init(project=project, location=location)
    logger.info(f"Initialized Vertex AI with project={project}, location={location}")


def generate_placeholder_evals(
    model: GenerativeModel,
    goal: str,
    workflow_guide: str,
    website_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Generate placeholder evals using Vertex AI.
    
    IMPORTANT: These are placeholder evals with TODO markers.
    They must be manually refined using the actual /finish JSON structure.
    
    Args:
        model: Vertex AI GenerativeModel instance
        goal: Task goal/prompt
        workflow_guide: Step-by-step workflow description
        website_ids: List of website IDs involved
        
    Returns:
        List of eval dicts with placeholder queries
    """
    is_multi_app = len(website_ids) > 1
    
    prompt = f"""Task Goal: {goal}

Workflow: {workflow_guide}

Websites: {', '.join(website_ids)}

Generate placeholder evaluation criteria (evals) for this {'multi-app' if is_multi_app else 'single-app'} task.
These will be manually refined using the actual /finish JSON structure.

FINISH JSON STRUCTURE:
Each app's finish JSON contains: actionhistory, initialstate, finalstate, state, initialfinaldiff (raw diff), and differences (semantic diff by entity type).
{"For multi-app tasks, the finish JSON is structured as: {{appname1: {{...}}, appname2: {{...}}}}" if is_multi_app else "For single-app tasks, the finish JSON is the app's state directly."}

GENERAL QUERY PATTERNS:
- Checking for added items: appname.differences.entity_type.added[0]
- Counting items: length(appname.differences.entity_type.added) >= `1` (use backticks around numbers, use >= `1` not > 0)
- Field value checks: contains(appname.differences.entity_type.added[0].field, 'value')
- Dict to list conversion: appname.differences.entity_type.added.values(@)[0] (for dicts, convert with values(@))

APP-SPECIFIC PATTERNS:

gomail (Email):
Structure: differences.emails.added[], differences.emails.sent[], differences.emails.drafts[]
Common queries:
  - gomail.differences.emails.sent[0] (check email was sent)
  - contains(gomail.differences.emails.sent[0].content, 'text') (check email contains text)
  - contains(gomail.differences.emails.sent[0].subject, 'Subject Text') (check subject)
  - contains(gomail.differences.emails.sent[0].recipient, 'email@example.com') (check recipient)
  - length(gomail.differences.emails.sent) >= `1` (count sent emails)

gocalendar (Calendar):
Structure: differences.events.added{{}} (dict, use values()), differences.calendars{{}}, differences.joinedEvents{{}}
Common queries:
  - gocalendar.differences.events.added.values(@)[0] (get first event)
  - contains(gocalendar.differences.events.added.values(@)[0].title, 'Meeting') (check event title)
  - gocalendar.differences.events.added.values(@)[0].start.date (check event date)
  - gocalendar.differences.events.added.values(@)[0].start.dateTime (check event time)
  - length(gocalendar.differences.events.added) >= `1` (count events)

opendining (Restaurant Booking):
Structure: differences.bookings.added[], differences.reviews.added[], differences.savedRestaurants.added[]
Common queries:
  - opendining.differences.bookings.added[0] (check booking exists)
  - contains(opendining.differences.bookings.added[0].restaurantName, 'Restaurant') (check restaurant name)
  - opendining.differences.bookings.added[0].partySize >= `2` (check party size)
  - opendining.differences.bookings.added[0].date (check date)

udriver (Ride Booking):
Structure: initialfinaldiff.added.ride.bookedTrips{{}}, finalstate.ride.bookedTrip, finalstate.ride.calculatedPrice
Common queries:
  - udriver.finalstate.ride.bookedTrip (check trip exists)
  - contains(udriver.finalstate.ride.bookedTrip.pickup, 'Location') (check pickup location)
  - contains(udriver.finalstate.ride.bookedTrip.destination, 'Location') (check destination)
  - udriver.initialfinaldiff.added.ride.bookedTrips.values(@) (get all booked trips)

flyunified (Flight Booking):
Structure: differences.bookedFlights[], differences.selectedFlightCartIds[], differences.purchaseDetails{{}}
Common queries:
  - flyunified.differences.bookedFlights[0] (check flight booked)
  - contains(flyunified.differences.bookedFlights[0].destination, 'City') (check destination)
  - flyunified.differences.bookedFlights[0].departureDate (check departure date)
  - length(flyunified.differences.bookedFlights) >= `1` (count flights)

dashdish (Food Delivery):
Structure: differences.foodOrders.added[], initialfinaldiff.added.cart{{}}
Common queries:
  - dashdish.differences.foodOrders.added[0] (check order exists)
  - contains(dashdish.differences.foodOrders.added[0].restaurant, 'Name') (check restaurant)
  - dashdish.initialfinaldiff.added.cart.values(@) (get cart items)

staynb (Hotel Booking):
Structure: differences.bookings.added[]
Common queries:
  - staynb.differences.bookings.added[0] (check booking exists)
  - contains(staynb.differences.bookings.added[0].hotelName, 'Hotel') (check hotel name)
  - staynb.differences.bookings.added[0].checkIn (check check-in date)
  - staynb.differences.bookings.added[0].checkOut (check check-out date)

topwork (Job Applications):
Structure: initialfinaldiff.added.jobs{{}} (dict, use values())
Common queries:
  - topwork.initialfinaldiff.added.jobs.values(@) (get applied jobs)
  - contains(topwork.initialfinaldiff.added.jobs.values(@)[0].title, 'Position') (check job title)
  - length(topwork.initialfinaldiff.added.jobs) >= `1` (count applications)

COMMON MISTAKES TO AVOID:
1. Wrong comparison operator: Use length(array) >= `1` NOT length(array) > 0 (backticks required around numbers)
2. Dict vs list confusion: For dicts like events.added{{}}, use .values(@)[0] NOT [0] directly
3. Wrong field names: Use .content NOT .body for emails, use .start.date NOT .date for events
4. Missing app prefix (multi-app): Use gomail.differences.emails.sent[0] NOT differences.emails.sent[0]
5. Wrong date/time access: Events use .start.date or .start.dateTime, NOT .date directly
6. Invalid functions: Use contains() NOT intersection() (intersection is not valid JMESPath)
7. String escaping: Escape apostrophes by doubling: 'can''t' NOT 'can't'
8. Arithmetic operations: JMESPath does NOT support +, -, *, / operations
9. Null safety: Check for null before accessing nested paths. Use: (path != null && path.field != null) for nested access
10. Using untracked entities: Only query entities that are tracked by the app (see app-specific patterns above). If a path doesn't exist, check if similar paths exist (e.g., differences.emails vs initialfinaldiff.added.email)

QUERY RELIABILITY GUIDELINES:
- Prefer using 'differences' paths over 'initialfinaldiff' when both exist (differences is semantic, initialfinaldiff is raw)
- If a query path might not exist, consider checking for existence first: (appname.differences.entity != null && appname.differences.entity.added[0])
- Common failure modes: query bugs (syntax errors), data incomplete (path missing but similar paths exist), schema missing (entity not tracked)
- Always use tracked entities from the app-specific patterns above - don't invent new paths

{"For multi-app tasks, all queries must be prefixed with the website ID (e.g., gomail.differences.emails.sent[0])." if is_multi_app else ""}

Return ONLY a valid JSON array of evals with this structure (no markdown, no explanation):
[
  {{
    "description": "Clear description of what should be verified",
    "type": "jmespath",
    "query": "tentative.jmespath.query",
    "expected_value": "TODO: Add expected value after checking /finish JSON"
  }}
]

IMPORTANT: 
- These queries are placeholders. Use "TODO: Add expected value after checking /finish JSON" for all expected_value fields.
- Generate 2-4 evals that cover the key success criteria for this task.
- Focus on verifying: actions completed, data created/modified, cross-app data flow.
- Follow the patterns above for correct query syntax.
- Return ONLY the JSON array, no other text.
"""

    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Try to extract JSON from the response
    # Sometimes the model wraps it in markdown code blocks
    json_match = re.search(r"\[[\s\S]*\]", response_text)
    if json_match:
        response_text = json_match.group()
    
    evals = json.loads(response_text)
    
    # Validate and normalize evals structure
    validated_evals = []
    for eval_item in evals:
        validated_eval = {
            "description": eval_item.get("description", "TODO: Add description"),
            "type": "jmespath",
            "query": eval_item.get("query", "TODO: Add query"),
            "expected_value": "TODO: Add expected value after checking /finish JSON",
        }
        validated_evals.append(validated_eval)
    
    return validated_evals

# =============================================================================
# Task JSON Structure Mapping
# =============================================================================

def create_task_json(
    task_id: str,
    goal: str,
    websites: list[dict[str, str]],
    evals: list[dict[str, Any]],
    task_category: str,
) -> dict[str, Any]:
    """
    Create complete task JSON structure.
    
    Args:
        task_id: Unique task identifier
        goal: Task goal/prompt
        websites: List of website configuration dicts
        evals: List of eval dicts
        task_category: Category from CSV (e.g., 'BROWSER_OS')
        
    Returns:
        Complete task JSON dict
    """
    # Map task_category to challengeType
    challenge_type = "action"  # Default for multi-app tasks
    if task_category:
        category_lower = task_category.lower()
        if "retrieval" in category_lower:
            challenge_type = "retrieval"
        elif "navigation" in category_lower:
            challenge_type = "navigation"
    
    # Determine difficulty and points based on number of websites
    is_multi_app = len(websites) > 1
    difficulty = "hard" if is_multi_app else "medium"
    points = 2 if is_multi_app else 1
    
    task_json: dict[str, Any] = {
        "id": task_id,
        "goal": goal,
        "websites": websites,
        "difficulty": difficulty,
        "challengeType": challenge_type,
        "possible": True,
        "evals": evals,
        "points": points,
        "config": {},
    }
    
    return task_json


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    """Main entry point for generating task JSON files."""
    script_dir = Path(__file__).parent
    csv_path = script_dir / "tasks.csv"
    output_dir = script_dir / "tasks"
    output_dir.mkdir(exist_ok=True)
    
    init_vertex_ai()
    model = GenerativeModel("gemini-2.5-flash")
    
    # Read tasks from CSV
    tasks = read_tasks_csv(csv_path)
    
    # Initialize task ID generator
    id_generator = TaskIdGenerator()
    
    # Track statistics
    success_count = 0
    error_count = 0
    
    for i, task in enumerate(tasks):
        logger.info(f"Processing task {i + 1}/{len(tasks)}...")
        
        # Parse websites from comma-separated names
        websites = parse_websites_from_names(task["websites"])
        if not websites:
            logger.error(f"No valid websites found for task: {task['prompt'][:50]}...")
            error_count += 1
            continue
        
        website_ids = [w["id"] for w in websites]
        
        # Generate task ID
        task_id = id_generator.generate_id(website_ids)
        
        evals = generate_placeholder_evals(
            model=model,
            goal=task["prompt"],
            workflow_guide="",  # Not available in current CSV format
            website_ids=website_ids,
        )
        
        # Create complete task JSON
        task_json = create_task_json(
            task_id=task_id,
            goal=task["prompt"],
            websites=websites,
            evals=evals,
            task_category="",  # Not available in current CSV format
        )
        task_json["possible"] = task["possible"]
        output_path = output_dir / f"{task_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task_json, f, indent=2)
        
        logger.info(f"  Wrote: {output_path}")
        success_count += 1

if __name__ == "__main__":
    main()
