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
    },
    "omnizon": {
        "name": "Omnizon",
        "similarTo": "Amazon",
        "previewImage": "/websitePreviews/omnizon_preview.jpg",
    },
    "zilloft": {
        "name": "Zilloft",
        "similarTo": "Zillow",
        "previewImage": "/websitePreviews/zilloft_preview.jpg",
    },
    "gocalendar": {
        "name": "GoCalendar",
        "similarTo": "Google Calendar",
        "previewImage": "/websitePreviews/gocalendar_preview.jpg",
    },
    "opendining": {
        "name": "OpenDining",
        "similarTo": "OpenTable",
        "previewImage": "/websitePreviews/opendining_preview.jpg",
    },
    "udriver": {
        "name": "Udriver",
        "similarTo": "Uber",
        "previewImage": "/websitePreviews/udriver_preview.jpg",
    },
    "dashdish": {
        "name": "DashDish",
        "similarTo": "DoorDash",
        "previewImage": "/websitePreviews/dashdish_preview.jpg",
    },
    "topwork": {
        "name": "TopWork",
        "similarTo": "Indeed",
        "previewImage": "/websitePreviews/topwork_preview.jpg",
    },
    "staynb": {
        "name": "StaynB",
        "similarTo": "Airbnb",
        "previewImage": "/websitePreviews/staynb_preview.jpg",
    },
    "networkin": {
        "name": "NetworkIn",
        "similarTo": "LinkedIn",
        "previewImage": "/websitePreviews/networkin_preview.jpg",
    },
    "marrisuite": {
        "name": "Marrisuite",
        "similarTo": "Marriott",
        "previewImage": "/websitePreviews/marrisuite_preview.jpg",
    },
    "flyunified": {
        "name": "FlyUnified",
        "similarTo": "United Airlines",
        "previewImage": "/websitePreviews/flyunified_preview.jpg",
    },
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


def parse_assets_url(assets_url: str) -> list[dict[str, str]]:
    """
    Parse comma-separated assets_url field into list of website configurations.
    
    Args:
        assets_url: Comma-separated URL string from CSV
        
    Returns:
        List of website dicts with id, name, similarTo, previewImage, url
    """
    websites = []
    urls = [u.strip() for u in assets_url.split(",") if u.strip()]
    
    for url in urls:
        website_id = extract_website_id_from_url(url)
        if not website_id:
            logger.warning(f"Could not extract website ID from URL: {url}")
            continue
            
        if website_id not in WEBSITE_METADATA:
            logger.warning(f"Unknown website ID: {website_id} from URL: {url}")
            continue
        
        metadata = WEBSITE_METADATA[website_id]
        normalized_url = normalize_url(url)
        
        websites.append({
            "id": website_id,
            "name": metadata["name"],
            "similarTo": metadata["similarTo"],
            "previewImage": metadata["previewImage"],
            "url": normalized_url,
        })
    
    return websites


# =============================================================================
# CSV Reading and Parsing
# =============================================================================

def read_tasks_csv(csv_path: Path) -> list[dict[str, str]]:
    """
    Read tasks from CSV file.
    
    Args:
        csv_path: Path to tasks.csv
        
    Returns:
        List of task dicts with keys: prompt, assets_url, workflow_guide, task_category
    """
    assert csv_path.exists(), f"CSV file not found: {csv_path}"
    
    tasks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if not prompt:
                continue
            
            tasks.append({
                "prompt": prompt,
                "assets_url": row.get("assets_url", "").strip(),
                "workflow_guide": row.get("workflow_guide", "").strip(),
                "task_category": row.get("task_category", "").strip(),
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

{"For multi-app tasks, JMESPath queries must be prefixed with website IDs like:" if is_multi_app else "JMESPath queries check the final state from the /finish endpoint. Examples:"}
{f"- `{website_ids[0]}.orders.added[0]`" if is_multi_app else "- `orders.added[0]`"}
{f"- `{website_ids[1]}.emails.added[0].to`" if is_multi_app and len(website_ids) > 1 else "- `emails.added[0].to`"}

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
        
        # Parse websites from assets_url
        websites = parse_assets_url(task["assets_url"])
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
            workflow_guide=task["workflow_guide"],
            website_ids=website_ids,
        )
        
        # Create complete task JSON
        task_json = create_task_json(
            task_id=task_id,
            goal=task["prompt"],
            websites=websites,
            evals=evals,
            task_category=task["task_category"],
        )
        output_path = output_dir / f"{task_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task_json, f, indent=2)
        
        logger.info(f"  Wrote: {output_path}")
        success_count += 1

if __name__ == "__main__":
    main()
