from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

def init_vertex_ai() -> None:
    """Initialize Vertex AI with project and location from environment."""
    project = os.environ.get("VERTEX_PROJECT")
    assert project, "VERTEX_PROJECT environment variable must be set"
    
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    aiplatform.init(project=project, location=location)


def load_existing_multiapp_tasks(csv_path: Path) -> dict[str, list[str]]:
    """
    Load existing multi-app tasks from CSV, grouped by app combination.
    
    Args:
        csv_path: Path to existing multiapp_tasks CSV file
        
    Returns:
        Dict mapping app combination string (sorted, comma-separated) to list of prompts
    """
    if not csv_path.exists():
        return {}
    
    existing_by_combo: dict[str, list[str]] = defaultdict(list)
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if not prompt:
                continue
            
            # Infer app combination from assets_url (primary) or apps_involved (fallback)
            assets_url = row.get("assets_url", "").strip()
            apps_involved = None
            
            if assets_url:
                # Primary: infer from assets_url column
                apps_involved = _infer_apps_from_assets(assets_url)
            else:
                # Fallback: try apps_involved column if present
                apps_involved = row.get("apps_involved", "").strip()
                if apps_involved:
                    # Normalize: parse and sort
                    apps_list = [a.strip().lower() for a in apps_involved.split(",")]
                    apps_involved = ",".join(sorted(apps_list))
            
            if apps_involved:
                # Normalize: sort apps to create consistent key
                apps_list = [a.strip().lower() for a in apps_involved.split(",")]
                combo_key = ",".join(sorted(apps_list))
                existing_by_combo[combo_key].append(prompt)
            else:
                # Fallback: try to detect apps from prompt text
                prompt_lower = prompt.lower()
                detected_apps = []
                for app in APP_DESCRIPTIONS:
                    # Handle 'go' prefix variations
                    app_variants = [app, app.replace("go", "go "), app.replace("go", "gocalendar" if app == "gocalendar" else "go")]
                    if any(variant in prompt_lower for variant in app_variants):
                        detected_apps.append(app)
                if len(detected_apps) >= 2:
                    combo_key = ",".join(sorted(detected_apps))
                    existing_by_combo[combo_key].append(prompt)
    
    return dict(existing_by_combo)


