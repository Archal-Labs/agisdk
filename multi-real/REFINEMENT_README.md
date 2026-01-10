# Manual Eval Refinement Guide

## Overview

The `gen_jsons.py` script generates task JSON files with **placeholder evals** that must be manually refined. The LLM cannot generate accurate JMESPath queries because it doesn't have access to the actual JSON structure from the `/finish` endpoint.

## Workflow

### Step 1: Run the Generator Script

```bash
export VERTEX_PROJECT="your-gcp-project-id"
export VERTEX_LOCATION="us-central1"  # optional
python gen_jsons.py
```

This creates JSON files in `multi-real/tasks/` with placeholder evals like:

```json
{
  "description": "Order was placed on DashDish",
  "type": "jmespath",
  "query": "dashdish.foodOrders != null && length(dashdish.foodOrders.added) > `0`",
  "expected_value": "TODO: Add expected value after checking /finish JSON"
}
```

### Step 2: Collect /finish JSON

For each generated task:

1. **Open the website(s)** in a browser (use incognito/private mode for clean state)
2. **Complete the task manually** following the goal description
3. **Navigate to `{website_url}/finish`** (e.g., `https://real-gomail.vercel.app/finish`)
4. **Copy the JSON output** from the page

For multi-app tasks, you need the JSON from each website's `/finish` endpoint.

### Step 3: Use Eval Genie Tool

1. Go to: **https://eval-genie-checks-maker.lovable.app/**
2. Paste the `/finish` JSON
3. Paste the task goal
4. The tool will generate candidate JMESPath queries
5. Test each query to verify it returns the expected value
6. Select the queries you want to use

### Step 4: Update JSON Files

Replace placeholder evals in the generated JSON file with the refined queries from Eval Genie:

**Before:**
```json
{
  "description": "Email forwarded to finance@corp.com",
  "type": "jmespath",
  "query": "gomail.forwardedEmails.added[0].to",
  "expected_value": "TODO: Add expected value after checking /finish JSON"
}
```

**After:**
```json
{
  "description": "Email forwarded to finance@corp.com",
  "type": "jmespath",
  "query": "gomail.forwardedEmails.added[0].to",
  "expected_value": "finance@corp.com"
}
```

## Multi-App Task Considerations

For multi-app tasks, JMESPath queries must be prefixed with the website ID:

| Task Type | Query Format |
|-----------|--------------|
| Single-app | `orders.added[0].total` |
| Multi-app | `omnizon.orders.added[0].total` |

The combined `/finish` JSON for multi-app tasks has this structure:

```json
{
  "gomail": {
    "forwardedEmails": { "added": [...] },
    "sentEmails": { "added": [...] }
  },
  "omnizon": {
    "orders": { "added": [...] },
    "cart": { "items": [...] }
  }
}
```

## Common JMESPath Patterns

```
length(array) > `0`                    # Array has items
array[0].field                         # First item's field
object.nested.value                    # Nested value access
array[?field == 'value']               # Filter array
array[*].field                         # All values of a field
contains(string, 'substring')          # String contains check
object != null && length(object) > `0` # Object exists with items
```

## Verification

After refining evals, verify they work:

```python
from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig

# Load your task
config = TaskConfig("path/to/your/task.json", is_path=True)

# Paste the /finish JSON you collected
env_state = { ... }

# Run evaluation
evaluator = WebCloneEvaluator(task_config=config)
reward, done, message, info = evaluator.evaluate(
    env_state=env_state,
    model_response=""
)

print(f"Reward: {reward}")
print(f"Results: {info}")
```

## Resources

- **Eval Genie Tool**: https://eval-genie-checks-maker.lovable.app/
- **JMESPath Tutorial**: https://jmespath.org/tutorial.html
- **Task Development Guide**: `docs/Task Development Guide.md`

