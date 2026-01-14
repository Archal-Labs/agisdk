"""
Multi-REAL Task Registry

Loads and manages multi-app task definitions independently from main REAL registry.
Task IDs use the 'multi.' prefix: multi.gocalendar-gomail-1
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class MultiRealTask:
    """A multi-app benchmark task."""
    id: str  # e.g., "gocalendar-gomail-1"
    prefixed_id: str  # e.g., "multi.gocalendar-gomail-1"
    goal: str
    websites: list[dict]
    difficulty: str
    challenge_type: str
    evals: list[dict]
    points: int
    config: dict

    @property
    def website_ids(self) -> list[str]:
        return [w["id"] for w in self.websites]

    @property
    def is_multi_app(self) -> bool:
        return len(self.websites) > 1


class MultiRealRegistry:
    """Registry for multi-app benchmark tasks."""

    TASKS_DIR = Path(__file__).parent.parent / "tasks"
    PREFIX = "multi"

    def __init__(self):
        self._tasks: dict[str, MultiRealTask] = {}
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load all task JSON files from tasks directory."""
        for task_file in self.TASKS_DIR.glob("*.json"):
            if task_file.name.startswith("example"):
                continue  # Skip example files

            with open(task_file) as f:
                data = json.load(f)

            task_id = data["id"]
            task = MultiRealTask(
                id=task_id,
                prefixed_id=f"{self.PREFIX}.{task_id}",
                goal=data["goal"],
                websites=data["websites"],
                difficulty=data.get("difficulty", "hard"),
                challenge_type=data.get("challengeType", "action"),
                evals=data["evals"],
                points=data.get("points", 2),
                config=data.get("config", {}),
            )
            self._tasks[task_id] = task

    def get(self, task_id: str) -> MultiRealTask | None:
        """Get task by ID (with or without prefix)."""
        clean_id = task_id.removeprefix(f"{self.PREFIX}.")
        return self._tasks.get(clean_id)

    def all(self) -> Iterator[MultiRealTask]:
        """Iterate over all tasks."""
        yield from self._tasks.values()

    def filter(
        self,
        websites: list[str] | None = None,
        difficulty: str | None = None,
        min_apps: int | None = None,
        max_apps: int | None = None,
    ) -> Iterator[MultiRealTask]:
        """Filter tasks by criteria."""
        for task in self._tasks.values():
            if websites and not any(w in task.website_ids for w in websites):
                continue
            if difficulty and task.difficulty != difficulty:
                continue
            if min_apps and len(task.websites) < min_apps:
                continue
            if max_apps and len(task.websites) > max_apps:
                continue
            yield task

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[MultiRealTask]:
        return self.all()


# Global registry instance
registry = MultiRealRegistry()
