"""Skill growth manager for collecting and persisting proposed tool specs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SkillGrowthManager:
    """Manages skill proposals from the agent."""

    def __init__(self, run_dir: str | Path):
        """Initialize skill growth manager.

        Args:
            run_dir: Directory for the current run.
        """
        self.run_dir = Path(run_dir)
        self.skill_specs_dir = self.run_dir / "skill_specs"
        self.skill_specs_dir.mkdir(parents=True, exist_ok=True)
        self._proposals: list[dict] = []

    def propose(self, skill_spec: dict[str, Any], rationale: str = "") -> str:
        """Record a skill proposal.

        Args:
            skill_spec: Skill specification dictionary with at least 'name' and 'description'.
            rationale: Explanation for why this skill is proposed.

        Returns:
            Path to saved skill spec file.
        """
        # Validate required fields
        if "name" not in skill_spec:
            raise ValueError("skill_spec must have 'name' field")
        if "description" not in skill_spec:
            raise ValueError("skill_spec must have 'description' field")

        # Create proposal record
        proposal = {
            "skill_spec": skill_spec,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
        }
        self._proposals.append(proposal)

        # Save to file
        name = skill_spec["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{name}.json"
        filepath = self.skill_specs_dir / filename

        with open(filepath, "w") as f:
            json.dump(proposal, f, indent=2)

        return str(filepath)

    def get_proposals(self) -> list[dict]:
        """Get all proposals from this session.

        Returns:
            List of proposal records.
        """
        return self._proposals.copy()

    def load_existing_proposals(self) -> list[dict]:
        """Load all existing proposals from skill_specs directory.

        Returns:
            List of all proposal records from files.
        """
        proposals = []
        for filepath in sorted(self.skill_specs_dir.glob("*.json")):
            try:
                with open(filepath) as f:
                    proposal = json.load(f)
                    proposals.append(proposal)
            except (json.JSONDecodeError, IOError):
                continue
        return proposals

    def summarize(self) -> dict:
        """Get summary of skill proposals.

        Returns:
            Summary dictionary.
        """
        return {
            "total_proposals": len(self._proposals),
            "skills_proposed": [p["skill_spec"]["name"] for p in self._proposals],
            "skill_specs_dir": str(self.skill_specs_dir),
        }
