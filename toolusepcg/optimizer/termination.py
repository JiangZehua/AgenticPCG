"""Termination condition checking."""

from __future__ import annotations

from ..config import Config


class TerminationChecker:
    """Checks termination conditions for optimization."""

    def __init__(self, config: Config):
        """Initialize termination checker.

        Args:
            config: Configuration with termination settings.
        """
        self.config = config
        self.multiplier = config.termination.change_budget_multiplier
        self.height = config.env.height
        self.width = config.env.width

        # Compute budget
        self.change_budget = int(self.multiplier * self.height * self.width)

        # Track changes
        self._total_changes = 0

    @property
    def total_changes(self) -> int:
        """Total tiles changed so far."""
        return self._total_changes

    @property
    def remaining_budget(self) -> int:
        """Remaining change budget."""
        return max(0, self.change_budget - self._total_changes)

    def add_changes(self, num_changes: int) -> None:
        """Record tile changes.

        Args:
            num_changes: Number of tiles changed.
        """
        self._total_changes += num_changes

    def should_terminate(self) -> bool:
        """Check if optimization should terminate.

        Returns:
            True if change budget exhausted.
        """
        return self._total_changes >= self.change_budget

    def set_total_changes(self, total: int) -> None:
        """Set total changes directly (used for resume).

        Args:
            total: Total tiles changed to restore.
        """
        self._total_changes = total

    def reset(self) -> None:
        """Reset change counter."""
        self._total_changes = 0

    def status(self) -> dict:
        """Get current termination status.

        Returns:
            Dictionary with budget info.
        """
        return {
            "total_changes": self._total_changes,
            "change_budget": self.change_budget,
            "remaining_budget": self.remaining_budget,
            "budget_exhausted": self.should_terminate(),
            "budget_percent_used": (self._total_changes / self.change_budget * 100) if self.change_budget > 0 else 100,
        }

    def format_status(self) -> str:
        """Format status for display.

        Returns:
            Human-readable status string.
        """
        status = self.status()
        return (
            f"Changes: {status['total_changes']}/{status['change_budget']} "
            f"({status['budget_percent_used']:.1f}% used, "
            f"{status['remaining_budget']} remaining)"
        )
