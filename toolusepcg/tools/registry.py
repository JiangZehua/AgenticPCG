"""Tool registry for registering and dispatching tool calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..edit_manager import EditManager


@dataclass
class ToolResult:
    """Result of executing a tool."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
        }


class Tool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for registration and invocation."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_spec(self) -> dict:
        """Convert tool to specification dict for LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class EditTool(Tool):
    """Base class for tools that modify the level.

    All edit tools MUST use EditManager for modifications.
    This ensures consistent validation and tracking.
    """

    def __init__(self, edit_manager: "EditManager"):
        """Initialize with EditManager.

        Args:
            edit_manager: Central edit manager for level modifications.
        """
        self._edit_manager = edit_manager

    @property
    def edit_manager(self) -> "EditManager":
        """Access the edit manager."""
        return self._edit_manager

    def set_edit_manager(self, edit_manager: "EditManager") -> None:
        """Update the edit manager reference.

        Args:
            edit_manager: New edit manager.
        """
        self._edit_manager = edit_manager


class ToolRegistry:
    """Registry for managing and dispatching tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If tool with same name already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name.
            **kwargs: Tool parameters.

        Returns:
            ToolResult from tool execution.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                errors=[f"Unknown tool: {name}"],
            )

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                errors=[f"Tool execution failed: {e}"],
            )

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_specs(self) -> list[dict]:
        """Get specifications for all registered tools."""
        return [tool.to_spec() for tool in self._tools.values()]

    def format_tools_for_prompt(self) -> str:
        """Format all tools for inclusion in LLM prompt."""
        lines = ["Available Tools:"]
        for tool in self._tools.values():
            lines.append(f"\n## {tool.name}")
            lines.append(f"Description: {tool.description}")
            lines.append("Parameters:")
            schema = tool.parameters_schema
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for param_name, param_spec in properties.items():
                req_marker = "(required)" if param_name in required else "(optional)"
                param_type = param_spec.get("type", "any")
                param_desc = param_spec.get("description", "")
                lines.append(f"  - {param_name} ({param_type}) {req_marker}: {param_desc}")
        return "\n".join(lines)
