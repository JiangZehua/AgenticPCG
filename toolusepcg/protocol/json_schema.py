"""JSON schema definitions for agent messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """Types of messages the agent can produce."""

    STEP = "STEP"
    PROPOSE_SKILL = "PROPOSE_SKILL"
    STOP = "STOP"


@dataclass
class ToolCall:
    """Represents a single tool call."""

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> ToolCall:
        return cls(
            tool_name=d.get("tool_name", d.get("name", "")),
            parameters=d.get("parameters", d.get("params", d.get("arguments", {}))),
        )

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
        }


@dataclass
class StepMessage:
    """STEP message: agent wants to execute tools."""

    type: MessageType = MessageType.STEP
    rationale: str = ""
    plan: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    acceptance_hint: str = ""  # Hint about whether to accept result

    @classmethod
    def from_dict(cls, d: dict) -> StepMessage:
        tool_calls_raw = d.get("tool_calls", [])
        tool_calls = [ToolCall.from_dict(tc) for tc in tool_calls_raw]
        return cls(
            rationale=d.get("rationale", ""),
            plan=d.get("plan", ""),
            tool_calls=tool_calls,
            acceptance_hint=d.get("acceptance_hint", ""),
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "rationale": self.rationale,
            "plan": self.plan,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "acceptance_hint": self.acceptance_hint,
        }


@dataclass
class ProposeSkillMessage:
    """PROPOSE_SKILL message: agent proposes a new tool."""

    type: MessageType = MessageType.PROPOSE_SKILL
    rationale: str = ""
    skill_spec: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> ProposeSkillMessage:
        return cls(
            rationale=d.get("rationale", ""),
            skill_spec=d.get("skill_spec", {}),
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "rationale": self.rationale,
            "skill_spec": self.skill_spec,
        }


@dataclass
class StopMessage:
    """STOP message: agent wants to terminate."""

    type: MessageType = MessageType.STOP
    rationale: str = ""
    final_notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> StopMessage:
        return cls(
            rationale=d.get("rationale", ""),
            final_notes=d.get("final_notes", ""),
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "rationale": self.rationale,
            "final_notes": self.final_notes,
        }


def validate_message(d: dict) -> list[str]:
    """Validate a message dictionary against schema.

    Args:
        d: Dictionary to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    if "type" not in d:
        errors.append("Missing required field: type")
        return errors

    msg_type = d.get("type", "").upper()

    if msg_type == "STEP":
        if "tool_calls" not in d:
            errors.append("STEP message missing required field: tool_calls")
        elif not isinstance(d["tool_calls"], list):
            errors.append("tool_calls must be a list")
        else:
            for i, tc in enumerate(d["tool_calls"]):
                if not isinstance(tc, dict):
                    errors.append(f"tool_calls[{i}] must be an object")
                elif "tool_name" not in tc and "name" not in tc:
                    errors.append(f"tool_calls[{i}] missing tool_name")

    elif msg_type == "PROPOSE_SKILL":
        if "skill_spec" not in d:
            errors.append("PROPOSE_SKILL message missing required field: skill_spec")
        elif not isinstance(d["skill_spec"], dict):
            errors.append("skill_spec must be an object")
        else:
            spec = d["skill_spec"]
            if "name" not in spec:
                errors.append("skill_spec missing required field: name")
            if "description" not in spec:
                errors.append("skill_spec missing required field: description")

    elif msg_type == "STOP":
        # STOP has no required fields beyond type
        pass

    else:
        errors.append(f"Invalid message type: {msg_type}")

    return errors


def parse_message(d: dict) -> StepMessage | ProposeSkillMessage | StopMessage:
    """Parse a validated message dictionary into typed object.

    Args:
        d: Validated message dictionary.

    Returns:
        Typed message object.

    Raises:
        ValueError: If message type is invalid.
    """
    msg_type = d.get("type", "").upper()

    if msg_type == "STEP":
        return StepMessage.from_dict(d)
    elif msg_type == "PROPOSE_SKILL":
        return ProposeSkillMessage.from_dict(d)
    elif msg_type == "STOP":
        return StopMessage.from_dict(d)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")


# JSON Schema definitions for documentation/validation
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "STEP"},
        "rationale": {"type": "string", "description": "Explanation of reasoning"},
        "plan": {"type": "string", "description": "High-level plan for this step"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "parameters": {"type": "object"},
                },
                "required": ["tool_name"],
            },
        },
        "acceptance_hint": {"type": "string", "description": "Hint about accepting result"},
    },
    "required": ["type", "tool_calls"],
}

PROPOSE_SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "PROPOSE_SKILL"},
        "rationale": {"type": "string"},
        "skill_spec": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "parameters": {"type": "object"},
                "implementation_hint": {"type": "string"},
            },
            "required": ["name", "description"],
        },
    },
    "required": ["type", "skill_spec"],
}

STOP_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "STOP"},
        "rationale": {"type": "string"},
        "final_notes": {"type": "string"},
    },
    "required": ["type"],
}
