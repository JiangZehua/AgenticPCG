"""JSON extraction and repair utilities."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> str | None:
    """Extract JSON from text that may contain other content.

    Handles:
    - JSON wrapped in markdown code blocks
    - JSON with leading/trailing text
    - Multiple JSON objects (returns first valid one)

    Args:
        text: Raw text that may contain JSON.

    Returns:
        Extracted JSON string, or None if not found.
    """
    if not text:
        return None

    # Try to parse the whole text first
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    # Try to extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_patterns = [
        r"```json\s*\n?(.*?)\n?```",
        r"```\s*\n?(.*?)\n?```",
    ]

    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            match = match.strip()
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                # Try repair
                repaired = repair_json(match)
                if repaired:
                    return repaired

    # Try to find JSON object by matching braces
    brace_start = text.find("{")
    if brace_start != -1:
        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[brace_start:], brace_start):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        repaired = repair_json(candidate)
                        if repaired:
                            return repaired
                        break

    return None


def repair_json(text: str) -> str | None:
    """Attempt to repair malformed JSON.

    Handles common issues:
    - Trailing commas
    - Single quotes instead of double quotes
    - Unquoted keys
    - Missing closing braces/brackets

    Args:
        text: Malformed JSON string.

    Returns:
        Repaired JSON string, or None if repair fails.
    """
    if not text:
        return None

    original = text
    text = text.strip()

    # Try parsing first - maybe it's already valid
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Fix 1: Replace single quotes with double quotes (but not within strings)
    # This is tricky - we'll use a simple heuristic
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    # Fix 2: Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Fix 3: Add missing closing braces/brackets
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    if open_braces > 0 or open_brackets > 0:
        text = text + "}" * open_braces + "]" * open_brackets
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    # Fix 4: Quote unquoted keys
    # Pattern: word followed by colon, not already quoted
    def quote_key(match):
        key = match.group(1)
        return f'"{key}":'

    text = re.sub(r'(?<=[{,\s])(\w+)\s*:', quote_key, original)
    text = re.sub(r",\s*([}\]])", r"\1", text)  # Remove trailing commas again
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Fix 5: Try to fix common typos
    text = original.strip()
    # True/False/None -> true/false/null
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    return None


def repair_truncated_json(text: str) -> str | None:
    """Aggressively repair JSON that was truncated mid-generation.

    Unlike repair_json(), this handles cases where the response was cut off
    by a max_tokens limit, leaving incomplete strings, values, or structures.

    Strategy:
    1. Find the start of the JSON object
    2. Truncate back to the last complete key-value pair or array element
    3. Close all open structures

    Args:
        text: Truncated text that may contain partial JSON.

    Returns:
        Repaired JSON string, or None if repair fails.
    """
    if not text:
        return None

    # First try normal repair - maybe it's close enough
    normal = repair_json(text)
    if normal:
        return normal

    # Find the JSON start
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    json_text = text[brace_start:]

    # Strategy: progressively truncate from the end until we can close the JSON
    # We look for the last position where we can cleanly close all structures

    # Step 1: Try to find the last complete value boundary
    # Work backwards from the end, looking for positions after which we can
    # close all open brackets/braces
    best_result = None

    # Find positions of potential truncation points (after complete values)
    # These are positions right after: }, ], "...", number, true, false, null
    truncation_points = []

    i = brace_start
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape_next = False

    while i < len(json_text) + brace_start:
        idx = i - brace_start
        if idx >= len(json_text):
            break
        char = json_text[idx]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == "\\":
            escape_next = True
            i += 1
            continue

        if char == '"' and not escape_next:
            if in_string:
                # End of string - this is a truncation point
                in_string = False
                truncation_points.append(i + 1)
            else:
                in_string = True
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
            truncation_points.append(i + 1)
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket -= 1
            truncation_points.append(i + 1)
        elif char == ",":
            truncation_points.append(i)  # Before the comma

        i += 1

    # If we ended inside a string, try closing it to salvage more content
    if in_string:
        # Close the unclosed string, then try to close structures
        closed_str = text[brace_start:] + '"'
        # Remove trailing commas and close structures
        ob = closed_str.count("{") - closed_str.count("}")
        obrk = closed_str.count("[") - closed_str.count("]")
        if ob >= 0 and obrk >= 0:
            candidate = closed_str.rstrip()
            if candidate.endswith(","):
                candidate = candidate[:-1]
            closed = candidate + "]" * obrk + "}" * ob
            try:
                result = json.loads(closed)
                if isinstance(result, dict) and "type" in result:
                    return closed
            except json.JSONDecodeError:
                pass

    # Try truncation points in reverse order (prefer keeping more content)
    for tp in reversed(truncation_points):
        candidate = text[brace_start:tp]

        # Remove any trailing comma
        candidate = candidate.rstrip()
        if candidate.endswith(","):
            candidate = candidate[:-1]

        # Count open structures
        ob = candidate.count("{") - candidate.count("}")
        obrk = candidate.count("[") - candidate.count("]")

        if ob < 0 or obrk < 0:
            continue  # Over-closed, skip

        # Close open structures (brackets first, then braces - inner to outer)
        closed = candidate + "]" * obrk + "}" * ob

        try:
            result = json.loads(closed)
            # Verify it has at minimum a "type" field to be useful
            if isinstance(result, dict) and "type" in result:
                return closed
            # Keep looking for a better candidate if no "type" field
            if best_result is None:
                best_result = closed
        except json.JSONDecodeError:
            continue

    return best_result


def parse_with_repair(text: str, max_retries: int = 3, truncated: bool = False) -> tuple[dict | None, list[str]]:
    """Extract and parse JSON with repair attempts.

    Args:
        text: Raw text containing JSON.
        max_retries: Maximum repair attempts.
        truncated: Whether the response was truncated by max_tokens limit.

    Returns:
        Tuple of (parsed dict or None, list of error messages).
    """
    errors = []

    # Try extraction
    extracted = extract_json(text)
    if extracted is not None:
        # Try parsing
        try:
            return json.loads(extracted), []
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")

        # Try repair
        for attempt in range(max_retries):
            repaired = repair_json(extracted)
            if repaired:
                try:
                    return json.loads(repaired), [f"JSON repaired after {attempt + 1} attempt(s)"]
                except json.JSONDecodeError as e:
                    errors.append(f"Repair attempt {attempt + 1} failed: {e}")
                    extracted = repaired  # Try repairing the repaired version
            else:
                errors.append(f"Repair attempt {attempt + 1} returned None")
                break

    # If truncated, try aggressive truncation repair on the raw text
    if truncated:
        repaired = repair_truncated_json(text)
        if repaired:
            try:
                result = json.loads(repaired)
                return result, ["JSON salvaged from truncated response"]
            except json.JSONDecodeError as e:
                errors.append(f"Truncation repair failed: {e}")

    if not errors:
        errors.append("Could not extract JSON from response")

    return None, errors
