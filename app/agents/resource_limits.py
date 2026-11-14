"""Deterministic per-turn budgets for front-desk conversations, not rate limits."""

# Normal demo cases use under 200 caller characters, 1 KiB of tool JSON, and
# 7 KiB of instructions. Leave room for detailed concerns and policy evidence.
MAX_USER_MESSAGE_CHARS = 4_000
MAX_TOOL_ARGUMENT_BYTES = 16_384
MAX_TOOL_RESULT_BYTES = 32_768
MAX_RECENT_MESSAGES = 6
MAX_TOOL_ROUNDS = 3
# A message-count window alone cannot bound a large saved assistant message or
# structured context. Reject oversized rendered instructions before API calls.
MAX_INSTRUCTION_BYTES = 65_536


class ResourceLimitError(ValueError):
    """A local resource budget was exceeded; messages contain no payload data."""


class CallerMessageTooLargeError(ResourceLimitError):
    """The caller message exceeds the character budget."""


class ToolArgumentsTooLargeError(ResourceLimitError):
    """Raw tool argument JSON exceeds the UTF-8 byte budget."""


class ToolResultTooLargeError(ResourceLimitError):
    """Serialized tool result JSON exceeds the UTF-8 byte budget."""


class InstructionsTooLargeError(ResourceLimitError):
    """Rendered instructions, including saved context, exceed the byte budget."""


def validate_user_message_size(user_message: str) -> None:
    """Check the original message; blank-message rules remain at each boundary."""
    if len(user_message) > MAX_USER_MESSAGE_CHARS:
        raise CallerMessageTooLargeError(
            f"Caller message exceeds the limit of {MAX_USER_MESSAGE_CHARS} characters."
        )


def _validate_utf8_size(text: str, limit: int, error_type: type[ResourceLimitError], label: str) -> None:
    # UTF-8 needs at least one byte per character. Avoid another large allocation
    # when the character count alone already proves the payload is oversized.
    if len(text) > limit or len(text.encode("utf-8")) > limit:
        raise error_type(f"{label} exceeds the limit of {limit} UTF-8 bytes.")


def validate_tool_arguments_size(raw_arguments: str) -> None:
    """Check raw JSON before parsing it or invoking a handler."""
    _validate_utf8_size(raw_arguments, MAX_TOOL_ARGUMENT_BYTES, ToolArgumentsTooLargeError, "Tool arguments")


def validate_tool_result_size(serialized_result: str) -> None:
    """Check the one serialized result before it enters model input."""
    _validate_utf8_size(serialized_result, MAX_TOOL_RESULT_BYTES, ToolResultTooLargeError, "Tool result")


def validate_instructions_size(instructions: str) -> None:
    """Bound rendered history and context without silently truncating facts."""
    _validate_utf8_size(instructions, MAX_INSTRUCTION_BYTES, InstructionsTooLargeError, "Instructions")
