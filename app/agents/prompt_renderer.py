"""Load PawLine's Jinja templates and turn typed context into prompt text."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from .prompt_context import PromptContext

# This identifies the exact instructions used for an LLM call, like a version number for application code.
FRONT_DESK_PROMPT_VERSION = "front-desk-v1"
ROUTER_PROMPT_VERSION = "router-v1"
VERIFICATION_PROMPT_VERSION = "verification-v1"
# Templates live beside this module so their instructions can be reviewed and versioned with the code.
DEFAULT_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent.parent / "prompts"


class PromptRenderingError(RuntimeError):
    """Raised when PawLine cannot build a prompt from its template and context.

    For example, this occurs when a required Jinja variable is missing or the
    template cannot be loaded.
    """


class PromptRenderer:
    """Build the complete instructions that PawLine will send to the LLM.

    It takes a PromptContext, fills the matching blanks in a Jinja prompt
    template, and returns the finished prompt as text. It does not retrieve
    conversation data, call the LLM, or make agent decisions.
    """

    def __init__(self, template_directory: Path | None = None) -> None:
        """Prepare to load templates from the package or a supplied directory.

        The optional directory is useful for tests. Creating a renderer does not
        load conversation data or render a prompt.
        """
        self._environment = Environment(
            loader=FileSystemLoader(template_directory or DEFAULT_TEMPLATE_DIRECTORY),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    @property
    def prompt_version(self) -> str:
        """Return the version label attached to the front-desk instructions."""
        return FRONT_DESK_PROMPT_VERSION

    def render_front_desk(self, context: PromptContext) -> str:
        """Render front-desk instructions from the supplied PromptContext.

        The context is prepared by trusted application code, usually from
        ConversationState. This method returns prompt text; the caller decides
        whether and when to send it to an LLM.
        """
        template_values = context.model_dump(mode="json")
        template_values["prompt_version"] = self.prompt_version

        try:
            template = self._environment.get_template("front_desk.md.j2")
            return template.render(**template_values)
        except (OSError, TemplateError) as exc:
            raise PromptRenderingError(
                f"Unable to render prompt version '{self.prompt_version}'."
            ) from exc

    def render_router(self, context: PromptContext, user_message: str) -> str:
        """Render router instructions from allowlisted context and the caller message.

        The context contains only conversation facts needed for routing. This
        method returns prompt text; it does not select a destination or execute a
        specialist.
        """
        if not user_message or not user_message.strip():
            raise PromptRenderingError("User message cannot be blank.")

        template_values = context.model_dump(mode="json")
        template_values["prompt_version"] = ROUTER_PROMPT_VERSION
        template_values["user_message"] = user_message.strip()

        try:
            template = self._environment.get_template("router.md.j2")
            return template.render(**template_values)
        except (OSError, TemplateError) as exc:
            raise PromptRenderingError(
                f"Unable to render prompt version '{ROUTER_PROMPT_VERSION}'."
            ) from exc

    def render_verification(self, context: PromptContext, user_message: str) -> str:
        """Render verification instructions from allowlisted context and the caller message.

        The context contains only conversation facts needed for safe customer
        identification. This method returns prompt text; it does not call the CRM
        or execute another agent.
        """
        if not user_message or not user_message.strip():
            raise PromptRenderingError("User message cannot be blank.")

        template_values = context.model_dump(mode="json")
        template_values["prompt_version"] = VERIFICATION_PROMPT_VERSION
        template_values["user_message"] = user_message.strip()

        try:
            template = self._environment.get_template("verification.md.j2")
            return template.render(**template_values)
        except (OSError, TemplateError) as exc:
            raise PromptRenderingError(
                f"Unable to render prompt version '{VERIFICATION_PROMPT_VERSION}'."
            ) from exc


__all__ = [
    "FRONT_DESK_PROMPT_VERSION",
    "ROUTER_PROMPT_VERSION",
    "VERIFICATION_PROMPT_VERSION",
    "PromptRenderer",
    "PromptRenderingError",
]