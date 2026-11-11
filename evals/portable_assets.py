"""Version existing, loader-validated evaluation datasets without copying cases."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.conversation_eval import DEFAULT_CASES_PATH as CONVERSATION_CASES_PATH
from evals.conversation_eval import load_conversation_cases
from evals.response_eval import DEFAULT_CASES_PATH as RESPONSE_CASES_PATH
from evals.response_eval import load_response_cases
from evals.router_eval import DEFAULT_CASES_PATH as ROUTER_CASES_PATH
from evals.router_eval import load_router_cases
from evals.specialist_eval import DEFAULT_CASES_PATH as SPECIALIST_CASES_PATH
from evals.specialist_eval import load_specialist_cases

SuiteName = Literal["router", "specialist", "response", "conversation"]
SUITE_ORDER = get_args(SuiteName)
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
NonblankString = Annotated[str, Field(min_length=1)]


class PortableEvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    suite_name: SuiteName
    relative_path: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    case_ids: tuple[NonblankString, ...]
    tags: tuple[NonblankString, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_suite(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute() or ".." in path.parts or "\\" in self.relative_path
            or path == PurePosixPath(".") or path.as_posix() != self.relative_path
            or ":" in self.relative_path
        ):
            raise ValueError("relative_path must be a normalized repository-relative POSIX path")
        if self.case_count != len(self.case_ids):
            raise ValueError("case_count must equal the number of case IDs")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Case IDs must be unique within each suite")
        if self.case_ids != tuple(sorted(self.case_ids)):
            raise ValueError("Case IDs must be sorted")
        if self.tags != tuple(sorted(set(self.tags))):
            raise ValueError("Tags must be unique and sorted")
        return self


class PortableEvalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["pawline-evals-v1"]
    suites: tuple[PortableEvalSuite, ...]

    @model_validator(mode="after")
    def validate_suite_order(self) -> Self:
        names = [suite.suite_name for suite in self.suites]
        if len(set(names)) != len(names):
            raise ValueError("Suite names must be unique")
        if names != sorted(names, key=SUITE_ORDER.index):
            raise ValueError("Suites must follow router, specialist, response, conversation order")
        return self


def build_portable_eval_manifest() -> PortableEvalManifest:
    """Use existing loaders and hash exact bytes; fail if a file changes during validation."""
    suites = []
    for name, source, loader in (
        ("router", ROUTER_CASES_PATH, load_router_cases),
        ("specialist", SPECIALIST_CASES_PATH, load_specialist_cases),
        ("response", RESPONSE_CASES_PATH, load_response_cases),
        ("conversation", CONVERSATION_CASES_PATH, load_conversation_cases),
    ):
        path = source.resolve()
        relative_path = path.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
        file_bytes = path.read_bytes()
        cases = loader(path)
        if path.read_bytes() != file_bytes:
            raise ValueError("An evaluation dataset changed while building the manifest")
        suites.append(PortableEvalSuite(
            suite_name=name, relative_path=relative_path, case_count=len(cases),
            case_ids=tuple(sorted(case.case_id for case in cases)),
            tags=tuple(sorted({tag for case in cases for tag in case.tags})),
            sha256=hashlib.sha256(file_bytes).hexdigest(),
        ))
    return PortableEvalManifest(schema_version="pawline-evals-v1", suites=tuple(suites))


def format_portable_eval_manifest(manifest: PortableEvalManifest) -> str:
    """Return canonical indented JSON with a trailing newline and no machine metadata."""
    return manifest.model_dump_json(indent=2) + "\n"
