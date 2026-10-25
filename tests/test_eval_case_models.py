"""Check case-file validation directly, independently of scoring and tool execution."""

import json
from dataclasses import FrozenInstanceError, is_dataclass

import pytest
from pydantic import BaseModel, ValidationError

from app.agents.prompt_context import PromptContext
from evals.router_eval import DEFAULT_CASES_PATH as ROUTER_CASES_PATH
from evals.router_eval import RouterEvalCase, load_router_cases
from evals.specialist_eval import DEFAULT_CASES_PATH as SPECIALIST_CASES_PATH
from evals.specialist_eval import SpecialistEvalCase, SpecialistFirstAction, load_specialist_cases


@pytest.fixture(params=[
    (RouterEvalCase, ROUTER_CASES_PATH, load_router_cases),
    (SpecialistEvalCase, SPECIALIST_CASES_PATH, load_specialist_cases),
], ids=["router", "specialist"])
def case_setup(request):
    model, path, loader = request.param
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return model, row, loader


def test_case_models_are_pydantic_and_require_every_declared_field(case_setup) -> None:
    model, row, _ = case_setup
    assert isinstance(model.model_validate(row), BaseModel)
    for field in row:
        missing = {name: value for name, value in row.items() if name != field}
        with pytest.raises(ValidationError) as caught:
            model.model_validate(missing)
        assert any(error["loc"] == (field,) and error["type"] == "missing" for error in caught.value.errors())


def test_extra_fields_are_forbidden_by_the_case_model(case_setup) -> None:
    model, row, _ = case_setup
    with pytest.raises(ValidationError) as caught:
        model.model_validate({**row, "unexpected": True})
    assert caught.value.errors()[0]["loc"] == ("unexpected",)
    assert caught.value.errors()[0]["type"] == "extra_forbidden"


def test_metadata_and_tag_whitespace_are_stripped(case_setup) -> None:
    model, row, _ = case_setup
    for field in ("case_id", "user_message", "reason"):
        row[field] = f" \t{row[field]}\n "
    row["tags"] = ["  first  ", "\tsecond\n"]

    case = model.model_validate(row)

    for field in ("case_id", "user_message", "reason"):
        assert getattr(case, field) == row[field].strip()
    assert case.tags == ["first", "second"]
    assert isinstance(case.context, PromptContext)


@pytest.mark.parametrize("field, value", [
    ("case_id", "  "), ("user_message", "\n"), ("reason", ""),
    ("tags", []), ("tags", ["valid", " \t "]),
    ("case_id", 7), ("user_message", False), ("reason", ["text"]),
    ("tags", None), ("tags", "tag"), ("tags", [7]),
    ("context", []), ("context", {"customer_verified": "not-a-bool"}),
])
def test_invalid_or_blank_fields_are_rejected_without_custom_field_validators(case_setup, field, value) -> None:
    model, row, _ = case_setup
    row[field] = value
    with pytest.raises(ValidationError) as caught:
        model.model_validate(row)
    assert caught.value.errors()[0]["loc"][0] == field


def test_frozen_cases_cannot_be_reassigned(case_setup) -> None:
    model, row, _ = case_setup
    case = model.model_validate(row)
    with pytest.raises(ValidationError) as caught:
        case.case_id = "replacement"
    assert caught.value.errors()[0]["type"] == "frozen_instance"
    assert case.case_id == row["case_id"]


def test_pydantic_errors_keep_filename_and_physical_line_number(case_setup, tmp_path) -> None:
    _, row, loader = case_setup
    path = tmp_path / "cases.jsonl"
    path.write_text("\n \t\n" + json.dumps({**row, "unexpected": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        loader(path)
    assert str(caught.value).startswith(f"{path}:3:")
    assert isinstance(caught.value.__cause__, ValidationError)
    assert caught.value.__cause__.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("action, tool, arguments, error_field", [
    ("speech", "lookup_customer", None, None),
    ("speech", None, {}, None),
    ("tool", None, {}, None),
    ("tool", "lookup_customer", None, None),
    ("tool", " \t ", {}, "expected_tool"),
    ("tool", 7, {}, "expected_tool"),
    ("tool", "lookup_customer", [], "expected_arguments"),
    ("invented", None, None, "expected_action"),
])
def test_specialist_action_tool_relationships_and_types(action, tool, arguments, error_field) -> None:
    row = json.loads(SPECIALIST_CASES_PATH.read_text(encoding="utf-8").splitlines()[0])
    row.update(expected_action=action, expected_tool=tool, expected_arguments=arguments)
    with pytest.raises(ValidationError) as caught:
        SpecialistEvalCase.model_validate(row)
    error = caught.value.errors()[0]
    if error_field is None:
        assert error["loc"] == ()
        assert error["type"] == "value_error"
    else:
        assert error["loc"] == (error_field,)


def test_valid_specialist_actions_preserve_exact_argument_values() -> None:
    row = json.loads(SPECIALIST_CASES_PATH.read_text(encoding="utf-8").splitlines()[0])
    row["expected_tool"] = "  lookup_customer  "
    row["expected_arguments"] = {"phone": "  202-555-0142  "}
    tool_case = SpecialistEvalCase.model_validate(row)
    assert tool_case.expected_tool == "lookup_customer"
    assert tool_case.expected_arguments == row["expected_arguments"]
    speech_case = SpecialistEvalCase.model_validate({
        **row, "expected_action": "speech", "expected_tool": None, "expected_arguments": None,
    })
    assert speech_case.expected_action == "speech"
    assert speech_case.expected_tool is speech_case.expected_arguments is None


def test_specialist_first_action_remains_a_frozen_dataclass() -> None:
    assert is_dataclass(SpecialistFirstAction)
    action = SpecialistFirstAction("speech")
    with pytest.raises(FrozenInstanceError):
        action.action = "tool"
