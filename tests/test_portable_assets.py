"""Portable manifests identify existing validated assets using exact file hashes."""

import hashlib
import socket
from pathlib import Path
from unittest.mock import Mock

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from evals import export_portable_manifest, portable_assets
from evals.portable_assets import (
    PortableEvalManifest,
    PortableEvalSuite,
    build_portable_eval_manifest,
    format_portable_eval_manifest,
)

SOURCES = (
    ("router", "ROUTER_CASES_PATH", "load_router_cases"),
    ("specialist", "SPECIALIST_CASES_PATH", "load_specialist_cases"),
    ("response", "RESPONSE_CASES_PATH", "load_response_cases"),
    ("conversation", "CONVERSATION_CASES_PATH", "load_conversation_cases"),
)


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Manifest generation must not construct clients or use the network")

    for cls in (AsyncOpenAI, OpenAI):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_MANIFEST_API_KEY")
    monkeypatch.setenv("DATABASE_URL", "PRIVATE_MANIFEST_DATABASE_URL")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "PRIVATE_MANIFEST_MODEL")


@pytest.fixture
def dataset_copy(tmp_path, monkeypatch):
    destination = tmp_path / "evals"
    destination.mkdir()
    for _, field, _ in SOURCES:
        source = getattr(portable_assets, field)
        copied = destination / source.name
        copied.write_bytes(source.read_bytes())
        monkeypatch.setattr(portable_assets, field, copied)
    monkeypatch.setattr(portable_assets, "REPOSITORY_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def suite_data():
    return {
        "suite_name": "router", "relative_path": "evals/router_cases.jsonl", "case_count": 2,
        "case_ids": ("case_a", "case_b"), "tags": ("first", "second"), "sha256": "a" * 64,
    }


def test_unchanged_assets_have_identical_manifest_and_json_regardless_of_cwd(monkeypatch, tmp_path):
    first = build_portable_eval_manifest()
    monkeypatch.chdir(tmp_path)
    second = build_portable_eval_manifest()
    assert first == second
    assert format_portable_eval_manifest(first) == format_portable_eval_manifest(second)
    assert PortableEvalManifest.model_validate_json(format_portable_eval_manifest(first)) == first


def test_all_four_suites_use_existing_loaders_paths_sorted_metadata_and_exact_hashes(monkeypatch):
    spies = {}
    for name, _, loader_name in SOURCES:
        spies[name] = Mock(wraps=getattr(portable_assets, loader_name))
        monkeypatch.setattr(portable_assets, loader_name, spies[name])
    manifest = build_portable_eval_manifest()
    assert manifest.schema_version == "pawline-evals-v1"
    assert tuple(suite.suite_name for suite in manifest.suites) == ("router", "specialist", "response", "conversation")
    assert tuple(suite.case_count for suite in manifest.suites) == (27, 17, 9, 6)
    for suite, (name, field, loader_name) in zip(manifest.suites, SOURCES, strict=True):
        source = getattr(portable_assets, field).resolve()
        spies[name].assert_called_once_with(source)
        cases = spies[name]._mock_wraps(source)
        assert suite.case_ids == tuple(sorted(case.case_id for case in cases))
        assert suite.tags == tuple(sorted({tag for case in cases for tag in case.tags}))
        assert suite.relative_path == f"evals/{name}_cases.jsonl"
        assert not Path(suite.relative_path).is_absolute()
        assert suite.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_formatter_contains_only_manifest_metadata_and_no_machine_or_environment_data():
    manifest = build_portable_eval_manifest()
    text = format_portable_eval_manifest(manifest)
    assert text.startswith('{\n  "schema_version": "pawline-evals-v1",\n  "suites": [\n')
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert set(manifest.model_dump()) == {"schema_version", "suites"}
    for suite in manifest.suites:
        assert set(suite.model_dump()) == {"suite_name", "relative_path", "case_count", "case_ids", "tags", "sha256"}
    for forbidden in (
        str(portable_assets.REPOSITORY_ROOT), "PRIVATE_MANIFEST", "timestamp", "created_at",
        '"user_message"', '"tool_result"', '"response"' + ":", '"api_key"', '"customer_id"',
    ):
        assert forbidden not in text


def test_hash_tracks_byte_changes_even_when_loader_metadata_is_identical(dataset_copy):
    before = build_portable_eval_manifest()
    source = portable_assets.ROUTER_CASES_PATH
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
    after = build_portable_eval_manifest()
    assert before.suites[0].case_ids == after.suites[0].case_ids
    assert before.suites[0].tags == after.suites[0].tags
    assert before.suites[0].case_count == after.suites[0].case_count
    assert before.suites[0].sha256 != after.suites[0].sha256
    assert before.suites[1:] == after.suites[1:]


@pytest.mark.parametrize("field", [item[1] for item in SOURCES])
def test_missing_dataset_files_are_rejected(dataset_copy, field):
    getattr(portable_assets, field).unlink()
    with pytest.raises(FileNotFoundError):
        build_portable_eval_manifest()


@pytest.mark.parametrize("name, field, loader_name", SOURCES)
def test_invalid_cases_fail_through_each_existing_loader(dataset_copy, monkeypatch, name, field, loader_name):
    source = getattr(portable_assets, field)
    source.write_text('{"case_id":"demo-invalid"}\n', encoding="utf-8")
    loader = Mock(wraps=getattr(portable_assets, loader_name))
    monkeypatch.setattr(portable_assets, loader_name, loader)
    with pytest.raises(ValueError) as caught:
        build_portable_eval_manifest()
    assert str(caught.value).startswith(f"{source}:1:")
    loader.assert_called_once_with(source)


def test_duplicate_dataset_case_ids_are_rejected_by_the_existing_loader(dataset_copy):
    source = portable_assets.ROUTER_CASES_PATH
    original = source.read_bytes()
    source.write_bytes(original + original.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="duplicate case_id"):
        build_portable_eval_manifest()


def test_dataset_changes_during_loader_validation_abort_the_manifest(dataset_copy, monkeypatch):
    loader = portable_assets.load_router_cases

    def changed_source(path):
        cases = loader(path)
        path.write_bytes(path.read_bytes() + b"\n")
        return cases

    monkeypatch.setattr(portable_assets, "load_router_cases", changed_source)
    with pytest.raises(ValueError, match="dataset changed"):
        build_portable_eval_manifest()


@pytest.mark.parametrize("update", [
    {"suite_name": "unknown"}, {"relative_path": " "}, {"relative_path": "/private/local/cases.jsonl"},
    {"relative_path": "C:\\private\\cases.jsonl"}, {"relative_path": "../cases.jsonl"},
    {"relative_path": "evals/../cases.jsonl"}, {"relative_path": "./evals/cases.jsonl"},
    {"relative_path": "evals//cases.jsonl"}, {"relative_path": "."},
    {"case_count": "2"}, {"case_count": True}, {"case_count": -1}, {"case_count": 3},
    {"case_ids": ["case_a", "case_b"]}, {"case_ids": (" ", "case_b")},
    {"case_ids": (1, "case_b")}, {"case_ids": ("case_a", "case_a")},
    {"case_ids": ("case_a", " case_a ")}, {"case_ids": ("case_b", "case_a")},
    {"tags": ["first", "second"]}, {"tags": (" ",)}, {"tags": (1,)},
    {"tags": ("second", "first")}, {"tags": ("first", "first")},
    {"sha256": ""}, {"sha256": " "}, {"sha256": "hash"}, {"sha256": "A" * 64},
    {"sha256": "g" * 64}, {"extra": "not permitted"},
])
def test_suite_models_are_strict_and_reject_noncanonical_or_inconsistent_values(suite_data, update):
    with pytest.raises(ValidationError):
        PortableEvalSuite.model_validate({**suite_data, **update})


def test_models_require_all_fields_and_freeze_nested_metadata(suite_data):
    for field in PortableEvalSuite.model_fields:
        with pytest.raises(ValidationError):
            PortableEvalSuite.model_validate({key: value for key, value in suite_data.items() if key != field})
    manifest = build_portable_eval_manifest()
    for field in PortableEvalManifest.model_fields:
        with pytest.raises(ValidationError):
            PortableEvalManifest.model_validate({key: value for key, value in manifest.model_dump().items() if key != field})
    with pytest.raises(ValidationError, match="frozen"):
        manifest.schema_version = "other"
    with pytest.raises(ValidationError, match="frozen"):
        manifest.suites[0].sha256 = "b" * 64
    with pytest.raises(TypeError):
        manifest.suites[0].case_ids[0] = "replacement"


def test_manifest_rejects_duplicate_unsorted_or_non_tuple_suites_and_unknown_schema():
    manifest = build_portable_eval_manifest()
    with pytest.raises(ValidationError, match="Suite names must be unique"):
        PortableEvalManifest(schema_version="pawline-evals-v1", suites=(manifest.suites[0], manifest.suites[0]))
    with pytest.raises(ValidationError, match="Suites must follow"):
        PortableEvalManifest(schema_version="pawline-evals-v1", suites=tuple(reversed(manifest.suites)))
    with pytest.raises(ValidationError):
        PortableEvalManifest(schema_version="pawline-evals-v1", suites=list(manifest.suites))
    with pytest.raises(ValidationError):
        PortableEvalManifest(schema_version="unsupported", suites=manifest.suites)
    with pytest.raises(ValidationError):
        PortableEvalManifest(**manifest.model_dump(), timestamp="not permitted")


def test_cli_prints_exact_deterministic_json_without_credentials(capsys):
    assert export_portable_manifest.main([]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == format_portable_eval_manifest(build_portable_eval_manifest())
    assert "PRIVATE_MANIFEST" not in output.out


def test_check_succeeds_for_committed_manifest_without_rewriting_it(capsys):
    committed = portable_assets.REPOSITORY_ROOT / "evals" / "portable_manifest.json"
    original = committed.read_bytes()
    assert export_portable_manifest.main(["--check", str(committed)]) == 0
    assert committed.read_bytes() == original
    output = capsys.readouterr()
    assert output.out == output.err == ""


@pytest.mark.parametrize("content", [b"PRIVATE_STALE_CONTENT", b'{"schema_version":"outdated"}\n'])
def test_check_fails_safely_for_stale_manifest_without_rewriting_it(tmp_path, capsys, content):
    committed = tmp_path / "stale.json"
    committed.write_bytes(content)
    assert export_portable_manifest.main(["--check", str(committed)]) == 1
    assert committed.read_bytes() == content
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Portable evaluation manifest is stale.\n"
    assert "PRIVATE" not in output.err


def test_check_detects_actual_dataset_changes(dataset_copy, capsys):
    committed = dataset_copy / "portable_manifest.json"
    committed.write_text(format_portable_eval_manifest(build_portable_eval_manifest()), encoding="utf-8")
    original = committed.read_bytes()
    portable_assets.ROUTER_CASES_PATH.write_bytes(portable_assets.ROUTER_CASES_PATH.read_bytes() + b"\n")
    assert export_portable_manifest.main(["--check", str(committed)]) == 1
    assert committed.read_bytes() == original
    assert "stale" in capsys.readouterr().err


def test_check_missing_file_fails_without_creating_it(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    assert export_portable_manifest.main(["--check", str(missing)]) == 1
    assert not missing.exists()
    output = capsys.readouterr()
    assert output.out == "" and output.err == "Unable to read the portable evaluation manifest for checking.\n"


def test_cli_dataset_errors_never_echo_loader_errors_or_case_data(dataset_copy, capsys):
    portable_assets.ROUTER_CASES_PATH.write_text('{"PRIVATE_MANIFEST_API_KEY":"private case"}\n', encoding="utf-8")
    assert export_portable_manifest.main([]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Unable to build the portable evaluation manifest.\n"


def test_export_has_no_automatic_write_option(tmp_path):
    target = tmp_path / "never-written.json"
    with pytest.raises(SystemExit) as caught:
        export_portable_manifest.main(["--write", str(target)])
    assert caught.value.code == 2 and not target.exists()
