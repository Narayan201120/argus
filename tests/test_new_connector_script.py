"""Stage P3-6 - connector scaffold script (pure file generation, no network)."""

import ast

import pytest

from scripts import new_connector


def _point_at_tmp(tmp_path, monkeypatch):
    """Redirect scaffold output into tmp so the real repo is never touched."""
    monkeypatch.setattr(new_connector, "CONNECTORS_DIR", tmp_path / "app" / "connectors")
    monkeypatch.setattr(new_connector, "TESTS_DIR", tmp_path / "tests")


def test_main_generates_both_files(tmp_path, monkeypatch):
    _point_at_tmp(tmp_path, monkeypatch)

    code = new_connector.main(["cohere", "--display-name", "Cohere"])
    assert code == new_connector.EXIT_OK

    connector_file = tmp_path / "app" / "connectors" / "cohere.py"
    test_file = tmp_path / "tests" / "test_connectors_cohere.py"
    assert connector_file.exists()
    assert test_file.exists()

    source = connector_file.read_text(encoding="utf-8")
    assert "class CohereConnector(BaseConnector):" in source
    assert 'connector_id = "cohere"' in source
    assert "cohere_api_key" in source
    assert "classify_provider_exception" in source
    assert "import asyncio" not in source  # template must be lint-clean as generated


def test_generated_code_is_lint_clean_shape(tmp_path, monkeypatch):
    _point_at_tmp(tmp_path, monkeypatch)
    assert new_connector.main(["lintprov"]) == new_connector.EXIT_OK

    for path in (tmp_path / "app" / "connectors" / "lintprov.py",
                 tmp_path / "tests" / "test_connectors_lintprov.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert len(line) <= 110, f"{path.name}: line exceeds 110 chars"


def test_main_refuses_overwrite_without_force(tmp_path, monkeypatch):
    _point_at_tmp(tmp_path, monkeypatch)

    assert new_connector.main(["mistral2"]) == new_connector.EXIT_OK
    assert new_connector.main(["mistral2"]) == new_connector.EXIT_EXISTS
    assert new_connector.main(["mistral2", "--force"]) == new_connector.EXIT_OK


@pytest.mark.parametrize("bad", ["1starts", "has space", "", "-lead"])
def test_invalid_names_rejected(bad):
    assert new_connector.main([bad]) == new_connector.EXIT_USAGE


def test_mixed_case_name_is_normalized(tmp_path, monkeypatch):
    _point_at_tmp(tmp_path, monkeypatch)

    assert new_connector.main(["MyProvider"]) == new_connector.EXIT_OK
    source = (tmp_path / "app" / "connectors" / "myprovider.py").read_text(encoding="utf-8")
    assert 'connector_id = "myprovider"' in source


def test_generated_code_compiles(tmp_path, monkeypatch):
    _point_at_tmp(tmp_path, monkeypatch)

    assert new_connector.main(["sampleprov"]) == new_connector.EXIT_OK
    source = (tmp_path / "app" / "connectors" / "sampleprov.py").read_text(encoding="utf-8")
    ast.parse(source)  # raises on syntax errors
