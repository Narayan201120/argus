"""P5-0 corpus loader + mock runner tests (DEC-054, mock-only, no network)."""

from pathlib import Path

import pytest

from app.eval.corpus import EvalCase, load_corpus
from scripts import eval_mock

CORPUS = Path("evals/corpus.yaml")


def test_corpus_loads_twenty_items_across_four_buckets() -> None:
    cases = load_corpus(CORPUS)
    assert len(cases) == 20
    buckets = {case.bucket for case in cases}
    assert buckets == {"radar-answerable", "rag-answerable", "needs-web", "unanswerable"}
    for bucket in buckets:
        assert sum(1 for case in cases if case.bucket == bucket) == 5


def test_corpus_ids_unique_and_expectations_sane() -> None:
    cases = load_corpus(CORPUS)
    ids = [case.id for case in cases]
    assert len(set(ids)) == len(ids)
    for case in cases:
        assert isinstance(case, EvalCase)
        assert case.query
        if case.bucket == "unanswerable":
            assert case.expect.verdict == "refused-correctly"
            assert case.expect.min_evidence == 0
            assert case.expect.must_cite is False
        else:
            assert case.expect.min_evidence >= 1
            assert case.expect.must_cite is True


def test_corpus_missing_file_raises_clear_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus("evals/does-not-exist.yaml")


def test_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    first = load_corpus(CORPUS)[0]
    dupes = [first.model_dump(), first.model_dump()]
    target = tmp_path / "dupes.yaml"
    import yaml

    target.write_text(yaml.safe_dump(dupes), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(target)


def test_mock_runner_passes_clean_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    assert eval_mock.main(["--corpus", str(CORPUS)]) == 0
    out = capsys.readouterr().out
    assert "20/20 passed" in out


def test_mock_runner_sabotage_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert eval_mock.main(["--corpus", str(CORPUS), "--sabotage", "drop-citations"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert eval_mock.main(["--corpus", str(CORPUS), "--sabotage", "empty-board"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_mock_runner_only_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert eval_mock.main(["--corpus", str(CORPUS), "--only", "radar-01", "unans-01"]) == 0
    out = capsys.readouterr().out
    assert "2/2 passed" in out
    assert eval_mock.main(["--corpus", str(CORPUS), "--list"]) == 0
    out = capsys.readouterr().out
    assert "radar-01" in out


def test_mock_runner_unknown_id_is_usage_error() -> None:
    assert eval_mock.main(["--corpus", str(CORPUS), "--only", "nope-99"]) == 2


def test_mock_runner_bad_corpus_is_usage_error() -> None:
    assert eval_mock.main(["--corpus", "evals/does-not-exist.yaml"]) == 2
