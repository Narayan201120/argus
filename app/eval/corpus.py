"""P5-0 eval corpus loader (DEC-054, mock-only).

No network, provider SDK, or store imports: parsing a YAML file only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

Bucket = Literal["radar-answerable", "rag-answerable", "needs-web", "unanswerable"]
Verdict = Literal["supported", "partial", "unsupported", "refused-correctly"]


class EvalExpectation(BaseModel):
    min_evidence: int = Field(ge=0)
    verdict: Verdict
    must_cite: bool


class EvalCase(BaseModel):
    id: str = Field(min_length=1)
    bucket: Bucket
    query: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)
    expect: EvalExpectation
    notes: str = ""


def load_corpus(path: str | Path) -> list[EvalCase]:
    """Load and validate the eval corpus at *path*.

    Raises:
        FileNotFoundError: when the corpus file does not exist.
        ValueError: when the file is not a non-empty YAML list, when any item
            fails schema validation, or when case ids are not unique.
    """
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"eval corpus not found: {corpus_path}")
    try:
        raw: Any = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"eval corpus is not valid YAML ({corpus_path}): {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"eval corpus must be a non-empty YAML list: {corpus_path}")
    cases: list[EvalCase] = []
    problems: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"item {index}: expected a mapping, got {type(item).__name__}")
            continue
        try:
            cases.append(EvalCase.model_validate(item))
        except ValidationError as exc:
            problems.append(f"item {index}: {exc.errors(include_url=False)}")
    if problems:
        detail = "\n".join(problems)
        raise ValueError(f"eval corpus has {len(problems)} invalid item(s):\n{detail}")
    seen: set[str] = set()
    dupes: list[str] = []
    for case in cases:
        if case.id in seen:
            dupes.append(case.id)
        seen.add(case.id)
    if dupes:
        raise ValueError(f"eval corpus has duplicate case ids: {sorted(set(dupes))}")
    return cases
