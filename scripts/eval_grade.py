"""P5-2 human grading CLI (DEC-054). No LLM judge in this stage.

Reads a live-eval run (``runs/<ts>.jsonl``), shows each ungraded row to a
person on stdin, and appends one judgment per line to the sibling file
``runs/<ts>.grades.jsonl``. The grades file is the offline truth: reruns skip
rows that already have a grade, so grading is resumable after interruption.

Usage:
    python scripts/eval_grade.py grade --run runs/20260908T120000Z.jsonl
    python scripts/eval_grade.py grade --run runs/x.jsonl --only radar-01 rag-01
    python scripts/eval_grade.py --run runs/x.jsonl --non-interactive --verdict supported

(The leading "grade" word is accepted and ignored; both spellings work.)

Interactive verdict keys: s (supported), p (partial), u (unsupported),
r (refused-correctly). Full verdict names work too. An empty verdict skips
the row and leaves it ungraded. --non-interactive --verdict X applies X to
every selected ungraded row without prompting (tests and scripted use).

Exit codes: 0 done, 2 bad args or missing/corrupt files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.corpus import load_corpus  # noqa: E402
from app.eval.grades import (  # noqa: E402
    GRADE_VERDICTS,
    VERDICT_HELP,
    GradeJoin,
    GradeRecord,
    GradeVerdict,
    join_grades,
    parse_verdict_token,
)
from app.eval.normalize import board_summary  # noqa: E402

DEFAULT_CORPUS = "evals/corpus.yaml"
SYNTHESIS_DISPLAY_CHARS = 2000
STATEMENT_DISPLAY_CHARS = 300

_VERDICT_CHOICES: tuple[str, ...] = tuple(GRADE_VERDICTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human grading workflow for eval runs.")
    parser.add_argument("--run", default=None, help="Run rows file (runs/<ts>.jsonl).")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help="Corpus for query lookup (best effort).")
    parser.add_argument("--only-ungraded", action="store_true",
                        help="Grade only rows without a grade yet (this is the default).")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of case ids to grade.")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Apply --verdict to every selected ungraded row, no prompting.")
    parser.add_argument("--verdict", default=None, choices=_VERDICT_CHOICES,
                        help="Verdict for --non-interactive mode.")
    parser.add_argument("--notes", default="",
                        help="Notes stored with each grade in --non-interactive mode.")
    parser.add_argument("--grades", default=None,
                        help="Grades output path (default: sibling <run-stem>.grades.jsonl).")
    return parser


def grades_path_for(run_path: Path) -> Path:
    """Sibling grades file for a run file: <stem>.grades.jsonl."""
    return run_path.with_name(f"{run_path.stem}.grades.jsonl")


def _load_run_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read run {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"run {path} line {lineno}: bad JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"run {path} line {lineno}: expected a JSON object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"run {path} line {lineno}: missing case_id")
        rows.append(row)
    return rows


def _load_grades(path: Path) -> list[GradeRecord]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read grades {path}: {exc}") from exc
    grades: list[GradeRecord] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"grades {path} line {lineno}: bad JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"grades {path} line {lineno}: expected a JSON object")
        try:
            grades.append(GradeRecord.model_validate(raw))
        except ValueError as exc:
            raise ValueError(f"grades {path} line {lineno}: bad grade: {exc}") from exc
    return grades


def _corpus_queries(corpus_path: str) -> dict[str, str]:
    try:
        return {case.id: case.query for case in load_corpus(corpus_path)}
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Warning: corpus lookup disabled ({exc}).", file=sys.stderr)
        return {}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _synthesis_markdown(row: dict[str, Any]) -> str:
    """Best-effort final synthesis text from a run row.

    Live rows carry entry markdown plus a top-level final_markdown; older
    rows may record flags only. Never fails: grading display must not depend
    on row shape.
    """
    for key in ("final_synthesis", "final_report", "synthesis_markdown", "report_markdown"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value, SYNTHESIS_DISPLAY_CHARS)
    syntheses = row.get("syntheses")
    if isinstance(syntheses, list):
        items = [s for s in syntheses if isinstance(s, dict)]
        ordered = [s for s in items if s.get("final") is True] + [s for s in items if not s.get("final")]
        for entry in ordered:
            for key in ("markdown", "report", "text", "content"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return _truncate(value, SYNTHESIS_DISPLAY_CHARS)
    return "(no synthesis markdown recorded in row; synthesis flags only)"


def _row_features(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features")
    if isinstance(features, dict):
        return {
            "terminal": features.get("terminal", row.get("status")),
            "claim_count": features.get("claim_count"),
            "grounded_ratio": features.get("grounded_ratio"),
            "has_final_synthesis": features.get("has_final_synthesis"),
            "refused_correctly": features.get("refused_correctly"),
        }
    return dict(board_summary(row))


def display_row(row: dict[str, Any], queries: dict[str, str], position: str) -> None:
    case_id = str(row.get("case_id"))
    query = row.get("query")
    if not isinstance(query, str) or not query:
        query = queries.get(case_id, "(query not in row; see evals/corpus.yaml)")
    print(f"=== [{position}] case_id={case_id} "
          f"investigation={row.get('investigation_id', '?')} "
          f"status={row.get('status', '?')} ===")
    print(f"query: {query}")
    print(f"synthesis: {_synthesis_markdown(row)}")
    counts = row.get("counts")
    print(f"counts: {counts if isinstance(counts, dict) else '(none)'}")
    evidence = row.get("evidence_ids")
    if isinstance(evidence, list):
        print(f"evidence_ids ({len(evidence)}): "
              f"{', '.join(str(e) for e in evidence[:12])}"
              f"{' ...' if len(evidence) > 12 else ''}")
    else:
        print("evidence_ids: (none)")
    claims = row.get("claims")
    if isinstance(claims, list) and claims:
        print(f"claims ({len(claims)}):")
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            statement = claim.get("statement", "")
            text = _truncate(str(statement), STATEMENT_DISPLAY_CHARS)
            print(f"  - {claim.get('id', '?')} [{claim.get('status', '?')}] "
                  f"cites={claim.get('evidence_ids', [])}: {text}")
    else:
        print("claims: (none)")
    print(f"features: {_row_features(row)}")
    failures = row.get("failures")
    if isinstance(failures, list) and failures:
        print(f"failures: {failures}")
    print("rubric: " + "; ".join(f"{v} = {VERDICT_HELP[v]}" for v in GRADE_VERDICTS))


def prompt_verdict() -> GradeVerdict | None:
    """Prompt for one verdict. Returns None when the grader skips the row."""
    try:
        raw = input("verdict [s]upported/[p]artial/[u]nsupported/[r]efused-correctly "
                    "(empty skips): ")
    except EOFError:
        print()
        return None
    if not raw.strip():
        return None
    verdict = parse_verdict_token(raw)
    while verdict is None:
        print(f"Unknown verdict {raw.strip()!r}; use s/p/u/r or a full verdict name.")
        try:
            raw = input("verdict [s]/[p]/[u]/[r] (empty skips): ")
        except EOFError:
            print()
            return None
        if not raw.strip():
            return None
        verdict = parse_verdict_token(raw)
    return verdict


def prompt_notes() -> str:
    try:
        return input("notes (optional): ").strip()
    except EOFError:
        print()
        return ""


def _append_grade(path: Path, grade: GradeRecord) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(grade.model_dump_json() + "\n")


def print_join_summary(join: GradeJoin) -> None:
    parts = " ".join(f"{verdict}={join.histogram.get(verdict, 0)}" for verdict in GRADE_VERDICTS)
    print(f"grades: {parts} graded={join.graded}/{join.total} "
          f"mean_grounded_ratio={join.mean_grounded_ratio:.2f}")


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "grade":
        raw_args = raw_args[1:]
    args = build_parser().parse_args(raw_args)
    if not args.run:
        print("Missing --run runs/<ts>.jsonl.", file=sys.stderr)
        return 2
    if args.non_interactive and not args.verdict:
        print("Missing --verdict X (required with --non-interactive).", file=sys.stderr)
        return 2
    if args.verdict and not args.non_interactive:
        print("--verdict needs --non-interactive (interactive mode prompts per row).",
              file=sys.stderr)
        return 2
    run_path = Path(args.run)
    if not run_path.is_file():
        print(f"Missing run file: {run_path}", file=sys.stderr)
        return 2
    grades_path = Path(args.grades) if args.grades else grades_path_for(run_path)
    try:
        rows = _load_run_rows(run_path)
    except ValueError as exc:
        print(f"Bad run: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"Bad run: {run_path} has no rows.", file=sys.stderr)
        return 2
    try:
        grades = _load_grades(grades_path)
    except ValueError as exc:
        print(f"Bad grades: {exc}", file=sys.stderr)
        return 2
    graded_ids = {grade.case_id for grade in grades}
    selected = list(rows)
    if args.only is not None:
        wanted = set(args.only)
        known = {str(row.get("case_id")) for row in rows}
        unknown = sorted(wanted - known)
        if unknown:
            print(f"Unknown case ids: {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected = [row for row in rows if str(row.get("case_id")) in wanted]
    # Grading is resumable: rows with a grade are always skipped.
    todo = [row for row in selected if str(row.get("case_id")) not in graded_ids]
    skipped = len(selected) - len(todo)
    if args.only_ungraded:
        print(f"Only ungraded: {len(todo)} to grade, {skipped} already graded.")
    elif skipped:
        print(f"Resume: {skipped} already graded, {len(todo)} to grade.")
    if not todo:
        print_join_summary(join_grades(rows, grades))
        return 0

    if args.non_interactive:
        verdict = parse_verdict_token(args.verdict or "")
        if verdict is None:
            print(f"Bad --verdict {args.verdict!r}.", file=sys.stderr)
            return 2
        for row in todo:
            grade = GradeRecord(
                case_id=str(row.get("case_id")),
                investigation_id=str(row.get("investigation_id", "")),
                verdict=verdict,
                notes=args.notes,
                graded_at=datetime.now(UTC).isoformat(),
            )
            _append_grade(grades_path, grade)
            grades.append(grade)
            print(f"[graded] {grade.case_id} verdict={grade.verdict}")
        print_join_summary(join_grades(rows, grades))
        return 0

    queries = _corpus_queries(args.corpus)
    try:
        for index, row in enumerate(todo, start=1):
            display_row(row, queries, f"{index}/{len(todo)}")
            verdict = prompt_verdict()
            if verdict is None:
                print(f"[skipped] {row.get('case_id')} (still ungraded)")
                continue
            notes = prompt_notes()
            grade = GradeRecord(
                case_id=str(row.get("case_id")),
                investigation_id=str(row.get("investigation_id", "")),
                verdict=verdict,
                notes=notes,
                graded_at=datetime.now(UTC).isoformat(),
            )
            _append_grade(grades_path, grade)
            grades.append(grade)
            print(f"[graded] {grade.case_id} verdict={grade.verdict}")
    except KeyboardInterrupt:
        print("\nGrading interrupted; progress saved, rerun to resume.")
    print_join_summary(join_grades(rows, grades))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
