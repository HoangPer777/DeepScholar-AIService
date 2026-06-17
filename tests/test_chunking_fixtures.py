import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "papers"


def test_lamaparse_markdown_fixture_loads():
    sample = FIXTURE_DIR / "llamaparse_sample_ieee.md"
    text = sample.read_text(encoding="utf-8")

    assert "# Introduction" in text
    assert "TABLE I." in text
    assert "## References" in text


def test_evaluation_questions_fixture_loads():
    questions = json.loads((FIXTURE_DIR / "evaluation_questions.json").read_text(encoding="utf-8"))

    assert len(questions) >= 4
    assert {q["expected_section"] for q in questions} >= {"introduction", "methodology", "results"}
