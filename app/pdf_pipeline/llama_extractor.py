"""
LlamaParse-based PDF Extractor for IEEE scientific papers.

Parsing logic mirrors the proven Colab script:
  1. Parse PDF → markdown via LlamaParse API
  2. Merge all pages into one text blob
  3. Extract TITLE  → first # heading
  4. Extract ABSTRACT → lines between "# abstract" heading and next heading
  5. Extract CONTENT  → lines from "# i. introduction" / "# 1. introduction" onward
"""

import os
import re
import tempfile
from typing import Optional
from app.core.config import settings


def _clean_text(text: str) -> str:
    """Collapse excessive newlines and trim trailing spaces."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _parse_paper_sections(full_text: str) -> dict:
    """
    Parse plain markdown text (all pages merged) into title / abstract / content.

    Strategy (same as proven Colab script):
    - TITLE    : first line that starts with '# '
    - ABSTRACT : content between the line matching /# abstract/i  
                 and the next heading that starts with '#'
    - CONTENT  : content from the line matching
                 /# (i\.|1)\s*introduction/i  onward.
                 Falls back to the full text if no introduction heading found.
    """
    lines = full_text.split("\n")

    title: Optional[str] = None
    abstract = ""
    content = ""

    # ── 1. Title ─────────────────────────────────────────────────────────────
    for line in lines:
        if line.startswith("# "):
            title = line.replace("#", "").strip()
            break

    # ── 2. Abstract ──────────────────────────────────────────────────────────
    abstract_start = None
    abstract_end = None

    for i, line in enumerate(lines):
        # Match any heading that contains the word "abstract"
        if re.match(r"#+\s*abstract", line.strip(), re.IGNORECASE):
            abstract_start = i + 1
            continue
        # Stop at the next heading after abstract block begins
        if abstract_start is not None and line.strip().startswith("#"):
            abstract_end = i
            break

    if abstract_start is not None:
        if abstract_end is None:
            abstract_end = len(lines)
        abstract = "\n".join(lines[abstract_start:abstract_end]).strip()

    # ── 3. Content ───────────────────────────────────────────────────────────
    content_start = None

    for i, line in enumerate(lines):
        # Match "# I. Introduction", "# 1. Introduction", "# 1 Introduction" etc.
        if re.match(r"#+\s*(i\.|1\.?)\s*introduction", line.strip(), re.IGNORECASE):
            content_start = i
            break

    if content_start is not None:
        content = "\n".join(lines[content_start:]).strip()
    else:
        # Fallback: use everything from after the abstract block
        if abstract_end is not None:
            content = "\n".join(lines[abstract_end:]).strip()
        else:
            content = full_text

    return {
        "title": title or "",
        "abstract": abstract,
        "content": content,
    }


def extract_sections_with_llamaparse(file_bytes: bytes) -> Optional[dict]:
    """
    Use LlamaParse API to extract title, abstract, and content from an IEEE PDF.

    Args:
        file_bytes: Raw PDF bytes

    Returns:
        dict with keys: title, abstract, content
        Returns None if extraction fails completely.
    """
    api_key = settings.LLAMAPARSE_API_KEY
    if not api_key:
        print("[LlamaParse] LLAMAPARSE_API_KEY is not set. Falling back to PyPDF2 extractor.")
        return None

    try:
        from llama_parse import LlamaParse

        print("[LlamaParse] Initializing parser...")

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            num_workers=1,
            verbose=False,
            language="en",
            system_prompt=(
                "This is an IEEE format scientific research paper. "
                "Please extract and preserve all sections: "
                "Title, Abstract, Introduction, Methodology, Results, "
                "Discussion, Conclusion, References. "
                "Use proper markdown headings (# for title, ## for sections). "
                "Do NOT insert '# Session N' or '## Session N' markers. "
                "Preserve all mathematical equations, tables, and figure captions."
            ),
        )

        # LlamaParse requires a file path → write bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            documents = parser.load_data(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not documents:
            print("[LlamaParse] Returned no documents.")
            return None

        # Merge all pages → one markdown blob → clean
        full_markdown = "\n\n".join(doc.text for doc in documents)
        full_markdown = _clean_text(full_markdown)
        print(f"[LlamaParse] Got {len(full_markdown)} characters of raw markdown.")

        # Parse into sections
        sections = _parse_paper_sections(full_markdown)

        print(f"[LlamaParse] Title   : {sections['title'][:80] if sections['title'] else '(empty)'}")
        print(f"[LlamaParse] Abstract: {len(sections['abstract'])} chars")
        print(f"[LlamaParse] Content : {len(sections['content'])} chars")

        # Return None if title is missing — triggers fallback
        if not sections["title"]:
            print("[LlamaParse] Could not extract title. Will fall back to LLM extractor.")
            return None

        return sections

    except ImportError:
        print("[LlamaParse] Package not installed. Run: pip install llama-parse")
        return None
    except Exception as e:
        print(f"[LlamaParse] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
