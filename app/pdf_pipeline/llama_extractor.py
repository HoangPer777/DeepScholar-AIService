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
    """Trim trailing spaces but preserve most newlines for content integrity."""
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def remove_references(text: str) -> str:
    """
    Remove the 'References', 'Appendix', 'Acknowledgements' sections and everything after them.
    """
    if not text:
        return ""
        
    lines = text.split("\n")
    cleaned_lines = []
    
    # Improved pattern to match:
    # 1. References / Reference
    # 2. Appendix / Appendices
    # 3. Acknowledgements / Acknowledgment
    # Supports markdown headings (#) and numbering (1., VI.)
    stop_pattern = re.compile(
        r"^(#+\s*)?([ivx]+\.?|\d+\.?\s*)?(references?|appendix|appendices|acknowledgments?)\s*$", 
        re.IGNORECASE
    )

    found_stop = False
    for line in lines:
        stripped = line.strip()
        if stop_pattern.match(stripped):
            print(f"[Extractor] Found stop section at: '{stripped}'. Cutting text here.")
            found_stop = True
            break
        cleaned_lines.append(line)
        
    result = "\n".join(cleaned_lines).strip()
    if found_stop:
        print(f"[Extractor] Content reduced from {len(text)} to {len(result)} chars.")
    return result


def _parse_paper_sections(full_text: str) -> dict:
    """
    Parse plain markdown text (all pages merged) into title / abstract / content.
    Uses a 'greedy' approach to ensure NO TEXT is lost between sections.
    """
    lines = full_text.split("\n")

    title: Optional[str] = None
    abstract_lines = []
    content_lines = []

    # Regex patterns for finding boundaries
    # We allow headings (#) or just standalone words
    abstract_pattern = re.compile(r"^(#+\s*)?abstract\s*$", re.IGNORECASE)
    intro_pattern = re.compile(r"^(#+\s*)?([ivx]+\.?|\d+\.?\s*)?introduction\s*$", re.IGNORECASE)
    stop_pattern = re.compile(
        r"^(#+\s*)?([ivx]+\.?|\d+\.?\s*)?(references?|appendix|appendices|acknowledgments?)\s*$", 
        re.IGNORECASE
    )

    abstract_start_idx = -1
    intro_start_idx = -1
    stop_idx = len(lines)

    # 1. Find boundaries first
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped: continue

        if abstract_start_idx == -1 and abstract_pattern.match(stripped):
            abstract_start_idx = i
        elif intro_start_idx == -1 and intro_pattern.match(stripped):
            intro_start_idx = i
        elif stop_pattern.match(stripped):
            # Only set stop_idx if it's after introduction (to avoid early stopping)
            if i > intro_start_idx and intro_start_idx != -1:
                stop_idx = i
                print(f"[Parser] Stop section found at line {i}: {stripped}")
                break

    # 2. Extract Title (first meaningful line before Abstract)
    search_limit = abstract_start_idx if abstract_start_idx != -1 else 20
    for i in range(min(search_limit, len(lines))):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            title = stripped
            break
        elif stripped.startswith("#"):
            title = re.sub(r"^#+\s*", "", stripped)
            break

    # 3. Extract Abstract (everything between 'Abstract' and 'Introduction')
    if abstract_start_idx != -1:
        end_idx = intro_start_idx if intro_start_idx != -1 else abstract_start_idx + 20
        # Skip the 'Abstract' heading itself
        abstract_lines = lines[abstract_start_idx+1 : end_idx]

    # 4. Extract Content (everything between 'Introduction' and 'Stop Section')
    # If no intro found, we start after abstract
    start_content = intro_start_idx if intro_start_idx != -1 else (abstract_start_idx + len(abstract_lines) + 1 if abstract_start_idx != -1 else 0)
    
    # Capture EVERYTHING in this range
    content_lines = lines[start_content : stop_idx]

    # Clean up title
    if title:
        title = re.sub(r"^(Paper Title|Title):?\s*", "", title, flags=re.IGNORECASE).strip()
    else:
        title = "Untitled Extract"

    print(f"[Parser] Extraction complete. Abstract: {len(abstract_lines)} lines, Content: {len(content_lines)} lines.")

    return {
        "title": title,
        "abstract": "\n".join(abstract_lines).strip(),
        "content": "\n".join(content_lines).strip(),
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
            # We remove the system_prompt here to prevent LlamaParse from "extracting" or "summarizing".
            # This ensures we get a faithful, full-text Markdown conversion.
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
