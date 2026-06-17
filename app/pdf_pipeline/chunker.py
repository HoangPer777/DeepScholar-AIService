from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


@dataclass
class PaperChunk:
    content: str
    content_for_embedding: str
    chunk_index: int
    chunk_type: str
    section: str
    section_title: str
    section_level: int
    heading_path: list[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: int = 0
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunking_version: str = "v2"


_STOP_SECTION_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:[ivxlcdm]+\.?|\d+(?:\.\d+)*\.?)?\s*"
    r"(references?|bibliography|appendix|appendices|acknowledg(?:e)?ments?)\s*$",
    re.IGNORECASE,
)
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*((?:[ivxlcdm]+|\d+(?:\.\d+)*|[A-Z])\.?)\s+"
    r"([A-Z][A-Za-z0-9 ,:;()/\-&]+)\s*$",
    re.IGNORECASE,
)
_PLAIN_SECTION_RE = re.compile(
    r"^\s*(abstract|introduction|related work|background|method|methods|methodology|"
    r"approach|proposed method|proposed approach|experiments?|evaluation|results?|"
    r"ablation(?: study)?|discussion|analysis|conclusion|future work)\s*$",
    re.IGNORECASE,
)
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_TABLE_CAPTION_RE = re.compile(r"^\s*(?:table|TABLE)\s+(?:[IVXLCDM]+|\d+)[\.:]?\s+.+")
_FIGURE_CAPTION_RE = re.compile(r"^\s*(?:fig\.|figure)\s*\d+[\.:]?\s+.+", re.IGNORECASE)
_ALGORITHM_RE = re.compile(r"^\s*(?:algorithm|procedure)\s+\d*[\.:]?", re.IGNORECASE)
_DISPLAY_MATH_RE = re.compile(r"^\s*(?:\$\$|\\\[|\\begin\{(?:equation|align|algorithmic)\})")


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    """
    Legacy fallback splitter. Kept for backward compatibility and failure fallback.
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_text(text)


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.3))


def normalize_section_name(section_title: str | None) -> str:
    raw = (section_title or "").strip().lower()
    if re.match(r"^(?:table|figure|fig\.)\s*[a-z0-9]+", raw):
        return "other"
    raw = re.sub(r"^(?:#+\s*)?(?:[ivxlcdm]+\.?|\d+(?:\.\d+)*\.?|[a-z]\.?)\s+", "", raw)
    raw = re.sub(r"[^a-z0-9 ]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    if not raw:
        return "unknown"
    if "abstract" in raw:
        return "abstract"
    if "intro" in raw:
        return "introduction"
    if "related" in raw or "background" in raw:
        return "related_work"
    if any(term in raw for term in ("method", "methodology", "approach", "framework", "architecture", "model")):
        return "methodology"
    if any(term in raw for term in ("experiment", "evaluation", "result", "ablation", "benchmark")):
        return "results"
    if any(term in raw for term in ("discussion", "analysis")):
        return "discussion"
    if any(term in raw for term in ("conclusion", "future work", "limitation")):
        return "conclusion"
    normalized = raw.replace(" ", "_")
    return normalized[:128].rstrip("_") or "other"


def _is_stop_heading(line: str) -> bool:
    return bool(_STOP_SECTION_RE.match(line.strip()))


def _parse_heading(line: str) -> Optional[tuple[int, str]]:
    stripped = line.strip()
    if not stripped:
        return None
    markdown_match = _MARKDOWN_HEADING_RE.match(stripped)
    if markdown_match:
        return len(markdown_match.group(1)), markdown_match.group(2).strip()
    plain_match = _PLAIN_SECTION_RE.match(stripped)
    if plain_match:
        return 2, plain_match.group(1).strip()
    numbered_match = _NUMBERED_HEADING_RE.match(stripped)
    if numbered_match and len(stripped.split()) <= 10:
        return 2, stripped
    return None


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]


def _is_table_separator(line: str) -> bool:
    return bool(_TABLE_SEPARATOR_RE.match(line.strip()))


def _is_caption(line: str) -> bool:
    stripped = line.strip()
    return bool(_TABLE_CAPTION_RE.match(stripped) or _FIGURE_CAPTION_RE.match(stripped))


def _is_formula_or_algorithm_start(line: str) -> bool:
    stripped = line.strip()
    return bool(_ALGORITHM_RE.match(stripped) or _DISPLAY_MATH_RE.match(stripped))


def _clean_join(lines: Iterable[str]) -> str:
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _last_words(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[-max_tokens:])


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    if estimate_token_count(text) <= max_tokens:
        return [text.strip()]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence.strip()
        if estimate_token_count(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if estimate_token_count(sentence) <= max_tokens:
            current = sentence.strip()
        else:
            words = sentence.split()
            current_words: list[str] = []
            for word in words:
                current_words.append(word)
                if estimate_token_count(" ".join(current_words)) >= max_tokens:
                    chunks.append(" ".join(current_words[:-1]).strip())
                    current_words = [word]
            current = " ".join(current_words).strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _content_for_embedding(title: str, heading_path: list[str], chunk_type: str, content: str) -> str:
    section = " > ".join(heading_path) if heading_path else "Unknown"
    return f"Title: {title}\nSection: {section}\nType: {chunk_type}\n\n{content}".strip()


def _make_chunk(
    *,
    title: str,
    content: str,
    chunk_type: str,
    section_title: str,
    section_level: int,
    heading_path: list[str],
    chunk_index: int,
    metadata: Optional[dict[str, Any]] = None,
) -> PaperChunk:
    normalized = normalize_section_name(section_title)
    clean_content = content.strip()
    return PaperChunk(
        content=clean_content,
        content_for_embedding=_content_for_embedding(title, heading_path, chunk_type, clean_content),
        chunk_index=chunk_index,
        chunk_type=chunk_type,
        section=normalized,
        section_title=section_title,
        section_level=section_level,
        heading_path=list(heading_path),
        token_count=estimate_token_count(clean_content),
        metadata=metadata or {},
        chunking_version=settings.CHUNKING_VERSION,
    )


def _split_table(
    *,
    title: str,
    caption: str,
    table_lines: list[str],
    section_title: str,
    section_level: int,
    heading_path: list[str],
    start_index: int,
) -> list[PaperChunk]:
    if not table_lines:
        return []

    header = table_lines[:2] if len(table_lines) >= 2 and _is_table_separator(table_lines[1]) else table_lines[:1]
    rows = table_lines[len(header) :]
    max_rows = max(1, settings.TABLE_MAX_ROWS_PER_CHUNK)
    chunks: list[PaperChunk] = []

    if len(rows) <= max_rows:
        body = "\n".join(table_lines)
        content = f"{caption}\n\n{body}".strip() if caption else body
        return [
            _make_chunk(
                title=title,
                content=content,
                chunk_type="table",
                section_title=section_title,
                section_level=section_level,
                heading_path=heading_path,
                chunk_index=start_index,
                metadata={"table_caption": caption} if caption else {},
            )
        ]

    for row_start in range(0, len(rows), max_rows):
        row_group = rows[row_start : row_start + max_rows]
        body = "\n".join(header + row_group)
        content = f"{caption}\n\n{body}".strip() if caption else body
        chunks.append(
            _make_chunk(
                title=title,
                content=content,
                chunk_type="table",
                section_title=section_title,
                section_level=section_level,
                heading_path=heading_path,
                chunk_index=start_index + len(chunks),
                metadata={
                    "table_caption": caption,
                    "row_start": row_start,
                    "row_end": row_start + len(row_group) - 1,
                },
            )
        )
    return chunks


def _split_text_block(
    *,
    title: str,
    text: str,
    section_title: str,
    section_level: int,
    heading_path: list[str],
    start_index: int,
) -> list[PaperChunk]:
    target = max(128, settings.CHUNK_TARGET_TOKENS)
    max_tokens = max(target, settings.CHUNK_MAX_TOKENS)
    overlap = max(0, settings.CHUNK_OVERLAP_TOKENS)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]

    chunks: list[PaperChunk] = []
    current = ""
    for paragraph in paragraphs:
        pieces = _split_long_text(paragraph, max_tokens)
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and estimate_token_count(candidate) > target:
                chunks.append(
                    _make_chunk(
                        title=title,
                        content=current,
                        chunk_type="section_text",
                        section_title=section_title,
                        section_level=section_level,
                        heading_path=heading_path,
                        chunk_index=start_index + len(chunks),
                    )
                )
                current = _last_words(current, overlap)
                current = f"{current}\n\n{piece}".strip() if current else piece
            else:
                current = candidate
    if current:
        chunks.append(
            _make_chunk(
                title=title,
                content=current,
                chunk_type="section_text",
                section_title=section_title,
                section_level=section_level,
                heading_path=heading_path,
                chunk_index=start_index + len(chunks),
            )
        )
    return chunks


def _chunk_section(
    *,
    title: str,
    section_lines: list[str],
    section_title: str,
    section_level: int,
    heading_path: list[str],
    start_index: int,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    text_buffer: list[str] = []
    pending_caption = ""
    i = 0

    def flush_text() -> None:
        nonlocal text_buffer
        text = _clean_join(text_buffer)
        text_buffer = []
        if text:
            chunks.extend(
                _split_text_block(
                    title=title,
                    text=text,
                    section_title=section_title,
                    section_level=section_level,
                    heading_path=heading_path,
                    start_index=start_index + len(chunks),
                )
            )

    while i < len(section_lines):
        line = section_lines[i]
        stripped = line.strip()
        if not stripped:
            text_buffer.append(line)
            i += 1
            continue

        if _TABLE_CAPTION_RE.match(stripped):
            flush_text()
            pending_caption = stripped
            i += 1
            continue

        if _FIGURE_CAPTION_RE.match(stripped):
            flush_text()
            chunks.append(
                _make_chunk(
                    title=title,
                    content=stripped,
                    chunk_type="figure_caption",
                    section_title=section_title,
                    section_level=section_level,
                    heading_path=heading_path,
                    chunk_index=start_index + len(chunks),
                    metadata={"caption": stripped},
                )
            )
            i += 1
            continue

        if _is_table_line(line):
            flush_text()
            table_lines = []
            while i < len(section_lines) and (_is_table_line(section_lines[i]) or _is_table_separator(section_lines[i])):
                table_lines.append(section_lines[i].strip())
                i += 1
            chunks.extend(
                _split_table(
                    title=title,
                    caption=pending_caption,
                    table_lines=table_lines,
                    section_title=section_title,
                    section_level=section_level,
                    heading_path=heading_path,
                    start_index=start_index + len(chunks),
                )
            )
            pending_caption = ""
            continue

        if _is_formula_or_algorithm_start(line):
            flush_text()
            block_lines = [line]
            i += 1
            while i < len(section_lines) and section_lines[i].strip():
                if _parse_heading(section_lines[i]) or _is_caption(section_lines[i]):
                    break
                block_lines.append(section_lines[i])
                i += 1
            block = _clean_join(block_lines)
            chunks.append(
                _make_chunk(
                    title=title,
                    content=block,
                    chunk_type="formula_or_algorithm",
                    section_title=section_title,
                    section_level=section_level,
                    heading_path=heading_path,
                    chunk_index=start_index + len(chunks),
                    metadata={"block_type": "formula_or_algorithm"},
                )
            )
            continue

        text_buffer.append(line)
        i += 1

    flush_text()
    return chunks


def chunk_paper(
    title: str,
    abstract: str,
    content: str,
    source_format: str = "llamaparse_markdown",
) -> list[PaperChunk]:
    """
    Structure-aware chunking for scientific papers extracted as Markdown.
    """
    chunks: list[PaperChunk] = []
    clean_title = (title or "Untitled Paper").strip()
    clean_abstract = (abstract or "").strip()

    if clean_abstract:
        chunks.append(
            _make_chunk(
                title=clean_title,
                content=clean_abstract,
                chunk_type="abstract",
                section_title="Abstract",
                section_level=1,
                heading_path=[clean_title, "Abstract"],
                chunk_index=0,
                metadata={"source_format": source_format},
            )
        )

    lines = (content or "").splitlines()
    section_title = "Content"
    section_level = 1
    heading_path = [clean_title, section_title]
    section_lines: list[str] = []

    def flush_section() -> None:
        nonlocal section_lines
        if not section_lines:
            return
        chunks.extend(
            _chunk_section(
                title=clean_title,
                section_lines=section_lines,
                section_title=section_title,
                section_level=section_level,
                heading_path=heading_path,
                start_index=len(chunks),
            )
        )
        section_lines = []

    for line in lines:
        if _is_stop_heading(line):
            flush_section()
            break
        parsed_heading = _parse_heading(line)
        if parsed_heading:
            flush_section()
            level, heading = parsed_heading
            section_title = heading.strip()
            section_level = level
            heading_path = heading_path[: max(1, level)]
            if heading_path and heading_path[0] != clean_title:
                heading_path.insert(0, clean_title)
            if not heading_path:
                heading_path = [clean_title]
            if len(heading_path) >= level:
                heading_path = heading_path[:level]
            heading_path.append(section_title)
            continue
        section_lines.append(line)
    flush_section()

    # Re-number defensively after all split branches.
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
        chunk.metadata.setdefault("source_format", source_format)
    return chunks
