from app.pdf_pipeline.llama_extractor import _parse_paper_sections


def test_abstract_repairs_pdf_line_wraps_and_preserves_paragraphs():
    parsed = _parse_paper_sections(
        """# GraphRAG

# Abstract
The use of retrieval augmented generation (RAG) enables large language models
to answer questions over private document collections.

Prior methods do not scale to large corpora.

# 1. Introduction
This is the introduction.
"""
    )

    assert parsed["abstract"] == (
        "The use of retrieval augmented generation (RAG) enables large language models "
        "to answer questions over private document collections.\n\n"
        "Prior methods do not scale to large corpora."
    )
