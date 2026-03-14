from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    """
    Split text into overlapping semantic chunks.
    Uses Langchain's RecursiveCharacterTextSplitter for optimal NLP boundaries.
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks
