import io
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bypasses using PyPDF2.
    """
    if not file_bytes:
        return ""

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""
