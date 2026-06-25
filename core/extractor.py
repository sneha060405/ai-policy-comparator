import fitz  # PyMuPDF
import docx
import os
from utils.text_cleaner import clean_text

def extract_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return clean_text(text)

def extract_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return clean_text(text)

def extract_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file."""
    return clean_text(file_bytes.decode('utf-8', errors='ignore'))

def extract_text(uploaded_file) -> str:
    """
    Auto-detect file type and extract text.
    Works with Streamlit UploadedFile objects.
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith('.pdf'):
        return extract_from_pdf(file_bytes)
    elif name.endswith('.docx'):
        return extract_from_docx(file_bytes)
    elif name.endswith('.txt'):
        return extract_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {name}. Please upload PDF, DOCX, or TXT.")