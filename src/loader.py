# src/loader.py
# Loads text from a .txt or .pdf file in the data/ folder

import os

def load_document(filepath: str) -> str:
    """
    Loads a document from the given filepath.
    Supports .txt and .pdf files.
    Returns the full text as a string.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            import PyPDF2
            text = ""
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            raise ImportError("PyPDF2 not installed. Run: pip install pypdf2")

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")
