try:
    import PyPDF2
except ImportError:  # pragma: no cover - fallback for environments with pypdf only
    import pypdf as PyPDF2


def _read_pdf_text_with_pypdf(pdf_path: str) -> str:
    pages = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def read_pdf_text(
    pdf_path: str,
    *,
    pdf_text_backend: str = "pypdf",
) -> str:
    backend = pdf_text_backend.strip().lower()
    if backend != "pypdf":
        raise ValueError(f"Unsupported PDF text backend: {pdf_text_backend}")
    return _read_pdf_text_with_pypdf(pdf_path)
