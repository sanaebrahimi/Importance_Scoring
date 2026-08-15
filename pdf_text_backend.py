import re
import unicodedata

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
    return _clean_extracted_text("\n\n".join(pages))


def _clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\uE000-\uF8FF]", "", text)
    text = text.replace("\ufffd", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_paragraphs = []
    for raw_paragraph in re.split(r"\n\s*\n", text):
        lines = []
        for raw_line in raw_paragraph.splitlines():
            line = raw_line.strip()
            if not line or re.fullmatch(r"[.·•]+", line):
                continue
            lines.append(line)
        if not lines:
            continue
        cleaned_paragraphs.append(" ".join(lines))

    text = "\n\n".join(cleaned_paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_text(
    pdf_path: str,
    *,
    pdf_text_backend: str = "pypdf",
) -> str:
    backend = pdf_text_backend.strip().lower()
    if backend != "pypdf":
        raise ValueError(f"Unsupported PDF text backend: {pdf_text_backend}")
    return _read_pdf_text_with_pypdf(pdf_path)
