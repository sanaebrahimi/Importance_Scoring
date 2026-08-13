import hashlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

try:
    import PyPDF2
except ImportError:  # pragma: no cover - fallback for environments with pypdf only
    import pypdf as PyPDF2


DEFAULT_MINERU_BACKEND = "pipeline"
DEFAULT_MINERU_METHOD = "auto"
DEFAULT_MINERU_OUTPUT_ROOT = Path(".mineru_cache/mineru")
DEFAULT_MINERU_TIMEOUT_SECONDS = 1800


def _read_pdf_text_with_pypdf(pdf_path: str) -> str:
    pages = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _normalize_markdown_text(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"<summary[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</summary>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?details[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?table[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(tr|thead|tbody|tfoot|ul|ol|li|p|div|br|hr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(td|th|span|code|pre|em|strong)[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("```", "\n")
    text = text.replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _split_cli_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("MinerU CLI command is empty.")
    return parts


def _resolve_cli_command(command: str) -> Optional[list[str]]:
    parts = _split_cli_command(command)
    executable = shutil.which(parts[0])
    if executable is None:
        candidate = Path(parts[0]).expanduser()
        if not candidate.exists():
            return None
        executable = str(candidate.resolve())
    return [executable, *parts[1:]]


def _expected_markdown_path(
    output_root: Path,
    pdf_stem: str,
    mineru_backend: str,
    mineru_method: str,
) -> Path:
    if mineru_backend == "pipeline":
        parse_dir = output_root / pdf_stem / mineru_method
    elif mineru_backend.startswith("hybrid"):
        parse_dir = output_root / pdf_stem / f"hybrid_{mineru_method}"
    elif mineru_backend.startswith("vlm"):
        parse_dir = output_root / pdf_stem / "vlm"
    else:
        parse_dir = output_root / pdf_stem / mineru_method
    return parse_dir / f"{pdf_stem}.md"


def _build_cache_dir(pdf_path: str, output_root: str) -> Path:
    pdf = Path(pdf_path)
    stat = pdf.stat()
    identity = f"{pdf.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return Path(output_root) / f"{pdf.stem}_{digest}"


def read_pdf_text_with_mineru(
    pdf_path: str,
    *,
    mineru_cli: str = "mineru",
    mineru_output_root: str = str(DEFAULT_MINERU_OUTPUT_ROOT),
    mineru_backend: str = DEFAULT_MINERU_BACKEND,
    mineru_method: str = DEFAULT_MINERU_METHOD,
    mineru_timeout: int = DEFAULT_MINERU_TIMEOUT_SECONDS,
) -> str:
    resolved_cli = _resolve_cli_command(mineru_cli)
    if resolved_cli is None:
        raise FileNotFoundError(
            f"MinerU CLI '{mineru_cli}' was not found. "
            "Install MinerU or pass --mineru-cli with the correct executable."
        )

    pdf = Path(pdf_path)
    cache_dir = _build_cache_dir(pdf_path, mineru_output_root)
    markdown_path = _expected_markdown_path(cache_dir, pdf.stem, mineru_backend, mineru_method)

    if not markdown_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *resolved_cli,
            "-p",
            str(pdf),
            "-o",
            str(cache_dir),
            "-b",
            mineru_backend,
            "-m",
            mineru_method,
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, mineru_timeout),
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            raise RuntimeError(f"MinerU extraction failed for {pdf.name}: {detail}")

    if not markdown_path.exists():
        matches = sorted(cache_dir.rglob(f"{pdf.stem}.md"))
        if matches:
            markdown_path = matches[0]
        else:
            raise FileNotFoundError(
                f"MinerU completed but no markdown output was found under {cache_dir}."
            )

    return _normalize_markdown_text(markdown_path.read_text(encoding="utf-8", errors="ignore"))


def read_pdf_text(
    pdf_path: str,
    *,
    pdf_text_backend: str = "auto",
    mineru_cli: str = "mineru",
    mineru_output_root: str = str(DEFAULT_MINERU_OUTPUT_ROOT),
    mineru_backend: str = DEFAULT_MINERU_BACKEND,
    mineru_method: str = DEFAULT_MINERU_METHOD,
    mineru_timeout: int = DEFAULT_MINERU_TIMEOUT_SECONDS,
) -> str:
    backend = pdf_text_backend.strip().lower()
    if backend not in {"auto", "pypdf", "mineru"}:
        raise ValueError(f"Unsupported PDF text backend: {pdf_text_backend}")

    if backend == "pypdf":
        return _read_pdf_text_with_pypdf(pdf_path)

    if backend == "mineru":
        return read_pdf_text_with_mineru(
            pdf_path,
            mineru_cli=mineru_cli,
            mineru_output_root=mineru_output_root,
            mineru_backend=mineru_backend,
            mineru_method=mineru_method,
            mineru_timeout=mineru_timeout,
        )

    resolved_cli = _resolve_cli_command(mineru_cli)
    if resolved_cli is not None:
        try:
            return read_pdf_text_with_mineru(
                pdf_path,
                mineru_cli=mineru_cli,
                mineru_output_root=mineru_output_root,
                mineru_backend=mineru_backend,
                mineru_method=mineru_method,
                mineru_timeout=mineru_timeout,
            )
        except Exception:
            pass

    return _read_pdf_text_with_pypdf(pdf_path)
