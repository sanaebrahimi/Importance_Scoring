import re
from pathlib import Path
from typing import Optional

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit(
        "This script requires `pypdf`. Install it with `python3 -m pip install pypdf`."
    ) from exc


PAPERS_DIR = Path("papers")
OUTPUT_PATH = Path("papers_section_titles.txt")
CONNECTOR_WORDS = {"and", "or", "of", "the", "on", "in", "with", "vs", "for", "to", "a"}


def clean_title(raw_title: str) -> str:
    title = " ".join(raw_title.replace("\n", " ").split())

    if re.match(r"^[A-Z](?:\.\d+)*\s+", title):
        title = re.sub(r"^[A-Z](?:\.\d+)*\s+", "", title)
    elif re.match(r"^\d+(?:\.\d+)*\s+", title):
        title = re.sub(r"^\d+(?:\.\d+)*\s+", "", title)

    return title.strip()


def title_var_name(path: Path) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").upper()
    return f"{base}_SECTIONS"


def find_reference_page(reader: PdfReader) -> Optional[int]:
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        if any(re.match(r"^references(?:\d+)?$", line) for line in lines):
            return idx
    return None


def outline_has_references(outline_items) -> bool:
    for item in outline_items:
        if isinstance(item, list):
            if outline_has_references(item):
                return True
            continue
        title = getattr(item, "title", "")
        if clean_title(title).lower() == "references":
            return True
    return False


def parse_outline(reader: PdfReader):
    outline_items = reader.outline or []
    if not isinstance(outline_items, list):
        outline_items = [outline_items]

    ref_page = find_reference_page(reader)
    has_outline_refs = outline_has_references(outline_items)

    def walk(items, start_idx=0):
        result = []
        i = start_idx
        while i < len(items):
            item = items[i]

            if isinstance(item, list):
                i += 1
                continue

            raw_title = getattr(item, "title", "") or ""
            title = clean_title(raw_title)
            lower_title = title.lower()

            if lower_title == "references":
                return result, True, i

            try:
                page_num = reader.get_destination_page_number(item) + 1
            except Exception:
                page_num = None

            if not has_outline_refs and ref_page is not None and page_num is not None and page_num > ref_page:
                i += 1
                continue

            child_dict = None
            next_index = i + 1
            if next_index < len(items) and isinstance(items[next_index], list):
                children, stop, _ = walk(items[next_index], 0)
                if stop:
                    child_dict = children if children else None
                    if lower_title != "abstract":
                        result.append((title, child_dict))
                    return result, True, next_index
                child_dict = children if children else None
                i = next_index

            if lower_title != "abstract":
                result.append((title, child_dict))

            i += 1

        return result, False, i

    parsed, _, _ = walk(outline_items, 0)
    return parsed


def is_heading_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 35 or stripped.endswith((".", ":", ";")):
        return False
    if re.match(r"^\d+(?:\.\d+){0,2}\s+", stripped):
        return False
    if not re.search(r"[A-Za-z]", stripped):
        return False

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]*", stripped)
    if not words:
        return False
    if max(len(word) for word in words) > 20:
        return False

    good = 0
    for word in words:
        if word.lower() in CONNECTOR_WORDS or word[:1].isupper() or word.isupper():
            good += 1
    return good >= max(1, len(words) - 1)


def parse_text_fallback(reader: PdfReader):
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    lines = full_text.splitlines()
    entries = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "References":
            break

        match = re.match(r"^(\d+(?:\.\d+){0,2})\s+(.+)$", stripped)
        if match:
            number = match.group(1)
            title = match.group(2).strip()

            if not re.search(r"[A-Za-z]", title):
                i += 1
                continue
            if any(symbol in title for symbol in ["=", "+", "*", "<", ">", "|"]):
                i += 1
                continue

            level = number.count(".") + 1
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt or nxt == "References":
                    break
                if re.match(r"^\d+(?:\.\d+){0,2}\s+.+$", nxt):
                    break
                if is_heading_continuation(nxt):
                    title = f"{title} {nxt}"
                    j += 1
                    continue
                break

            entries.append((level, clean_title(f"{number} {title}")))
            i = j
            continue

        i += 1

    root = []
    stack = [(0, root)]
    for level, title in entries:
        node = [title, []]
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack[-1][1].append(node)
        stack.append((level, node[1]))
    return collapse_nodes(root)


def collapse_nodes(nodes):
    result = []
    for title, children in nodes:
        result.append((title, collapse_nodes(children) if children else None))
    return result


def format_entries(entries, indent=1):
    lines = []
    pad = "    " * indent
    for title, children in entries:
        if children:
            lines.append(f'{pad}"{title}": {{')
            lines.extend(format_entries(children, indent + 1))
            lines.append(f"{pad}}},")
        else:
            lines.append(f'{pad}"{title}": None,')
    return lines


def extract_sections(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    outline_entries = parse_outline(reader)
    if outline_entries:
        return outline_entries
    return parse_text_fallback(reader)


def main():
    output_blocks = []

    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        entries = extract_sections(pdf_path)
        variable_name = title_var_name(pdf_path)
        block = [f"# {pdf_path.name}", f"{variable_name} = {{"] + format_entries(entries) + ["}"]
        output_blocks.append("\n".join(block))

    OUTPUT_PATH.write_text("\n\n\n".join(output_blocks) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
