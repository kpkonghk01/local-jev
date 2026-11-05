"""Read a Markdown report the way its reader does: by its tables."""

from __future__ import annotations


def tables(markdown: str) -> list[list[dict[str, str]]]:
    """Every pipe table in `markdown`, each as its rows keyed by the header's cells.

    A cell reads as it renders: without its bold markers.
    """
    found, block = [], []
    for line in [*markdown.splitlines(), ""]:
        if line.startswith("|"):
            cells = line.strip().strip("|").split("|")
            block.append([cell.replace("**", "").strip() for cell in cells])
            continue
        if len(block) >= 2:
            header, _rule, *rows = block
            found.append([dict(zip(header, row, strict=True)) for row in rows])
        block = []
    return found


def table_with(markdown: str, column: str) -> list[dict[str, str]]:
    """The one table in `markdown` that has `column`."""
    (table,) = [t for t in tables(markdown) if t and column in t[0]]
    return table
