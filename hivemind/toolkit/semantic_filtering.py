"""Filter HTML to keep only key semantic elements."""

from functools import lru_cache
from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Tag, PageElement


def traverse_filter(node: Tag) -> None:
    """Traverse and filter HTML tree recursively."""
    new_children: list[PageElement] = []
    for child in list(node.children):
        # Skip non-tag nodes (e.g., NavigableString)
        if not isinstance(child, Tag):
            new_children.append(child)
            continue

        # Keep tags that are interactive or denote a section or are headings
        keep_tag = (
            child.name in {"button", "a", "input", "select", "textarea"}
            or child.has_attr("role")
            and child["role"] in {"heading", "section"}
            or child.name
            in {"section", "nav", "article", "aside", "main", "header", "footer"}
            or child.name.startswith("h")
            and child.name[1:].isdigit()
        )

        traverse_filter(child)

        if keep_tag:
            # Clean attributes and append to new_children
            attrs_to_keep = {
                k: v
                for k, v in child.attrs.items()
                if k in {"alt", "title", "id"}
                or k.startswith("aria-")
                or k == "role"
                or k == "aria-level"
            }
            child.attrs = attrs_to_keep
            new_children.append(child)
        else:
            # Move children up to parent (flatten), if any
            new_children.extend(list(child.children))

    node.clear()
    for new_child in new_children:
        node.append(new_child)


@lru_cache
def filter_semantic_html(html_string: str) -> BeautifulSoup:
    """Filter HTML to keep only key semantic elements."""
    soup = BeautifulSoup(html_string, "html.parser")
    if not soup.html:
        raise ValueError("HTML must have an html tag.")
    traverse_filter(soup.html)
    return soup


def test_filter_semantic_html_1() -> None:
    """Test small HTML."""
    example_html_1 = """
    <html>
        <head>
            <title>Example Page</title>
        </head>
        <body>
            <div id="container">
                <h1 role="heading">Main Title</h1>
                <section aria-label="Intro">
                    <p>Welcome to the site.</p>
                </section>
                <div class="menu">
                    <a href="#">Home</a>
                    <a href="#">About</a>
                </div>
                <button>Click me!</button>
                <img src="image.jpg" alt="An image">
            </div>
        </body>
    </html>
    """
    print(filter_semantic_html(example_html_1))


def test_filter_semantic_html_2() -> None:
    """Test bigger HTML."""
    example_html_2 = Path(".data/test_page.html").read_text(encoding="utf-8")
    print(filter_semantic_html(example_html_2))


# if __name__ == "__main__":
# test_filter_semantic_html_1()
# test_filter_semantic_html_2()
