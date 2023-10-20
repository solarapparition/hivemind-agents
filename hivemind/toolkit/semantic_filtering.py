from bs4 import BeautifulSoup
from bs4.element import Tag

def filter_semantic_html(html_string):

    # Parse HTML string using BeautifulSoup
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # Recursive function to traverse and filter nodes
    def traverse_filter(node):
        # Initialize list to keep filtered children
        new_children = []
        
        for child in list(node.children):  # Convert to list for proper traversal while modifying
            # Skip non-tag nodes (e.g., NavigableString)
            if not isinstance(child, Tag):
                new_children.append(child)
                continue
            
            # Keep tags that are interactive or denote a section or are headings
            keep_tag = child.name in {'button', 'a', 'input', 'select', 'textarea'} or \
                       child.has_attr('role') and child['role'] in {'heading', 'section'} or \
                       child.name in {'section', 'nav', 'article', 'aside', 'main', 'header', 'footer'} or \
                       child.name.startswith('h') and child.name[1:].isdigit()
            
            # Recurse into child to filter its children
            traverse_filter(child)
            
            # If tag should be kept, clean attributes and append to new_children
            if keep_tag:
                # Filter attributes
                attrs_to_keep = {k: v for k, v in child.attrs.items() if k in {'alt', 'title'} or k.startswith('aria-') or k == 'role' or k == 'aria-level'}
                child.attrs = attrs_to_keep
                new_children.append(child)
            else:
                # Move children up to parent (flatten), if any
                new_children.extend(list(child.children))
        
        # Replace old children with new filtered children
        node.clear()
        for new_child in new_children:
            node.append(new_child)
    
    # Start traversal from root node
    traverse_filter(soup.html)
    
    # Return BeautifulSoup object
    return soup

example_html = '''
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
'''

# Apply modified function
filtered_soup = filter_semantic_html(example_html)
from pathlib import Path
html = Path(".data/test_page.html").read_text(encoding="utf-8")
filtered_soup_2 = filter_semantic_html(html)

breakpoint()

