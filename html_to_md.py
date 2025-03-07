import os
import glob
from pathlib import Path
from html2text import HTML2Text

def convert_to_markdown(html_content):
    """
    Convert HTML content to Markdown format
    
    Args:
        html_content (str): HTML content to convert
        
    Returns:
        str: Converted markdown content
    """
    converter = HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.body_width = 0  # Don't wrap text
    markdown_content = converter.handle(html_content)
    
    # Remove all content before "## On this page" if it exists
    if "## On this page" in markdown_content:
        markdown_content = markdown_content.split("## On this page", 1)[-1]
        markdown_content = "## On this page" + markdown_content
    
    return markdown_content

def main():
    # Create the output directory if it doesn't exist
    md_dir = Path("./md")
    md_dir.mkdir(exist_ok=True)
    
    # Find all HTML files in the html directory
    html_files = glob.glob("./html/*.html")
    
    if not html_files:
        print("No HTML files found in ./html/ directory")
        return
    
    # Process each HTML file
    for html_file in html_files:
        file_path = Path(html_file)
        output_file = md_dir / f"{file_path.stem}.md"
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Convert to markdown
        markdown_content = convert_to_markdown(html_content)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Converted {html_file} to {output_file}")

if __name__ == "__main__":
    main()
