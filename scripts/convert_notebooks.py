#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Markdown for MkDocs documentation.

This script scans docs/examples/ for .ipynb files, extracts images,
and generates corresponding .md files with proper formatting for
Material for MkDocs theme.

Usage:
    python scripts/convert_notebooks.py [--notebooks PATTERN]

Examples:
    python scripts/convert_notebooks.py                    # Convert all notebooks
    python scripts/convert_notebooks.py --notebooks "1_*.ipynb"  # Convert specific pattern
"""

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


class NotebookConverter:
    """Convert Jupyter notebooks to Markdown."""

    def __init__(self, notebooks_dir: Path, images_dir: Path):
        self.notebooks_dir = notebooks_dir
        self.images_dir = images_dir
        self.images_dir.mkdir(exist_ok=True)

    def convert_all(self, pattern: str = "*.ipynb") -> None:
        """Convert all notebooks matching the pattern."""
        notebooks = sorted(self.notebooks_dir.glob(pattern))

        if not notebooks:
            print(f"No notebooks found matching: {pattern}")
            return

        print(f"Found {len(notebooks)} notebook(s) to convert\n")

        for notebook_path in notebooks:
            self.convert_notebook(notebook_path)

    def convert_notebook(self, notebook_path: Path) -> None:
        """Convert a single notebook to markdown."""
        print(f"Converting: {notebook_path.name}")

        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Generate base name for images (e.g., "1_prediction" from "1_prediction.ipynb")
        base_name = notebook_path.stem

        # Convert cells to markdown
        markdown_lines = []
        image_counter = 1

        for cell_idx, cell in enumerate(notebook['cells']):
            cell_type = cell['cell_type']

            if cell_type == 'markdown':
                # Process markdown cells
                content = ''.join(cell['source'])
                # Convert blockquote notes to Material admonitions
                content = self._convert_admonitions(content)
                markdown_lines.append(content)

            elif cell_type == 'code':
                # Add code block
                source = ''.join(cell['source'])
                markdown_lines.append(f"```python\n{source}\n```")

                # Process outputs
                outputs = cell.get('outputs', [])
                for output in outputs:
                    output_type = output.get('output_type')

                    # Handle text output
                    if output_type in ('stream', 'execute_result'):
                        text = self._get_text_output(output)
                        # Skip matplotlib Axes repr strings (not useful in docs)
                        if text and not self._is_matplotlib_repr(text):
                            markdown_lines.append(f"\n```\n{text.rstrip()}\n```")

                    # Handle image output
                    if output.get('data', {}).get('image/png'):
                        image_name = f"{base_name}_{image_counter}.png"
                        image_path = self.images_dir / image_name

                        # Extract and save image
                        png_data = output['data']['image/png'].replace('\n', '')
                        with open(image_path, 'wb') as img_file:
                            img_file.write(base64.b64decode(png_data))

                        # Add markdown image reference
                        markdown_lines.append(f"\n![](images/{image_name})")
                        image_counter += 1
                        print(f"  Extracted: {image_name}")

                markdown_lines.append("")  # Add spacing after code blocks

        # Write markdown file
        output_path = notebook_path.with_suffix('.md')
        markdown_content = '\n'.join(markdown_lines).strip() + '\n'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"  Created: {output_path.name}")
        print(f"  Images extracted: {image_counter - 1}\n")

    def _convert_admonitions(self, content: str) -> str:
        """Convert blockquote-style notes to Material admonitions."""
        # Pattern: > **Note:** text
        pattern = r'^>\s*\*\*(\w+):\*\*\s*(.+)$'

        def replace_admonition(match):
            admonition_type = match.group(1).lower()
            text = match.group(2)
            return f'!!! {admonition_type}\n    {text}'

        lines = content.split('\n')
        converted_lines = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                converted_lines.append(replace_admonition(match))
            else:
                converted_lines.append(line)

        return '\n'.join(converted_lines)

    def _get_text_output(self, output: Dict) -> str:
        """Extract text output from a cell output."""
        if 'text' in output:
            text = output['text']
        elif output.get('data', {}).get('text/plain'):
            text = output['data']['text/plain']
        else:
            return ""

        if isinstance(text, list):
            return ''.join(text)
        return text

    def _is_matplotlib_repr(self, text: str) -> bool:
        """Check if text is a matplotlib repr string (not useful in docs)."""
        text = text.strip()
        # Common matplotlib repr patterns
        patterns = [
            r'^<.*Axes.*>$',
            r'^<matplotlib\..*>$',
            r'^<Figure size.*>$',
        ]
        return any(re.match(pattern, text) for pattern in patterns)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Jupyter notebooks to Markdown for MkDocs'
    )
    parser.add_argument(
        '--notebooks',
        default='*.ipynb',
        help='Glob pattern for notebooks to convert (default: *.ipynb)'
    )
    parser.add_argument(
        '--examples-dir',
        type=Path,
        default=Path('docs/examples'),
        help='Directory containing notebooks (default: docs/examples)'
    )

    args = parser.parse_args()

    # Setup paths
    notebooks_dir = args.examples_dir
    images_dir = notebooks_dir / 'images'

    if not notebooks_dir.exists():
        print(f"Error: Directory not found: {notebooks_dir}")
        return 1

    # Convert notebooks
    converter = NotebookConverter(notebooks_dir, images_dir)
    converter.convert_all(args.notebooks)

    print("✓ Conversion complete!")
    return 0


if __name__ == '__main__':
    exit(main())
