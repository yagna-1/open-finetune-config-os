from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbformat import validate as validate_notebook
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class NotebookTemplateEngine:
    def __init__(self, template_dir: str | Path) -> None:
        self.template_dir = Path(template_dir)
        self.environment = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False,
            undefined=StrictUndefined,
        )

    def render_template(self, template_name: str, context: dict) -> dict:
        template = self.environment.get_template(f"{template_name}.j2")
        rendered = template.render(**context)
        return self._script_to_notebook(rendered)

    def write_notebook(self, template_name: str, context: dict, output_path: str | Path) -> dict:
        notebook = self.render_template(template_name=template_name, context=context)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(notebook, handle, indent=2, ensure_ascii=True)
        return notebook

    def _script_to_notebook(self, script: str) -> dict:
        raw_blocks = []
        current: list[str] = []
        for line in script.splitlines():
            if line.startswith("# %%"):
                if current:
                    raw_blocks.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            raw_blocks.append(current)

        cells = []
        for block in raw_blocks:
            header = block[0].strip()
            body = "\n".join(block[1:]).strip("\n")
            if not body:
                continue

            if header.startswith("# %% [markdown]"):
                markdown_lines = []
                for line in body.splitlines():
                    if line.startswith("# "):
                        markdown_lines.append(line[2:])
                    elif line.startswith("#"):
                        markdown_lines.append(line[1:])
                    else:
                        markdown_lines.append(line)
                source = "\n".join(markdown_lines).strip("\n") + "\n"
                cells.append(nbf.v4.new_markdown_cell(source=source))
                continue

            source = body.strip("\n") + "\n"
            cells.append(nbf.v4.new_code_cell(source=source))

        notebook = nbf.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3",
                },
            },
        )
        validate_notebook(notebook)
        return json.loads(nbf.writes(notebook, version=4))
