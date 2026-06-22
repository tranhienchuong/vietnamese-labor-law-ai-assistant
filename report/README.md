# Report Workspace

This directory is organized to keep the report sources easy to navigate on GitHub.

## Recommended entry points

- `draft/`: current working draft and main writing workspace.
- `common/`: shared style, bibliography, and logo used by the report sources.
- `style_reference/`: formatting reference material.

## Directory layout

```text
report/
|-- common/
|-- draft/
`-- style_reference/
```

Legacy duplicate report sources and top-level LaTeX build outputs were removed so readers can immediately see which version to open.

## Build commands

Compile the working draft:

```powershell
cd report/draft
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

Clean generated files inside a report folder:

```powershell
latexmk -C
```

## Tracking policy

- Keep report source files such as `.tex`, `.bib`, images, and shared assets in git.
- Keep output `.pdf` files when they are useful to review or submit.
- Ignore LaTeX build artifacts such as `.aux`, `.log`, `.bcf`, `.run.xml`, and related temporary files.
