# Thesis LaTeX Template

This directory contains a modular LaTeX thesis skeleton for the project:

**Vietnamese Labor Law Question Answering System Using Retrieval-Augmented Generation and Graph-Augmented Retrieval with Neo4j**

## File Tree

```text
thesis/
├── main.tex
├── chapters/
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_methodology.tex
│   ├── 04_implementation.tex
│   ├── 05_evaluation.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
├── appendices/
│   └── appendix_a.tex
├── references.bib
└── README.md
```

## Compile with latexmk

Use XeLaTeX:

```bash
cd thesis
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

Or use LuaLaTeX:

```bash
cd thesis
latexmk -lualatex -interaction=nonstopmode -file-line-error main.tex
```

Clean generated files:

```bash
latexmk -C
```

## Compile on Overleaf

1. Upload the contents of this `thesis/` directory to Overleaf.
2. Set the main document to `main.tex`.
3. Set the compiler to **XeLaTeX** or **LuaLaTeX**.
4. Set the bibliography tool to **Biber** if Overleaf asks.
5. Recompile twice if references or the table of contents are not updated.

## Writing Notes

- Replace all `TODO`, `TBD`, and "to be filled after evaluation" placeholders after experiments are complete.
- Keep quantitative results out of the thesis until they are produced by the evaluation pipeline.
- Add real figures later by replacing the placeholder boxes with `\includegraphics`.
- Add or update bibliography entries in `references.bib` as the related work section matures.
