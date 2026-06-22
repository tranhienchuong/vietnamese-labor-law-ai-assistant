# Draft Thesis

This folder contains the concise draft rewrite of the thesis as an AI/software system thesis. It preserves the requested six-chapter structure and reads official metrics from the final evaluation artifacts under `artifacts/evaluation/final/`.

Compile from this folder with XeLaTeX or LuaLaTeX:

```powershell
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

