# Thesis Float and Label Fix Summary

Date: 2026-05-28

## Labels Fixed

- Fixed Figure 3.5 label from `fig:fig_answer_generation_validation_flow` to `fig:answer-generation-citation-validation-flow`.
- Verified Chapter 3 `\ref{...}` entries resolve to existing Chapter 3 labels.
- Added the missing TODO-style bibliography key `todo_reranker` so the existing Chapter 3 reranker citation no longer produces an unresolved citation warning.

## Figure Float Specifiers Changed

- Figure 3.1 Overall System Architecture: changed to `[t!]` and moved later within Section 3.1 so it no longer floats above the Chapter 3 heading.
- Figure 3.2 Data Processing Pipeline: changed to `[tbp]`.
- Figure 3.3 Legal Graph Schema: changed to `[t!]`.
- Figure 3.4 Graph-Augmented Retrieval Flow: changed to `[tbp]`.
- Figure 3.5 Answer Generation and Citation Validation Flow: kept `[htbp]`, with caption unchanged.

## Table Float Specifiers Changed

- Table 3.3 graph node and edge counts: changed from `[H]` to `[tbp]`.
- Table 3.3 metrics were not changed.

## Chapter-End Float Flushing

- Added `\clearpage` at the end of `chapters/03_methodology.tex` after the final Chapter 3 figure.

## Compilation Result

- `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` passed.
- `main.pdf` was regenerated successfully.

## Unresolved References

- No unresolved Chapter 3 figure/table references remain.
- No `Figure ??` or `Table ??` markers were found in extracted PDF text.
- Figure 3.5 appears in the List of Figures.

## Remaining Layout Issues

- Chapter 3 floats do not spill into Chapter 4; Figure 3.5 is on page 27 and Chapter 4 starts on page 28.
- Figure 3.1 no longer appears above the Chapter 3 heading.
- Existing overfull/underfull box warnings remain in draft prose and long inline code strings.
- Existing `xdvipdfmx` warning `Object @page.1 already defined` remains.
