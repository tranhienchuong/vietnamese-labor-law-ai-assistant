# Thesis Figure Generation Summary

**Date:** 2026-05-28

## What Was Redesigned

All six thesis TikZ figures under `assets/figures/tikz/` were refined as clean academic boxed diagrams using subtle grayscale-friendly color coding. The update keeps metrics in tables, preserves the thesis structure, and avoids adding new system components.

1. **Figure 2.1: Standard RAG Pipeline**
   - Rebuilt as a two-lane diagram with boxed `Offline Indexing` and `Online Question Answering` containers.
   - Shows the requested flow from documents to vector index, and from user query to answer with sources.
   - Kept generic and non-legal-specific; vector/retrieval boxes use a subtle light-blue fill.

2. **Figure 3.1: Overall System Architecture**
   - Rebuilt as three visible stage containers: `Corpus Processing`, `Knowledge Stores`, and `Question Answering`.
   - Shows Qdrant and Neo4j as storage/index boxes and keeps reference edges feeding Neo4j.
   - Added a dashed bidirectional `shared chunk_id` link between `Qdrant Vector Index` and `Neo4j Legal Graph`.
   - Applied light blue to Qdrant and light green to Neo4j.

3. **Figure 3.2: Data Processing Pipeline**
   - Rebuilt as a boxed preprocessing pipeline ending in an `Output Artifacts` group.
   - Emphasizes the semantic preservation layer with a light gray fill.
   - Removed database drawings from this preprocessing figure.
   - Added a side branch from `Reference Extraction` to `Unresolved References Log`, labeled `111 unresolved references`.

4. **Figure 3.3: Legal Graph Schema**
   - Rebuilt as four conceptual zones: `Legal Hierarchy`, `Evidence`, `Metadata`, and `Legal Relations`.
   - Removed detailed counts so the figure remains conceptual and the metrics stay in Table 3.3.
   - Added directed graph notation with explicit arrow direction for structural and metadata relations.
   - Added a `Directed legal knowledge graph` note.

5. **Figure 3.4: Graph-Augmented Retrieval Flow**
   - Rebuilt as a boxed retrieval pipeline with a distinct `Graph Expansion Module`.
   - Uses a compact note box for the graph edge categories instead of listing edge names in the main flow.
   - Added mathematical notation: `q`, `S_k`, `C_graph`, and `C_final = S_k union C_graph`.
   - Applied light blue to hybrid seed retrieval, light green to graph expansion, and light gray to merge/final context boxes.

6. **Figure 3.5: Answer Generation and Citation Validation Flow**
   - Rebuilt as a clean validation pipeline with only two check boxes.
   - Includes a note stating that validation is deterministic and rule-based, not LLM-as-Judge.
   - Preserved the deterministic validation note exactly and added subtle validation/failure color coding.

## Compilation

Compilation command:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Status: passed. The command completed successfully and regenerated `main.pdf`.

## Remaining Layout Issues

No figure-specific overlap or page-width overflow remains after PDF rendering checks. The compile log still contains unrelated thesis-draft warnings from TODO text, one long section heading, and MiKTeX's existing `Object @page.1 already defined` warning; these are not caused by the redesigned TikZ figures.
