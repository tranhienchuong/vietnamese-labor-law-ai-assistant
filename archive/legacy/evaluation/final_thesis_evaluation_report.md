# Final Thesis Evaluation Report

Prepared: 2026-05-27

This report summarizes the thesis evaluation package for the Vietnamese labor-law RAG assistant. It consolidates the corpus, chunking, metadata enrichment, cross-reference graph, vector index, retrieval-mode comparison, and end-to-end benchmark results generated from the existing repository artifacts.

Source artifacts:

- `artifacts/evaluation/retrieval_modes_expanded_report.md`
- `artifacts/evaluation/retrieval_modes_expanded_summary.json`
- `artifacts/evaluation/end_to_end_expanded_report.md`
- `artifacts/evaluation/end_to_end_expanded_results.json`
- `artifacts/evaluation/end_to_end_expanded_benchmark.jsonl`
- `artifacts/graph/legal_graph_build_summary.md`
- `artifacts/index/vector_index_summary.md`
- `artifacts/chunks/legal_chunks_summary.md`
- `artifacts/chunks/legal_chunks_enriched_summary.md`
- `artifacts/graph/reference_edges_summary.md`

## Dataset / Corpus Summary

The indexed corpus contains 6 legal documents focused on Vietnamese labor law and closely related labor-dispute procedure sources.

| Document ID | Type | Normative rank | Chunks |
| --- | --- | ---: | ---: |
| `45-2019-qh14` | `bo_luat` | 1 | 697 |
| `92-2015-qh13-labor-only` | `bo_luat` | 1 | 159 |
| `nghi-dinh-135-2020-nd-cp` | `nghi_dinh` | 2 | 41 |
| `nghi-dinh-145-2020-nd-cp` | `nghi_dinh` | 2 | 520 |
| `thong-tu-09-2020-tt-bldtbxh` | `thong_tu` | 3 | 73 |
| `thong-tu-10-2020-tt-bldtbxh` | `thong_tu` | 3 | 66 |

The corpus is intentionally scoped. It supports questions covered by the indexed labor-law documents, implementing decrees, circular guidance, retirement appendices, and selected labor-procedure provisions.

## Chunking Summary

The chunking artifact contains 1,556 hierarchy-aware legal chunks.

| Validation item | Value |
| --- | ---: |
| Chunk count | 1,556 |
| Duplicate chunk IDs | 0 |
| Missing citation text | 0 |
| Missing article number | 0 |
| Very short chunks | 0 |
| Very long chunks | 0 |
| Very long normal legal chunks | 0 |

Chunks by hierarchy level:

| Level | Chunks |
| --- | ---: |
| Appendix | 111 |
| Article | 121 |
| Clause | 1,324 |

Chunks by type:

| Chunk type | Chunks |
| --- | ---: |
| `clause` | 1,198 |
| `clause_part` | 126 |
| `article_full` | 64 |
| `appendix_form` | 54 |
| `article_intro` | 47 |
| `appendix` | 40 |
| `appendix_table` | 15 |
| `article_sequential` | 10 |
| `appendix_list` | 2 |

## Metadata Enrichment Summary

The enriched chunk artifact preserves the same 1,556 chunks and adds retrieval-oriented metadata such as topic, actor, issue type, document type, and normative rank.

| Enrichment field | Missing chunks |
| --- | ---: |
| Topic | 235 |
| Actor | 404 |
| Issue type | 690 |

Document-type distribution:

| Document type | Chunks |
| --- | ---: |
| `bo_luat` | 856 |
| `nghi_dinh` | 561 |
| `thong_tu` | 139 |

Normative-rank distribution:

| Normative rank | Chunks |
| --- | ---: |
| 1 | 856 |
| 2 | 561 |
| 3 | 139 |

The most frequent enriched topics include `hop_dong_lao_dong` with 377 chunks, `tranh_chap_lao_dong` with 210 chunks, `tien_luong` with 189 chunks, `ky_luat_lao_dong` with 178 chunks, and `to_tung_lao_dong` with 194 chunks. These distributions match the benchmark focus on contract formation, termination, discipline, overtime pay, allowances, retirement, and labor-dispute procedure.

## Cross-Reference Graph Summary

The cross-reference edge artifact contains extracted legal references before Neo4j loading.

| Item | Count |
| --- | ---: |
| Total reference edges extracted | 1,059 |
| Resolved edges | 948 |
| Unresolved edges | 111 |
| Duplicate edges removed | 1,565 |

Edges by extracted type:

| Edge type | Edges |
| --- | ---: |
| `DETAILS` | 334 |
| `GUIDED_BY` | 334 |
| `REFERENCES` | 391 |

The graph build loads only resolved references. This prevents unresolved legal citations from becoming graph traversal paths during retrieval.

## Vector Index Summary

The vector index uses dense multilingual sentence-transformer embeddings and stores chunk metadata for hybrid retrieval.

| Item | Value |
| --- | --- |
| Build ID | `20260526T121738Z` |
| Generated at | `2026-05-26T12:28:15.032286+00:00` |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Chunk count | 1,556 |
| Document count | 6 |
| Vector dimension | 384 |
| Collection | `vietnamese_labor_law_chunks` |
| Qdrant storage | `cloud` |
| Source chunks file | `artifacts/chunks/legal_chunks_enriched.jsonl` |

Index validation passed: all chunks were indexed, duplicate chunk IDs were 0, missing retrieval text was 0, missing citation text was 0, missing document IDs were 0, missing normative ranks were 0, and empty vector payloads were 0.

## Neo4j Graph Summary

The Neo4j legal graph was built and validated from the enriched chunks plus resolved reference edges.

Node counts:

| Node type | Count |
| --- | ---: |
| Documents | 6 |
| Articles | 411 |
| Clauses | 1,235 |
| Points | 774 |
| Appendices | 35 |
| Evidence chunks | 1,556 |
| Topic nodes | 28 |
| Actor nodes | 10 |
| Issue type nodes | 33 |

Loaded edge counts:

| Edge group | Count |
| --- | ---: |
| Resolved reference edges | 948 |
| `REFERENCES` | 290 |
| `DETAILS` | 329 |
| `GUIDED_BY` | 329 |
| Taxonomy edges | 6,099 |
| Normative hierarchy edges | 20 |

Neo4j validation passed. The loaded graph has no orphan evidence chunks, no unresolved reference edges loaded, balanced `DETAILS` and `GUIDED_BY` links, and correct normative-rank assignments.

## Retrieval Modes Comparison

The expanded retrieval evaluation compares vector-only retrieval, hybrid retrieval, and graph-augmented retrieval on the same 69-query benchmark.

| Mode | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR | Required citation coverage | Forbidden citation violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector-only | 0.012 | 0.012 | 0.026 | 0.020 | 0.022 | 0.012 | 0.000 |
| Hybrid | 0.860 | 0.920 | 0.571 | 0.380 | 0.833 | 0.920 | 0.000 |
| Graph-augmented | 0.961 | 1.000 | 0.562 | 0.359 | 0.872 | 1.000 | 0.000 |

Graph-augmented retrieval improved Recall@10 from 0.920 to 1.000 compared with hybrid retrieval. Required citation coverage improved from 0.920 to 1.000. The forbidden citation violation rate remained 0.000.

Vector-only diagnostics show that vector-only retrieval returned nonempty results for all 69 queries and preserved metadata for all 689 returned contexts. Its low score should therefore be interpreted as a weak baseline result on this benchmark, not as an empty-result or metadata-evaluation failure.

## End-to-End Evaluation Summary

The end-to-end evaluation measures graph-augmented retrieval plus deterministic extractive answer generation and citation validation.

| Metric | Value |
| --- | ---: |
| Benchmark queries | 69 |
| End-to-end passed | True |
| End-to-end pass rate | 1.000 |
| Retrieval pass rate | 1.000 |
| Answer pass rate | 1.000 |
| Citation pass rate | 1.000 |
| Quality pass rate | 1.000 |
| Average final quality score | 100.00 |
| Low-information quotes | 0 |
| Unsupported article numbers | None |
| Unretrieved citations | None |
| Graph expansion used | 68 queries |
| Average graph depth | 2.333 |

This result separates two claims: retrieval-mode evaluation shows that graph augmentation improves required citation retrieval over hybrid retrieval, while end-to-end evaluation shows that the complete graph-augmented RAG pipeline passes the constructed answer-quality and citation-grounding checks.

## Category-Level Analysis

| Category | Queries | Vector Recall@10 | Hybrid Recall@10 | Graph Recall@10 | Hybrid coverage | Graph coverage | E2E pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `calculation_or_table_lookup` | 5 | 0.067 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `comparison_qa` | 4 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `definition_qa` | 7 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `direct_qa` | 19 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `document_guidance_qa` | 10 | 0.000 | 0.683 | 1.000 | 0.683 | 1.000 | 1.000 |
| `exception_based_qa` | 11 | 0.045 | 0.955 | 1.000 | 0.955 | 1.000 | 1.000 |
| `multi_hop_qa` | 2 | 0.000 | 0.583 | 1.000 | 0.583 | 1.000 | 1.000 |
| `procedure_qa` | 7 | 0.000 | 0.857 | 1.000 | 0.857 | 1.000 | 1.000 |
| `scenario_based_qa` | 4 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The largest graph-augmented gains over hybrid retrieval appear in `document_guidance_qa`, `multi_hop_qa`, and `procedure_qa`. These query types often require connecting Labor Code provisions to decrees, circulars, appendices, or procedural rules.

## Topic-Level Analysis

All evaluated topics reached 1.000 retrieval pass, answer pass, citation pass, and end-to-end pass rates in the final graph-augmented end-to-end run.

| Topic | Queries | Retrieval pass | Answer pass | Citation pass | E2E pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ky luat lao dong` | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| `don phuong cham dut hop dong` | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| `nguoi chua du 15 tuoi` | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| `tuoi nghi huu` | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| `thoi gio lam viec nghi ngoi` | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| `thay doi co cau cong nghe` | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `lam them gio` | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `noi dung hop dong lao dong` | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `don phuong cham dut trai phap luat` | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `thu viec` | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `lao dong nu` | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `tien luong lam them gio` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hanh vi bi cam khi giao ket` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| `tro cap thoi viec` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| `loai hop dong lao dong` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| `tranh chap lao dong` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hoa giai truoc khi kien` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| `lao dong chua thanh nien` | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| Single-query topics | 14 | 1.000 | 1.000 | 1.000 | 1.000 |

Single-query topics include `nghia vu khi cham dut`, `tro cap khi cham dut`, `tien luong thu viec`, `tro cap mat viec lam`, `khai niem quan he lao dong`, `khai niem to chuc dai dien nguoi lao dong`, `khai niem nguoi su dung lao dong`, `khai niem hop dong lao dong`, `khai niem nguoi lao dong`, `so sanh trach nhiem cham dut`, `tham quyen cua Toa an theo BLTTDS`, `sa thai`, `hinh thuc hop dong lao dong`, and `phu luc hop dong lao dong`.

## Difficulty-Level Analysis

| Difficulty | Queries | Retrieval pass | Answer pass | Citation pass | E2E pass | Avg quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| Medium | 32 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| Hard | 22 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |

The final pipeline passes all benchmark difficulty levels. The hard subset includes comparison, multi-hop, table lookup, and document-guidance questions where retrieval completeness is more important than lexical similarity alone.

## Graph-Required vs Non-Graph-Required Analysis

| Requires graph | Queries | Retrieval pass | Answer pass | Citation pass | E2E pass | Avg quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 44 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| True | 25 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |

The graph-required subset is where graph augmentation is most meaningful. Retrieval-mode metrics show that graph augmentation closes gaps left by hybrid retrieval in document guidance, procedure, and multi-hop categories.

## Normative-Hierarchy-Required Analysis

| Requires normative hierarchy | Queries | Retrieval pass | Answer pass | Citation pass | E2E pass | Avg quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 47 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| True | 22 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |

The normative-hierarchy-required subset evaluates whether the system can combine higher-rank Labor Code provisions with lower-rank implementing decrees or circulars without losing citation grounding. The final run passes this subset with 1.000 end-to-end pass rate and 1.000 citation pass rate.

## Careful Interpretation

- The 100% result is measured on the constructed 69-query benchmark.
- The benchmark is not a guarantee of universal legal correctness.
- The system is strongest for questions covered by the labor-law corpus and benchmark scope.
- External laws outside the indexed corpus may still require additional sources.
- Legal consultation should still be verified by qualified professionals for real cases.

## Remaining Limitations

- The benchmark is manually and heuristically constructed, so it may reflect the design priorities and coverage choices of this project.
- Answer quality is evaluated using deterministic rule-based validation, not human legal expert review or LLM-as-Judge evaluation.
- Answer generation is currently extractive and deterministic. This improves citation safety but may be less fluent or less adaptive than a carefully constrained LLM-based answer generator.
- The system depends on chunk quality, citation parsing, metadata enrichment, and graph-edge extraction. Errors in any of these upstream artifacts can affect retrieval and answers.
- The vector-only baseline is weak compared with hybrid and graph-augmented retrieval on this benchmark, although diagnostics confirm it returns nonempty results with preserved metadata.
- The corpus is scoped to selected Vietnamese labor-law materials. Questions requiring external statutes, newer legal changes, administrative guidance outside the indexed corpus, or case-specific factual analysis may require more sources.
- Future work should include an expert-validated benchmark, human legal review, LLM-as-judge evaluation with strict grounding checks, larger adversarial tests, and longitudinal checks when source law changes.

## Thesis-Ready Conclusion

On the constructed 69-query benchmark, the graph-augmented RAG pipeline achieved 100% end-to-end pass rate, 100% citation validation pass rate, and 100% required citation coverage. Compared with hybrid retrieval, graph-augmented retrieval improved Recall@10 from 0.920 to 1.000 and required citation coverage from 0.920 to 1.000. The improvement is most visible in document-guidance, multi-hop, procedure, and calculation/table-lookup questions, where answers often require linking provisions from the Labor Code with decrees, circulars, or appendices.

These results support the thesis claim that graph-augmented retrieval can improve citation-grounded legal question answering in a scoped Vietnamese labor-law corpus. The claim should be interpreted as benchmark-specific evidence, not as proof of universal legal correctness.
