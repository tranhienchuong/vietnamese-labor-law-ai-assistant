# Final 100-Query Benchmark Metric Reference

## Benchmark Composition

- Total benchmark queries: 100
- In-corpus queries: 94
- Out-of-corpus refusal tests: 6

## Final Metrics

| Metric | Final value |
| --- | ---: |
| In-corpus Recall@10 | 0.828 |
| In-corpus Required Citation Coverage | 0.828 |
| Forbidden Citation Violation Rate | 0.043 |
| In-corpus retrieval pass rate | 0.755 |
| Adjusted end-to-end pass rate | 0.710 |
| Answer pass rate | 0.970 |
| Citation grounding pass rate | 1.000 |
| Quality pass rate | 0.970 |
| Out-of-corpus refusal pass rate | 0.000 |
| Low-information quotes | 1 |

## Metric Scope

Recall@10 and Required Citation Coverage are computed only over the 94 in-corpus queries that have non-empty `required_citations`. The 6 out-of-corpus queries are excluded from citation-retrieval coverage metrics so empty required-citation sets do not create artificial retrieval passes.

Out-of-corpus queries are evaluated separately using refusal or insufficient-context logic. A correct out-of-corpus response should identify insufficient indexed context, avoid unsupported legal citations, and avoid giving specific legal values that are not present in the indexed corpus.

The final end-to-end score is the adjusted end-to-end pass rate of 0.710. The built-in 100-query end-to-end pass rate of 0.760 is not used as the final benchmark score because it allows empty required-citation cases to pass retrieval coverage automatically.

## Main Limitation

The main limitation exposed by the final 100-query benchmark is retrieval robustness and out-of-corpus refusal behavior, not citation hallucination. Citation grounding remains strong, with a citation grounding pass rate of 1.000, while in-corpus retrieval coverage drops to 0.828 and out-of-corpus refusal pass rate is 0.000.
