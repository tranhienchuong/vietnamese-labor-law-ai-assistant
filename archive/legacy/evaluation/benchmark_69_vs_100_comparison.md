# Benchmark 69 vs 100 Comparison

## Commands Executed

- Initial attempt, failed before evaluation because Neo4j was not running: `.\.venv\Scripts\python.exe scripts\evaluate_end_to_end_rag.py --benchmark-path artifacts\evaluation\end_to_end_expanded_benchmark.jsonl --output-dir artifacts\evaluation --output-prefix end_to_end_69_current --top-k 10 --prefetch-limit 24 --max-answer-contexts 8`
- `Start-Process -FilePath 'C:\Program Files\Docker\Docker\Docker Desktop.exe' -WindowStyle Hidden`
- `docker compose -f docker-compose.neo4j.yml up -d neo4j`
- `docker exec vietnamese-labor-law-neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n) AS node_count"`
- `.\.venv\Scripts\python.exe scripts\evaluate_end_to_end_rag.py --benchmark-path artifacts\evaluation\end_to_end_expanded_benchmark.jsonl --output-dir artifacts\evaluation --output-prefix end_to_end_69_current --top-k 10 --prefetch-limit 24 --max-answer-contexts 8`
- `.\.venv\Scripts\python.exe scripts\evaluate_end_to_end_rag.py --benchmark-path artifacts\evaluation\golden_benchmark_100_extended.jsonl --output-dir artifacts\evaluation --output-prefix end_to_end_100_extended --top-k 10 --prefetch-limit 24 --max-answer-contexts 8`
- Post-processed `artifacts/evaluation/end_to_end_100_extended_results.json` with separate in-corpus and out-of-corpus metric rules.

## Empty Required-Citation Handling

- `scripts/evaluate_retrieval_modes.py`: `coverage()` returns `0.0` when `citations` is empty, avoiding division by zero.
- `scripts/evaluate_end_to_end_rag.py`: `required_citation_coverage` is set to `1.0` when `item.required_citations` is empty, and `retrieval_passed` becomes true if there are no forbidden/order violations. Therefore the built-in 100-query aggregate is not valid for out-of-corpus retrieval coverage.

## Output Files

- `artifacts/evaluation/end_to_end_69_current_results.json`
- `artifacts/evaluation/end_to_end_69_current_results.csv`
- `artifacts/evaluation/end_to_end_69_current_report.md`
- `artifacts/evaluation/end_to_end_100_extended_results.json`
- `artifacts/evaluation/end_to_end_100_extended_results.csv`
- `artifacts/evaluation/end_to_end_100_extended_report.md`
- `artifacts/evaluation/benchmark_100_split_metrics.json`
- `artifacts/evaluation/benchmark_100_adjusted_failed_cases.csv`
- `artifacts/evaluation/benchmark_69_vs_100_comparison.md`

## Old 69-Query Metrics

| Metric | Value |
| --- | ---: |
| Query count | 69 |
| Recall@10 | 1.000 |
| Required Citation Coverage | 1.000 |
| Forbidden Citation Violation Rate | 0.000 |
| End-to-end pass rate | 1.000 |
| Citation grounding pass rate | 1.000 |

## New 94-Query In-Corpus Retrieval Metrics

| Metric | Value |
| --- | ---: |
| In-corpus query count | 94 |
| Recall@10 | 0.828 |
| Required Citation Coverage | 0.828 |
| Forbidden Citation Violation Rate | 0.043 |
| Retrieval pass rate | 0.755 |

## New 100-Query End-To-End Metrics

| Metric | Value |
| --- | ---: |
| Total query count | 100 |
| In-corpus end-to-end pass count | 71 / 94 |
| Out-of-corpus refusal pass count | 0 / 6 |
| Adjusted end-to-end pass count | 71 / 100 |
| Adjusted end-to-end pass rate | 0.710 |
| Built-in script end-to-end pass rate | 0.760 |
| Citation grounding pass rate | 1.000 |

The built-in 100-query pass rate is shown only for auditability; it is not the corrected benchmark score because it lets empty required-citation cases pass retrieval automatically.

## Out-Of-Corpus Refusal Metrics

| Metric | Value |
| --- | ---: |
| Out-of-corpus query count | 6 |
| Insufficient-context/refusal pass rate | 0.000 |
| Full out-of-corpus pass rate | 0.000 |
| Unsupported citation count | 0 |
| Legal-basis citations emitted | 21 |
| Specific out-of-corpus value failures detected | 1 |

The unsupported citation count uses the existing citation validator (`unsupported_article_numbers` plus `unretrieved_citations`). The system did not invent unretrieved citations, but it did emit in-corpus legal-basis citations for all six out-of-corpus questions instead of refusing.

## Failed Cases

- In-corpus failures: 23
- Out-of-corpus failures: 6

### In-Corpus Failure IDs

`robust_user_temp_transfer_other_job`, `robust_user_salary_deduction_damage`, `robust_user_late_monthly_salary`, `robust_user_wedding_paid_leave`, `robust_user_pregnant_night_overtime`, `robust_multihop_partial_year_annual_leave`, `robust_multihop_sexual_harassment_rules`, `robust_multihop_discipline_procedure_nd145`, `robust_multihop_domestic_worker_contract_notice`, `robust_multihop_reproductive_bad_work_list`, `robust_appendix_early_retirement_female_1974_05`, `robust_appendix_minor_light_work_bamboo_craft`, `robust_appendix_minor_night_work_packaging`, `robust_appendix_domestic_worker_contract_template`, `robust_hn_overtime_holiday_pay_not_limits`, `robust_hn_severance_not_job_loss`, `robust_hn_no_notice_resignation_not_employer`, `robust_hn_business_secret_agreement_not_contract_content`, `robust_dispute_unlawful_termination_court`, `robust_dispute_salary_deduction_court_choice`, `robust_dispute_job_loss_allowance_no_mediation`, `robust_dispute_domestic_worker_no_mediation`, `robust_dispute_collective_agreement_invalid_request`

### Out-Of-Corpus Failure IDs

`robust_ooc_minimum_wage_region_2026`, `robust_ooc_bhxh_contribution_rate_2026`, `robust_ooc_admin_fine_late_salary`, `robust_ooc_foreign_work_permit_dossier_2026`, `robust_ooc_trade_union_dues_rate`, `robust_ooc_pit_on_severance`

## Thesis Conclusion Impact

- The original 69-query result still holds at `1.000` when rerun under the current configuration.
- The expanded benchmark does not preserve the `1.000` result: in-corpus retrieval drops to Recall@10 `0.828` and Required Citation Coverage `0.828`; adjusted 100-query end-to-end pass rate is `0.710`.
- The thesis conclusion should be updated if the 100-query benchmark is adopted: the system remains strong on the original diagnostic set, but the robustness extension exposes citation-retrieval gaps and poor out-of-corpus refusal behavior.

## Recommendation

Do not replace the thesis results with the 100-query score without explaining the benchmark change. Recommended wording is to keep the 69-query result as the original controlled diagnostic result and add the 100-query robustness extension as a stress test. The 100-query result should be reported because it better captures robustness risk, especially refusal behavior for unsupported questions.
