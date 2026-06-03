# Benchmark Extension 31 Summary

## Source benchmark

- Source file used: `artifacts/evaluation/end_to_end_expanded_benchmark.jsonl`
- Source count: 69 queries
- Schema preserved from source:
  `id`, `query`, `category`, `topic`, `expected_documents`, `required_citations`, `forbidden_citations`, `citation_order_rules`, `expected_answer_points`, `difficulty`, `requires_graph`, `requires_normative_hierarchy`

## Output files

- Extension only: `artifacts/evaluation/benchmark_extension_31.jsonl`
- Merged benchmark: `artifacts/evaluation/golden_benchmark_100_extended.jsonl`
- Summary report: `artifacts/evaluation/benchmark_extension_31_summary.md`

## Count verification

- Original benchmark: 69
- New extension: 31
- Merged benchmark: 100
- Arithmetic check: 69 + 31 = 100

## Distribution of new items

| Category | Count |
| --- | ---: |
| `out_of_corpus_qa` | 6 |
| `paraphrased_real_user_qa` | 5 |
| `multi_hop_qa` | 5 |
| `calculation_or_table_lookup` | 5 |
| `hard_negative_citation_qa` | 5 |
| `labor_dispute_procedure_qa` | 5 |

## New query IDs

- `robust_ooc_minimum_wage_region_2026`
- `robust_ooc_bhxh_contribution_rate_2026`
- `robust_ooc_admin_fine_late_salary`
- `robust_ooc_foreign_work_permit_dossier_2026`
- `robust_ooc_trade_union_dues_rate`
- `robust_ooc_pit_on_severance`
- `robust_user_temp_transfer_other_job`
- `robust_user_salary_deduction_damage`
- `robust_user_late_monthly_salary`
- `robust_user_wedding_paid_leave`
- `robust_user_pregnant_night_overtime`
- `robust_multihop_partial_year_annual_leave`
- `robust_multihop_sexual_harassment_rules`
- `robust_multihop_discipline_procedure_nd145`
- `robust_multihop_domestic_worker_contract_notice`
- `robust_multihop_reproductive_bad_work_list`
- `robust_appendix_retirement_male_1964_10`
- `robust_appendix_early_retirement_female_1974_05`
- `robust_appendix_minor_light_work_bamboo_craft`
- `robust_appendix_minor_night_work_packaging`
- `robust_appendix_domestic_worker_contract_template`
- `robust_hn_night_hours_not_night_pay`
- `robust_hn_overtime_holiday_pay_not_limits`
- `robust_hn_severance_not_job_loss`
- `robust_hn_no_notice_resignation_not_employer`
- `robust_hn_business_secret_agreement_not_contract_content`
- `robust_dispute_unlawful_termination_court`
- `robust_dispute_salary_deduction_court_choice`
- `robust_dispute_job_loss_allowance_no_mediation`
- `robust_dispute_domestic_worker_no_mediation`
- `robust_dispute_collective_agreement_invalid_request`

## Out-of-corpus items

These items intentionally have empty `expected_documents` and `required_citations`; their expected behavior is encoded in `expected_answer_points` with an explicit `insufficient_context` instruction.

- `robust_ooc_minimum_wage_region_2026`
- `robust_ooc_bhxh_contribution_rate_2026`
- `robust_ooc_admin_fine_late_salary`
- `robust_ooc_foreign_work_permit_dossier_2026`
- `robust_ooc_trade_union_dues_rate`
- `robust_ooc_pit_on_severance`

## Hard-negative items

| ID | Forbidden citations |
| --- | --- |
| `robust_hn_night_hours_not_night_pay` | BLLD Dieu 98 tien luong ban dem; ND145 Dieu 56 tien luong lam viec vao ban dem |
| `robust_hn_overtime_holiday_pay_not_limits` | BLLD Dieu 107 lam them gio; ND145 Dieu 60 gioi han so gio lam them |
| `robust_hn_severance_not_job_loss` | BLLD Dieu 47 tro cap mat viec lam |
| `robust_hn_no_notice_resignation_not_employer` | BLLD Dieu 36 quyen don phuong cua nguoi su dung lao dong; BLLD Dieu 37 truong hop nguoi su dung lao dong khong duoc don phuong |
| `robust_hn_business_secret_agreement_not_contract_content` | BLLD Dieu 21 noi dung hop dong lao dong chung; TT10 Dieu 3 noi dung chu yeu hop dong lao dong |

## Validation results

- Source count validated: 69
- Extension count validated: 31
- Merged count validated: 100
- Duplicate IDs: none
- Duplicate questions: none
- Schema mismatches against source schema: none
- In-corpus extension items without required citations: none
- Out-of-corpus items without clear insufficient-context expected behavior: none
- Required and forbidden citation specs for extension items resolve to indexed corpus coordinates: yes
- Merged file sequence matches source benchmark followed by extension: yes

## Assumptions

- The current 69-query source benchmark is `artifacts/evaluation/end_to_end_expanded_benchmark.jsonl`, because it is the 69-line benchmark artifact referenced by the thesis evaluation workflow.
- The project schema has no explicit `expected_insufficient_context` field, so out-of-corpus behavior is represented using the existing schema: empty citation/document lists plus `expected_answer_points` containing `insufficient_context`.
- New questions follow the current benchmark's unaccented Vietnamese text style to preserve local format consistency.
- Citation availability was checked against `artifacts/chunks/legal_chunks_enriched.jsonl`, the current six-document indexed corpus source.

## Evaluation status

No retrieval evaluation was run.

No end-to-end evaluation was run.
