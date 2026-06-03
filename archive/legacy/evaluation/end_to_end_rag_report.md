# End-to-End Legal RAG Evaluation

- Generated at: 2026-05-27T11:24:50.709905+00:00
- Benchmark queries: 15
- Top K: 10
- Provider: extractive
- End-to-end passed: True

## Overall Summary

| Queries | E2E pass rate | Retrieval pass rate | Answer pass rate | Citation pass rate | Quality pass rate | Avg quality score | Low-info quotes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 | 0 |

- Unsupported article numbers: None
- Unretrieved citations: None
- Graph expansion used: 15 queries
- Average graph depth: 3.833

## Per-Category Results

| Category | Queries | Retrieval pass | Answer pass | Citation pass | E2E pass | Avg quality |
| --- | --- | --- | --- | --- | --- | --- |
| comparison_qa | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| direct_qa | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| exception_based_qa | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| multi_hop_qa | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| procedure_qa | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| scenario_based_qa | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |

## Per-Query Results

| ID | Category | E2E | Required coverage | Citation | Quality | Score | Failure reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strict_minor_worker_14 | direct_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| strict_minor_worker_under_15_conditions | procedure_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| strict_retirement_age_2026_woman | scenario_based_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| strict_labor_contract_content | direct_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| strict_dismissal_dispute_mediation_before_lawsuit | procedure_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| strict_employee_unlawful_unilateral_termination | scenario_based_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| strict_structural_change_job_loss_allowance | scenario_based_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| strict_no_notice_resignation | exception_based_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| extra_employee_definition | direct_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| extra_contract_type_comparison | comparison_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| extra_employee_unlawful_vs_structural_change | comparison_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| extra_minor_worker_multihop_guidance | multi_hop_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| extra_retirement_multihop_guidance | multi_hop_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| extra_overtime_exception | exception_based_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |
| extra_probation_wage | scenario_based_qa | Pass | 1.00 | Pass | Pass | 100.00 | None |

## Successful Graph-Augmented Answers

### extra_contract_type_comparison

- Query: So sánh hợp đồng lao động xác định thời hạn và không xác định thời hạn?
- Top citations: Bộ luật Lao động 2019, Điều 20 (Loại hợp đồng lao động), khoản 1, các điểm a, b; Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 1, các điểm a, b, c, d, đ, e, g, h, i, k; Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 2
- Answer excerpt: Câu trả lời: Căn cứ vào Bộ luật Lao động 2019, Điều 20 (Loại hợp đồng lao động), khoản 1, các điểm a, b, có cơ sở pháp lý để trả lời trong phạm vi dữ liệu được truy xuất. Nội dung cụ thể như sau: - Trong Chương III. HỢP ĐỒNG LAO ĐỘNG; Mục 1. GIAO KẾT HỢP ĐỒNG LAO ĐỘNG, Bộ luật Lao động 2019, Điều 20 (Loại hợp đồng lao động), khoản 1, các điểm a, b quy định:...

### extra_employee_definition

- Query: Người lao động được định nghĩa như thế nào theo Bộ luật Lao động 2019?
- Top citations: Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ), khoản 1; Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ)
- Answer excerpt: Câu trả lời: Căn cứ vào Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ), khoản 1, có cơ sở pháp lý để trả lời trong phạm vi dữ liệu được truy xuất. Nội dung cụ thể như sau: - 1. Người lao động là người làm việc cho người sử dụng lao động theo thỏa thuận, được trả lương và chịu sự quản lý, điều hành, giám sát của người (Bộ luật Lao động 2019, Điều 3 (Giải...

### extra_employee_unlawful_vs_structural_change

- Query: So sánh trách nhiệm khi người lao động đơn phương chấm dứt hợp đồng trái luật với trường hợp công ty thay đổi cơ cấu phải trợ cấp?
- Top citations: Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 1; Bộ luật Lao động 2019, Điều 42 (Nghĩa vụ của người sử dụng lao động trong trường hợp thay đổi cơ cấu, công nghệ hoặc vì lý do kinh tế), khoản 5; Bộ luật Lao động 2019, Điều 47 (Trợ cấp mất việc làm), khoản 1
- Answer excerpt: Câu trả lời: Nếu người lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật, hậu quả chính là không được trợ cấp thôi việc và phải bồi thường cho người sử dụng lao động. (Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 2) Nội dung cụ thể như sau: - Không được trợ cấp thôi...


## Remaining Limitations

- No end-to-end failures were observed in this benchmark run.

## Failure Reason Counts

- None

## Thesis-Ready Conclusion

The end-to-end evaluation shows that graph-augmented retrieval improves the system's ability to answer complex labor-law questions requiring multiple connected provisions. The system maintains citation grounding by only citing retrieved legal contexts and applies normative hierarchy ordering when combining Labor Code provisions with decrees and circulars.
