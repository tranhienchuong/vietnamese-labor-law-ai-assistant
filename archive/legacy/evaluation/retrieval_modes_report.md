# Retrieval Modes Evaluation

- Generated at: 2026-05-27T11:28:09.241395+00:00
- Benchmark queries: 15
- Modes: vector_only, hybrid, graph_augmented

## Overall Comparison

| Mode | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR | Required coverage | Forbidden rate | Avg graph chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_only | 0.600 | 0.661 | 0.453 | 0.320 | 0.872 | 0.583 | 0.000 | 0.00 |
| hybrid | 0.683 | 0.750 | 0.560 | 0.380 | 0.800 | 0.750 | 0.067 | 0.00 |
| graph_augmented | 0.900 | 1.000 | 0.747 | 0.513 | 1.000 | 1.000 | 0.000 | 6.73 |

## Per-Category Comparison

| Category | Mode | Queries | Recall@10 | Precision@10 | MRR | Required coverage |
| --- | --- | --- | --- | --- | --- | --- |
| comparison_qa | vector_only | 2 | 0.833 | 0.400 | 1.000 | 0.833 |
| comparison_qa | hybrid | 2 | 1.000 | 0.500 | 1.000 | 1.000 |
| comparison_qa | graph_augmented | 2 | 1.000 | 0.600 | 1.000 | 1.000 |
| direct_qa | vector_only | 3 | 0.583 | 0.333 | 0.750 | 0.417 |
| direct_qa | hybrid | 3 | 0.583 | 0.300 | 0.667 | 0.583 |
| direct_qa | graph_augmented | 3 | 1.000 | 0.533 | 1.000 | 1.000 |
| exception_based_qa | vector_only | 2 | 0.750 | 0.150 | 1.000 | 0.500 |
| exception_based_qa | hybrid | 2 | 0.750 | 0.350 | 1.000 | 0.750 |
| exception_based_qa | graph_augmented | 2 | 1.000 | 0.400 | 1.000 | 1.000 |
| multi_hop_qa | vector_only | 2 | 0.500 | 0.400 | 0.667 | 0.500 |
| multi_hop_qa | hybrid | 2 | 0.500 | 0.300 | 0.667 | 0.500 |
| multi_hop_qa | graph_augmented | 2 | 1.000 | 0.400 | 1.000 | 1.000 |
| procedure_qa | vector_only | 2 | 0.500 | 0.250 | 0.750 | 0.500 |
| procedure_qa | hybrid | 2 | 0.667 | 0.350 | 0.333 | 0.667 |
| procedure_qa | graph_augmented | 2 | 1.000 | 0.450 | 1.000 | 1.000 |
| scenario_based_qa | vector_only | 4 | 0.750 | 0.350 | 1.000 | 0.708 |
| scenario_based_qa | hybrid | 4 | 0.917 | 0.450 | 1.000 | 0.917 |
| scenario_based_qa | graph_augmented | 4 | 1.000 | 0.600 | 1.000 | 1.000 |

## Per-Query Comparison

| ID | Category | Query | Vector R/M/C | Hybrid R/M/C | Graph R/M/C |
| --- | --- | --- | --- | --- | --- |
| strict_minor_worker_14 | direct_qa | Người 14 tuổi có được làm việc không? | 0.75/1.00/0.75 | 0.75/1.00/0.75 | 1.00/1.00/1.00 |
| strict_minor_worker_under_15_conditions | procedure_qa | Người chưa đủ 15 tuổi làm việc cần điều kiện gì? | 0.67/0.50/0.67 | 1.00/0.33/1.00 | 1.00/1.00/1.00 |
| strict_retirement_age_2026_woman | scenario_based_qa | Nữ nghỉ hưu năm 2026 thì bao nhiêu tuổi? | 0.67/1.00/0.67 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| strict_labor_contract_content | direct_qa | Hợp đồng lao động cần có những nội dung gì? | 0.50/0.25/0.50 | 0.00/0.00/0.00 | 1.00/1.00/1.00 |
| strict_dismissal_dispute_mediation_before_lawsuit | procedure_qa | Tranh chấp sa thải có cần hòa giải trước khi kiện không? | 0.33/1.00/0.33 | 0.33/0.33/0.33 | 1.00/1.00/1.00 |
| strict_employee_unlawful_unilateral_termination | scenario_based_qa | Người lao động đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì? | 0.67/1.00/0.50 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| strict_structural_change_job_loss_allowance | scenario_based_qa | Công ty thay đổi cơ cấu thì phải trả trợ cấp gì? | 0.67/1.00/0.67 | 0.67/1.00/0.67 | 1.00/1.00/1.00 |
| strict_no_notice_resignation | exception_based_qa | Khi nào người lao động được nghỉ việc không cần báo trước? | 0.50/1.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| extra_employee_definition | direct_qa | Người lao động được định nghĩa như thế nào theo Bộ luật Lao động 2019? | 0.50/1.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| extra_contract_type_comparison | comparison_qa | So sánh hợp đồng lao động xác định thời hạn và không xác định thời hạn? | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| extra_employee_unlawful_vs_structural_change | comparison_qa | So sánh trách nhiệm khi người lao động đơn phương chấm dứt hợp đồng trái luật với trường hợp công ty thay đổi cơ cấu phải trợ cấp? | 0.67/1.00/0.67 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| extra_minor_worker_multihop_guidance | multi_hop_qa | Người chưa đủ 15 tuổi được làm việc theo Bộ luật Lao động và thông tư nào hướng dẫn điều kiện? | 0.50/0.33/0.50 | 0.50/0.33/0.50 | 1.00/1.00/1.00 |
| extra_retirement_multihop_guidance | multi_hop_qa | Tuổi nghỉ hưu theo BLLĐ Điều 169 được Nghị định 135 hướng dẫn thế nào? | 0.50/1.00/0.50 | 0.50/1.00/0.50 | 1.00/1.00/1.00 |
| extra_overtime_exception | exception_based_qa | Trường hợp nào được làm thêm giờ và giới hạn làm thêm theo tháng là bao nhiêu? | 1.00/1.00/1.00 | 0.50/1.00/0.50 | 1.00/1.00/1.00 |
| extra_probation_wage | scenario_based_qa | Thử việc thì tiền lương tối thiểu phải bằng bao nhiêu phần trăm lương của công việc? | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |

## Graph Diagnostics

- Graph expansion used count: 15
- Average graph depth: 3.792
- Graph hit ratio: 0.802
- Direct vector hit ratio in graph mode: 0.198
- Improved over vector-only: 12
- Worse than vector-only: 0
- Top graph edge types: SOURCE_OF:9, DETAILS:9, HAS_SOURCE_CHUNK:9

## Graph Improvements

- `strict_minor_worker_14` (direct_qa): +0.500
  - Query: Người 14 tuổi có được làm việc không?
  - Vector top 3: Bộ luật Lao động 2019, Điều 143 (Lao động chưa thành niên), khoản 3; Bộ luật Lao động 2019, Điều 143 (Lao động chưa thành niên), khoản 2; Bộ luật Lao động 2019, Điều 145 (Sử dụng người chưa đủ 15 tuổi làm việc), khoản 2
  - Graph top 3: Thông tư 09/2020/TT-BLĐTBXH, Điều 3 (Điều kiện sử dụng người chưa đủ 15 tuổi làm việc); Bộ luật Lao động 2019, Điều 143 (Lao động chưa thành niên), khoản 1; Bộ luật Lao động 2019, Điều 145 (Sử dụng người chưa đủ 15 tuổi làm việc), khoản 1, các điểm a, b, c, d
- `strict_minor_worker_under_15_conditions` (procedure_qa): +1.167
  - Query: Người chưa đủ 15 tuổi làm việc cần điều kiện gì?
  - Vector top 3: Bộ luật Lao động 2019, Điều 143 (Lao động chưa thành niên), khoản 2; Bộ luật Lao động 2019, Điều 145 (Sử dụng người chưa đủ 15 tuổi làm việc), khoản 2; Bộ luật Lao động 2019, Điều 143 (Lao động chưa thành niên), khoản 3
  - Graph top 3: Thông tư 09/2020/TT-BLĐTBXH, Điều 3 (Điều kiện sử dụng người chưa đủ 15 tuổi làm việc); Bộ luật Lao động 2019, Điều 143 (Lao động chưa thành niên), khoản 1; Bộ luật Lao động 2019, Điều 145 (Sử dụng người chưa đủ 15 tuổi làm việc), khoản 1, các điểm a, b, c, d
- `strict_retirement_age_2026_woman` (scenario_based_qa): +0.667
  - Query: Nữ nghỉ hưu năm 2026 thì bao nhiêu tuổi?
  - Vector top 3: Bộ luật Lao động 2019, Điều 169 (Tuổi nghỉ hưu), khoản 2; Nghị định 135/2020/NĐ-CP, Điều 4 (Tuổi nghỉ hưu trong điều kiện lao động bình thường); Nghị định 135/2020/NĐ-CP, Điều 7 (Quy định chuyển tiếp), khoản 3, điểm a
  - Graph top 3: Bộ luật Lao động 2019, Điều 169 (Tuổi nghỉ hưu), khoản 2; Bộ luật Lao động 2019, Điều 169 (Tuổi nghỉ hưu), khoản 1; Nghị định 135/2020/NĐ-CP, Điều 4 (Tuổi nghỉ hưu trong điều kiện lao động bình thường)
- `strict_labor_contract_content` (direct_qa): +1.750
  - Query: Hợp đồng lao động cần có những nội dung gì?
  - Vector top 3: Bộ luật Lao động 2019, Điều 14 (Hình thức hợp đồng lao động), khoản 1; Bộ luật Lao động 2019, Điều 22 (Phụ lục hợp đồng lao động), khoản 1; Bộ luật Lao động 2019, Điều 178 (Quyền và nghĩa vụ của tổ chức đại diện người lao động tại cơ sở trong quan hệ lao động), khoản 7
  - Graph top 3: Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 1, các điểm a, b, c, d, đ, e, g, h, i, k; Thông tư 10/2020/TT-BLĐTBXH, Điều 3 (Nội dung chủ yếu của hợp đồng lao động); Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 2
- `strict_dismissal_dispute_mediation_before_lawsuit` (procedure_qa): +1.333
  - Query: Tranh chấp sa thải có cần hòa giải trước khi kiện không?
  - Vector top 3: Bộ luật Lao động 2019, Điều 190 (Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân), khoản 3; Nghị định 145/2020/NĐ-CP, Điều 98 (Tiêu chuẩn, điều kiện trọng tài viên lao động), khoản 3; Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 352 (Căn cứ để kháng nghị theo thủ tục tái thẩm)
  - Graph top 3: Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 32 (Những tranh chấp về lao động thuộc thẩm quyền giải quyết của); Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 1, các điểm a, b, c, d, đ, e; Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 2
- `strict_employee_unlawful_unilateral_termination` (scenario_based_qa): +0.833
  - Query: Người lao động đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?
  - Vector top 3: Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 2; Bộ luật Lao động 2019, Điều 41 (Nghĩa vụ của người sử dụng lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 3; Bộ luật Lao động 2019, Điều 41 (Nghĩa vụ của người sử dụng lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 1
  - Graph top 3: Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 2; Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 1; Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 3
- `strict_structural_change_job_loss_allowance` (scenario_based_qa): +0.667
  - Query: Công ty thay đổi cơ cấu thì phải trả trợ cấp gì?
  - Vector top 3: Bộ luật Lao động 2019, Điều 42 (Nghĩa vụ của người sử dụng lao động trong trường hợp thay đổi cơ cấu, công nghệ hoặc vì lý do kinh tế), khoản 1, các điểm a, b, c; Nghị định 145/2020/NĐ-CP, Điều 8 (Trợ cầp thôi việc, trợ cấp mất việc làm), khoản 4, điểm b; Nghị định 145/2020/NĐ-CP, Điều 8 (Trợ cầp thôi việc, trợ cấp mất việc làm), khoản 6
  - Graph top 3: Bộ luật Lao động 2019, Điều 47 (Trợ cấp mất việc làm), khoản 1; Bộ luật Lao động 2019, Điều 47 (Trợ cấp mất việc làm), khoản 3; Bộ luật Lao động 2019, Điều 47 (Trợ cấp mất việc làm), khoản 2
- `strict_no_notice_resignation` (exception_based_qa): +1.500
  - Query: Khi nào người lao động được nghỉ việc không cần báo trước?
  - Vector top 3: Bộ luật Lao động 2019, Điều 35 (Quyền đơn phương chấm dứt hợp đồng lao động của người lao động), khoản 1, các điểm a, b, c, d; Bộ luật Lao động 2019, Điều 46 (Trợ cấp thôi việc), khoản 2; Bộ luật Lao động 2019, Điều 46 (Trợ cấp thôi việc), khoản 1
  - Graph top 3: Bộ luật Lao động 2019, Điều 35 (Quyền đơn phương chấm dứt hợp đồng lao động của người lao động), khoản 2, các điểm a, b, c, d, đ, e, g; Bộ luật Lao động 2019, Điều 35 (Quyền đơn phương chấm dứt hợp đồng lao động của người lao động), khoản 1, các điểm a, b, c, d; Nghị định 145/2020/NĐ-CP, Điều 7 (Thời hạn báo trước khi đơn phương chấm dứt hợp đồng lao động đối với một số ngành, nghề, công việc đặc thù)

## Remaining Failures Or Over-Expansion

- No query scored worse than vector-only under the current scoring formula.

## Conclusion

Graph-augmented retrieval improves complex legal retrieval when the answer requires connecting a primary Labor Code provision to implementing decrees, circular guidance, exceptions, or labor litigation jurisdiction. Vector-only search remains useful for direct definition lookups, while hybrid dense+sparse retrieval improves lexical legal-reference matching. The graph layer is most valuable for multi-hop, procedure, exception, and scenario questions because it can promote legally connected provisions even when the query wording does not exactly match every required citation.
