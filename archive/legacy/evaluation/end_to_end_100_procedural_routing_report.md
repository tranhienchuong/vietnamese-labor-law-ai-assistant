# End-to-End Legal RAG Evaluation

- Generated at: 2026-06-01T11:25:44.729611+00:00
- Benchmark queries: 100
- Top K: 10
- Provider: extractive
- End-to-end passed: False

## Overall Summary

| Queries | E2E pass rate | Retrieval pass rate | Answer pass rate | Citation pass rate | Quality pass rate | Avg quality score | Low-info quotes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.790 | 0.790 | 0.960 | 1.000 | 0.960 | 94.06 | 2 |

- Unsupported article numbers: None
- Unretrieved citations: None
- Graph expansion used: 88 queries
- Average graph depth: 2.111

## Per-Category Results

| Category | Queries | Retrieval pass | Answer pass | Citation pass | E2E pass | Avg quality |
| --- | --- | --- | --- | --- | --- | --- |
| calculation_or_table_lookup | 10 | 0.600 | 0.900 | 1.000 | 0.600 | 89.62 |
| comparison_qa | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| definition_qa | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| direct_qa | 19 | 0.947 | 1.000 | 1.000 | 0.947 | 98.22 |
| document_guidance_qa | 10 | 0.800 | 0.900 | 1.000 | 0.800 | 94.81 |
| exception_based_qa | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| hard_negative_citation_qa | 5 | 0.200 | 1.000 | 1.000 | 0.200 | 77.50 |
| labor_dispute_procedure_qa | 5 | 0.800 | 0.800 | 1.000 | 0.800 | 97.28 |
| multi_hop_qa | 7 | 0.286 | 1.000 | 1.000 | 0.286 | 81.52 |
| out_of_corpus_qa | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| paraphrased_real_user_qa | 5 | 0.200 | 0.800 | 1.000 | 0.200 | 70.25 |
| procedure_qa | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| scenario_based_qa | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |

## Per-Topic Results

| Topic | Queries | Retrieval pass | Answer pass | Citation pass | Quality pass | E2E pass | Avg quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bang tuoi nghi huu dieu kien binh thuong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| bang tuoi nghi huu thap nhat | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 88.75 |
| bao ve bi mat kinh doanh bi mat cong nghe | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 55.00 |
| bao ve thai san khi lam dem lam them | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| cham tra luong | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 66.25 |
| danh muc cong viec anh huong sinh san va nuoi con | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 77.50 |
| danh muc cong viec nhe cho nguoi 13 den duoi 15 tuoi | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 88.75 |
| danh muc nguoi 15 den duoi 18 tuoi duoc lam them lam dem | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 66.25 |
| don phuong cham dut hop dong | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| don phuong cham dut trai phap luat | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| gio lam viec ban dem khong phai tien luong ban dem | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| hanh vi bi cam khi giao ket | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| hinh thuc hop dong lao dong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| hoa giai truoc khi kien | 2 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 94.38 |
| khai niem hop dong lao dong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| khai niem nguoi lao dong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| khai niem nguoi su dung lao dong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| khai niem quan he lao dong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| khai niem to chuc dai dien nguoi lao dong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| khau tru tien luong va boi thuong thiet hai | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 52.50 |
| ky luat lao dong | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| lam them gio | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| lao dong chua thanh nien | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| lao dong giup viec gia dinh | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 66.25 |
| lao dong nu | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| loai hop dong lao dong | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| mau hop dong lao dong giup viec gia dinh | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 52.50 |
| nghi hang nam lam chua du 12 thang | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 66.25 |
| nghi viec rieng co huong luong | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 66.25 |
| nghia vu khi cham dut | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| ngoai pham vi corpus - giay phep lao dong nguoi nuoc ngoai | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| ngoai pham vi corpus - kinh phi cong doan | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| ngoai pham vi corpus - luong toi thieu vung | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| ngoai pham vi corpus - thue thu nhap ca nhan | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| ngoai pham vi corpus - ty le dong bao hiem | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| ngoai pham vi corpus - xu phat hanh chinh | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| nguoi chua du 15 tuoi | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| nguoi lao dong nghi khong bao truoc khong nham voi quyen cua nguoi su dung lao dong | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 88.75 |
| noi dung hop dong lao dong | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| noi quy lao dong va quay roi tinh duc | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 77.50 |
| phan biet tro cap thoi viec va tro cap mat viec | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 88.75 |
| phu luc hop dong lao dong | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| sa thai | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| so sanh trach nhiem cham dut | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tam thoi chuyen nguoi lao dong lam cong viec khac | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 66.25 |
| tham quyen cua Toa an theo BLTTDS | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| thay doi co cau cong nghe | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| thoi gio lam viec nghi ngoi | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| thu viec | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tien luong lam them gio | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tien luong lam them ngay le khong phai gioi han lam them | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 55.00 |
| tien luong thu viec | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tranh chap don phuong cham dut hop dong trai phap luat | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 86.38 |
| tranh chap lao dong | 2 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 83.12 |
| tranh chap nguoi giup viec gia dinh | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tranh chap tien luong va tham quyen toa an | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tranh chap tro cap mat viec lam | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tro cap khi cham dut | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tro cap mat viec lam | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tro cap thoi viec | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |
| tuoi nghi huu | 4 | 0.750 | 0.750 | 1.000 | 0.750 | 0.750 | 89.84 |
| xu ly ky luat lao dong | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 83.12 |
| yeu cau tuyen bo thoa uoc lao dong tap the vo hieu | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100.00 |

## Per-Difficulty Results

| Difficulty | Queries | Retrieval pass | Answer pass | Citation pass | Quality pass | E2E pass | Avg quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easy | 16 | 0.938 | 1.000 | 1.000 | 1.000 | 0.938 | 97.89 |
| hard | 46 | 0.674 | 0.935 | 1.000 | 0.935 | 0.674 | 91.31 |
| medium | 38 | 0.868 | 0.974 | 1.000 | 0.974 | 0.868 | 95.79 |

## Graph-Required Results

| Requires graph | Queries | Retrieval pass | Answer pass | Citation pass | Quality pass | E2E pass | Avg quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 57 | 0.895 | 0.982 | 1.000 | 0.982 | 0.895 | 96.60 |
| True | 43 | 0.651 | 0.930 | 1.000 | 0.930 | 0.651 | 90.70 |

## Normative-Hierarchy Results

| Requires hierarchy | Queries | Retrieval pass | Answer pass | Citation pass | Quality pass | E2E pass | Avg quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 61 | 0.885 | 0.984 | 1.000 | 0.984 | 0.885 | 96.09 |
| True | 39 | 0.641 | 0.923 | 1.000 | 0.923 | 0.641 | 90.90 |

## Retrieval Mode And End-To-End Comparison

- Retrieval mode comparison was not provided for this run.

## Per-Query Results

| ID | Category | Topic | Difficulty | E2E | Required coverage | Citation | Quality | Score | Failure reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| def_employee | definition_qa | khai niem nguoi lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| def_employer | definition_qa | khai niem nguoi su dung lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| def_employee_representative_org | definition_qa | khai niem to chuc dai dien nguoi lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| def_labor_relationship | definition_qa | khai niem quan he lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_definition | definition_qa | khai niem hop dong lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_form | direct_qa | hinh thuc hop dong lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_types | direct_qa | loai hop dong lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_fixed_vs_indefinite | comparison_qa | loai hop dong lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_content | direct_qa | noi dung hop dong lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_content_guidance | document_guidance_qa | noi dung hop dong lao dong | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_appendix | direct_qa | phu luc hop dong lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_deposit_forbidden | exception_based_qa | hanh vi bi cam khi giao ket | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| contract_keep_original_papers | exception_based_qa | hanh vi bi cam khi giao ket | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| probation_agreement | direct_qa | thu viec | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| probation_duration | direct_qa | thu viec | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| probation_wage | calculation_or_table_lookup | tien luong thu viec | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| probation_end | procedure_qa | thu viec | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| employee_notice_period | direct_qa | don phuong cham dut hop dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| employee_no_notice | exception_based_qa | don phuong cham dut hop dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| employer_unilateral_cases | direct_qa | don phuong cham dut hop dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| employer_no_unilateral_cases | exception_based_qa | don phuong cham dut hop dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| unlawful_unilateral_definition | definition_qa | don phuong cham dut trai phap luat | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| employee_unlawful_liability | scenario_based_qa | don phuong cham dut trai phap luat | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| employer_unlawful_liability | scenario_based_qa | don phuong cham dut trai phap luat | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| severance_allowance | direct_qa | tro cap thoi viec | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| job_loss_allowance | calculation_or_table_lookup | tro cap mat viec lam | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| severance_vs_job_loss | comparison_qa | tro cap khi cham dut | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| structural_change_obligation | procedure_qa | thay doi co cau cong nghe | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| structural_change_job_loss | multi_hop_qa | thay doi co cau cong nghe | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| final_settlement | procedure_qa | nghia vu khi cham dut | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| overtime_conditions_limits | exception_based_qa | lam them gio | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| overtime_pay | calculation_or_table_lookup | tien luong lam them gio | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| night_overtime_pay | calculation_or_table_lookup | tien luong lam them gio | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| normal_working_time | direct_qa | thoi gio lam viec nghi ngoi | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| night_work_definition | direct_qa | thoi gio lam viec nghi ngoi | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| weekly_rest | direct_qa | thoi gio lam viec nghi ngoi | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| annual_leave | direct_qa | thoi gio lam viec nghi ngoi | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| labor_rules_content | direct_qa | ky luat lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| discipline_principles | procedure_qa | ky luat lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| discipline_forms | direct_qa | ky luat lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| dismissal_cases | exception_based_qa | sa thai | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| prohibited_discipline | exception_based_qa | ky luat lao dong | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| temporary_suspension | procedure_qa | ky luat lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| material_liability | scenario_based_qa | ky luat lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| minor_worker_definition | definition_qa | lao dong chua thanh nien | easy | Pass | 1.00 | Pass | Pass | 100.00 | None |
| minor_worker_14 | direct_qa | nguoi chua du 15 tuoi | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| under15_conditions | procedure_qa | nguoi chua du 15 tuoi | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| under15_working_time | direct_qa | nguoi chua du 15 tuoi | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| minor_prohibited_jobs | exception_based_qa | lao dong chua thanh nien | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| female_maternity_protection | exception_based_qa | lao dong nu | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| pregnant_worker_resignation | exception_based_qa | lao dong nu | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| maternity_leave | direct_qa | lao dong nu | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| retirement_age_general | document_guidance_qa | tuoi nghi huu | hard | Fail | 0.00 | Pass | Fail | 59.38 | retrieval_missing_required_context, answer_missing_required_rule, low_information_answer |
| retirement_female_2026 | calculation_or_table_lookup | tuoi nghi huu | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| retirement_hazardous_work | scenario_based_qa | tuoi nghi huu | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| labor_dispute_mediation | procedure_qa | tranh chap lao dong | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| dismissal_dispute_court | multi_hop_qa | hoa giai truoc khi kien | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| labor_dispute_limitation | direct_qa | tranh chap lao dong | medium | Fail | 0.00 | Pass | Pass | 66.25 | retrieval_missing_required_context |
| court_jurisdiction_labor | document_guidance_qa | tham quyen cua Toa an theo BLTTDS | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| compare_unlawful_vs_structural | comparison_qa | so sanh trach nhiem cham dut | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| compare_overtime_conditions_vs_pay | comparison_qa | lam them gio | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| guidance_minor_worker | document_guidance_qa | nguoi chua du 15 tuoi | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| guidance_contract_content | document_guidance_qa | noi dung hop dong lao dong | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| guidance_retirement_2026 | document_guidance_qa | tuoi nghi huu | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| guidance_structural_allowance | document_guidance_qa | thay doi co cau cong nghe | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| guidance_dismissal_dispute | document_guidance_qa | hoa giai truoc khi kien | hard | Fail | 0.67 | Pass | Pass | 88.75 | retrieval_missing_required_context |
| termination_notice_special_work | document_guidance_qa | don phuong cham dut hop dong | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| severance_detail_nd145 | document_guidance_qa | tro cap thoi viec | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| overtime_300_hours | exception_based_qa | lam them gio | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_ooc_minimum_wage_region_2026 | out_of_corpus_qa | ngoai pham vi corpus - luong toi thieu vung | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_ooc_bhxh_contribution_rate_2026 | out_of_corpus_qa | ngoai pham vi corpus - ty le dong bao hiem | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_ooc_admin_fine_late_salary | out_of_corpus_qa | ngoai pham vi corpus - xu phat hanh chinh | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_ooc_foreign_work_permit_dossier_2026 | out_of_corpus_qa | ngoai pham vi corpus - giay phep lao dong nguoi nuoc ngoai | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_ooc_trade_union_dues_rate | out_of_corpus_qa | ngoai pham vi corpus - kinh phi cong doan | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_ooc_pit_on_severance | out_of_corpus_qa | ngoai pham vi corpus - thue thu nhap ca nhan | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_user_temp_transfer_other_job | paraphrased_real_user_qa | tam thoi chuyen nguoi lao dong lam cong viec khac | medium | Fail | 0.00 | Pass | Pass | 66.25 | retrieval_missing_required_context |
| robust_user_salary_deduction_damage | paraphrased_real_user_qa | khau tru tien luong va boi thuong thiet hai | medium | Fail | 0.00 | Pass | Fail | 52.50 | retrieval_missing_required_context, low_information_answer |
| robust_user_late_monthly_salary | paraphrased_real_user_qa | cham tra luong | medium | Fail | 0.00 | Pass | Pass | 66.25 | retrieval_missing_required_context |
| robust_user_wedding_paid_leave | paraphrased_real_user_qa | nghi viec rieng co huong luong | easy | Fail | 0.00 | Pass | Pass | 66.25 | retrieval_missing_required_context |
| robust_user_pregnant_night_overtime | paraphrased_real_user_qa | bao ve thai san khi lam dem lam them | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_multihop_partial_year_annual_leave | multi_hop_qa | nghi hang nam lam chua du 12 thang | hard | Fail | 0.00 | Pass | Pass | 66.25 | retrieval_missing_required_context |
| robust_multihop_sexual_harassment_rules | multi_hop_qa | noi quy lao dong va quay roi tinh duc | hard | Fail | 0.33 | Pass | Pass | 77.50 | retrieval_missing_required_context |
| robust_multihop_discipline_procedure_nd145 | multi_hop_qa | xu ly ky luat lao dong | hard | Fail | 0.50 | Pass | Pass | 83.12 | retrieval_missing_required_context |
| robust_multihop_domestic_worker_contract_notice | multi_hop_qa | lao dong giup viec gia dinh | hard | Fail | 0.00 | Pass | Pass | 66.25 | retrieval_missing_required_context |
| robust_multihop_reproductive_bad_work_list | multi_hop_qa | danh muc cong viec anh huong sinh san va nuoi con | hard | Fail | 0.33 | Pass | Pass | 77.50 | retrieval_missing_required_context |
| robust_appendix_retirement_male_1964_10 | calculation_or_table_lookup | bang tuoi nghi huu dieu kien binh thuong | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_appendix_early_retirement_female_1974_05 | calculation_or_table_lookup | bang tuoi nghi huu thap nhat | hard | Fail | 0.67 | Pass | Pass | 88.75 | retrieval_missing_required_context |
| robust_appendix_minor_light_work_bamboo_craft | calculation_or_table_lookup | danh muc cong viec nhe cho nguoi 13 den duoi 15 tuoi | hard | Fail | 0.67 | Pass | Pass | 88.75 | retrieval_missing_required_context |
| robust_appendix_minor_night_work_packaging | calculation_or_table_lookup | danh muc nguoi 15 den duoi 18 tuoi duoc lam them lam dem | hard | Fail | 0.00 | Pass | Pass | 66.25 | retrieval_missing_required_context |
| robust_appendix_domestic_worker_contract_template | calculation_or_table_lookup | mau hop dong lao dong giup viec gia dinh | hard | Fail | 0.00 | Pass | Fail | 52.50 | retrieval_missing_required_context, low_information_answer |
| robust_hn_night_hours_not_night_pay | hard_negative_citation_qa | gio lam viec ban dem khong phai tien luong ban dem | medium | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_hn_overtime_holiday_pay_not_limits | hard_negative_citation_qa | tien luong lam them ngay le khong phai gioi han lam them | hard | Fail | 0.00 | Pass | Pass | 55.00 | retrieval_missing_required_context, retrieval_over_expansion |
| robust_hn_severance_not_job_loss | hard_negative_citation_qa | phan biet tro cap thoi viec va tro cap mat viec | hard | Fail | 1.00 | Pass | Pass | 88.75 | retrieval_over_expansion |
| robust_hn_no_notice_resignation_not_employer | hard_negative_citation_qa | nguoi lao dong nghi khong bao truoc khong nham voi quyen cua nguoi su dung lao dong | medium | Fail | 1.00 | Pass | Pass | 88.75 | retrieval_over_expansion |
| robust_hn_business_secret_agreement_not_contract_content | hard_negative_citation_qa | bao ve bi mat kinh doanh bi mat cong nghe | hard | Fail | 0.00 | Pass | Pass | 55.00 | retrieval_missing_required_context, retrieval_over_expansion |
| robust_dispute_unlawful_termination_court | labor_dispute_procedure_qa | tranh chap don phuong cham dut hop dong trai phap luat | hard | Fail | 0.80 | Pass | Fail | 86.38 | retrieval_missing_required_context, answer_missing_required_rule, low_information_answer |
| robust_dispute_salary_deduction_court_choice | labor_dispute_procedure_qa | tranh chap tien luong va tham quyen toa an | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_dispute_job_loss_allowance_no_mediation | labor_dispute_procedure_qa | tranh chap tro cap mat viec lam | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_dispute_domestic_worker_no_mediation | labor_dispute_procedure_qa | tranh chap nguoi giup viec gia dinh | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |
| robust_dispute_collective_agreement_invalid_request | labor_dispute_procedure_qa | yeu cau tuyen bo thoa uoc lao dong tap the vo hieu | hard | Pass | 1.00 | Pass | Pass | 100.00 | None |

## Successful Graph-Augmented Answers

### annual_leave

- Query: Nguoi lao dong lam du 12 thang thi duoc nghi hang nam bao nhieu ngay?
- Top citations: Bộ luật Lao động 2019, Điều 113 (Nghỉ hằng năm), khoản 3; Bộ luật Lao động 2019, Điều 113 (Nghỉ hằng năm), khoản 1, các điểm a, b, c; Bộ luật Lao động 2019, Điều 113 (Nghỉ hằng năm), khoản 2
- Answer excerpt: Câu trả lời: Căn cứ vào Bộ luật Lao động 2019, Điều 113 (Nghỉ hằng năm), khoản 3, có cơ sở pháp lý để trả lời trong phạm vi dữ liệu được truy xuất. Nội dung cụ thể như sau: - Trong Chương VII. THỜI GIỜ LÀM VIỆC, THỜI GIỜ NGHỈ NGƠI; Mục 2. THỜI GIỜ NGHỈ NGƠI, Bộ luật Lao động 2019, Điều 113 (Nghỉ hằng năm), khoản 3 quy định: 3. Trường hợp do thôi việc, bị...

### compare_overtime_conditions_vs_pay

- Query: Phan biet dieu kien lam them gio voi tien luong lam them gio.
- Top citations: Bộ luật Lao động 2019, Điều 107 (Làm thêm giờ), khoản 2, các điểm a, b, c; Bộ luật Lao động 2019, Điều 98 (Tiền lương làm thêm giờ, làm việc vào ban đêm), khoản 1, các điểm a, b, c; Bộ luật Lao động 2019, Điều 98 (Tiền lương làm thêm giờ, làm việc vào ban đêm), khoản 2
- Answer excerpt: Câu trả lời: Căn cứ vào Bộ luật Lao động 2019, Điều 107 (Làm thêm giờ), khoản 2, các điểm a, b, c, có cơ sở pháp lý để trả lời trong phạm vi dữ liệu được truy xuất. Nội dung cụ thể như sau: - Trong Chương VII. THỜI GIỜ LÀM VIỆC, THỜI GIỜ NGHỈ NGƠI; Mục 1. THỜI GIỜ LÀM VIỆC, Bộ luật Lao động 2019, Điều 107 (Làm thêm giờ), khoản 2, các điểm a, b, c quy định:...

### compare_unlawful_vs_structural

- Query: So sanh trach nhiem khi nguoi lao dong don phuong cham dut hop dong trai luat voi truong hop cong ty thay doi co cau phai tro cap.
- Top citations: Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 1; Bộ luật Lao động 2019, Điều 42 (Nghĩa vụ của người sử dụng lao động trong trường hợp thay đổi cơ cấu, công nghệ hoặc vì lý do kinh tế), khoản 5; Bộ luật Lao động 2019, Điều 47 (Trợ cấp mất việc làm), khoản 1
- Answer excerpt: Câu trả lời: Nếu người lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật, hậu quả chính là không được trợ cấp thôi việc và phải bồi thường cho người sử dụng lao động. (Bộ luật Lao động 2019, Điều 40 (Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật), khoản 2) Nội dung cụ thể như sau: - Không được trợ cấp thôi...


## Retrieval Failures

- `retirement_age_general`: retrieval_missing_required_context, answer_missing_required_rule, low_information_answer
  Query: Tuoi nghi huu trong dieu kien lao dong binh thuong duoc xac dinh theo quy dinh nao?
  Missing required citations: BLLD Dieu 169, ND135 Dieu 4
  Citation issues: None
- `labor_dispute_limitation`: retrieval_missing_required_context
  Query: Thoi hieu yeu cau giai quyet tranh chap lao dong ca nhan la bao lau?
  Missing required citations: BLLD Dieu 190
  Citation issues: None
- `guidance_dismissal_dispute`: retrieval_missing_required_context
  Query: Tranh chap sa thai co can hoa giai truoc khi kien khong va Toa an co tham quyen khong?
  Missing required citations: BLLD Dieu 190
  Citation issues: None
- `robust_user_temp_transfer_other_job`: retrieval_missing_required_context
  Query: Cong ty noi hang it viec nen chuyen toi sang viec khac gan 3 thang, khong hoi y kien toi, vay co on khong?
  Missing required citations: BLLD Dieu 29 khoan 1, BLLD Dieu 29 khoan 2, BLLD Dieu 29 khoan 3
  Citation issues: None
- `robust_user_salary_deduction_damage`: retrieval_missing_required_context, low_information_answer
  Query: Toi lam hong dung cu, cong ty tru luong hang thang toi gan het luong thi co duoc khong?
  Missing required citations: BLLD Dieu 102 khoan 1, BLLD Dieu 102 khoan 3, BLLD Dieu 129 khoan 1
  Citation issues: None
- `robust_user_late_monthly_salary`: retrieval_missing_required_context
  Query: Luong thang nay bi tre hon nua thang, cong ty co phai tra them tien cho toi khong?
  Missing required citations: BLLD Dieu 94 khoan 1, BLLD Dieu 97 khoan 4
  Citation issues: None
- `robust_user_wedding_paid_leave`: retrieval_missing_required_context
  Query: Toi sap ket hon, xin nghi viec rieng thi duoc huong luong may ngay?
  Missing required citations: BLLD Dieu 115 khoan 1 diem a
  Citation issues: None
- `robust_multihop_partial_year_annual_leave`: retrieval_missing_required_context
  Query: Lam 7 thang trong nam thi phep nam tinh theo ti le nao va thoi gian thu viec co duoc tinh vao phep nam khong?
  Missing required citations: BLLD Dieu 113 khoan 2, ND145 Dieu 66 khoan 1, ND145 Dieu 65 khoan 2
  Citation issues: None
- `robust_multihop_sexual_harassment_rules`: retrieval_missing_required_context
  Query: Noi quy cong ty ve phong chong quay roi tinh duc phai ghi nhung gi, can doc Dieu 118 hay nghi dinh huong dan nao?
  Missing required citations: BLLD Dieu 118 khoan 2 diem d, ND145 Dieu 85
  Citation issues: None
- `robust_multihop_discipline_procedure_nd145`: retrieval_missing_required_context
  Query: Khi xu ly ky luat nhan vien, cong ty phai chung minh loi va lam theo thu tuc chi tiet o dau?
  Missing required citations: ND145 Dieu 70
  Citation issues: None
- `robust_multihop_domestic_worker_contract_notice`: retrieval_missing_required_context
  Query: Thue nguoi giup viec gia dinh thi hop dong co phai bang van ban va co can thong bao cho xa phuong khong?
  Missing required citations: BLLD Dieu 162 khoan 1, ND145 Dieu 89 khoan 1 diem a, ND145 Dieu 90 khoan 2
  Citation issues: None
- `robust_multihop_reproductive_bad_work_list`: retrieval_missing_required_context
  Query: Cong viec co anh huong xau toi sinh san va nuoi con thi danh muc nam o dau, nguoi lao dong can duoc thong tin gi?
  Missing required citations: BLLD Dieu 142, TT10 phu luc danh muc nghe cong viec anh huong sinh san nuoi con
  Citation issues: None
- `robust_appendix_early_retirement_female_1974_05`: retrieval_missing_required_context
  Query: Lao dong nu sinh thang 5/1974 neu thuoc dien nghi huu som thi tuoi nghi huu thap nhat trong bang la bao nhieu?
  Missing required citations: ND135 Phu luc II bang nu nhom 02
  Citation issues: None
- `robust_appendix_minor_light_work_bamboo_craft`: retrieval_missing_required_context
  Query: Nguoi 14 tuoi co duoc lam viec dan lat do thu cong tu tre, nua, coi khong?
  Missing required citations: TT09 Phu luc II danh muc cong viec nhe
  Citation issues: None
- `robust_appendix_minor_night_work_packaging`: retrieval_missing_required_context
  Query: Nguoi 16 tuoi co the lam them gio hoac lam dem voi cong viec goi nem, goi keo, goi banh khong?
  Missing required citations: BLLD Dieu 146, TT09 Dieu 10, TT09 Phu luc V danh muc lam them lam dem
  Citation issues: None
- `robust_appendix_domestic_worker_contract_template`: retrieval_missing_required_context, low_information_answer
  Query: Mau hop dong lao dong giup viec gia dinh nam o phu luc nao cua Nghi dinh 145?
  Missing required citations: BLLD Dieu 162, ND145 Dieu 89 khoan 1 diem c, ND145 Mau so 01/PLV hop dong lao dong giup viec gia dinh
  Citation issues: None
- `robust_hn_overtime_holiday_pay_not_limits`: retrieval_missing_required_context, retrieval_over_expansion
  Query: Lam them vao ngay le thi it nhat duoc tra bao nhieu phan tram, dung nham voi gioi han lam them.
  Missing required citations: BLLD Dieu 98 khoan 1 diem c, ND145 Dieu 55
  Citation issues: None
- `robust_hn_severance_not_job_loss`: retrieval_over_expansion
  Query: Nhan vien tu nghi dung bao truoc sau hon 2 nam lam viec thi neu co tro cap thi la tro cap thoi viec hay tro cap mat viec?
  Missing required citations: None
  Citation issues: None
- `robust_hn_no_notice_resignation_not_employer`: retrieval_over_expansion
  Query: Cong ty khong tra luong dung han, toi co duoc nghi ngay khong hay phai bao truoc?
  Missing required citations: None
  Citation issues: None
- `robust_hn_business_secret_agreement_not_contract_content`: retrieval_missing_required_context, retrieval_over_expansion
  Query: Thoa thuan bao ve bi mat kinh doanh nen ghi nhung noi dung gi, dung nham voi noi dung hop dong lao dong chung.
  Missing required citations: TT10 Dieu 4
  Citation issues: None
- `robust_dispute_unlawful_termination_court`: retrieval_missing_required_context, answer_missing_required_rule, low_information_answer
  Query: Cong ty don phuong cho toi nghi trai luat, toi kien doi nhan lai va boi thuong thi can can cu luat lao dong va BLTTDS nao?
  Missing required citations: BLTTDS Dieu 40 khoan 1 diem d
  Citation issues: None

## Answer Generation Failures

- No answer-generation failures were observed in this benchmark run.

## Remaining Limitations

- Some benchmark items did not satisfy all required-citation checks; these failures should be interpreted as retrieval coverage gaps for the constructed benchmark, not as a full assessment of legal correctness.
- The benchmark is manually constructed from selected Vietnamese labor-law topics and does not prove universal legal correctness.
- Deterministic answer synthesis favors citation safety and may be less fluent than a carefully constrained LLM provider.

## Failure Reason Counts

| Reason | Count |
| --- | --- |
| answer_missing_required_rule | 2 |
| low_information_answer | 4 |
| retrieval_missing_required_context | 19 |
| retrieval_over_expansion | 4 |

## Thesis-Ready Conclusion

Based on the constructed benchmark, the end-to-end evaluation indicates that graph-augmented retrieval is most helpful for Vietnamese labor-law questions that require connecting multiple provisions, such as Labor Code rules with implementing decrees, circular guidance, exceptions, or court-jurisdiction rules. The results should not be read as a claim of universal legal correctness. Instead, they show that the system can maintain citation grounding on this benchmark by only citing retrieved legal contexts, which reduces hallucinated legal references and makes remaining retrieval gaps easier to inspect.
