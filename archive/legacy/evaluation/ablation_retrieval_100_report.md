# Retrieval Ablation on Frozen 100-Query Benchmark

- Generated at: 2026-06-01T11:17:07.502621+00:00
- Benchmark: `artifacts\evaluation\golden_benchmark_100_extended.jsonl`
- In-corpus queries used for retrieval metrics: 94
- Top-k window: 10

| Mode | Queries | Recall@10 | Required Citation Coverage | Forbidden Citation Violation Rate | Retrieval Pass Rate |
| --- | --- | --- | --- | --- | --- |
| hybrid_only | 94 | 0.789 | 0.789 | 0.032 | 0.713 |
| graph_augmented | 94 | 0.840 | 0.840 | 0.043 | 0.777 |

## hybrid_only

- Missing required citation cases: 26
- `structural_change_job_loss` (multi_hop_qa): ND145 Dieu 8
- `discipline_principles` (procedure_qa): BLLD Dieu 122
- `retirement_age_general` (document_guidance_qa): BLLD Dieu 169, ND135 Dieu 4
- `labor_dispute_limitation` (direct_qa): BLLD Dieu 190
- `guidance_structural_allowance` (document_guidance_qa): ND145 Dieu 8
- `guidance_dismissal_dispute` (document_guidance_qa): BLLD Dieu 190
- `termination_notice_special_work` (document_guidance_qa): ND145 Dieu 7
- `overtime_300_hours` (exception_based_qa): BLLD Dieu 107
- `robust_user_temp_transfer_other_job` (paraphrased_real_user_qa): BLLD Dieu 29 khoan 1, BLLD Dieu 29 khoan 2, BLLD Dieu 29 khoan 3
- `robust_user_salary_deduction_damage` (paraphrased_real_user_qa): BLLD Dieu 102 khoan 1, BLLD Dieu 102 khoan 3, BLLD Dieu 129 khoan 1
- `robust_user_late_monthly_salary` (paraphrased_real_user_qa): BLLD Dieu 94 khoan 1, BLLD Dieu 97 khoan 4
- `robust_multihop_partial_year_annual_leave` (multi_hop_qa): BLLD Dieu 113 khoan 2, ND145 Dieu 66 khoan 1, ND145 Dieu 65 khoan 2
- `robust_multihop_sexual_harassment_rules` (multi_hop_qa): ND145 Dieu 69, ND145 Dieu 85
- `robust_multihop_discipline_procedure_nd145` (multi_hop_qa): BLLD Dieu 122 khoan 1, ND145 Dieu 70
- `robust_multihop_domestic_worker_contract_notice` (multi_hop_qa): BLLD Dieu 162 khoan 1, ND145 Dieu 89 khoan 1 diem a, ND145 Dieu 90 khoan 2
- `robust_multihop_reproductive_bad_work_list` (multi_hop_qa): BLLD Dieu 142, TT10 phu luc danh muc nghe cong viec anh huong sinh san nuoi con
- `robust_appendix_retirement_male_1964_10` (calculation_or_table_lookup): ND135 Dieu 4
- `robust_appendix_early_retirement_female_1974_05` (calculation_or_table_lookup): ND135 Phu luc II bang nu nhom 02
- `robust_appendix_minor_light_work_bamboo_craft` (calculation_or_table_lookup): TT09 Dieu 8, TT09 Phu luc II danh muc cong viec nhe
- `robust_appendix_minor_night_work_packaging` (calculation_or_table_lookup): BLLD Dieu 146, TT09 Dieu 10, TT09 Phu luc V danh muc lam them lam dem
- `robust_appendix_domestic_worker_contract_template` (calculation_or_table_lookup): BLLD Dieu 162, ND145 Dieu 89 khoan 1 diem c, ND145 Mau so 01/PLV hop dong lao dong giup viec gia dinh
- `robust_hn_night_hours_not_night_pay` (hard_negative_citation_qa): BLLD Dieu 106
- `robust_hn_overtime_holiday_pay_not_limits` (hard_negative_citation_qa): BLLD Dieu 98 khoan 1 diem c, ND145 Dieu 55
- `robust_hn_severance_not_job_loss` (hard_negative_citation_qa): BLLD Dieu 46, ND145 Dieu 8
- `robust_hn_business_secret_agreement_not_contract_content` (hard_negative_citation_qa): TT10 Dieu 4
- `robust_dispute_unlawful_termination_court` (labor_dispute_procedure_qa): BLTTDS Dieu 40 khoan 1 diem d

- Forbidden citation violations: 3
- `robust_hn_severance_not_job_loss` (hard_negative_citation_qa): BLLD Dieu 47 tro cap mat viec lam
- `robust_hn_no_notice_resignation_not_employer` (hard_negative_citation_qa): BLLD Dieu 36 quyen don phuong cua nguoi su dung lao dong, BLLD Dieu 37 truong hop nguoi su dung lao dong khong duoc don phuong
- `robust_hn_business_secret_agreement_not_contract_content` (hard_negative_citation_qa): BLLD Dieu 21 noi dung hop dong lao dong chung, TT10 Dieu 3 noi dung chu yeu hop dong lao dong

## graph_augmented

- Missing required citation cases: 19
- `retirement_age_general` (document_guidance_qa): BLLD Dieu 169, ND135 Dieu 4
- `labor_dispute_limitation` (direct_qa): BLLD Dieu 190
- `guidance_dismissal_dispute` (document_guidance_qa): BLLD Dieu 190
- `robust_user_temp_transfer_other_job` (paraphrased_real_user_qa): BLLD Dieu 29 khoan 1, BLLD Dieu 29 khoan 2, BLLD Dieu 29 khoan 3
- `robust_user_salary_deduction_damage` (paraphrased_real_user_qa): BLLD Dieu 102 khoan 1, BLLD Dieu 102 khoan 3, BLLD Dieu 129 khoan 1
- `robust_user_late_monthly_salary` (paraphrased_real_user_qa): BLLD Dieu 94 khoan 1, BLLD Dieu 97 khoan 4
- `robust_user_wedding_paid_leave` (paraphrased_real_user_qa): BLLD Dieu 115 khoan 1 diem a
- `robust_multihop_partial_year_annual_leave` (multi_hop_qa): BLLD Dieu 113 khoan 2, ND145 Dieu 66 khoan 1, ND145 Dieu 65 khoan 2
- `robust_multihop_sexual_harassment_rules` (multi_hop_qa): BLLD Dieu 118 khoan 2 diem d, ND145 Dieu 85
- `robust_multihop_discipline_procedure_nd145` (multi_hop_qa): ND145 Dieu 70
- `robust_multihop_domestic_worker_contract_notice` (multi_hop_qa): BLLD Dieu 162 khoan 1, ND145 Dieu 89 khoan 1 diem a, ND145 Dieu 90 khoan 2
- `robust_multihop_reproductive_bad_work_list` (multi_hop_qa): BLLD Dieu 142, TT10 phu luc danh muc nghe cong viec anh huong sinh san nuoi con
- `robust_appendix_early_retirement_female_1974_05` (calculation_or_table_lookup): ND135 Phu luc II bang nu nhom 02
- `robust_appendix_minor_light_work_bamboo_craft` (calculation_or_table_lookup): TT09 Phu luc II danh muc cong viec nhe
- `robust_appendix_minor_night_work_packaging` (calculation_or_table_lookup): BLLD Dieu 146, TT09 Dieu 10, TT09 Phu luc V danh muc lam them lam dem
- `robust_appendix_domestic_worker_contract_template` (calculation_or_table_lookup): BLLD Dieu 162, ND145 Dieu 89 khoan 1 diem c, ND145 Mau so 01/PLV hop dong lao dong giup viec gia dinh
- `robust_hn_overtime_holiday_pay_not_limits` (hard_negative_citation_qa): BLLD Dieu 98 khoan 1 diem c, ND145 Dieu 55
- `robust_hn_business_secret_agreement_not_contract_content` (hard_negative_citation_qa): TT10 Dieu 4
- `robust_dispute_unlawful_termination_court` (labor_dispute_procedure_qa): BLTTDS Dieu 40 khoan 1 diem d

- Forbidden citation violations: 4
- `robust_hn_overtime_holiday_pay_not_limits` (hard_negative_citation_qa): BLLD Dieu 107 lam them gio
- `robust_hn_severance_not_job_loss` (hard_negative_citation_qa): BLLD Dieu 47 tro cap mat viec lam
- `robust_hn_no_notice_resignation_not_employer` (hard_negative_citation_qa): BLLD Dieu 36 quyen don phuong cua nguoi su dung lao dong, BLLD Dieu 37 truong hop nguoi su dung lao dong khong duoc don phuong
- `robust_hn_business_secret_agreement_not_contract_content` (hard_negative_citation_qa): BLLD Dieu 21 noi dung hop dong lao dong chung, TT10 Dieu 3 noi dung chu yeu hop dong lao dong
