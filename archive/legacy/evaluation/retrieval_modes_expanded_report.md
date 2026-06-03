# Retrieval Modes Evaluation

- Generated at: 2026-05-27T15:46:36.669105+00:00
- Benchmark queries: 69
- Modes: vector_only, hybrid, graph_augmented

## Overall Comparison

| Mode | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR | Required coverage | Forbidden rate | Avg graph chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_only | 0.012 | 0.012 | 0.026 | 0.020 | 0.022 | 0.012 | 0.000 | 0.00 |
| hybrid | 0.860 | 0.920 | 0.571 | 0.380 | 0.833 | 0.920 | 0.000 | 0.00 |
| graph_augmented | 0.961 | 1.000 | 0.562 | 0.359 | 0.872 | 1.000 | 0.000 | 5.71 |

## Per-Category Comparison

| Category | Mode | Queries | Recall@10 | Precision@10 | MRR | Required coverage |
| --- | --- | --- | --- | --- | --- | --- |
| calculation_or_table_lookup | vector_only | 5 | 0.067 | 0.180 | 0.100 | 0.067 |
| calculation_or_table_lookup | hybrid | 5 | 1.000 | 0.480 | 1.000 | 1.000 |
| calculation_or_table_lookup | graph_augmented | 5 | 1.000 | 0.500 | 1.000 | 1.000 |
| comparison_qa | vector_only | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| comparison_qa | hybrid | 4 | 1.000 | 0.625 | 1.000 | 1.000 |
| comparison_qa | graph_augmented | 4 | 1.000 | 0.475 | 1.000 | 1.000 |
| definition_qa | vector_only | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| definition_qa | hybrid | 7 | 1.000 | 0.143 | 1.000 | 1.000 |
| definition_qa | graph_augmented | 7 | 1.000 | 0.114 | 0.786 | 1.000 |
| direct_qa | vector_only | 19 | 0.000 | 0.000 | 0.000 | 0.000 |
| direct_qa | hybrid | 19 | 1.000 | 0.342 | 0.845 | 1.000 |
| direct_qa | graph_augmented | 19 | 1.000 | 0.295 | 0.882 | 1.000 |
| document_guidance_qa | vector_only | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| document_guidance_qa | hybrid | 10 | 0.683 | 0.470 | 0.653 | 0.683 |
| document_guidance_qa | graph_augmented | 10 | 1.000 | 0.490 | 0.933 | 1.000 |
| exception_based_qa | vector_only | 11 | 0.045 | 0.045 | 0.091 | 0.045 |
| exception_based_qa | hybrid | 11 | 0.955 | 0.291 | 0.882 | 0.955 |
| exception_based_qa | graph_augmented | 11 | 1.000 | 0.327 | 0.836 | 1.000 |
| multi_hop_qa | vector_only | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| multi_hop_qa | hybrid | 2 | 0.583 | 0.850 | 0.750 | 0.583 |
| multi_hop_qa | graph_augmented | 2 | 1.000 | 0.800 | 1.000 | 1.000 |
| procedure_qa | vector_only | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| procedure_qa | hybrid | 7 | 0.857 | 0.329 | 0.528 | 0.857 |
| procedure_qa | graph_augmented | 7 | 1.000 | 0.300 | 0.702 | 1.000 |
| scenario_based_qa | vector_only | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| scenario_based_qa | hybrid | 4 | 1.000 | 0.475 | 1.000 | 1.000 |
| scenario_based_qa | graph_augmented | 4 | 1.000 | 0.450 | 0.875 | 1.000 |

## Per-Query Comparison

| ID | Category | Query | Vector R/M/C | Hybrid R/M/C | Graph R/M/C |
| --- | --- | --- | --- | --- | --- |
| def_employee | definition_qa | Nguoi lao dong duoc dinh nghia nhu the nao theo Bo luat Lao dong 2019? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| def_employer | definition_qa | Nguoi su dung lao dong la gi theo Bo luat Lao dong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| def_employee_representative_org | definition_qa | To chuc dai dien nguoi lao dong tai co so la gi? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| def_labor_relationship | definition_qa | Quan he lao dong duoc hieu nhu the nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_definition | definition_qa | Hop dong lao dong la gi? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.50/1.00 |
| contract_form | direct_qa | Hop dong lao dong co bat buoc phai lap bang van ban khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_types | direct_qa | Co may loai hop dong lao dong theo Bo luat Lao dong 2019? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_fixed_vs_indefinite | comparison_qa | So sanh hop dong xac dinh thoi han va khong xac dinh thoi han. | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_content | direct_qa | Hop dong lao dong can co nhung noi dung chu yeu nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_content_guidance | document_guidance_qa | Noi dung hop dong lao dong duoc Thong tu 10 huong dan chi tiet nhu the nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_appendix | direct_qa | Phu luc hop dong lao dong co gia tri nhu hop dong khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_deposit_forbidden | exception_based_qa | Cong ty yeu cau nguoi lao dong dat coc tien khi ky hop dong co dung luat khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| contract_keep_original_papers | exception_based_qa | Nguoi su dung lao dong co duoc giu ban chinh can cuoc hoac bang cap cua nguoi lao dong khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| probation_agreement | direct_qa | Thu viec duoc thoa thuan nhu the nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| probation_duration | direct_qa | Thoi gian thu viec toi da la bao lau? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| probation_wage | calculation_or_table_lookup | Tien luong thu viec toi thieu bang bao nhieu phan tram luong cua cong viec? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| probation_end | procedure_qa | Khi ket thuc thoi gian thu viec thi nguoi su dung lao dong phai lam gi? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| employee_notice_period | direct_qa | Nguoi lao dong don phuong cham dut hop dong phai bao truoc bao lau? | 0.00/0.00/0.00 | 1.00/0.25/1.00 | 1.00/1.00/1.00 |
| employee_no_notice | exception_based_qa | Khi nao nguoi lao dong duoc nghi viec khong can bao truoc? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| employer_unilateral_cases | direct_qa | Nguoi su dung lao dong duoc don phuong cham dut hop dong trong nhung truong hop nao? | 0.00/0.00/0.00 | 1.00/0.14/1.00 | 1.00/0.50/1.00 |
| employer_no_unilateral_cases | exception_based_qa | Truong hop nao nguoi su dung lao dong khong duoc don phuong cham dut hop dong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.50/1.00 |
| unlawful_unilateral_definition | definition_qa | Don phuong cham dut hop dong lao dong trai phap luat la gi? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.50/1.00 |
| employee_unlawful_liability | scenario_based_qa | Nguoi lao dong don phuong cham dut hop dong trai luat thi phai boi thuong gi? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| employer_unlawful_liability | scenario_based_qa | Cong ty don phuong cham dut hop dong trai phap luat thi phai nhan nguoi lao dong lai khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| severance_allowance | direct_qa | Tro cap thoi viec duoc tra trong truong hop nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| job_loss_allowance | calculation_or_table_lookup | Tro cap mat viec lam duoc tinh nhu the nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| severance_vs_job_loss | comparison_qa | So sanh tro cap thoi viec va tro cap mat viec lam. | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| structural_change_obligation | procedure_qa | Cong ty thay doi co cau cong nghe thi phai lam gi voi nguoi lao dong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| structural_change_job_loss | multi_hop_qa | Cong ty thay doi co cau thi phai tra tro cap gi va can cu cach tinh o dau? | 0.00/0.00/0.00 | 0.67/1.00/0.67 | 1.00/1.00/1.00 |
| final_settlement | procedure_qa | Sau khi cham dut hop dong, cong ty phai thanh toan va tra giay to trong thoi han nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.25/1.00 |
| overtime_conditions_limits | exception_based_qa | Truong hop nao duoc lam them gio va gioi han lam them theo thang la bao nhieu? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| overtime_pay | calculation_or_table_lookup | Luong lam them gio duoc tra nhu the nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| night_overtime_pay | calculation_or_table_lookup | Lam them gio vao ban dem thi tien luong duoc tinh theo dieu nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| normal_working_time | direct_qa | Thoi gio lam viec binh thuong toi da la bao nhieu? | 0.00/0.00/0.00 | 1.00/0.33/1.00 | 1.00/0.50/1.00 |
| night_work_definition | direct_qa | Gio lam viec ban dem duoc tinh tu may gio den may gio? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| weekly_rest | direct_qa | Nguoi lao dong duoc nghi hang tuan nhu the nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| annual_leave | direct_qa | Nguoi lao dong lam du 12 thang thi duoc nghi hang nam bao nhieu ngay? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| labor_rules_content | direct_qa | Noi quy lao dong phai co nhung noi dung chu yeu nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| discipline_principles | procedure_qa | Khi xu ly ky luat lao dong phai tuan thu nguyen tac nao? | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/0.33/1.00 |
| discipline_forms | direct_qa | Cac hinh thuc xu ly ky luat lao dong gom nhung gi? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.50/1.00 |
| dismissal_cases | exception_based_qa | Nguoi lao dong bi sa thai trong nhung truong hop nao? | 0.00/0.00/0.00 | 1.00/0.50/1.00 | 1.00/0.50/1.00 |
| prohibited_discipline | exception_based_qa | Cong ty co duoc phat tien hoac cat luong thay cho ky luat lao dong khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| temporary_suspension | procedure_qa | Tam dinh chi cong viec de xu ly ky luat duoc quy dinh the nao? | 0.00/0.00/0.00 | 1.00/0.11/1.00 | 1.00/0.33/1.00 |
| material_liability | scenario_based_qa | Nguoi lao dong lam hu hong dung cu thiet bi thi boi thuong theo dieu nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.50/1.00 |
| minor_worker_definition | definition_qa | Lao dong chua thanh nien la nguoi lao dong duoi bao nhieu tuoi? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.50/1.00 |
| minor_worker_14 | direct_qa | Nguoi 14 tuoi co duoc lam viec khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| under15_conditions | procedure_qa | Nguoi chua du 15 tuoi lam viec can dieu kien gi? | 0.00/0.00/0.00 | 1.00/0.33/1.00 | 1.00/1.00/1.00 |
| under15_working_time | direct_qa | Thoi gio lam viec cua nguoi chua du 15 tuoi bi gioi han the nao? | 0.00/0.00/0.00 | 1.00/0.33/1.00 | 1.00/0.25/1.00 |
| minor_prohibited_jobs | exception_based_qa | Nguoi chua thanh nien khong duoc lam nhung cong viec nao? | 0.50/1.00/0.50 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| female_maternity_protection | exception_based_qa | Cong ty co duoc sa thai hoac don phuong cham dut hop dong voi lao dong nu mang thai khong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| pregnant_worker_resignation | exception_based_qa | Lao dong nu mang thai co quyen don phuong cham dut hoac tam hoan hop dong khi cong viec anh huong thai nhi khong? | 0.00/0.00/0.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 |
| maternity_leave | direct_qa | Lao dong nu sinh con duoc nghi thai san bao lau theo Bo luat Lao dong? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| retirement_age_general | document_guidance_qa | Tuoi nghi huu trong dieu kien lao dong binh thuong duoc xac dinh theo quy dinh nao? | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/1.00/1.00 |
| retirement_female_2026 | calculation_or_table_lookup | Nu nghi huu nam 2026 thi bao nhieu tuoi va can cu van ban nao? | 0.33/0.50/0.33 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| retirement_hazardous_work | scenario_based_qa | Nguoi lao dong lam nghe nang nhoc doc hai co the nghi huu som theo can cu nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| labor_dispute_mediation | procedure_qa | Tranh chap lao dong ca nhan co bat buoc hoa giai truoc khi yeu cau Toa an khong? | 0.00/0.00/0.00 | 1.00/0.25/1.00 | 1.00/1.00/1.00 |
| dismissal_dispute_court | multi_hop_qa | Tranh chap sa thai co can hoa giai truoc khi kien va Toa an co tham quyen khong? | 0.00/0.00/0.00 | 0.50/0.50/0.50 | 1.00/1.00/1.00 |
| labor_dispute_limitation | direct_qa | Thoi hieu yeu cau giai quyet tranh chap lao dong ca nhan la bao lau? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| court_jurisdiction_labor | document_guidance_qa | Nhung tranh chap lao dong nao thuoc tham quyen giai quyet cua Toa an theo BLTTDS? | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 1.00/1.00/1.00 |
| compare_unlawful_vs_structural | comparison_qa | So sanh trach nhiem khi nguoi lao dong don phuong cham dut hop dong trai luat voi truong hop cong ty thay doi co cau phai tro cap. | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| compare_overtime_conditions_vs_pay | comparison_qa | Phan biet dieu kien lam them gio voi tien luong lam them gio. | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| guidance_minor_worker | document_guidance_qa | Nguoi 14 tuoi duoc lam viec trong dieu kien nao va van ban nao huong dan chi tiet? | 0.00/0.00/0.00 | 1.00/0.33/1.00 | 1.00/1.00/1.00 |
| guidance_contract_content | document_guidance_qa | Hop dong lao dong can noi dung gi va thong tu nao huong dan chi tiet? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| guidance_retirement_2026 | document_guidance_qa | Nu nghi huu nam 2026 thi can cu luat va nghi dinh nao? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/0.33/1.00 |
| guidance_structural_allowance | document_guidance_qa | Cong ty thay doi co cau phai tra tro cap gi va cach tinh theo van ban nao? | 0.00/0.00/0.00 | 0.67/1.00/0.67 | 1.00/1.00/1.00 |
| guidance_dismissal_dispute | document_guidance_qa | Tranh chap sa thai co can hoa giai truoc khi kien khong va Toa an co tham quyen khong? | 0.00/0.00/0.00 | 0.67/1.00/0.67 | 1.00/1.00/1.00 |
| termination_notice_special_work | document_guidance_qa | Nganh nghe dac thu thi thoi han bao truoc khi nghi viec duoc huong dan o van ban nao? | 0.00/0.00/0.00 | 0.50/0.20/0.50 | 1.00/1.00/1.00 |
| severance_detail_nd145 | document_guidance_qa | Thoi gian lam viec de tinh tro cap thoi viec duoc huong dan chi tiet o dau? | 0.00/0.00/0.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 |
| overtime_300_hours | exception_based_qa | Truong hop nao duoc lam them den 300 gio trong mot nam? | 0.00/0.00/0.00 | 0.50/1.00/0.50 | 1.00/1.00/1.00 |

## Graph Diagnostics

- Graph expansion used count: 67
- Average graph depth: 2.419
- Graph hit ratio: 0.631
- Direct vector hit ratio in graph mode: 0.369
- Improved over vector-only: 69
- Worse than vector-only: 0
- Top graph edge types: MENTIONS_TOPIC:1

## Vector-Only Diagnostics

- Nonempty vector-only queries: 69 / 69
- Vector-only contexts: 689
- Metadata document-id rate: 1.000
- Metadata article-or-appendix rate: 1.000
- All-zero recall/coverage: False
- All-zero interpretation: nonzero_metrics

## Graph Improvements

- `def_employee` (definition_qa): +3.000
  - Query: Nguoi lao dong duoc dinh nghia nhu the nao theo Bo luat Lao dong 2019?
  - Vector top 3: Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 1; Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 2; Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 3
  - Graph top 3: Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ), khoản 1; Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ)
- `def_employer` (definition_qa): +3.000
  - Query: Nguoi su dung lao dong la gi theo Bo luat Lao dong?
  - Vector top 3: Bộ luật Lao động 2019, Điều 152 (Điều kiện tuyển dụng, sử dụng người lao động nước ngoài làm việc tại Việt Nam), khoản 1; Bộ luật Lao động 2019, Điều 152 (Điều kiện tuyển dụng, sử dụng người lao động nước ngoài làm việc tại Việt Nam), khoản 3; Bộ luật Lao động 2019, Điều 152 (Điều kiện tuyển dụng, sử dụng người lao động nước ngoài làm việc tại Việt Nam), khoản 2
  - Graph top 3: Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ), khoản 2; Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ)
- `def_employee_representative_org` (definition_qa): +3.000
  - Query: To chuc dai dien nguoi lao dong tai co so la gi?
  - Vector top 3: Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 1; Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 2; Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 3
  - Graph top 3: Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ), khoản 3; Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ); Nghị định 135/2020/NĐ-CP, Điều 8 (Hiệu lực thi hành), khoản 2, các điểm a, b, c
- `def_labor_relationship` (definition_qa): +3.000
  - Query: Quan he lao dong duoc hieu nhu the nao?
  - Vector top 3: Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 1; Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 2; Bộ luật Lao động 2019, Điều 154 (Người lao động nước ngoài làm việc tại Việt Nam không thuộc diện cấp giấy phép lao động), khoản 3
  - Graph top 3: Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ), khoản 5; Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ); Nghị định 135/2020/NĐ-CP, Điều 8 (Hiệu lực thi hành), khoản 2, các điểm a, b, c
- `contract_definition` (definition_qa): +2.500
  - Query: Hop dong lao dong la gi?
  - Vector top 3: Nghị định 145/2020/NĐ-CP, Điều 18 (Rút tiền ký quỹ), khoản 1, điểm đ; Nghị định 145/2020/NĐ-CP, Điều 18 (Rút tiền ký quỹ), khoản 1, các điểm a, b, c, d; Nghị định 145/2020/NĐ-CP, Điều 18 (Rút tiền ký quỹ), khoản 2, các điểm a, b, c, d
  - Graph top 3: Bộ luật Lao động 2019, Điều 3 (Giải thích từ ngữ); Bộ luật Lao động 2019, Điều 13 (Hợp đồng lao động), khoản 1; Bộ luật Lao động 2019, Điều 13 (Hợp đồng lao động), khoản 2
- `contract_form` (direct_qa): +3.000
  - Query: Hop dong lao dong co bat buoc phai lap bang van ban khong?
  - Vector top 3: Bộ luật Lao động 2019, Điều 151 (Điều kiện người lao động nước ngoài làm việc tại Việt Nam), khoản 2; Bộ luật Lao động 2019, Điều 156 (Các trường hợp giấy phép lao động hết hiệu lực), khoản 2; Bộ luật Lao động 2019, Điều 150 (Người lao động Việt Nam đi làm việc ở nước ngoài, lao động cho các tổ chức, cá nhân nước ngoài tại Việt Nam), khoản 1
  - Graph top 3: Bộ luật Lao động 2019, Điều 14 (Hình thức hợp đồng lao động), khoản 1; Bộ luật Lao động 2019, Điều 14 (Hình thức hợp đồng lao động), khoản 2; Nghị định 145/2020/NĐ-CP, Điều 17 (Quản lý tiền ký quỹ), khoản 2
- `contract_types` (direct_qa): +3.000
  - Query: Co may loai hop dong lao dong theo Bo luat Lao dong 2019?
  - Vector top 3: Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 5; Bộ luật Lao động 2019, Điều 151 (Điều kiện người lao động nước ngoài làm việc tại Việt Nam), khoản 2; Bộ luật Lao động 2019, Điều 156 (Các trường hợp giấy phép lao động hết hiệu lực), khoản 2
  - Graph top 3: Bộ luật Lao động 2019, Điều 20 (Loại hợp đồng lao động), khoản 1, các điểm a, b; Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 1, các điểm a, b, c, d, đ, e, g, h, i, k; Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 2
- `contract_fixed_vs_indefinite` (comparison_qa): +3.000
  - Query: So sanh hop dong xac dinh thoi han va khong xac dinh thoi han.
  - Vector top 3: Bộ luật Lao động 2019, Điều 151 (Điều kiện người lao động nước ngoài làm việc tại Việt Nam), khoản 2; Bộ luật Lao động 2019, Điều 151 (Điều kiện người lao động nước ngoài làm việc tại Việt Nam), khoản 3; Nghị định 145/2020/NĐ-CP, Mẫu số 02/PLI
  - Graph top 3: Bộ luật Lao động 2019, Điều 20 (Loại hợp đồng lao động), khoản 1, các điểm a, b; Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 1, các điểm a, b, c, d, đ, e, g, h, i, k; Bộ luật Lao động 2019, Điều 21 (Nội dung hợp đồng lao động), khoản 2

## Remaining Failures Or Over-Expansion

- No query scored worse than vector-only under the current scoring formula.

## Conclusion

On this constructed benchmark, graph-augmented retrieval provides the strongest citation coverage because it can connect primary Labor Code provisions to implementing decrees, circular guidance, exceptions, and labor-litigation provisions. Hybrid dense+sparse retrieval is a stronger baseline than dense vector-only retrieval for these legal-reference questions. The graph layer is most valuable for multi-hop, procedure, exception, and scenario questions because it can promote legally connected provisions even when the query wording does not exactly match every required citation. These results should be interpreted as benchmark evidence, not as proof of universal legal correctness.
