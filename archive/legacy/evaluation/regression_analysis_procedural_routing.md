# Procedural Routing Regression Analysis

Sources:
- `artifacts/evaluation/end_to_end_100_intent_gating_results.json` generated `2026-06-01T10:09:17.685276+00:00`
- `artifacts/evaluation/end_to_end_100_procedural_routing_results.json` generated `2026-06-01T11:25:44.729611+00:00`
- `artifacts/evaluation/ablation_retrieval_100_results_before_procedural_routing.json` generated `2026-06-01T10:12:23.568947+00:00`
- `artifacts/evaluation/ablation_retrieval_100_results.json` generated `2026-06-01T11:17:07.502621+00:00`

## Summary

Intent-gating end-to-end pass rate: `78/100`.

Procedural-routing end-to-end pass rate: `79/100`.

Net result improved by one case, but there are `3` pass-to-fail regressions and `4` fail-to-pass fixes.

Pass-to-fail regressions:

| ID | Category | Topic | Difficulty | Missing required citations after procedural routing | Injected BLTTDS 32/33/35/40? | Crowd-out? |
| --- | --- | --- | --- | --- | --- | --- |
| `retirement_age_general` | `document_guidance_qa` | `tuoi nghi huu` | `hard` | `BLLD Dieu 169`; `ND135 Dieu 4` | Yes: `BLTTDS 32` | Yes |
| `labor_dispute_limitation` | `direct_qa` | `tranh chap lao dong` | `medium` | `BLLD Dieu 190` | Yes: `BLTTDS 32` | Yes |
| `guidance_dismissal_dispute` | `document_guidance_qa` | `hoa giai truoc khi kien` | `hard` | `BLLD Dieu 190` | Yes: `BLTTDS 32`, `BLTTDS 35` | Yes, partially |

## Ablation Signal

The ablation totals improved overall, but the affected graph-augmented cases regressed:

| Mode | Before procedural routing pass rate | After procedural routing pass rate |
| --- | ---: | ---: |
| `hybrid_only` | `0.6596` | `0.7128` |
| `graph_augmented` | `0.7660` | `0.7766` |

Per-regression graph-augmented retrieval status:

| ID | Before procedural routing | After procedural routing | Interpretation |
| --- | --- | --- | --- |
| `retirement_age_general` | Passed; required `BLLD Dieu 169` and `ND135 Dieu 4` found | Failed; both required citations missing | Procedural route replaced retirement anchors with dispute-procedure anchors |
| `labor_dispute_limitation` | Passed; required `BLLD Dieu 190` found | Failed; `BLLD Dieu 190` missing | Limitation article was displaced by mediation/court anchors |
| `guidance_dismissal_dispute` | Passed; required `BLLD Dieu 188`, `BLTTDS Dieu 32`, and `BLLD Dieu 190` found | Failed; `BLLD Dieu 190` missing | Procedural court anchors preserved `BLTTDS 32` but crowded out the limitation anchor |

## Regression Details

### `retirement_age_general`

Query: `Tuoi nghi huu trong dieu kien lao dong binh thuong duoc xac dinh theo quy dinh nao?`

Category: `document_guidance_qa`

Required citations: `BLLD Dieu 169`, `ND135 Dieu 4`

After procedural routing:
- Missing required citations: `BLLD Dieu 169`, `ND135 Dieu 4`
- Failure reasons: `retrieval_missing_required_context`, `answer_missing_required_rule`, `low_information_answer`
- Injected procedural anchors: `BLTTDS 32`
- Graph expansion changed from enabled in the intent-gating run to disabled in the procedural-routing run.

Top citations before vs after:

| Rank | Intent-gating run | Procedural-routing run |
| ---: | --- | --- |
| 1 | Bộ luật Lao động 2019, Điều 169 (Tuổi nghỉ hưu), khoản 1 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 1, các điểm a, b, c, d, đ, e |
| 2 | Bộ luật Lao động 2019, Điều 169 (Tuổi nghỉ hưu), khoản 2 | Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 32 (Những tranh chấp về lao động thuộc thẩm quyền giải quyết của) |
| 3 | Nghị định 135/2020/NĐ-CP, Điều 4 (Tuổi nghỉ hưu trong điều kiện lao động bình thường) | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 2 |
| 4 | Nghị định 135/2020/NĐ-CP, PHỤ LỤC I. LỘ TRÌNH TUỔI NGHỈ HƯU TRONG ĐIỀU KIỆN LAO ĐỘNG BÌNH THƯỜNG GẮN VỚI THÁNG, NĂM SINH TƯƠNG ỨNG. Lao động nữ | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 5 |
| 5 | Nghị định 135/2020/NĐ-CP, PHỤ LỤC I. LỘ TRÌNH TUỔI NGHỈ HƯU TRONG ĐIỀU KIỆN LAO ĐỘNG BÌNH THƯỜNG GẮN VỚI THÁNG, NĂM SINH TƯƠNG ỨNG. Lao động nữ | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 3 |
| 6 | Nghị định 135/2020/NĐ-CP, Điều 7 (Quy định chuyển tiếp), khoản 1 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 6 |
| 7 | Nghị định 145/2020/NĐ-CP, Điều 48 (Trách nhiệm ban hành quy chê dân chủ ở cơ sở tại nơi làm việc), khoản 3 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 4 |
| 8 | Nghị định 135/2020/NĐ-CP, Điều 5 (Nghỉ hưu ở tuổi thấp hơn tuổi nghỉ hưu trong điều kiện lao động bình thường) | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 7, các điểm a, b |
| 9 | Nghị định 135/2020/NĐ-CP, Điều 6 (Nghỉ hưu ở tuổi cao hơn tuổi nghỉ hưu trong điều kiện lao động bình thường) | None returned |
| 10 | Nghị định 135/2020/NĐ-CP, PHỤ LỤC III. DANH MỤC CÔNG VIỆC KHAI THÁC THAN TRONG HẦM LÒ | None returned |

Cause judgment: yes, this is caused by procedural anchors crowding out required citations. The new top set is entirely mediation/court procedure material (`BLLD 188`, `BLTTDS 32`) for a retirement-age query, while the prior run had `BLLD 169` at ranks 1-2 and `ND135 Dieu 4` at rank 3.

### `labor_dispute_limitation`

Query: `Thoi hieu yeu cau giai quyet tranh chap lao dong ca nhan la bao lau?`

Category: `direct_qa`

Required citations: `BLLD Dieu 190`

After procedural routing:
- Missing required citations: `BLLD Dieu 190`
- Failure reasons: `retrieval_missing_required_context`
- Injected procedural anchors: `BLTTDS 32`

Top citations before vs after:

| Rank | Intent-gating run | Procedural-routing run |
| ---: | --- | --- |
| 1 | Bộ luật Lao động 2019, Điều 190 (Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân), khoản 1 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 1, các điểm a, b, c, d, đ, e |
| 2 | Bộ luật Lao động 2019, Điều 190 (Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân), khoản 2 | Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 32 (Những tranh chấp về lao động thuộc thẩm quyền giải quyết của) |
| 3 | Bộ luật Lao động 2019, Điều 179 (Tranh chấp lao động), khoản 1, các điểm a, b | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 5 |
| 4 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 1, các điểm a, b, c, d, đ, e | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 6 |
| 5 | Bộ luật Lao động 2019, Điều 190 (Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân), khoản 3 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 2 |
| 6 | Bộ luật Lao động 2019, Điều 179 (Tranh chấp lao động), khoản 3, các điểm a, b | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 3 |
| 7 | Bộ luật Lao động 2019, Điều 190 (Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân), khoản 4 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 4 |
| 8 | Bộ luật Lao động 2019, Điều 179 (Tranh chấp lao động), khoản 2, các điểm a, b, c | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 7, các điểm a, b |
| 9 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 6 | None returned |
| 10 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 2 | None returned |

Cause judgment: yes, this is caused by procedural anchors crowding out the required limitation citation. In the intent-gating run, `BLLD Dieu 190` occupied ranks 1, 2, 5, and 7. In the procedural-routing run, `BLLD Dieu 190` is absent and the list is dominated by `BLLD Dieu 188` plus injected `BLTTDS Dieu 32`.

### `guidance_dismissal_dispute`

Query: `Tranh chap sa thai co can hoa giai truoc khi kien khong va Toa an co tham quyen khong?`

Category: `document_guidance_qa`

Required citations: `BLLD Dieu 188`, `BLTTDS Dieu 32`, `BLLD Dieu 190`

After procedural routing:
- Missing required citations: `BLLD Dieu 190`
- Failure reasons: `retrieval_missing_required_context`
- Injected procedural anchors: `BLTTDS 32`, `BLTTDS 35`

Top citations before vs after:

| Rank | Intent-gating run | Procedural-routing run |
| ---: | --- | --- |
| 1 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 1, các điểm a, b, c, d, đ, e | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 1, các điểm a, b, c, d, đ, e |
| 2 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 2 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 2 |
| 3 | Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 32 (Những tranh chấp về lao động thuộc thẩm quyền giải quyết của) | Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 32 (Những tranh chấp về lao động thuộc thẩm quyền giải quyết của), khoản 1, các điểm a, b, c, d, đ |
| 4 | Bộ luật Lao động 2019, Điều 190 (Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân), khoản 2 | Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 32 (Những tranh chấp về lao động thuộc thẩm quyền giải quyết của) |
| 5 | Bộ luật Lao động 2019, Điều 190 (Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân), khoản 1 | Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 35 (Thẩm quyền của Tòa án nhân dân cấp huyện), khoản 1, các điểm a, b, c |
| 6 | Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động, Điều 119 (Tạm đình chỉ thi hành quyết định đơn phương chấm dứt hợp đồng lao động, quyết định sa thải người lao động) | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 6 |
| 7 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 6 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 4 |
| 8 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 4 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 5 |
| 9 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 5 | Bộ luật Lao động 2019, Điều 188 (Trình tự, thủ tục hòa giải tranh chấp lao động cá nhân của hòa giải viên lao động), khoản 3 |
| 10 | None returned | None returned |

Cause judgment: yes, but only partially. This query legitimately needs procedural anchors, and `BLLD Dieu 188` plus `BLTTDS Dieu 32` remain present. The regression is that `BLLD Dieu 190`, previously present at ranks 4 and 5, is displaced after procedural routing by additional court-competence material, especially `BLTTDS Dieu 35`.

## Conclusion

All three regressions are retrieval regressions, not citation-grounding regressions. Procedural routing improved several labor-dispute-procedure cases overall, but it over-applies the procedural anchor set in these pass-to-fail cases.

The clearest bad route is `retirement_age_general`, where a non-dispute retirement query is routed into the dispute-procedure anchor pattern. The two dispute-related regressions show a narrower problem: procedural anchors preserve or add `BLLD Dieu 188` and `BLTTDS Dieu 32/35`, but crowd out `BLLD Dieu 190`, which the benchmark requires for limitation-period coverage.
