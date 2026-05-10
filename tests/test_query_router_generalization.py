from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.heuristic_router import route_query


class QueryRouterGeneralizationTests(unittest.TestCase):
    def test_direct_reference_rules_match_paraphrased_legal_concepts(self) -> None:
        cases = (
            (
                "Nguoi su dung lao dong duoc hieu nhu the nao?",
                ("45-2019-qh14", "3", "2", ""),
            ),
            (
                "The nao la hop dong lao dong theo quy dinh hien hanh?",
                ("45-2019-qh14", "13", "1", ""),
            ),
            (
                "Thoi gian thu viec khong duoc qua bao lau?",
                ("45-2019-qh14", "25", "", ""),
            ),
            (
                "Doanh nghiep can dap ung dieu kien nao khi yeu cau lam them gio?",
                ("45-2019-qh14", "107", "", ""),
            ),
            (
                "Luong thu viec chi bang 60 phan tram luong chinh thuc co hop phap khong?",
                ("45-2019-qh14", "26", "", ""),
            ),
            (
                "Hợp đồng lao động ghi lương thử việc chỉ bằng 60% lương chính thức có hợp pháp không?",
                ("45-2019-qh14", "26", "", ""),
            ),
            (
                "Hop dong lao dong phai co nhung noi dung chu yeu nao?",
                ("45-2019-qh14", "21", "1", ""),
            ),
        )

        for query, expected in cases:
            with self.subTest(query=query):
                intent = route_query(query)
                actual = {
                    (
                        reference.document_id,
                        reference.article,
                        reference.clause,
                        reference.point,
                    )
                    for reference in intent.forced_references
                }

                self.assertIn(expected, actual)

    def test_paraphrase_legal_concepts_route_to_expected_articles(self) -> None:
        cases = (
            (
                "Cong ty giu ho chieu ban goc cua nhan vien nuoc ngoai de ho khong tu y nghi thi co rui ro gi khong?",
                {"17"},
                {"giu_giay_to_goc"},
                {"17"},
            ),
            (
                "Quan yeu cau nhan vien nop tien the chan 3 trieu truoc khi nhan viec, vay co hop phap khong?",
                {"17"},
                {"dat_coc_bao_dam"},
                {"17"},
            ),
            (
                "HR co the tu choi mot ung vien vi biet ho co HIV du nang luc phu hop khong?",
                {"3", "8", "11"},
                {"phan_biet_doi_xu", "tuyen_dung_lao_dong"},
                {"3", "8", "11"},
            ),
            (
                "Mot be 14 tuoi lam phuc vu tu 7 gio toi den nua dem o quan an thi co vi pham gi khong?",
                {"143", "145", "146", "106"},
                {"lao_dong_chua_thanh_nien", "lam_ban_dem"},
                {"143", "145", "146", "106"},
            ),
            (
                "Ca lam tu 23h den 5h sang co duoc tinh la lam ban dem khong?",
                {"106"},
                {"lam_ban_dem"},
                set(),
            ),
            (
                "Toi bi sep nhan tin ga gam o noi lam viec va muon nghi ngay thi co phai bao truoc khong?",
                {"3", "8", "35", "118"},
                {"quay_roi_tinh_duc"},
                {"3", "8", "35", "118"},
            ),
            (
                "Noi quy cong ty ghi di muon bi tru 200 nghin vao luong thi co dung luat khong?",
                {"127", "124", "118"},
                {"xu_ly_ky_luat_lao_dong", "noi_quy_lao_dong"},
                {"127", "124", "118"},
            ),
            (
                "Quan ly bat toi ky don tu nguyen nghi viec de cong ty khong phai ra quyet dinh cho nghi, toi nen hieu the nao?",
                {"7", "15", "34", "36", "39", "41"},
                {"ep_nghi_viec"},
                {"7", "15", "34", "36", "39", "41"},
            ),
            (
                "Cong ty bat toi cam ket sau khi nghi khong duoc lam cung nganh suot doi, dieu khoan nay co van de gi khong?",
                {"10", "21", "15"},
                {"han_che_viec_lam_sau_nghi", "bao_mat_bi_mat_kinh_doanh"},
                {"10", "21", "15"},
            ),
            (
                "Cong ty noi toi nghi roi nhung khong co quyet dinh gi, co chac la ho sai luat khong?",
                {"34", "36", "37", "39", "41", "48"},
                {"can_cu_cham_dut", "trai_phap_luat"},
                set(),
            ),
        )

        for query, expected_articles, expected_issues, expected_force in cases:
            with self.subTest(query=query):
                intent = route_query(query)

                self.assertTrue(expected_articles.issubset(set(intent.inferred_article_numbers)))
                self.assertTrue(expected_issues.issubset(set(intent.issue_filters)))
                self.assertTrue(expected_force.issubset(set(intent.force_reference_article_numbers)))

    def test_short_soft_hints_do_not_force_reference_fallback(self) -> None:
        soft_cases = (
            "HIV co phai thong tin suc khoe trong ho so nhan su khong?",
            "Em 14 tuoi co di lam them duoc khong?",
            "Ca dem duoc tinh tu may gio?",
        )

        for query in soft_cases:
            with self.subTest(query=query):
                intent = route_query(query)

                self.assertFalse(intent.force_reference_article_numbers)

        minor_intent = route_query("Em 14 tuoi co di lam them duoc khong?")
        self.assertIn("143", minor_intent.inferred_article_numbers)
        self.assertNotIn("106", minor_intent.inferred_article_numbers)

        night_intent = route_query("Ca dem duoc tinh tu may gio?")
        self.assertIn("106", night_intent.inferred_article_numbers)


if __name__ == "__main__":
    unittest.main()
