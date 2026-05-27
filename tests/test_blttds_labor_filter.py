from __future__ import annotations

import json
from pathlib import Path

from src.vn_labor_law_ai_assistant.blttds_labor_filter import (
    FILTER_HEADER,
    filter_blttds_labor_subset,
    parse_articles,
    render_labor_subset,
    select_labor_related_articles,
)


def sample_blttds_text() -> str:
    return "\n".join(
        [
            "BỘ LUẬT",
            "TỐ TỤNG DÂN SỰ",
            "",
            "PHẦN THỨ NHẤT",
            "NHỮNG QUY ĐỊNH CHUNG",
            "",
            "CHƯƠNG III",
            "THẨM QUYỀN CỦA TÒA ÁN",
            "",
            "Mục 1",
            "NHỮNG VỤ VIỆC DÂN SỰ THUỘC THẨM QUYỀN CỦA TÒA ÁN",
            "",
            "Điều 28. Những tranh chấp về hôn nhân và gia đình thuộc thẩm quyền giải quyết của Tòa án",
            "1. Ly hôn, tranh chấp về nuôi con, chia tài sản khi ly hôn.",
            "",
            "Điều 31. Những yêu cầu về kinh doanh, thương mại thuộc thẩm quyền giải quyết của Tòa án",
            "1. Yêu cầu liên quan đến Trọng tài thương mại Việt Nam.",
            "",
            "Điều 32. Những tranh chấp về lao động thuộc thẩm quyền giải quyết của",
            "Tòa án",
            "1. Tranh chấp lao động cá nhân giữa người lao động với người sử dụng lao động.",
            "a) Về xử lý kỷ luật lao động theo hình thức sa thải.",
            "",
            "Điều 33. Những yêu cầu về lao động thuộc thẩm quyền giải quyết của Tòa án",
            "1. Yêu cầu tuyên bố hợp đồng lao động, thỏa ước lao động tập thể vô hiệu.",
            "2. Yêu cầu xét tính hợp pháp của cuộc đình công.",
            "",
            "CHƯƠNG VIII",
            "CÁC BIỆN PHÁP KHẨN CẤP TẠM THỜI",
            "",
            "Điều 118. Buộc người sử dụng lao động tạm ứng tiền lương, tiền bảo hiểm y tế, bảo hiểm xã hội, bảo hiểm thất nghiệp, chi phí cứu chữa tai nạn lao động hoặc bệnh nghề nghiệp, tiền bồi thường, trợ cấp tai nạn lao động hoặc bệnh nghề nghiệp cho người lao động",
            "Buộc người sử dụng lao động tạm ứng khoản tiền cần thiết cho người lao động.",
            "",
            "Điều 119. Tạm đình chỉ thi hành quyết định đơn phương chấm dứt hợp đồng lao động, quyết định sa thải người lao động",
            "Tòa án tạm đình chỉ quyết định của người sử dụng lao động.",
            "",
            "PHẦN THỨ SÁU",
            "THỦ TỤC GIẢI QUYẾT VIỆC DÂN SỰ",
            "",
            "CHƯƠNG XXIX",
            "THỦ TỤC GIẢI QUYẾT YÊU CẦU TUYÊN BỐ",
            "VĂN BẢN CÔNG CHỨNG VÔ HIỆU",
            "",
            "Điều 400. Quyết định tuyên bố văn bản công chứng vô hiệu",
            "Tòa án có thể chấp nhận hoặc không chấp nhận đơn yêu cầu.",
            "",
            "CHƯƠNG XXX",
            "THỦ TỤC GIẢI QUYẾT YÊU CẦU TUYÊN BỐ HỢP ĐỒNG",
            "LAO ĐỘNG VÔ HIỆU; THỎA ƯỚC LAO ĐỘNG TẬP THỂ VÔ HIỆU",
            "",
            "Điều 401. Yêu cầu tuyên bố hợp đồng lao động vô hiệu, thỏa ước lao động tập thể vô hiệu",
            "Người lao động, người sử dụng lao động có quyền yêu cầu Tòa án tuyên bố hợp đồng lao động vô hiệu.",
            "",
            "Điều 402. Xem xét yêu cầu tuyên bố hợp đồng lao động vô hiệu, thỏa ước lao động tập thể vô hiệu",
            "Tòa án mở phiên họp để xét yêu cầu tuyên bố hợp đồng lao động vô hiệu.",
        ]
    )


def selected_article_numbers(text: str) -> set[str]:
    articles = parse_articles(text)
    selections = select_labor_related_articles(articles)
    return {selection.article.number for selection in selections}


def test_required_labor_articles_are_kept() -> None:
    numbers = selected_article_numbers(sample_blttds_text())

    assert {"32", "33", "118", "119"}.issubset(numbers)


def test_invalid_labor_contract_provisions_are_kept() -> None:
    numbers = selected_article_numbers(sample_blttds_text())

    assert {"401", "402"}.issubset(numbers)


def test_unrelated_civil_family_business_only_articles_are_removed() -> None:
    numbers = selected_article_numbers(sample_blttds_text())

    assert "28" not in numbers
    assert "31" not in numbers
    assert "400" not in numbers


def test_hierarchy_is_preserved_in_rendered_subset() -> None:
    articles = parse_articles(sample_blttds_text())
    selections = select_labor_related_articles(articles)
    output = render_labor_subset(selections)

    assert output.startswith(FILTER_HEADER)
    assert output.index("PHẦN THỨ NHẤT") < output.index("CHƯƠNG III")
    assert output.index("CHƯƠNG III") < output.index("Mục 1")
    assert output.index("Mục 1") < output.index("Điều 32")
    assert output.index("CHƯƠNG XXX") < output.index("Điều 401")


def test_filter_writes_expected_report_format(tmp_path: Path) -> None:
    input_path = tmp_path / "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt"
    output_path = tmp_path / "92_2015_QH13_labor_only.txt"
    report_dir = tmp_path / "validation"
    input_path.write_text(sample_blttds_text(), encoding="utf-8")

    report = filter_blttds_labor_subset(input_path, output_path, report_dir)
    payload = json.loads((report_dir / "blttds_labor_filter_report.json").read_text(encoding="utf-8"))

    assert output_path.exists()
    assert (report_dir / "blttds_labor_filter_report.md").exists()
    assert report["kept_article_count"] == payload["kept_article_count"]
    assert payload["removed_article_count"] == 3
    assert {"kept_articles", "filter_keywords", "uncertain_articles_for_manual_review"}.issubset(
        payload
    )
