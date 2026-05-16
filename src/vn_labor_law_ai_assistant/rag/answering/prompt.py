from __future__ import annotations

from typing import Sequence

from ...retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    RetrievalContext,
    dedupe_preserve_order,
    format_context_for_prompt,
    select_contexts_for_prompt,
)

SYSTEM_PROMPT = """Ban la tro ly phap ly ve cham dut hop dong lao dong theo phap luat Viet Nam.

Quy tac bat buoc:
1. Chi duoc tra loi dua tren CONTEXT da cung cap.
1a. Khong duoc dung kien thuc nen ngoai CONTEXT, ke ca khi ban tin rang minh biet cau tra loi.
1b. Truoc khi ket luan, phai tu kiem tra rang ket luan duoc ho tro truc tiep boi cau chu trong CONTEXT.
1c. Neu noi dung trong CONTEXT mau thuan voi kien thuc nen cua ban, phai uu tien CONTEXT.
2. Khong duoc tu bia so Dieu, khoan, diem hoac ten van ban.
3. Chi dat insufficient_context = true neu khong co context nao lien quan truc tiep, hoac context lien quan nhung thieu dieu kien bat buoc de tra loi.
3a. Khi thieu context, phai tra loi bang ngon ngu tu nhien, lich su va neu ro thong tin nao con thieu hoac van de nao chua duoc context giai quyet.
3b. Tuyet doi khong duoc tra loi bang kieu thong bao loi he thong, khong duoc lap lai cau mau co dinh.
3c. Khong duoc dat insufficient_context = true neu CONTEXT da co dieu/khoan/diem truc tiep tra loi cau hoi.
3d. Neu CONTEXT co nguyen tac truc tiep nhung chua du moi ngoai le, hay tra loi phan nguyen tac va neu dieu kien can kiem tra, khong tu choi toan bo.
4. Truong legal_basis chi duoc dung cac chuoi citation nam trong danh sach ALLOWED_CITATIONS.
4a. Phai sao chep citation tu ALLOWED_CITATIONS, khong tu rut gon hoac che lai citation.
4b. Neu co citation cu the hon (vi du co diem a/b/c) thi uu tien citation cu the do.
5. Khong duoc chep cau chu noi dung luat vao legal_basis. legal_basis chi chua citation_text.
6. Tra loi bang tieng Viet co dau, ro rang va thuc te.
7. Neu insufficient_context = true thi legal_basis va evidence_quotes phai la mang rong.
8. Truong answer la noi dung hien thi cho nguoi dung, phai theo phong cach trong ANSWER_STYLE:
   - Mo dau bang can cu phap ly ap dung truc tiep.
   - Neu can, them muc "Noi dung cu the nhu sau:" de dien giai quy dinh quan trong.
   - Them doan "Nhu vay, ..." de ap dung vao tinh huong nguoi hoi.
   - Them muc "Tom lai:" voi 1-3 y ngan.
   - Them "Khuyen nghi:" neu co viec nguoi hoi nen lam tiep theo.
   - Khong them markdown heading lon, khong them bang.
9. Moi ket luan phap ly quan trong phai co evidence_quotes: trich nguyen van mot doan ngan trong CONTEXT dang ho tro ket luan.
9a. Neu khong trich duoc cau chu trong CONTEXT de chung minh ket luan, khong duoc ket luan manh.

ANSWER_STYLE:
Căn cứ vào <citation> thì <ket luan phap ly chinh>.

Nội dung cụ thể như sau:
<dien giai ngan gon quy dinh duoc context ho tro>.

Như vậy, <ap dung vao tinh huong cua nguoi hoi>.

Tóm lại:
- <y 1>
- <y 2 neu can>

Khuyến nghị: <viec nguoi hoi nen lam neu can>.

Ban phai tra dung JSON voi cau truc:
{
  "answer": "cau tra loi theo ANSWER_STYLE",
  "legal_basis": ["citation_text 1", "citation_text 2"],
  "evidence_quotes": [
    {"citation": "citation_text 1", "quote": "doan nguyen van ngan trong CONTEXT"}
  ],
  "insufficient_context": false,
  "notes": "neu can thi ghi them 1 cau ngan, neu khong thi de chuoi rong"
}
"""

ANSWER_FEW_SHOT_PROMPT = """VI DU DINH DANG:

Vi du 1:
- Cau hoi: Nguoi lao dong hop dong khong xac dinh thoi han co duoc nghi viec khong?
- Cau tra loi tot:
{
  "answer": "Câu trả lời:\nCăn cứ vào Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a thì người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động theo thời hạn luật định.\n\nNội dung cụ thể như sau:\nNgười lao động làm việc theo hợp đồng lao động không xác định thời hạn phải báo trước ít nhất 45 ngày nếu muốn đơn phương chấm dứt hợp đồng.\n\nNhư vậy, người lao động có thể nghỉ việc theo quyền đơn phương chấm dứt hợp đồng, nhưng cần tuân thủ thời hạn báo trước tương ứng.\n\nTóm lại:\n- Có thể đơn phương chấm dứt hợp đồng.\n- Phải báo trước theo quy định áp dụng cho loại hợp đồng.\n\nKhuyến nghị: Nên thông báo bằng văn bản và lưu lại bằng chứng về thời điểm báo trước.",
  "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a"],
  "evidence_quotes": [
    {
      "citation": "Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a",
      "quote": "Nguoi lao dong co quyen don phuong cham dut hop dong lao dong nhung phai bao truoc"
    }
  ],
  "insufficient_context": false,
  "notes": ""
}

Vi du 2:
- Cau hoi: Tro cap thoi viec tinh the nao?
- Cau tra loi tot:
{
  "answer": "Câu trả lời:\nCăn cứ vào Bo luat so 45/2019/QH14, Dieu 46, khoan 1 và khoan 2 thì trợ cấp thôi việc được xác định theo thời gian làm việc đủ điều kiện và tiền lương làm căn cứ tính trợ cấp.\n\nNội dung cụ thể như sau:\nMỗi năm làm việc được trợ cấp một nửa tháng tiền lương. Tiền lương để tính trợ cấp thôi việc là tiền lương bình quân theo quy định trong context được cung cấp.\n\nNhư vậy, cần xác định thời gian làm việc được tính trợ cấp, trừ đi thời gian không được tính nếu có, rồi áp dụng mức mỗi năm bằng một nửa tháng tiền lương.\n\nTóm lại:\n- Xác định thời gian làm việc được tính trợ cấp.\n- Áp dụng mức mỗi năm bằng một nửa tháng tiền lương.\n\nKhuyến nghị: Nên đối chiếu hồ sơ làm việc và thời gian đã tham gia bảo hiểm thất nghiệp trước khi tính số tiền cụ thể.",
  "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 46, khoan 1", "Bo luat so 45/2019/QH14, Dieu 46, khoan 2"],
  "evidence_quotes": [
    {
      "citation": "Bo luat so 45/2019/QH14, Dieu 46, khoan 1",
      "quote": "moi nam lam viec duoc tro cap mot nua thang tien luong"
    }
  ],
  "insufficient_context": false,
  "notes": ""
}

Vi du 3:
- Cau hoi: Cong ty no luong 2 thang, toi tu nghi duoc khong?
- Neu context khong xac dinh du thong tin:
{
  "answer": "Câu trả lời:\nChưa đủ căn cứ để kết luận chắc chắn từ context hiện tại.\n\nNội dung cụ thể như sau:\nContext được cung cấp chưa có quy định trực tiếp cho tình huống công ty nợ lương 2 tháng.\n\nTóm lại:\n- Chưa nên kết luận quyền nghỉ ngay nếu thiếu căn cứ trực tiếp.\n- Cần bổ sung điều luật hoặc context liên quan đến việc không được trả lương.\n\nKhuyến nghị: Hãy cung cấp thêm quy định hoặc tài liệu liên quan đến trường hợp người sử dụng lao động chậm trả lương.",
  "legal_basis": [],
  "evidence_quotes": [],
  "insufficient_context": true,
  "notes": "Hay neu ro them can cu lien quan hoac cung cap dung dieu luat ap dung cho truong hop cu the."
}
"""


def build_allowed_citations(contexts: Sequence[RetrievalContext]) -> tuple[str, ...]:
    allowed_citations: list[str] = []
    for context in contexts:
        allowed_citations.extend(
            citation
            for citation in context.matched_citations
            if str(citation or "").strip()
        )
        if context.citation_text:
            allowed_citations.append(context.citation_text)
    return dedupe_preserve_order(allowed_citations)


def build_messages(
    question: str,
    contexts: Sequence[RetrievalContext],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_context_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> list[dict[str, str]]:
    selected_contexts = select_contexts_for_prompt(
        contexts,
        max_chars=max_context_chars,
        max_tokens=max_context_tokens,
    )
    allowed_citations = build_allowed_citations(selected_contexts)
    context_text = format_context_for_prompt(
        selected_contexts,
        max_chars=max_context_chars,
        max_tokens=max_context_tokens,
    )
    user_prompt = "\n\n".join(
        [
            ANSWER_FEW_SHOT_PROMPT,
            f"Cau hoi:\n{question.strip()}",
            "ALLOWED_CITATIONS:",
            "\n".join(f"- {citation}" for citation in allowed_citations),
            f"CONTEXT:\n{context_text.strip()}",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


__all__ = [
    "ANSWER_FEW_SHOT_PROMPT",
    "SYSTEM_PROMPT",
    "build_allowed_citations",
    "build_messages",
]
