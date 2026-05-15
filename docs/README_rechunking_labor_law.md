# Re-chunking cho RAG chatbot luật lao động

## Vấn đề trong chunk hiện tại

Bộ dữ liệu hiện tại tách rất nhỏ theo `điểm a/b/c` và đôi khi tách riêng câu mở đầu của `khoản`. Ví dụ Điều 107, khoản 2 có câu mở đầu "Người sử dụng lao động được sử dụng người lao động làm thêm giờ khi đáp ứng..." nhưng các điều kiện a/b/c bị tách thành chunk riêng. Khi query hỏi "điều kiện làm thêm giờ", retriever có thể lấy riêng điểm b hoặc c và bỏ mất câu mở đầu hoặc các điều kiện còn lại.

## Chiến lược mới

- Đơn vị chính để embedding: `khoản`.
- Nếu `khoản` có nhiều `điểm`, gộp câu mở đầu của khoản + toàn bộ các điểm liên quan vào cùng một chunk.
- Nếu khoản quá dài, tách theo nhóm điểm, nhưng mỗi chunk con vẫn lặp lại câu mở đầu của khoản.
- Với điều chỉ có một đoạn/không có khoản: giữ chunk cấp `article_full`.
- Với điều có đoạn mở đầu trước các khoản: tạo `article_intro`, đồng thời lặp đoạn mở đầu vào chunk khoản để giữ ngữ cảnh.
- Với điều sửa đổi/bổ sung phức tạp như Điều 219, không gộp theo `clause_ref` vì số khoản bị lặp trong phần trích luật khác; dùng chunk tuần tự có giới hạn độ dài.

## File đầu ra

- `labor_law_rechunked_hierarchical.jsonl`: dữ liệu đã re-chunk.
- `rechunk_labor_law.py`: script dùng để tái tạo hoặc chỉnh `--max-chars`.

## Field nên dùng

- Dùng `page_content` hoặc `retrieval_text` để embedding.
- Dùng `citation_text` để hiển thị căn cứ pháp lý.
- Dùng `document_id`, `article_number`, `clause_ref`, `point_refs`, `chunk_type` để filter/rerank.

## Gợi ý retrieval

1. Hybrid search: BM25 + vector search.
2. Rerank top 20-50 bằng cross-encoder hoặc LLM reranker.
3. Khi top chunk là một khoản, có thể fetch thêm các chunk cùng `section_id` hoặc cùng `article_number` để bổ sung ngữ cảnh nếu câu hỏi rộng.
4. Ưu tiên exact match cho cụm như "Điều 35", "khoản 2", "điểm b", "làm thêm giờ", "nghỉ lễ".
5. Không chỉ embed `text` thô; nên embed `retrieval_text` vì có tên văn bản, chương/mục, điều/khoản/điểm.
