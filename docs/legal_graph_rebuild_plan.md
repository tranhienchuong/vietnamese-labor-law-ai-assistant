# Rebuild Legal Corpus and Knowledge Graph for Vietnamese Labor Law RAG

## 1. Mục tiêu chính

Mục tiêu của task này là xây dựng lại toàn bộ dữ liệu và đồ thị tri thức pháp luật cho hệ thống RAG về **Luật Lao động Việt Nam**, thay vì chỉ dùng graph hiện tại gồm **Bộ luật Lao động 2019** và một phần **Nghị định 145/2020/NĐ-CP**.

Graph mới cần lấy **Bộ luật Lao động 2019** làm xương sống, sau đó liên kết với các văn bản hướng dẫn, nghị định, thông tư và một phần văn bản tố tụng có liên quan đến tranh chấp lao động.

Kết quả cuối cùng cần đạt được là một hệ thống **Graph-Augmented RAG** có khả năng truy xuất tốt hơn với các câu hỏi pháp lý phức tạp, đặc biệt là các câu hỏi cần kết hợp nhiều điều luật, nhiều khoản, nhiều văn bản hướng dẫn hoặc ngoại lệ pháp lý.

---

## 2. Phạm vi tài liệu cần xử lý

Corpus mới cần bao gồm các tài liệu sau:

| STT | Văn bản | Vai trò |
|---|---|---|
| 1 | Bộ luật Lao động 2019 | Văn bản gốc, làm backbone của graph |
| 2 | Nghị định 145/2020/NĐ-CP | Văn bản hướng dẫn chính về điều kiện lao động và quan hệ lao động |
| 3 | Nghị định 135/2020/NĐ-CP | Văn bản hướng dẫn về tuổi nghỉ hưu |
| 4 | Thông tư 09/2020/TT-BLĐTBXH | Văn bản hướng dẫn về lao động chưa thành niên |
| 5 | Thông tư 10/2020/TT-BLĐTBXH | Văn bản hướng dẫn về nội dung hợp đồng lao động |
| 6 | Bộ luật Tố tụng dân sự 2015 | Chỉ lấy phần liên quan đến tranh chấp lao động, thẩm quyền Tòa án và thủ tục tố tụng |

Lưu ý: Bộ luật Tố tụng dân sự không nên đưa toàn bộ vào graph chính. Chỉ nên giữ các phần liên quan trực tiếp đến tranh chấp lao động, ví dụ thẩm quyền của Tòa án, tranh chấp lao động cá nhân, hòa giải, khởi kiện và tố tụng.

---

## 3. Vấn đề hiện tại

Graph hiện tại chưa đủ mạnh vì:

1. Corpus mới chỉ có Bộ luật Lao động 2019 và một phần Nghị định 145/2020/NĐ-CP.
2. Thiếu các văn bản hướng dẫn quan trọng như Nghị định 135, Thông tư 09, Thông tư 10.
3. Một số PDF là dạng scan nên text extraction có thể thiếu hoặc lỗi.
4. Chunking hiện tại cần được kiểm tra lại để đảm bảo không phá vỡ cấu trúc pháp luật.
5. Cross-reference giữa các văn bản chưa đầy đủ.
6. Graph chưa thể hiện rõ quan hệ giữa luật gốc và văn bản hướng dẫn.
7. Với câu hỏi phức tạp, vector search đơn thuần có thể chỉ lấy được một điều luật, trong khi câu trả lời đúng cần nhiều điều luật liên quan.

---

## 4. Mục tiêu kỹ thuật

Task này cần đạt các mục tiêu kỹ thuật sau:

### 4.1. Rebuild corpus

Cần xử lý lại toàn bộ tài liệu pháp luật thành dạng text sạch, có cấu trúc rõ ràng.

Yêu cầu:

- OCR các file PDF dạng scan.
- Làm sạch header, footer, số trang, ký tự lỗi.
- Chuẩn hóa Unicode tiếng Việt.
- Giữ nguyên cấu trúc pháp luật: Chương, Mục, Điều, Khoản, Điểm.
- Không cắt chunk theo số token một cách ngẫu nhiên.
- Không làm mất mối quan hệ giữa Điều, Khoản, Điểm.

Cấu trúc dữ liệu đề xuất:

```text
data/
  raw/
    45_2019_QH14_bo_luat_lao_dong.pdf
    145_2020_ND_CP.pdf
    135_2020_ND_CP.pdf
    09_2020_TT_BLDTBXH.pdf
    10_2020_TT_BLDTBXH.pdf
    92_2015_QH13_blttds.pdf

  curated/
    45_2019_QH14.txt
    145_2020_ND_CP.txt
    135_2020_ND_CP.txt
    09_2020_TT_BLDTBXH.txt
    10_2020_TT_BLDTBXH.txt
    92_2015_QH13_lao_dong_only.txt
```

---

### 4.2. Rebuild chunking

Chunking phải bám sát cấu trúc tự nhiên của văn bản pháp luật.

Thứ tự phân cấp cần nhận diện:

```text
Document
  → Chapter
    → Section
      → Article
        → Clause
          → Point
```

Quy tắc chunking:

| Trường hợp | Cách chunk |
|---|---|
| Điều ngắn | Giữ nguyên cả Điều |
| Điều dài | Tách theo Khoản |
| Khoản dài | Tách theo Điểm |
| Danh sách dài | Tách theo từng nhóm logic |
| Phụ lục / bảng | Tách theo từng bảng hoặc từng nhóm dữ liệu |
| Văn bản tố tụng | Chỉ chunk phần liên quan đến tranh chấp lao động |

Mỗi chunk cần có metadata đầy đủ:

```json
{
  "chunk_id": "TT09_2020_Dieu_3_Khoan_5",
  "document_id": "thong-tu-09-2020-tt-bldtbxh",
  "document_title": "Thông tư 09/2020/TT-BLĐTBXH",
  "document_type": "thong_tu",
  "article_number": "3",
  "article_title": "Điều kiện sử dụng người chưa đủ 15 tuổi làm việc",
  "clause_ref": "5",
  "point_ref": null,
  "chapter_heading": "Chương II",
  "section_heading": null,
  "topic": ["lao_dong_chua_thanh_nien"],
  "actor": ["nguoi_su_dung_lao_dong", "nguoi_lao_dong_chua_thanh_nien"],
  "citation_text": "Thông tư 09/2020/TT-BLĐTBXH, Điều 3, khoản 5",
  "text": "..."
}
```

---

## 5. Thiết kế graph mới

Graph mới cần được xây theo 3 lớp chính.

---

### 5.1. Structural graph

Structural graph thể hiện cấu trúc pháp lý của văn bản.

Node types:

```text
LegalDocument
LegalChapter
LegalSection
LegalArticle
LegalClause
LegalPoint
EvidenceChunk
```

Relations:

```text
HAS_CHAPTER
HAS_SECTION
HAS_ARTICLE
HAS_CLAUSE
HAS_POINT
HAS_SOURCE_CHUNK
SOURCE_OF
```

Ví dụ:

```text
Bộ luật Lao động 2019
  → HAS_ARTICLE
    → Điều 145
      → HAS_CLAUSE
        → Khoản 1
          → HAS_SOURCE_CHUNK
            → chunk text gốc
```

Mục tiêu của lớp này là giúp hệ thống luôn biết chunk nào thuộc văn bản nào, điều nào, khoản nào và điểm nào.

---

### 5.2. Citation and reference graph

Citation graph thể hiện quan hệ viện dẫn giữa các điều luật và văn bản.

Relations cần có:

```text
REFERENCES
GUIDES
GUIDED_BY
DETAILS
AMENDS
REPLACES
```

Ví dụ:

```text
Thông tư 09/2020/TT-BLĐTBXH Điều 3
  → DETAILS
    → Bộ luật Lao động 2019 Điều 145

Nghị định 135/2020/NĐ-CP Điều 4
  → DETAILS
    → Bộ luật Lao động 2019 Điều 169

Thông tư 10/2020/TT-BLĐTBXH Điều 3
  → DETAILS
    → Bộ luật Lao động 2019 Điều 21
```

Cần parse các cụm như:

```text
theo quy định tại khoản 4 Điều 145 của Bộ luật Lao động
theo khoản 1 Điều 21 của Bộ luật Lao động
quy định tại Điều 169 của Bộ luật Lao động
theo Nghị định 145/2020/NĐ-CP
```

Mục tiêu của lớp này là hỗ trợ multi-hop retrieval. Khi retrieve một điều luật gốc, hệ thống có thể mở rộng sang văn bản hướng dẫn. Khi retrieve một thông tư hoặc nghị định, hệ thống có thể quay lại điều luật gốc.

---

### 5.3. Semantic legal graph

Semantic graph thể hiện nội dung pháp lý bên trong từng Điều/Khoản/Điểm.

Node types:

```text
LegalConcept
LegalSubject
LegalRight
LegalObligation
LegalCondition
LegalException
LegalDeadline
LegalProcedure
LegalFormula
LegalConsequence
LegalSanction
```

Relations:

```text
MENTIONS_CONCEPT
APPLIES_TO
GRANTS_RIGHT
IMPOSES_OBLIGATION
HAS_CONDITION
HAS_EXCEPTION
HAS_DEADLINE
HAS_PROCEDURE
HAS_FORMULA
TRIGGERS_CONSEQUENCE
PROHIBITS
PERMITS
REQUIRES
RELATED_TO
```

Ví dụ:

```text
Điều 145 BLLĐ 2019
  → APPLIES_TO
    → Người chưa đủ 15 tuổi

Điều 145 BLLĐ 2019
  → IMPOSES_OBLIGATION
    → Người sử dụng lao động phải giao kết hợp đồng lao động bằng văn bản

Điều 146 BLLĐ 2019
  → HAS_CONDITION
    → Thời giờ làm việc của người chưa thành niên

Nghị định 135/2020/NĐ-CP Điều 4
  → HAS_FORMULA
    → Lộ trình tuổi nghỉ hưu theo năm
```

Mọi semantic edge cần có evidence:

```json
{
  "source_chunk_id": "TT09_2020_Dieu_3_Khoan_5",
  "evidence_text": "...",
  "extraction_method": "regex | rule | llm",
  "confidence": 0.0
}
```

Không tạo edge semantic nếu không có evidence.

---

## 6. Topic map cần bổ sung

Cần mở rộng topic rules để bao phủ các nhóm vấn đề chính trong Luật Lao động.

Topic đề xuất:

```text
hop_dong_lao_dong
giao_ket_hop_dong_lao_dong
noi_dung_hop_dong_lao_dong
cham_dut_hop_dong_lao_dong
don_phuong_cham_dut
tro_cap_thoi_viec
tro_cap_mat_viec
tien_luong
thoi_gio_lam_viec_nghi_ngoi
lam_them_gio
ky_luat_lao_dong
trach_nhiem_vat_chat
lao_dong_chua_thanh_nien
lao_dong_nu
thai_san
tuoi_nghi_huu
an_toan_ve_sinh_lao_dong
doi_thoai_tai_noi_lam_viec
thuong_luong_tap_the
tranh_chap_lao_dong
to_tung_lao_dong
```

Các topic mới cần ưu tiên thêm:

```python
TOPIC_RULES.update({
    "tuoi_nghi_huu": [
        "tuoi nghi huu",
        "nghi huu",
        "huu tri",
        "lo trinh dieu chinh tuoi nghi huu"
    ],
    "lao_dong_chua_thanh_nien": [
        "lao dong chua thanh nien",
        "nguoi chua du 15 tuoi",
        "tu du 13 tuoi den chua du 15 tuoi",
        "nguoi chua thanh nien"
    ],
    "thoi_gio_lam_viec_nghi_ngoi": [
        "thoi gio lam viec",
        "thoi gio nghi ngoi",
        "lam them gio",
        "lam viec ban dem"
    ],
    "tranh_chap_lao_dong": [
        "tranh chap lao dong",
        "hoa giai vien lao dong",
        "toa an",
        "khoi kien"
    ],
    "noi_dung_hop_dong_lao_dong": [
        "noi dung chu yeu cua hop dong lao dong",
        "muc luong",
        "phu cap luong",
        "dia diem lam viec",
        "thoi han cua hop dong lao dong"
    ]
})
```

---

## 7. Document alias map

Cần có alias map để hệ thống nhận diện nhiều cách gọi khác nhau của cùng một văn bản.

```python
DOCUMENT_ALIASES = {
    "bo luat lao dong": "45-2019-qh14",
    "bo luat lao dong nam 2019": "45-2019-qh14",
    "luat lao dong 2019": "45-2019-qh14",
    "nghi dinh 145": "nghi-dinh-145-2020-nd-cp",
    "nghi dinh 145/2020": "nghi-dinh-145-2020-nd-cp",
    "nghi dinh 135": "nghi-dinh-135-2020-nd-cp",
    "nghi dinh 135/2020": "nghi-dinh-135-2020-nd-cp",
    "thong tu 09": "thong-tu-09-2020-tt-bldtbxh",
    "thong tu 09/2020": "thong-tu-09-2020-tt-bldtbxh",
    "thong tu 10": "thong-tu-10-2020-tt-bldtbxh",
    "thong tu 10/2020": "thong-tu-10-2020-tt-bldtbxh",
    "bo luat to tung dan su": "92-2015-qh13",
    "blttds": "92-2015-qh13"
}
```

Alias map này cần dùng trong:

1. Cross-reference parser.
2. Query router.
3. Graph expansion.
4. Metadata normalization.

---

## 8. Graph-augmented retrieval flow

Retrieval flow mới nên là:

```text
User query
  → Query intent classification
  → Vector search / BM25 search
  → Select seed chunks
  → Map seed chunks to graph nodes
  → Expand graph neighbors
  → Retrieve related chunks from graph
  → Merge direct hits + graph hits
  → Rerank
  → Generate answer with citation
```

Graph expansion nên ưu tiên các relation:

```text
REFERENCES
GUIDES
GUIDED_BY
DETAILS
HAS_CONDITION
HAS_EXCEPTION
HAS_DEADLINE
APPLIES_TO
MENTIONS_CONCEPT
HAS_SOURCE_CHUNK
SOURCE_OF
```

Không nên expand quá rộng. Mặc định:

```text
depth = 1 hoặc 2 cho câu hỏi đơn giản
depth = 3 cho câu hỏi phức tạp
depth = 4 cho câu hỏi nhiều điều luật hoặc nhiều văn bản
```

---

## 9. Ví dụ expected behavior

### 9.1. Câu hỏi về lao động chưa thành niên

Query:

```text
Người 14 tuổi có được làm việc không?
```

Graph cần retrieve:

```text
BLLĐ 2019 Điều 143
BLLĐ 2019 Điều 145
BLLĐ 2019 Điều 146
BLLĐ 2019 Điều 147
Thông tư 09/2020 Điều 3
Thông tư 09/2020 Điều 4
Thông tư 09/2020 Điều 8
```

Mục tiêu: câu trả lời không chỉ nói “có/không”, mà phải nêu điều kiện, giới hạn thời giờ làm việc, công việc nhẹ được phép làm và yêu cầu về sự đồng ý/hợp đồng.

---

### 9.2. Câu hỏi về tuổi nghỉ hưu

Query:

```text
Nữ nghỉ hưu năm 2026 thì bao nhiêu tuổi?
```

Graph cần retrieve:

```text
BLLĐ 2019 Điều 169
Nghị định 135/2020 Điều 4
Phụ lục/bảng lộ trình tuổi nghỉ hưu
```

Mục tiêu: câu trả lời phải lấy được cả luật gốc và bảng hướng dẫn cụ thể.

---

### 9.3. Câu hỏi về tranh chấp lao động

Query:

```text
Tranh chấp sa thải có cần hòa giải trước khi kiện không?
```

Graph cần retrieve:

```text
BLLĐ 2019 phần tranh chấp lao động
BLTTDS 2015 Điều 32
Các quy định về trường hợp không bắt buộc hòa giải
```

Mục tiêu: hệ thống phải kết nối được luật nội dung với luật tố tụng.

---

### 9.4. Câu hỏi về hợp đồng lao động

Query:

```text
Hợp đồng lao động cần có những nội dung gì?
```

Graph cần retrieve:

```text
BLLĐ 2019 Điều 21
Thông tư 10/2020 Điều 3
Các khoản liên quan đến thông tin người sử dụng lao động, người lao động, công việc, địa điểm, lương, thời hạn hợp đồng
```

Mục tiêu: câu trả lời phải có nội dung luật gốc và phần hướng dẫn chi tiết.

---

## 10. Các bước triển khai

### Phase 1: Chuẩn hóa corpus

- Thu thập đủ 6 văn bản trong phạm vi.
- OCR các file scan.
- Làm sạch text.
- Tạo curated text cho từng văn bản.
- Với Bộ luật Tố tụng dân sự, chỉ giữ phần liên quan đến lao động.
- Kiểm tra lại số Điều/Khoản/Điểm sau khi extract.

Output:

```text
data/curated/*.txt
```

---

### Phase 2: Rebuild chunks

- Parse Chương, Mục, Điều, Khoản, Điểm.
- Tạo chunk theo legal hierarchy.
- Gắn metadata đầy đủ cho mỗi chunk.
- Gắn topic, actor, issue_type.
- Tạo citation_text chuẩn.
- Xuất index records.

Output:

```text
artifacts/index/current.json
```

---

### Phase 3: Rebuild vector index

- Embed toàn bộ chunks mới.
- Upsert lại vào vector database.
- Không dùng chung index cũ nếu corpus đã thay đổi lớn.
- Tạo manifest mới để trace version.

Output:

```text
artifacts/index/
```

---

### Phase 4: Rebuild Neo4j graph

- Reset graph cũ.
- Build structural graph.
- Build citation/reference graph.
- Build concept/topic graph.
- Ghi summary build.

Command đề xuất:

```bash
python scripts/build_legal_graph.py \
  --index-path artifacts/index \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --neo4j-database neo4j \
  --reset \
  --with-concepts \
  --with-references
```

Output:

```text
artifacts/graph/legal_graph_build_summary.json
```

---

### Phase 5: Validate graph

Cần kiểm tra:

```text
Số lượng Document nodes
Số lượng Article nodes
Số lượng Clause nodes
Số lượng Point nodes
Số lượng EvidenceChunk nodes
Số lượng REFERENCES edges
Số lượng GUIDES / GUIDED_BY / DETAILS edges
Số lượng Concept nodes
Số lượng semantic edges
```

Các câu query kiểm tra trong Neo4j:

```cypher
MATCH (d:Legal_Document)
RETURN d.document_id, d.name, d.source_chunk_count;

MATCH (a:Legal_Article)-[r:REFERENCES]->(b)
RETURN a.name, type(r), b.name
LIMIT 50;

MATCH (a:Legal_Article)-[r:DETAILS]->(b:Legal_Article)
RETURN a.name, type(r), b.name
LIMIT 50;

MATCH (c:Evidence_Chunk)
RETURN count(c);
```

---

### Phase 6: Evaluate retrieval

Cần benchmark lại retrieval trước và sau khi dùng graph.

So sánh ít nhất các cấu hình:

```text
1. Vector-only retrieval
2. Hybrid retrieval
3. Graph-augmented retrieval
```

Metric nên dùng:

```text
Recall@k
Precision@k
MRR
Hit Rate
Context relevance
Citation correctness
Answer faithfulness
```

Nhóm câu hỏi benchmark nên có:

```text
Direct QA
Multi-hop QA
Exception-based QA
Procedure QA
Scenario-based QA
Comparison QA
```

---

## 11. Acceptance criteria

Task được coi là hoàn thành khi:

1. Corpus có đầy đủ các văn bản trong phạm vi.
2. Mỗi chunk có metadata pháp lý rõ ràng.
3. Chunk không phá vỡ cấu trúc Điều/Khoản/Điểm.
4. Neo4j graph có đủ Document, Article, Clause, Point, EvidenceChunk.
5. Các văn bản hướng dẫn được liên kết với Bộ luật Lao động 2019.
6. Cross-reference giữa các điều luật được parse tự động.
7. Graph expansion retrieve được các điều luật liên quan cho câu hỏi multi-hop.
8. Retrieval benchmark cho thấy graph-augmented retrieval tốt hơn vector-only ở nhóm câu hỏi khó.
9. Câu trả lời sinh ra có citation chính xác theo Điều/Khoản/Điểm.
10. Có file summary để trace graph build và index build.

---

## 12. Những việc không làm trong task này

Task này không bao gồm:

```text
Thiết kế UI graph visualization
Làm màn hình explorer cho graph
Tối ưu giao diện người dùng
Triển khai production server
Làm authentication/authorization
Tích hợp hệ thống pháp luật ngoài phạm vi Luật Lao động
```

Trọng tâm chỉ là:

```text
Corpus rebuild
Hierarchy-aware chunking
Knowledge graph reconstruction
Graph-augmented retrieval
Retrieval evaluation
```

---

## 13. Mô tả ngắn cho thesis/report

The legal corpus will be rebuilt using a hierarchy-aware chunking strategy that preserves the natural structure of Vietnamese legal documents, including chapters, sections, articles, clauses, and points. The new knowledge graph will use the Labor Code 2019 as the backbone and connect it with related decrees, circulars, and procedural provisions through structural, citation-based, and semantic relations. This graph will support graph-augmented retrieval for complex legal questions that require combining multiple connected legal provisions across different documents.
