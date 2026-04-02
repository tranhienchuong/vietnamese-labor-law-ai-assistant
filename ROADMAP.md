# Roadmap 8 Tuan

## Tuan 1-2: Nen tang du lieu va scope

- Khoa pham vi case study vao `employment contract termination`.
- Chuan hoa cau truc repo, tai lieu scope, quickstart.
- Xay pipeline `extract -> clean -> section -> chunk -> metadata`.
- Danh dau duoc van ban nao can OCR.

## Tuan 3-4: Baseline RAG

- Tao retrieval layer theo huong Qdrant native hybrid search:
  - embed `retrieval_text` bang sentence-transformers cho dense retrieval;
  - tao sparse vectors tu van ban da word-segment bang PyVi;
  - luu dense + sparse tren cung mot Qdrant collection;
  - dung hybrid search va RRF ngay trong Query API cua Qdrant.
- Them query understanding truoc retrieval:
  - phan loai nhanh cau hoi theo `topic`, `actor`, `issue_type`;
  - dung metadata pre-filter de khoa pham vi tim kiem truoc khi query vector DB.
- Them small-to-big context assembly co deduplication:
  - lay top-k chunk nho;
  - thu thap `parent_chunk_id` duy nhat;
  - truy xuat parent theo id map O(1) tu SQLite;
  - mo rong context theo parent ma khong lap lai chunk.
- Tao pipeline retrieve/generate voi Ollama co citation ro rang:
  - tra ve can cu theo van ban, dieu, khoan, diem;
  - ep format output co truong `co_so_phap_ly`;
  - neu context khong ghi ro dieu/khoan thi khong duoc tu sinh so dieu.
- Chuan bi CLI hoac script hoi dap toi thieu de test end-to-end truoc khi lam giao dien.

## Tuan 5-6: Evaluation

- Xay bo tinh huong phap ly cho cham dut hop dong lao dong.
- Do chat luong retrieval, citation, do huu ich cua cau tra loi.
- Do rieng:
  - dense vs sparse vs hybrid trong cung Qdrant collection;
  - ty le retrieve dung Dieu/Khoan/Diem;
  - ty le lap context sau khi expand parent;
  - hallucination rate o phan trich dan.
- Sua loi theo cac failure mode chinh.

## Tuan 7-8: Demo va bao cao

- Hoan thien giao dien demo.
- Viet methodology, thiet ke he thong, thuc nghiem, ket qua va han che.
- Chuan bi slide va script demo.
