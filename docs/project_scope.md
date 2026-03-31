# Scope Tuan 1-2

## De tai

AI Assistant for Vietnamese Labor Law: A Case Study on Employment Contract Termination.

## Pham vi nghien cuu

- Chu de trung tam: cham dut hop dong lao dong.
- Doi tuong su dung:
  - nguoi lao dong muon hieu quyen va nghia vu cua minh;
  - nguoi su dung lao dong muon doi chieu nghia vu phap ly co ban;
  - sinh vien/nghien cuu vien can mot he thong hoi dap co trich dan.

## Ngoai pham vi o giai doan nay

- Tu van phap ly ca nhan hoa.
- Toan bo luat lao dong ngoai chu de cham dut hop dong.
- Fine-tuning model.
- OCR quy mo lon.

## Cau hoi nghien cuu chinh

1. He thong RAG local co the tra loi cac cau hoi ve cham dut hop dong lao dong dua tren corpus phap ly Viet Nam chinh xac den muc nao?
2. Viec tach van ban theo dieu/khoan va gan metadata co cai thien kha nang truy hoi va trich dan khong?
3. Cac failure mode pho bien cua tro ly trong bai toan nay la gi?

## Tieu chi thanh cong cho MVP

- Co pipeline tai tao du lieu tu `corpus/raw`.
- Moi cau tra loi phai co can cu toi thieu o cap van ban va dieu.
- He thong biet tu choi khi khong du can cu.
- Co bo case thu nghiem de danh gia retrieval va answer quality.

## Deliverable ket thuc tuan 2

- Tai lieu scope va roadmap.
- Repo co cau truc ro rang cho data, source code, test va docs.
- Script build corpus tu dong.
- Metadata xac dinh van ban `ready` va van ban `needs_ocr`.
- Ollama smoke test dung UTF-8.

## Rui ro can quan ly som

- PDF scan lam mat kha nang ingest tu dong.
- Text extract tieng Viet co the bi dinh lien tu va mat dau cach.
- Corpus hien tai con hep, de gay missing citation khi hoi cac truong hop ngoai du lieu.
