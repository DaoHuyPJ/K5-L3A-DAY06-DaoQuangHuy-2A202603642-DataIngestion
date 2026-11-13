# Bài làm Day 06 · Bộ luật Lao động 2019 → Chroma

Pipeline ingest + benchmark truy xuất cho **Bộ luật Lao động số 45/2019/QH14**.

- Vector store: **Chroma 1.5.9** (`PersistentClient`, lưu trên đĩa tại `starter/vectorstore/`).
- Embedding: **`intfloat/multilingual-e5-small`** chạy local (384 chiều, Sentence Transformers), không cần API key, chi phí 0.
- Không dùng PostgreSQL/pgvector của starter vì máy không có Docker và không đủ dung lượng để cài; `docker-compose.yml`, `init.sql` và `scripts/check_setup.py` là file gốc của starter, **không dùng**.

Báo cáo: [report/individual/2A202603642_DaoQuangHuy.md](../report/individual/2A202603642_DaoQuangHuy.md).

## Cấu trúc

Theo [SUBMISSION.md](../SUBMISSION.md): mã nguồn trong `starter/src/`; dữ liệu, benchmark và kết quả ở thư mục gốc repo.

```text
<repo>/
├── data/
│   ├── raw/                  # snapshot HTML toàn văn + PDF scan chính thức (đối chiếu)
│   ├── processed/units.jsonl # 84 đơn vị trang đã chuẩn hóa (page = số đơn vị)
│   └── manifest.json         # nguồn, quyền, quy tắc tính trang, danh sách đơn vị
├── benchmark/
│   ├── questions.json        # 23 câu (17 dev / 6 test), gold page + evidence + filter
│   └── verification.json     # kiểm chứng evidence trước khi chạy
├── results/
│   ├── logs/ingest_log.json  # trang PDF không có text, phần bị loại, cảnh báo
│   ├── chunks/               # toàn bộ chunk A/B kèm metadata
│   ├── index_report.json     # model, phiên bản, số chunk, độ dài, thời gian, chi phí
│   ├── reindex_check.json    # nạp lại không nhân đôi + xóa chunk cũ khi tài liệu đổi
│   ├── selection.md          # lựa chọn chiến lược, chốt trên dev trước khi chạy test
│   ├── traces/               # top-3 đầy đủ cho từng câu × chiến lược
│   ├── metrics_dev.json / metrics_test.json
│   ├── manual/               # nhãn coherence / metadata / grounding
│   ├── validation.json       # kết quả `validate`
│   ├── analysis.json         # kết quả `analyze`
│   ├── hybrid/               # sweep trọng số trên dev + trace hybrid
│   └── ablation/             # A800, B0: chunk, trace dev, báo cáo tách yếu tố
├── artifacts/experiment_log.csv
├── report/individual/2A202603642_DaoQuangHuy.md
└── starter/
    ├── requirements.txt      # phiên bản đã dùng cho lần chạy báo cáo
    ├── .env.example          # copy thành .env
    ├── src/
    │   ├── ingestion.py      # đọc HTML/PDF, chuẩn hóa NFC, dựng đơn vị trang
    │   ├── chunking.py       # A: fixed-size · B: theo đoạn trong từng Điều
    │   ├── embeddings.py     # E5 local (prefix query:/passage:, normalize)
    │   ├── store.py          # Chroma: upsert idempotent, search + where filter, exact check
    │   ├── evaluation.py     # Hit@1 / Evidence Hit@3, trace, summarize
    │   ├── hybrid.py         # mở rộng: BM25 tự cài + weighted RRF
    │   └── run.py            # CLI điều phối toàn bộ pipeline
    └── vectorstore/          # Chroma (không commit; tạo lại bằng `index`)
```

## Setup (Windows PowerShell; macOS/Linux tương tự)

Yêu cầu: Python 3.11–3.14, khoảng 1,5 GB trống (venv CPU + model).

```powershell
cd starter
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Model được tải tự động từ Hugging Face trong lần chạy đầu (~470 MB). Có thể tải trước vào một thư mục rồi đặt
`EMBEDDING_MODEL_PATH` trong `.env`. Thư mục đó cần có `modules.json` và file `REVISION` chứa
`614241f622f53c4eeff9890bdc4f31cfecc418b3` (revision đã dùng). `HF_HOME` đổi nơi cache.

## Chạy

```powershell
python -m src.run ingest          # ../data/raw -> ../data/processed/units.jsonl, ../data/manifest.json
python -m src.run benchmark       # kiểm chứng evidence, điền gold_page (đã chạy trước thí nghiệm)
python -m src.run index           # chunk A, B -> embed -> upsert Chroma
python -m src.run reindex-check   # nạp lại cùng phiên bản + mô phỏng tài liệu đổi
python -m src.run evaluate --split dev
python -m src.run evaluate --split test   # chạy sau khi đã chốt lựa chọn trên dev
python -m src.run audit-sheets    # sinh bảng mẫu chấm thủ công
python -m src.run validate        # kiểm tra toàn vẹn: span, phủ ký tự, ranh giới Điều, store
python -m src.run analyze         # phân tích theo loại câu, MRR, token, latency
python -m src.run hybrid          # mở rộng: BM25 + dense (RRF), chỉnh trọng số trên dev
python -m src.run ablation        # dev: tách yếu tố A -> A800 (kích thước) -> B0 (ranh giới) -> B (tên Điều)
```

Lưu ý: lệnh chạy từ `starter/`, kết quả ghi vào `../results/`. `audit-sheets` từ chối ghi đè khi `results/manual/*.csv` đã có nhãn (cần `--force`).

## Ba chiến lược

| | Chunking | Collection | Truy xuất |
|---|---|---|---|
| **A** | Cửa sổ cố định 500 ký tự, overlap 50, không theo cấu trúc | `bld2019_A` | top-3 cosine, không lọc |
| **B** | Gom đoạn liên tiếp **trong cùng một Điều**, tối đa 800 ký tự; đoạn quá dài tách theo câu; chunk tiếp nối được thêm dòng tên Điều | `bld2019_B` | top-3 cosine, không lọc |
| **C** | Giống hệt B (dùng chung collection) | `bld2019_B` | thêm `where chapter = …` khi câu hỏi có filter; câu không có filter chạy như B |

B↔C chỉ khác ở filter. A↔B khác ở chiến lược chunking (ranh giới, kích thước tối đa, dòng tên Điều).
Cùng corpus, model, store, metric và bộ câu hỏi.

## Kết quả (chi tiết trong báo cáo)

| | Dev Hit@1 | Dev Hit@3 | Test Hit@1 | Test Hit@3 | Coherence /2 | Grounding /2 |
|---|---|---|---|---|---|---|
| A | 10/17 | 14/17 | 3/6 | 4/6 | 0,80 | 1,50 |
| B | 14/17 | 16/17 | 5/6 | 5/6 | 1,80 | 1,67 |
| C | 14/17 | 16/17 | 6/6 | 6/6 | 1,80 | 2,00 |

## Quyết định chính

- **Nguồn text**: PDF chính thức trên chinhphu.vn là bản scan (82/83 trang không có lớp text). Không OCR; dùng bản
  HTML toàn văn do Trang PBGDPL TP. Cần Thơ (cơ quan nhà nước) đăng; PDF giữ lại để đối chiếu.
- **Đơn vị trang**: gom các Điều trọn vẹn liên tiếp thành đơn vị ≥1.500 ký tự, không vượt Chương. 84 đơn vị; 82 đơn vị
  được tính (Chương II và IX ngắn hơn 1.500 ký tự cả chương nên không tính, không độn).
- **Metadata**: `document_id`, `source`, `page`, `char_start/char_end`, `chapter` (lọc), `chapter_title`, `section`, `articles`.
- **Nạp lại**: ID ổn định + `content_hash`; chunk không còn sinh ra thì bị xóa. Lỗi backend/model được raise, **không có fallback sang vector giả**.
- **ANN**: Chroma dùng HNSW; mỗi truy vấn được so với brute-force, top-3 trùng 69/69 lượt.
