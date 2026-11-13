# Day 06 · Data Ingestion Engineering

## Bài làm — Đào Quang Huy (2A202603642, L3A)

Corpus **Bộ luật Lao động 2019 (45/2019/QH14)** · embedding local `intfloat/multilingual-e5-small` (384 chiều) ·
vector store **Chroma 1.5.9** lưu trên đĩa · chi phí 0 USD.
Báo cáo: [report/individual/2A202603642_DaoQuangHuy.md](report/individual/2A202603642_DaoQuangHuy.md) ·
chi tiết kỹ thuật: [starter/README.md](starter/README.md).

| Thành phần (SUBMISSION.md) | Đường dẫn |
|---|---|
| Thông tin học viên | [STUDENT.md](STUDENT.md) |
| Mã nguồn pipeline | [starter/src/](starter/src/) |
| Manifest dữ liệu (nguồn, quyền, trang, cách lấy) | [data/manifest.json](data/manifest.json) |
| Corpus: snapshot gốc + đơn vị trang đã chuẩn hóa | [data/raw/](data/raw/) · [data/processed/units.jsonl](data/processed/units.jsonl) |
| Benchmark: 23 câu, split dev/test, gold evidence | [benchmark/questions.json](benchmark/questions.json) · [benchmark/verification.json](benchmark/verification.json) |
| Nhật ký cấu hình / giả thuyết A·B·C | [artifacts/experiment_log.csv](artifacts/experiment_log.csv) |
| Thống kê, trace top-3, nhãn audit, log | [results/](results/) (`traces/`, `manual/`, `logs/`, `metrics_*.json`, `selection.md`) |

### Setup (Windows PowerShell; macOS/Linux tương tự)

Yêu cầu Python 3.11–3.14 và khoảng 1,5 GB trống. Không cần Docker, không cần API key.

```powershell
cd starter
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env        # model tải tự động từ Hugging Face lần đầu (~470 MB)
```

### Chạy (từ thư mục `starter/`)

```powershell
python -m src.run ingest          # data/raw -> data/processed/units.jsonl + data/manifest.json
python -m src.run index           # chunk A/B -> embed -> Chroma (starter/vectorstore/)
python -m src.run reindex-check   # nạp lại không nhân đôi, xóa chunk cũ
python -m src.run evaluate --split dev
python -m src.run evaluate --split test
python -m src.run validate        # kiểm tra toàn vẹn
python -m src.run analyze         # phân tích theo loại câu
python -m src.run hybrid          # mở rộng tùy chọn BM25 + dense
```

Kết quả được ghi vào `results/` ở thư mục gốc repo. Nhãn thủ công nằm trong `results/manual/`; `audit-sheets` từ chối ghi đè nhãn đã có.

| | Dev Hit@1 | Dev Hit@3 | Test Hit@1 | Test Hit@3 | Coherence /2 | Grounding /2 |
|---|---|---|---|---|---|---|
| A · fixed 500/50 | 10/17 | 14/17 | 3/6 | 4/6 | 0,80 | 1,50 |
| B · theo Điều ≤800 | 14/17 | 16/17 | 5/6 | 5/6 | 1,80 | 1,67 |
| C · B + lọc Chương | 14/17 | 16/17 | 6/6 | 6/6 | 1,80 | 2,00 |

---

*Phần dưới đây là đề bài gốc.*

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
