# Báo cáo cá nhân — Day 6

**Đào Quang Huy · 2A202603642 · L3A** — Corpus: Bộ luật Lao động 2019 · Store: Chroma · Embedding: multilingual-e5-small

Mọi đường dẫn bên dưới tính từ thư mục `starter/`. Lệnh tái lập: xem [starter/README.md](../../starter/README.md).

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang.**
  - Văn bản: Bộ luật Lao động số 45/2019/QH14 (Quốc hội thông qua 20/11/2019, hiệu lực 01/01/2021), bản gốc như được ban hành, **không phải văn bản hợp nhất**.
  - Nguồn text: toàn văn HTML trên Trang PBGDPL TP. Cần Thơ (`https://pbgdpl.cantho.gov.vn/attachment/261`, truy cập 13/11/2026), snapshot lưu tại `data/raw/bld2019_pbgdpl_cantho.html` (sha256 `d72a20c8…`). Bản chính thức trên chinhphu.vn (`45.signed.pdf`, 83 trang) chỉ dùng để đối chiếu vì đó là bản scan: 82/83 trang không có lớp text (`results/logs/ingest_log.json`). Mình không OCR.
  - Quyền: văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả (Điều 15 Luật SHTT). Nguồn do cơ quan nhà nước đăng công khai, không chứa dữ liệu cá nhân.
  - Quy tắc tính trang: HTML không phân trang, nên các **Điều trọn vẹn liên tiếp** được gom thành "đơn vị trang" ≥1.500 ký tự sau chuẩn hóa NFC, không vượt ranh giới Chương. Kết quả: **84 đơn vị, 188.880 ký tự, 82 đơn vị ≥1.500 ký tự được tính**. Hai đơn vị không tính là đơn vị 6 (toàn bộ Chương II, 1.485 ký tự) và đơn vị 53 (toàn bộ Chương IX, 997 ký tự); mình không độn để đủ ngưỡng. `page` trong metadata và benchmark là số đơn vị. Danh sách đầy đủ ở `data/manifest.json`.
- **Chuẩn hóa, lỗi và giới hạn.**
  - Chuẩn hóa: Unicode NFC; bỏ NBSP/zero-width; gộp khoảng trắng; giữ xuống dòng giữa các đoạn `<p>`.
  - Parser kiểm tra đủ 220 Điều liên tục (thiếu thì dừng). Phần đầu trước Chương I (quốc hiệu, căn cứ; 8 đoạn) và chữ ký/footer sau Điều 220 (2 đoạn) bị loại, có ghi log.
  - Giới hạn:
    - Không phản ánh các lần sửa đổi sau 2019.
    - `page` là đơn vị tự định nghĩa, không phải số trang PDF Công báo.
    - Điều 219 chứa nguyên văn điều khoản của luật khác (Luật BHXH, BLTTDS), dễ nhầm khi trích dẫn.
- **Embedding, vector store, metric, lệnh chạy và chi phí.**
  - Embedding: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` (sha256 file model đã đối chiếu với Hugging Face). Model chạy local bằng sentence-transformers 6.3.0 / torch 2.15.0+cpu, **384 chiều**, max 512 token (không chunk nào bị cắt; dài nhất 238 token). Document dùng prefix `passage: `, query dùng `query: `, vector được L2-normalize.
  - Vector store: **Chroma 1.5.9** `PersistentClient` (lưu trên đĩa tại `starter/vectorstore/`, ~8 MB), collection `bld2019_A` và `bld2019_B`, metric `hnsw:space=cosine`. Search là **approximate (HNSW)**; mỗi truy vấn được so với brute-force numpy: top-3 trùng **51/51 (dev) và 18/18 (test)**.
  - Lý do bỏ pgvector: máy không có Docker và không đủ dung lượng để cài Docker Desktop + WSL2.
  - Lệnh: `python -m src.run ingest | benchmark | index | reindex-check | evaluate --split dev|test | audit-sheets | validate | analyze | hybrid`. Pipeline tất định: không lấy mẫu ngẫu nhiên, cùng dữ liệu và model cho cùng kết quả, nên không cần seed. `evaluate --split test` đã được chạy lại một lần lúc 17:50; top-3 và Hit giống hệt lần đầu, chỉ latency khác.
  - Thời gian index (CPU): A 16,9 s, B 16,2 s. **Chi phí 0 USD.**
- **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn.**
  - Bắt buộc: `document_id` (`BLLD-2019`), `source` (URL nguồn), `page` (số đơn vị, int), cộng thêm `char_start`/`char_end` để truy vết tới từng ký tự.
  - Bổ sung: `chapter` (str, số La Mã) + `chapter_title`, `section` (Mục, str, rỗng nếu không có hoặc chunk trải nhiều Mục), `articles` (danh sách số Điều), `content_hash`. Tất cả lấy từ heading "Chương/Mục/Điều" trong văn bản, không suy đoán.
  - Chọn `chapter` làm trường lọc vì Chương có giá trị đáng tin, 17 giá trị phân biệt, và tách được các quy định trùng từ ngữ (ví dụ "báo trước", "thời hiệu", "giờ làm việc") giữa HĐLĐ chung với nhóm lao động đặc thù.

## Cấu hình và benchmark

- **A / B / C** (`STRATEGIES` trong `src/run.py`):
  - **A**: cửa sổ cố định 500 ký tự, overlap 50, không theo cấu trúc, không lọc. 451 chunk, trung bình 460 ký tự.
  - **B**: gom các đoạn liên tiếp **trong cùng một Điều**, tối đa 800 ký tự; đoạn quá dài thì tách theo câu (`.;:`); chunk tiếp nối được thêm dòng tên Điều (141/361 chunk); không lọc. 361 chunk, trung bình 548 ký tự.
  - **C**: dùng chính index của B; **khác biệt duy nhất B–C** là `where chapter = …` khi câu hỏi có filter. Câu không có filter chạy y hệt B.
  - Lưu ý: A–B khác nhau về cả ranh giới chunk, kích thước tối đa (500 so với 800) và dòng tên Điều. Đó là so sánh "chiến lược chunking", không phải một biến duy nhất.
- **≥20 câu.**
  - `benchmark/questions.json` có **23 câu: 17 dev / 6 test**, chia trước khi chạy bất kỳ truy xuất nào. Loại câu: trực tiếp, diễn đạt lại, điều kiện/ngoại lệ, và 5 câu metadata filter (q15, q16, q17 thuộc dev; t05, t06 thuộc test). Câu test dùng các Điều không xuất hiện trong dev.
  - Mỗi evidence là trích nguyên văn, được `python -m src.run benchmark` kiểm chứng tồn tại trong corpus đã chuẩn hóa; script cũng kiểm tra giá trị filter có thật và trang gold nằm trong phạm vi filter (`benchmark/verification.json`: 23/23, mỗi câu khớp đúng một đơn vị).
  - Bộ câu hỏi được soạn với hỗ trợ của AI. Filter mô phỏng việc người dùng đang tra trong một Chương, được cung cấp trước khi truy xuất; không đưa đáp án hay trang gold vào query.
- **Tiêu chí chọn chiến lược đã chốt trước khi xem test**: Evidence Hit@3 dev → Hit@1 dev → coherence → ít chunk/thời gian. Quyết định được ghi tại `results/selection.md` trước khi chạy test.

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian index | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 10/17 | 14/17 | 451 | 16,9 s | 0 |
| B | dev | 14/17 | 16/17 | 361 | 16,2 s | 0 |
| C | dev | 14/17 | 16/17 | 361 (dùng chung B) | 0 (dùng index B) | 0 |
| A | test | 3/6 | 4/6 | 451 | | 0 |
| B | test | 5/6 | 5/6 | 361 | | 0 |
| C | test | **6/6** | **6/6** | 361 | | 0 |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | 0,80 (n=10; 0/1/2 = 2/8/0) | 18/18 | dev 0/3 · test 1/2 | 1/1/4 · 1,50 |
| B | 1,80 (n=10; 0/1/2 = 0/2/8) | 18/18 | dev 0/3 · test 1/2 | 1/0/5 · 1,67 |
| C | 1,80 (dùng chung chunk B) | 18/18 | dev 3/3 · test 2/2 | 0/0/6 · 2,00 |

"Filter tuân thủ" = số câu có filter mà **cả 3** kết quả top-3 nằm đúng Chương. Với A/B (không lọc), con số này cho thấy tìm kiếm không lọc thường để lẫn chunk ngoài phạm vi.

**Đường dẫn trace, nhãn thủ công và cách chọn mẫu.**
- Trace đầy đủ (câu hỏi, chunk ID, text, score, metadata, cờ hit, cờ nghi tách evidence, ANN = exact): `results/traces/traces_dev.jsonl`, `results/traces/traces_test.jsonl`. Chỉ số: `results/metrics_dev.json`, `results/metrics_test.json`. Index và nạp lại: `results/index_report.json`, `results/reindex_check.json`.
- Nhãn thủ công nằm trong `results/manual/`:
  - `coherence_labels.csv`: 10 vị trí cố định chọn trước khi xem điểm, là ký tự giữa của 10 đơn vị đầu (u001–u010). Mỗi vị trí chấm chunk A và chunk B chứa nó; C trùng B.
  - `metadata_audit.csv`: 6 câu test × top-3 × 3 chiến lược = 54 bản ghi. Mỗi bản ghi có đối chiếu tự động (text tại span đúng trang, chapter khớp đơn vị, các Điều có trong đơn vị) cộng kiểm tra thủ công.
  - `grounding_labels.csv`: 6 câu test × 3 chiến lược.
- **Người chấm**: AI (Claude) chấm nháp theo thang trong GUIDE, mỗi nhãn kèm lý do. Tác giả đã rà soát toàn bộ ngày 13/11/2026, tập trung vào 11 nhãn ở ranh giới (coherence P01/P09/P10 A, P03/P10 B, P04/P08/P09 B; grounding A t01/t03/t04/t05), và đồng ý giữ nguyên tất cả (cột `annotator`). Đây là chấm không mù: người chấm biết chunk thuộc chiến lược nào và đã thấy Hit@k, nên có nguy cơ thiên lệch.

**Nạp lại**: nạp lại cùng phiên bản cho 451/451 và 361/361 chunk `unchanged`, số bản ghi không đổi. Mô phỏng tài liệu đổi (xóa Điều 2 khỏi đơn vị 1 trong collection tạm) cho kết quả 5 chunk cập nhật, 7 giữ nguyên, 1 chunk cũ bị xóa; store sau đó khớp đúng 12 ID của phiên bản mới.

**Kiểm tra toàn vẹn** (`python -m src.run validate` → `results/validation.json`, tất cả đạt):
- Đơn vị trang: Điều 1–220 mỗi Điều xuất hiện đúng một lần; không đơn vị nào vượt Chương; số ký tự khớp manifest.
- Chunk: text của mọi chunk A/B khớp đúng span nguồn; phủ 100% ký tự; ID không trùng.
- Ranh giới Điều: **134/451 chunk A trải qua ≥2 Điều**, B là 0/361.
- Evidence của benchmark vẫn nằm đúng trang gold; ID và hash trong Chroma trùng với file chunk.

**Phân tích bổ sung** (`python -m src.run analyze` → `results/analysis.json`; gộp dev + test, N = 23):

| Loại câu (N) | A Hit@1 / Hit@3 | B Hit@1 / Hit@3 | C Hit@1 / Hit@3 |
|---|---|---|---|
| Trực tiếp (8) | 4/8 · 7/8 | 7/8 · 8/8 | 7/8 · 8/8 |
| Diễn đạt lại (5) | 3/5 · 5/5 | 4/5 · 5/5 | 4/5 · 5/5 |
| Điều kiện/ngoại lệ (5) | 3/5 · 3/5 | 4/5 · 4/5 | 4/5 · 4/5 |
| Metadata filter (5) | 3/5 · 3/5 | 4/5 · 4/5 | **5/5 · 5/5** |
| **Tổng** (MRR@3) | 13/23 · 18/23 (0,66) | 19/23 · 21/23 (0,87) | 20/23 · 22/23 (0,91) |

- Câu điều kiện/ngoại lệ khó nhất với mọi chiến lược, vì evidence dài và nhiều mệnh đề.
- A thua nhiều nhất ở câu trực tiếp (Hit@1 4/8): các chunk lẫn nhiều Điều làm loãng vector.
- Tài nguyên: A 451 chunk / 54.778 token, B 361 chunk / 51.784 token (đếm bằng tokenizer của e5 kèm prefix). Thời gian index ~16 s mỗi chiến lược trên CPU.
- Latency search trung bình (lần chạy test cuối): A 16,2 ms, B 9,0 ms, C 2,7 ms. Số liệu này có nhiễu do thứ tự chạy/warm-up, chỉ mang tính tham khảo.

## Mở rộng tùy chọn: hybrid BM25 + dense

`python -m src.run hybrid` → `results/hybrid/` (`src/hybrid.py`).
- Cách làm: giữ nguyên chunk, metadata và filter của C; thêm BM25 (Okapi, k1 = 1,5, b = 0,75, token theo âm tiết, tự cài đặt); gộp với dense bằng weighted RRF (k = 60, mỗi danh sách lấy top-20).
- Tham số: chỉ chỉnh trọng số BM25 `w ∈ {0; 0,25; 0,5; 0,75; 1; 1,5; 2}`, **chỉ trên dev**. Quy tắc định trước: Hit@3 dev cao nhất → Hit@1 → trọng số nhỏ nhất.

| w | Dev Hit@1 | Dev Hit@3 | Thay đổi thứ hạng so với w = 0 |
|---|---|---|---|
| 0 – 1,0 | 14/17 | 16/17 | không có |
| 1,5 – 2,0 | 14/17 | 16/17 | q05: 2 → 1 (tốt hơn) · **q01: 1 → 2 (xấu hơn)** |

- Kết quả chọn: **w = 0**, tức hybrid không được dùng; test giữ nguyên như C (6/6).
- Hybrid **làm xấu** q01 ("HĐ xác định thời hạn được ký tối đa bao nhiêu tháng?"): ở w = 1,5, BM25 xếp hạng 1 chunk Điều 155 "Thời hạn của giấy phép lao động tối đa là 02 năm" (khớp từ khóa "thời hạn… tối đa", dense chỉ xếp hạng 4), đẩy chunk Điều 20 chứa "không quá 36 tháng" xuống hạng 2. Đây là ví dụ điển hình của trùng từ khóa nhưng khác nghĩa.
- Hybrid không sửa được q06: ở w ≥ 1, chunk `u015:c000` (khoản 1 Điều 35) vào top-3 nhưng evidence nằm ở `u015:c001` (khoản 2 điểm b), nên vẫn không tính là hit.
- Kết luận: trên corpus này, từ khóa âm tiết không đủ phân biệt chủ thể "người lao động" với "người sử dụng lao động"; cần reranker cross-encoder hoặc tách từ tiếng Việt.
- Lưu ý: hybrid được thử **sau khi** đã xem kết quả test của A/B/C, nhưng không dùng test để chỉnh tham số.

## Phân tích và kết luận

- **So sánh A–B (chunking).**
  - B hơn A ở mọi khía cạnh: Hit@1 dev 14 so với 10, Hit@3 dev 16 so với 14; coherence 1,8 so với 0,8; grounding 1,67 so với 1,50; ít hơn 90 chunk; thời gian gần như nhau.
  - Nguyên nhân chính: A cắt ngang khoản/điểm, nên evidence dài bị tách qua hai chunk (cờ nghi tách: q02, q06, t05) và chunk mất chủ thể (bắt đầu giữa từ, không có tên Điều).
  - A vẫn đủ tốt với câu ngắn, evidence gọn trong một dòng (q03, q04, q08–q10, q12: Hit@1).
- **So sánh B–C (metadata filter).**
  - Trên dev, cả 3 câu filter B đã trúng top-1 nên Hit không đổi. Filter chỉ làm sạch top-2/3 (tuân thủ 0/3 → 3/3).
  - Trên test, filter quyết định kết quả ở t06. Ngoài ra filter giảm thời gian search, nhưng số đo latency có nhiễu do thứ tự chạy/warm-up nên mình không dùng để kết luận.
  - Hạn chế: filter chỉ có ích khi người dùng thật sự cung cấp phạm vi đúng; filter sai Chương sẽ loại mất evidence.
- **Hai ví dụ sai.**
  1. **q06 (dev, gold BLLD-2019 p15, Điều 35 khoản 2 điểm b)** — "Bị trả lương thiếu/chậm có được nghỉ ngay không cần báo trước không?". Cả A, B, C đều trượt. B trả về Điều 36 (quyền của **người sử dụng lao động**), Điều 41, Điều 115. Embedding bắt từ khóa "đơn phương chấm dứt", "báo trước", "lương" nhưng không phân biệt được chủ thể NLĐ hay NSDLĐ. Ở A, chunk Điều 35 bị cắt ngay giữa điểm b. Chunking tốt hơn không sửa được lỗi ngữ nghĩa này; cần hybrid/BM25 hoặc reranker.
  2. **t06 (test, gold p64, Điều 162 khoản 2)** — "Muốn đơn phương chấm dứt hợp đồng thì phải báo trước ít nhất bao nhiêu ngày?". A và B lấy Điều 35/36/38 (Chương III, 45/30/03 ngày), nên câu trả lời sinh ra từ đó sẽ **sai nhưng trông hợp lý** (grounding 0). C lọc Chương XI và lấy đúng Điều 162 (15 ngày, người giúp việc gia đình) ở top-1.
  - Thêm một ví dụ: **t05 với A**. Chunk chứa "Tòa án … là 01 năm kể từ ngày phát hiện ra hành vi…" bị cắt vài chữ cuối, nên Evidence Hit@3 tự động = miss, dù người chấm thấy đủ để trả lời (grounding 2). Định nghĩa Hit tự động chặt hơn đánh giá thủ công.
- **Chiến lược chọn, đánh đổi, điều kiện áp dụng.**
  - Chọn **C**, tức chunk theo cấu trúc Điều + lọc Chương khi có ngữ cảnh. Tiêu chí chốt trước loại được A nhưng hòa giữa B và C; C được chọn vì không kém B ở chỉ số nào và hơn ở tuân thủ filter (ghi trước khi xem test).
  - Đánh đổi: B/C phụ thuộc vào việc parse được cấu trúc Chương/Điều. Chunk có kích thước không đều (101–887 ký tự). Dòng tên Điều thêm vào làm chunk dài hơn chút. Filter cần người dùng/ngữ cảnh cho phạm vi đúng.
  - Điều kiện áp dụng: văn bản có cấu trúc heading rõ (luật, quy chế, handbook). Với văn bản không có cấu trúc, A là baseline an toàn hơn.
- **Hạn chế.**
  - Test chỉ có 6 câu, dev 17 câu: chênh lệch 1 câu = 17% (test) hoặc 6% (dev), nên kết quả chỉ mang tính minh họa trên corpus này, không suy rộng.
  - Một văn bản, cùng văn phong, nên dev/test vẫn có rủi ro rò rỉ về cách diễn đạt.
  - Câu hỏi và filter do tác giả (với hỗ trợ AI) thiết kế. Câu filter ở dev hóa ra dễ, vì B đã trúng top-1 sẵn.
  - Nhãn coherence/grounding do AI chấm nháp và một người (tác giả) rà soát; không có người chấm thứ hai độc lập, không chấm mù, mẫu nhỏ (10 vị trí, 6 câu).
  - Thời gian đo một lần trên CPU, gồm cả warm-up.
  - Corpus là bản gốc 2019, không gồm các sửa đổi sau đó.
- **Nếu làm lại hoặc mở rộng.**
  1. Hybrid BM25 đã thử nhưng không cải thiện (xem mục mở rộng). Bước tiếp theo là reranker cross-encoder đa ngữ trên top-20, cộng tách từ tiếng Việt cho BM25, để xử lý lỗi kiểu q06.
  2. Dùng văn bản hợp nhất mới nhất kèm `effective_date`/`version` để lọc theo hiệu lực.
  3. Thêm metadata `subject` (NLĐ/NSDLĐ) và cờ "văn bản được sửa đổi" cho Điều 219.
  4. Tăng test lên ≥20 câu, có người chấm thứ hai độc lập và chấm mù chiến lược.
  5. Thử model lớn hơn (e5-base/bge-m3) khi có thêm dung lượng.

## Công cụ AI

Dùng Claude (Claude Code) để viết code pipeline, soạn nháp bộ câu hỏi và chấm nháp nhãn thủ công. Tác giả đã rà soát nhãn trước khi nộp. Cách kiểm tra:
- Evidence được kiểm chứng tự động với corpus.
- Span của mọi chunk được kiểm tra khớp nguồn, phủ 100% ký tự, B không vượt ranh giới Điều.
- Kết quả ANN được đối chiếu với brute-force.
- Toàn bộ số liệu trong báo cáo lấy từ file kết quả đã chạy.
