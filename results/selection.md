# Lựa chọn chiến lược — chốt trên dev, TRƯỚC khi chạy test

Ghi lúc: 2026-11-13, sau `python -m src.run evaluate --split dev`, trước `--split test`.

Tiêu chí đã chốt trong `benchmark/questions.json` (`selection_criterion`): Evidence Hit@3 dev →
Hit@1 dev → coherence thủ công → ít chunk / thời gian index.

| Strategy | Dev Hit@1 | Dev Evidence Hit@3 | Top-3 tuân thủ filter (câu có filter) |
|---|---|---|---|
| A | 10/17 | 14/17 | 0/3 |
| B | 14/17 | 16/17 | 0/3 |
| C | 14/17 | 16/17 | 3/3 |

- A bị loại ở bước 1 (Hit@3 14/17 < 16/17).
- B và C hòa ở mọi bước của tiêu chí: cùng Hit@3, cùng Hit@1, cùng chunk (coherence giống hệt), cùng index
  (số chunk, thời gian như nhau). Tiêu chí chốt trước **không phân định được B và C**.
- Quyết định: chọn **C**, vì C không kém B ở chỉ số nào và hơn B ở tỷ lệ top-3 nằm đúng phạm vi filter
  (3/3 so với 0/3) — một chỉ số thuộc khía cạnh metadata usefulness trong GUIDE. Lý do bổ sung này được ghi
  ở đây trước khi xem test; không có tham số nào được chỉnh sau khi xem dev.
- Không thay đổi chunking, model, câu hỏi hay tham số nào sau khi xem kết quả dev.
