# Reflection — Lab 18

**Bùi Hoàng Anh · MSSV 2A202602697 · K4 Track 3A · 04/10/2026**

## 1. Từ bài giảng đến code

| Khái niệm | Chỗ áp dụng | Điều rút ra |
|---|---|---|
| Semantic chunking | `chunk_semantic()` (M1) | Ghép câu gần nghĩa, nhưng ngưỡng 0,85 cần thử trên văn bản tiếng Việt trước khi dùng rộng rãi. |
| Parent–child và cấu trúc | `chunk_hierarchical()`, `chunk_structure_aware()` (M1) | Tìm child rồi trả parent giữ đủ ngữ cảnh; bảng và tiêu đề không nên bị cắt ngang. |
| BM25 + Dense | `reciprocal_rank_fusion()` (M2) | BM25 giữ từ khóa chính xác; dense bổ sung tìm kiếm ngữ nghĩa, RRF gộp hai thứ hạng. |
| Cross-encoder | `CrossEncoderReranker.rerank()` (M3) | Chấm lại top 20 để chọn tối đa 3 đoạn, đổi thêm độ trễ lấy context gọn hơn. |
| RAGAS | `evaluate_ragas()`, `failure_analysis()` (M4) | Context Precision tăng từ 0,65 lên 0,80; Context Recall từ 0,80 lên 0,925. Answer Relevancy giảm từ 0,7972 xuống 0,7594, nên cần sửa câu trả lời nhiều ý. |
| Contextual enrichment | `_enrich_single_call()` (M5) | Thêm nguồn và bối cảnh vào chunk; một lời gọi Gemini tạo tóm tắt, câu hỏi và metadata. |

## 2. Vướng mắc và cách xử lý

- BM25 từng khó khớp cụm “nghỉ phép”: chuẩn hóa dấu gạch dưới của từ ghép về khoảng trắng khi index và tìm kiếm.
- `ModuleNotFoundError: langchain_community.chat_models.vertexai` do RAGAS 0.4.3 không hợp `langchain-community` 0.4.2 trong môi trường hiện tại; dùng bản 0.3.31 và kiểm tra lại trên một câu trước khi chạy toàn bộ.
- Gemini trả `Multiple candidates is not enabled for this model` khi RAGAS yêu cầu ba đáp án một lượt; cấu hình wrapper tạo ba lượt riêng. Sau đó gặp HTTP 429 ở giới hạn 15 yêu cầu/phút, nên giới hạn tốc độ chung cho các lời gọi.
- Hai PDF scan không có text layer, vì vậy tạm bỏ qua. Muốn truy hồi nội dung này phải OCR và kiểm tra lại bản trích xuất.

## 3. Kế hoạch áp dụng

Với trợ lý hỏi đáp chính sách nhân sự, tôi sẽ làm theo thứ tự:

1. **Tuần 1:** Chuẩn hóa tài liệu, ghi ngày hiệu lực và phiên bản; chunk theo mục, giữ bảng biểu nguyên vẹn.
2. **Tuần 2:** Dùng child để tìm kiếm BM25 + BGE-M3, rồi trả parent; lọc bản cũ khi người dùng hỏi quy định hiện hành.
3. **Tuần 3:** Rerank và tách câu hỏi đa ý thành các truy vấn riêng; câu trả lời phải nêu nguồn và không tự bổ sung quy định.
4. **Tuần 4:** Lập bộ câu hỏi có ground truth, chạy RAGAS định kỳ, đọc các ca điểm thấp rồi sửa đúng bước gây lỗi. Chỉ giữ enrichment nếu phép đo cho thấy lợi ích xứng với chi phí API.
