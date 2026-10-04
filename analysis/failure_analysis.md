# Phân tích kết quả RAG — Lab 18

**Bùi Hoàng Anh · 04/10/2026**

Đã chạy 20 câu hỏi cho mỗi pipeline. Gemini 3.1 Flash Lite sinh đáp án và chấm RAGAS; embedding và reranker chạy cục bộ. Bản production có thêm chunk cha–con, hybrid search và enrichment.

| Chỉ số RAGAS | Baseline | Production | Thay đổi |
|---|---:|---:|---:|
| Faithfulness | 0,8875 | 0,9125 | +0,0250 |
| Answer Relevancy | 0,7972 | 0,7594 | −0,0378 |
| Context Precision | 0,6500 | 0,8000 | +0,1500 |
| Context Recall | 0,8000 | 0,9250 | +0,1250 |

**Nhìn chung:** Truy hồi tốt hơn rõ nhất ở precision và recall. Answer Relevancy lại giảm nhẹ; cải thiện truy hồi chưa bảo đảm câu trả lời đầy đủ. Thời gian truy vấn production trung bình **6,16 giây/câu** (không tính index và chấm RAGAS).

## Năm câu điểm thấp nhất

### 1. Senior 9 năm: phép và lương
RAGAS cho Answer Relevancy và Context Precision bằng 0, Context Recall 0,5. Đáp án nói đúng **18 ngày phép** nhưng thiếu **lương Senior 20–35 triệu**. **Error Tree:** đáp án thiếu → context thiếu bảng lương → câu hỏi có hai ý nhưng top-2 đều là tài liệu nghỉ phép. **Sửa:** tách truy vấn phép/lương, lấy ít nhất một nguồn cho mỗi ý rồi mới sinh đáp án.

### 2. Phân loại thông tin lương
Đáp án “Bí mật” đúng, nhưng chưa nêu **cấp 3** và quy tắc xử lý dù context đã có; Answer Relevancy bị chấm 0. **Error Tree:** context đúng → đáp án quá ngắn → thiếu chi tiết so với ground truth. **Sửa:** yêu cầu trả lời cả nhãn, cấp độ và cách xử lý; kiểm tra thủ công điểm 0 vì câu trả lời hiện tại không sai về nhãn.

### 3. Ngày phép theo thâm niên
Đáp án đúng chính sách 2024 (**3 năm thêm 1 ngày**) nhưng bản 2023 lại đứng đầu context, khiến Context Precision bằng 0. **Error Tree:** đáp án đúng → context lẫn văn bản hết hiệu lực → lỗi chọn phiên bản. **Sửa:** gắn ngày hiệu lực, ưu tiên bản mới và chỉ lấy bản cũ khi câu hỏi yêu cầu so sánh.

### 4. Hoàn chi khóa học 25 triệu
Đáp án **hoàn trả 25 triệu** khớp ground truth; tài liệu cũng ghi nghỉ trước một năm phải trả **100% chi phí**. Faithfulness vẫn bị chấm 0. **Error Tree:** điểm thấp → kiểm tra context thấy có căn cứ → có thể là lỗi đánh giá phép suy ra 100% × 25 triệu; context thứ hai về nghỉ ốm là nhiễu. **Sửa:** loại context nhiễu, yêu cầu dẫn câu quy định và kiểm tra lại ca này bằng tay trước khi kết luận hallucination.

### 5. Chu kỳ đổi mật khẩu
Đáp án đúng **120 ngày** theo bản hiện hành, nhưng bản mật khẩu cũ vẫn xuất hiện ở context thứ hai; Context Precision bị chấm 0. **Error Tree:** đáp án đúng → có context cũ không cần thiết → lỗi lọc phiên bản. **Sửa:** bỏ hoặc giảm hạng chính sách đã bị thay thế khi hỏi quy định hiện hành.

**Nếu cải thiện thêm một bước:** ưu tiên xử lý câu hỏi nhiều ý và lọc văn bản cũ trước; sau đó chạy lại cùng 20 câu. Với các điểm 0 dù đáp án có căn cứ, đọc context và câu trả lời trước khi sửa pipeline theo chỉ số một cách máy móc.
