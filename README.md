# Xây dựng model nhận diện cảm xúc qua giọng nói
## 1. Mô tả bài toán
- **Input**: Một đoạn âm thanh có độ dài cố định.
- **Output**: Một trong 8 cảm xúc: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`.
- **Object function**: Đánh giá độ chính xác của model thông qua các metric như `accuracy`, `precision`, `recall`, `f1-score`.
- **Constrain**: 
    - Sử dụng các model Deep Learning.
    - Sử dụng MFCC để trích xuất đặc trưng.
    - Sử dụng dataset `RAVDESS` hoặc `SAVEE`.
## 2. Mô tả về bộ dữ liệu
Có nhiều bộ để tham khảo như:
- **RAVDESS**: Bộ dữ liệu gồm 1440 file âm thanh với 24 người thể hiện 8 cảm xúc khác nhau.
- **SAVEE**: Bộ dữ liệu gồm 480 file âm thanh với 4 người thể hiện 7 cảm xúc khác nhau.
- **TESS**: Bộ dữ liệu gồm 2800 file âm thanh với 2 người thể hiện 7 cảm xúc khác nhau.
- **CREMA-D**: Bộ dữ liệu gồm 7442 file âm thanh với 91 người thể hiện 12 cảm xúc khác nhau.
- **VN_emotions**: Là bộ nhóm tôi tự thu thập chỉ một ít mẫu để thử nghiệm.
## 3. Phân tích bài toán
- **Input**: Một đoạn âm thanh có độ dài cố định.
- **Output**: Một trong 8 cảm xúc: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`.
- **Kiến trúc model**: Sử dụng các model Deep Learning như `1D-CNN`.
- **Cách tiếp cận**: 
    - Sử dụng MFCC để trích xuất đặc trưng.
    - Sử dụng các model Deep Learning (1D-CNN) để phân loại cảm xúc.
## 4. Phân bổ dữ liệu

![alt text](/readme_img/data_distribution.png)

## 5. Tạo biểu đồ sóng và phổ âm thanh
- **Waveform**:

![alt text](/readme_img/Waveform.png)

- **Spectrogram**:

![alt text](/readme_img/Spectrogram.png)

## 6. Tăng cường dữ liệu
- **Shift**: Dịch file âm thanh theo trục thời gian.
- **Speed**: Tăng tốc độ file âm thanh.
- **Pitch**: Thay đổi tần số của file âm thanh.
- **Noise**: Thêm nhiễu vào file âm thanh.

## 7. Trích xuất đặc trưng
- **MFCC**: Trích xuất đặc trưng từ file âm thanh.

![alt text](/readme_img/mfcc.png)
## 8. Chuẩn bị dữ liệu mã hóa nhãn và xây dựng model

![alt text](/readme_img/model.png)
## 9. Huấn luyện model
- **Batch size**: 32.
- **Epochs**: 50.
## 10. Đánh giá model
- Đối với bộ dữ liệu TESS:

![alt text](/readme_img/accuracy.png)

- Confusion Matrix:

![alt text](/readme_img/confusion_matrix.png)
## 11. Kết luận
- Accuracy of our model on test data :  99.74285960197449 %

![alt text](/readme_img/training&testing_accuracy.png)


