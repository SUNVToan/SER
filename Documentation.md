# Speech emotion recognition

### shape=(None, 39, 1) hiểu thế nào?
Trong TensorFlow và Keras, hình dạng `(None, 39, 1)` được sử dụng để mô tả kích thước của dữ liệu đầu vào cho một mô hình mạng nơ-ron. Mỗi số trong bộ ba này có ý nghĩa cụ thể:

- **`None`**: Đại diện cho kích thước không xác định của batch. Nó cho phép bạn cung cấp một số lượng mẫu dữ liệu tùy ý khi đưa vào mô hình. Trong thực tế, `None` được sử dụng để biểu thị rằng bạn có thể đưa vào mô hình bất kỳ số lượng mẫu nào trong một lần chạy.

- **`39`**: Đại diện cho số lượng đặc trưng (features) trong mỗi mẫu dữ liệu. Trong trường hợp này, nó có thể đại diện cho số lượng hệ số MFCC được trích xuất từ mỗi đoạn âm thanh. Điều này cho biết mỗi mẫu dữ liệu sẽ có 39 hệ số MFCC.

- **`1`**: Đại diện cho chiều sâu của mỗi đặc trưng. Trong nhiều trường hợp, đặc biệt là với dữ liệu âm thanh hoặc hình ảnh xám, chiều sâu sẽ là 1, chỉ ra rằng mỗi đặc trưng là một giá trị vô hướng (không phải là một vector hay ma trận). Trong ngữ cảnh của dữ liệu âm thanh, điều này thường chỉ ra rằng dữ liệu đã được xử lý thành một dạng đơn kênh (mono) thay vì nhiều kênh (stereo).

Khi bạn thấy hình dạng `(None, 39, 1)` trong mô tả đầu vào của một mô hình mạng nơ-ron, điều này nói lên rằng mô hình đó có thể xử lý một lô (batch) dữ liệu với số lượng mẫu tùy ý, mỗi mẫu có 39 đặc trưng, và mỗi đặc trưng là một giá trị vô hướng.

### How can i save model?
from tensorflow.keras.models import load_model
model.save("model.h5")

### Tải mô hình đã lưu
model = load_model('model.h5')

### Lưu trọng số của mô hình
model.save_weights('model_weights.h5')

### Tải trọng số vào mô hình đã khởi tạo
model.load_weights('model_weights.h5')

