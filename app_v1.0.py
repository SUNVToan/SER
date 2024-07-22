import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import librosa
from keras.models import load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
ALLOWED_EXTENSIONS = {"wav", "mp3"}  # Các định dạng file âm thanh được chấp nhận


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Tạo thư mục lưu trữ nếu chưa tồn tại
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load mô hình đã huấn luyện
model = load_model("model_1.h5")


# Hàm xử lý file âm thanh và trích xuất đặc trưng MFCC
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)  # Trích xuất MFCCs
    mfcc = np.expand_dims(mfcc.T, axis=-1)  # Thêm chiều cuối cùng là 1
    return mfcc


def get_emotion_label(emotion_class):
    # Định nghĩa bản đồ từ chỉ số dự đoán sang nhãn cảm xúc
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # Kiểm tra nếu chỉ số emotion_class hợp lệ
    if 0 <= emotion_class < len(labels):
        return labels[emotion_class]
    else:
        return "Unknown"  # Trả về giá trị mặc định hoặc thông báo lỗi khi chỉ số không hợp lệ


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return (
            jsonify({"error": "No file part"}),
            400,
        )  # Trả về lỗi 400 Bad Request nếu không có phần tử file

    file = request.files["file"]
    if file.filename == "":
        return (
            jsonify({"error": "No selected file"}),
            400,
        )  # Trả về lỗi 400 Bad Request nếu không có file nào được chọn

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Xử lý âm thanh và dự đoán cảm xúc
        try:
            print("File path:", file_path)  # In ra đường dẫn tệp
            mfcc = process_audio(file_path)
            print("MFCC shape:", mfcc.shape)  # In ra hình dạng của MFCC
            try:
                prediction = model.predict(mfcc)
            except Exception as e:
                print("Error during model prediction:", e)
                raise
            print("Prediction:", prediction)  # In ra giá trị dự đoán

            average_probabilities = np.mean(prediction, axis=0)
            print(
                "Average probabilities:", average_probabilities
            )  # In ra xác suất trung bình của các nhãn cảm xúc
            # Xác định chỉ số nhãn cảm xúc có xác suất cao nhất
            emotion_index = np.argmax(average_probabilities)
            print("Emotion index:", emotion_index)  # In ra chỉ số của cảm xúc
            emotion_label = get_emotion_label(emotion_index)
            print("Emotion label:", emotion_label)  # In ra nhãn của cảm xúc

            # Trả về kết quả bao gồm cả xác suất của nhãn cảm xúc
            return jsonify({"emotion": emotion_label})

        except Exception as e:
            return (
                jsonify({"error": str(e)}),
                500,
            )  # Trả về lỗi 500 Internal Server Error nếu có lỗi xảy ra
    else:
        return (
            jsonify({"error": "File type not allowed"}),
            400,
        )  # Trả về lỗi 400 Bad Request nếu file không hợp lệ


if __name__ == "__main__":
    app.run(debug=True)
