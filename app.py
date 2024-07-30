from collections import Counter
import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
ALLOWED_EXTENSIONS = {"wav", "mp3"}  # Các định dạng file âm thanh được chấp nhận

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    filename="app_audio_7s.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Tạo thư mục lưu trữ nếu chưa tồn tại
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the encoder
encoder = joblib.load("encoder_cremad.joblib")

# Tải scaler đã lưu
scaler = joblib.load("scaler_cremad.joblib")

# Load mô hình đã huấn luyện
model = load_model("model_cremad.h5")


# Hàm xử lý file âm thanh và trích xuất đặc trưng MFCC
def process_audio(file_path):
    result = np.array([])
    try:
        y, sr = librosa.load(file_path, sr=None)
        logging.info(
            f"Audio loaded successfully: {file_path}, Sample rate: {sr}, Length: {len(y)}"
        )
    except Exception as e:
        logging.error(f"Error loading audio: {e}")
        raise e
    try:
        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0
        )  # Trích xuất MFCCs
        result = np.hstack((result, mfcc))  # stacking horizontally
        logging.info(f"MFCC extracted successfully: Shape: {mfcc.shape}")
    except Exception as e:
        logging.error(f"Error extracting MFCC: {e}")
        raise e
    # scaling our data with sklearn's Standard scaler
    try:
        result = result.reshape(1, -1)  # Chuyển đổi thành mảng 2D với một mẫu
        result = scaler.transform(result)  # Chuẩn hóa dữ liệu và chuyển vị MFCC
        logging.info(f"MFCC scaled successfully: Shape: {result.shape}")
    except Exception as e:
        logging.error(f"Error scaling MFCC: {e}")
        raise e

    result = np.expand_dims(result, axis=2)  # Thêm chiều cuối cùng là 1
    logging.info(f"MFCC expanded successfully: Shape: {result.shape}")
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "files[]" not in request.files:
        return (
            jsonify({"error": "No file part"}),
            400,
        )  # Trả về lỗi 400 Bad Request nếu không có phần tử file

    files = request.files.getlist("files[]")
    if not files:
        return (
            jsonify({"error": "No selected file"}),
            400,
        )  # Trả về lỗi 400 Bad Request nếu không có file nào được chọn

    emotions = []
    file_details = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Ghi log thông tin file
            logging.info(f"File uploaded: {filename}")

            # Xử lý âm thanh và dự đoán cảm xúc
            try:
                print("File path:", file_path)  # In ra đường dẫn tệp
                mfcc = process_audio(file_path)
                try:
                    prediction = model.predict(mfcc)
                    print("Prediction:", prediction)  # In ra giá trị dự đoán
                except Exception as e:
                    print("Error during model prediction:", e)
                    logging.error(f"Error during model prediction: {e}")
                    raise

                try:
                    # Mã hóa dự đoán
                    prediction = encoder.inverse_transform(
                        prediction
                    )  # Chuyển dự đoán về dạng số nguyên
                    print("Transformed Prediction:", prediction)
                    logging.info(f"Transformed Prediction: {prediction}")
                except Exception as e:
                    print("Error during prediction transformation:", e)
                    logging.error(f"Error during prediction transformation: {e}")
                    raise

                emotion = prediction[0][0]
                emotions.append(emotion)

                # Thêm chi tiết file vào danh sách
                file_details.append(
                    {
                        "file": filename,
                        "audio_path": file_path,
                        "mfcc_shape": mfcc.shape,
                        "emotion_detected": emotion,
                    }
                )

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

    return jsonify({"emotions": emotions, "file_details": file_details})


if __name__ == "__main__":
    app.run(debug=True)
