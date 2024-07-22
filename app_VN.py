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
    filename="app_VN.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Load the encoder
encoder = joblib.load("encoder.joblib")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Tạo thư mục lưu trữ nếu chưa tồn tại
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load mô hình đã huấn luyện
model = load_model("model_VN.h5")


# Hàm xử lý file âm thanh và trích xuất đặc trưng MFCC
def process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        logging.info(
            f"Audio loaded successfully: {file_path}, Sample rate: {sr}, Length: {len(y)}"
        )
    except Exception as e:
        logging.error(f"Error loading audio: {e}")
        raise e
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)  # Trích xuất MFCCs
        logging.info(f"MFCC extracted successfully: Shape: {mfcc.shape}")
    except Exception as e:
        logging.error(f"Error extracting MFCC: {e}")
        raise e
    # scaling our data with sklearn's Standard scaler
    try:
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc.T)  # Chuẩn hóa dữ liệu và chuyển vị MFCC
        logging.info(f"MFCC scaled successfully: Shape: {mfcc.shape}")
    except Exception as e:
        logging.error(f"Error scaling MFCC: {e}")
        raise e

    mfcc = np.expand_dims(mfcc, axis=2)  # Thêm chiều cuối cùng là 1
    logging.info(f"MFCC expanded successfully: Shape: {mfcc.shape}")
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

        # Ghi log thông tin file
        logging.info(f"File uploaded: {filename}")

        # Xử lý âm thanh và dự đoán cảm xúc
        try:
            print("File path:", file_path)  # In ra đường dẫn tệp
            mfcc = process_audio(file_path)
            print("MFCC shape:", mfcc.shape)  # In ra hình dạng của MFCC
            logging.info(f"MFCC shape: {mfcc.shape}")

            try:
                prediction = model.predict(mfcc)
                # print("Prediction:", prediction)  # In ra giá trị dự đoán
            except Exception as e:
                print("Error during model prediction:", e)
                logging.error(f"Error during model prediction: {e}")
                raise

            try:
                # mã hóa dự đoán
                prediction = encoder.inverse_transform(
                    prediction
                )  # Chuyển dự đoán về dạng số nguyên
                print("Transformed Prediction:", prediction)
                # logging.info(f"Transformed Prediction: {prediction}")
            except Exception as e:
                print("Error during prediction transformation:", e)
                logging.error(f"Error during prediction transformation: {e}")
                raise

            try:
                # Đếm số lần xuất hiện của từng nhãn cảm xúc
                emotion_counts = Counter([label[0] for label in prediction])
                print(
                    "Emotion counts:", emotion_counts
                )  # In ra số lượng của các cảm xúc
                logging.info(f"Emotion counts: {emotion_counts}")

                # Chọn nhãn có số lượng dự đoán cao nhất
                most_common_emotion = emotion_counts.most_common(1)[0][0]
                print("Most common emotion:", most_common_emotion)
                logging.info(f"Emotion detected: {most_common_emotion}")
            except Exception as e:
                print("Error during emotion counting and labeling:", e)
                logging.error(f"Error during emotion counting and labeling: {e}")
                raise

            # Trả về kết quả bao gồm cả xác suất của nhãn cảm xúc
            return jsonify({"emotion": most_common_emotion})

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
