import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Assuming `y_train` is your training labels
labels = np.array(["angry", "disgust", "fear", "happy", "neutral", "sad"]).reshape(
    -1, 1
)

# Initialize and fit the OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(labels)

# Save the encoder to a file
joblib.dump(encoder, "encoder_cremad.joblib")
