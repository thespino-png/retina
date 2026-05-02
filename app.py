from flask import Flask, request, jsonify, make_response
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import os

app = Flask(__name__)

# Manual CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

model = tf.keras.models.load_model("diabetic_retinopathy_model.keras")
IMG_SIZE = (224, 224)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Model API is running ✅"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = "Diseased" if prob >= 0.5 else "Healthy"
    confidence = prob if prob >= 0.5 else 1 - prob

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "raw_prob": round(prob, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
