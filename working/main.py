import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# ==============================
# LOAD MODELS
# ==============================
ENCODER_PATH = "encoder.h5"
DECODER_PATH = "decoder.h5"

encoder = load_model(ENCODER_PATH)
decoder = load_model(DECODER_PATH)

# ==============================
# IMAGE HELPERS
# ==============================
def load_and_preprocess(img_file, size=(256, 256)):
    img = Image.open(img_file).convert("RGB")
    img = img.resize(size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def array_to_image(arr):
    arr = np.clip(arr[0] * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(arr)

# ==============================
# ROUTE: Watermark Embedding
# ==============================
@app.route("/embed", methods=["POST"])
def embed_watermark():

    if "image" not in request.files or "watermark" not in request.files:
        return jsonify({"error": "You must upload 'image' and 'watermark'"}), 400

    host_image = request.files["image"]
    watermark_image = request.files["watermark"]

    # Preprocess
    host = load_and_preprocess(host_image)
    wm = load_and_preprocess(watermark_image)

    # Predict the encoded output
    encoded = encoder.predict([host, wm])

    # Convert to image
    output_image = array_to_image(encoded)

    # Return file
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# ==============================
# ROUTE: Extract Watermark
# ==============================
@app.route("/extract", methods=["POST"])
def extract_watermark():

    if "watermarked_image" not in request.files:
        return jsonify({"error": "Upload 'watermarked_image'"}), 400

    watermarked = request.files["watermarked_image"]

    wm_img = load_and_preprocess(watermarked)

    # Decode watermark
    decoded = decoder.predict(wm_img)

    decoded_img = array_to_image(decoded)

    buf = io.BytesIO()
    decoded_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# ==============================
# ROOT
# ==============================
@app.route("/", methods=["GET"])
def index():
    return """
    <h1>Deep Learning Image Watermarking API</h1>
    <p>Use POST /embed and POST /extract</p>
    """


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
