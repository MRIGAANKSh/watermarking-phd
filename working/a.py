import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ==============================
# USER CONFIG: Set your paths
# ==============================
HOST_IMAGE_PATH = "host.png"          # image to be watermarked
WATERMARK_IMAGE_PATH = "watermark.png"  # watermark logo/image
OUTPUT_PATH = "watermarked_output.png"  # saved output

MODEL_PATH = "encoder.h5"  # trained encoder model


# ==============================
# FUNCTIONS
# ==============================
def load_and_preprocess(path, size=(256, 256)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def array_to_image(arr):
    arr = np.clip(arr[0] * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(arr)


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    print("Loading model...")
    encoder = load_model(MODEL_PATH)

    print("Loading images...")
    host = load_and_preprocess(HOST_IMAGE_PATH)
    watermark = load_and_preprocess(WATERMARK_IMAGE_PATH)

    print("Embedding watermark...")
    encoded = encoder.predict([host, watermark])

    print("Saving output...")
    output_img = array_to_image(encoded)
    output_img.save(OUTPUT_PATH)

    print(f"Done! Watermarked image saved as: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
