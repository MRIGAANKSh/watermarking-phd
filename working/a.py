import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------
def load_image(path, size=(256,256)):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img)/255.0

def expand_watermark(wm, size=(256,256)):
    wm = Image.fromarray((wm*255).astype(np.uint8)).resize(size)
    return np.array(wm)/255.0

def save_image(arr, path):
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ------------------------------------------------
# Main Execution
# ------------------------------------------------
def main():
    print("Loading models...")
    encoder = load_model("encoder_model.h5")
    decoder = load_model("decoder_model.h5")

    print("Loading host image + watermark...")
    host = load_image("host.png")  # <--- YOU MUST PROVIDE THIS
    wm = load_image("watermark.png")

    # Resize watermark to 64×64 and expand back to 256×256
    wm_small = Image.fromarray((wm*255).astype(np.uint8)).resize((64,64))
    wm_small = np.array(wm_small) / 255.0
    wm_big = expand_watermark(wm_small)

    # Build 6-channel model input
    combined = np.concatenate([host, wm_big], axis=-1)
    combined = np.expand_dims(combined, 0)   # (1,256,256,6)

    print("Encoding watermark...")
    watermarked = encoder.predict(combined)[0]

    print("Decoding watermark...")
    extracted = decoder.predict(np.expand_dims(watermarked,0))[0]

    # Save output images
    save_image(watermarked, "watermarked_output.png")
    save_image(extracted, "extracted_watermark.png")

    print("✅ Watermarked image saved as 'watermarked_output.png'")
    print("✅ Extracted watermark saved as 'extracted_watermark.png'")


if __name__ == "__main__":
    main()
