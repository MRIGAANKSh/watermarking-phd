from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import uvicorn
import os

# ------------------------------------------------
# Load models
# ------------------------------------------------
encoder = load_model("encoder_model.h5")
decoder = load_model("decoder_model.h5")

# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def load_image_bytes(data, size=(256,256)):
    img = Image.open(data).convert("RGB").resize(size)
    return np.array(img)/255.0

def expand_watermark(wm, size=(256,256)):
    wm = Image.fromarray((wm*255).astype(np.uint8)).resize(size)
    return np.array(wm)/255.0

def array_to_image(arr):
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ------------------------------------------------
# FastAPI Application
# ------------------------------------------------
app = FastAPI()

@app.post("/embed")
async def embed_watermark(
    host: UploadFile = File(...),
    watermark: UploadFile = File(...)
):

    # Load images
    host_img = load_image_bytes(host.file)
    wm_img = load_image_bytes(watermark.file)

    # Shrink watermark to 64×64 and expand back
    wm_small = Image.fromarray((wm_img*255).astype(np.uint8)).resize((64,64))
    wm_small = np.array(wm_small)/255.0
    wm_big = expand_watermark(wm_small)

    # Build 6-channel input
    combined = np.concatenate([host_img, wm_big], axis=-1)
    combined = np.expand_dims(combined, 0)

    # Encode
    watermarked = encoder.predict(combined)[0]
    output = array_to_image(watermarked)

    output_path = "api_watermarked.png"
    output.save(output_path)

    return FileResponse(output_path, media_type="image/png", filename="watermarked.png")


@app.post("/extract")
async def extract_watermark(img: UploadFile = File(...)):
    watermarked = load_image_bytes(img.file)

    extracted = decoder.predict(np.expand_dims(watermarked,0))[0]
    out_img = array_to_image(extracted)

    output_path = "extracted_api.png"
    out_img.save(output_path)

    return FileResponse(output_path, media_type="image/png", filename="extracted.png")


# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
