import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import io

# ---------------------------------------------------
# Load Saved TF Models (SavedModel folders)
# ---------------------------------------------------
encoder = TFSMLayer("encoder_model", call_endpoint="serve")
decoder = TFSMLayer("decoder_model", call_endpoint="serve")

def add_small_noise_and_jpeg(x):
    x = np.clip(x + np.random.normal(0, 0.003, x.shape), 0, 1)
    img = Image.fromarray((x * 255).astype(np.uint8))
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=92)
    bio.seek(0)
    return np.array(Image.open(bio)).astype(np.float32) / 255.0


def watermark_process(host_img, wm_img):
    host_img = host_img.convert("RGB").resize((256, 256))
    host_array = np.array(host_img).astype(np.float32) / 255.0

    host_aug = add_small_noise_and_jpeg(host_array)
    host_input = np.expand_dims(host_aug, axis=0)

    wm_small = wm_img.convert("RGB").resize((64, 64), Image.BICUBIC)
    wm_big = wm_small.resize((256, 256), Image.BICUBIC)
    wm_array = np.array(wm_big).astype(np.float32) / 255.0
    wm_input = np.expand_dims(wm_array, axis=0)

    enc_input = np.concatenate([host_input, wm_input], axis=3)
    watermarked = encoder(enc_input)[0].numpy()
    watermarked = np.clip(watermarked, 0, 1)

    dec_input = np.expand_dims(watermarked, axis=0)
    recovered = decoder(dec_input)[0].numpy()
    recovered = np.clip(recovered, 0, 1)

    psnr_val = psnr(host_array, watermarked, data_range=1.0)
    ssim_val = ssim(host_array, watermarked, channel_axis=2, data_range=1.0)

    return watermarked, recovered, psnr_val, ssim_val


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.title("🔐 Deep Learning Watermarking System")

host_upload = st.file_uploader("📌 Upload Host Image", type=["png", "jpg", "jpeg"])
wm_upload = st.file_uploader("📌 Upload Watermark Image", type=["png", "jpg", "jpeg"])

if host_upload and wm_upload:
    host = Image.open(host_upload)
    wm = Image.open(wm_upload)

    if st.button("▶ Run Watermarking"):
        st.info("Processing... please wait ⏳")

        watermarked, recovered, psnr_val, ssim_val = watermark_process(host, wm)

        wm_img = Image.fromarray((watermarked * 255).astype(np.uint8))
        rec_img = Image.fromarray((recovered * 255).astype(np.uint8))

        st.success("Done!")

        st.subheader("📌 Results")
        st.image(wm_img, caption="Watermarked Image")
        st.image(rec_img, caption="Recovered Watermark")

        st.write(f"**PSNR:** {psnr_val:.2f} dB")
        st.write(f"**SSIM:** {ssim_val:.4f}")

        # Download Buttons
        st.download_button("⬇ Download Watermarked", wm_img.tobytes(), "watermarked.png")
        st.download_button("⬇ Download Recovered Watermark", rec_img.tobytes(), "recovered.png")
