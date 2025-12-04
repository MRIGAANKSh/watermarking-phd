import streamlit as st
import numpy as np
from PIL import Image
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
st.set_page_config(page_title="Deep Watermarking", layout="centered")
st.title("🔐 Deep Learning Image Watermarking System")

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

        # Save to buffer for downloading
        wm_buffer = io.BytesIO()
        rec_buffer = io.BytesIO()
        wm_img.save(wm_buffer, format="PNG"); wm_buffer.seek(0)
        rec_img.save(rec_buffer, format="PNG"); rec_buffer.seek(0)

        st.success("Watermarking Completed! 🎉")

        col1, col2 = st.columns(2)
        col1.image(wm_img, caption="Watermarked Image")
        col2.image(rec_img, caption="Recovered Watermark")

        st.markdown("### 📊 Quality Evaluation Metrics")
        m1, m2 = st.columns(2)
        m1.metric("PSNR", f"{psnr_val:.2f} dB", help="Higher is better")
        m2.metric("SSIM", f"{ssim_val:.4f}", help="Closer to 1 = better similarity")

        st.info(
            """
            **Interpretation**  
            ✔️ PSNR > 30 dB = High Quality  
            ✔️ SSIM close to **1** = Good Visual Similarity  
            """
        )

        # Download Buttons
        st.download_button("⬇ Download Watermarked Image", data=wm_buffer, file_name="watermarked.png", mime="image/png")
        st.download_button("⬇ Download Extracted Watermark", data=rec_buffer, file_name="recovered.png", mime="image/png")

        # Metrics report file
        report_text = f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}"
        st.download_button("⬇ Download Metrics Report", report_text, "metrics.txt")
