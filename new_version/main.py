import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import io

# ================================
# CONFIG: Update Input Image Paths
# ================================
HOST_IMAGE_PATH = "host.png"
WATERMARK_IMAGE_PATH = "watermark.png"
OUTPUT_WATERMARKED = "watermarked_output.png"
OUTPUT_RECOVERED = "recovered_output.png"

# ================================
# Load Encoder & Decoder Models
# ================================
encoder = TFSMLayer("encoder_model", call_endpoint="serve")
decoder = TFSMLayer("decoder_model", call_endpoint="serve")

# ================================
# Data Augmentation (Same as Training)
# ================================
def add_small_noise_and_jpeg(img):
    img = np.clip(img + np.random.normal(0, 0.003, img.shape), 0, 1)

    buff = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buff, format="JPEG", quality=92)
    buff.seek(0)

    return np.array(Image.open(buff)).astype(np.float32) / 255.0

# ================================
# Watermark Process
# ================================
def watermark_image(host_path, wm_path):

    # Host: Load + normalize
    host_img = Image.open(host_path).convert("RGB").resize((256, 256))
    host_arr = np.array(host_img).astype(np.float32) / 255.0

    host_aug = add_small_noise_and_jpeg(host_arr)
    host_input = np.expand_dims(host_aug, axis=0)

    # Watermark: Resize (64 → 256)
    wm_img = Image.open(wm_path).convert("RGB")
    wm_small = wm_img.resize((64, 64), Image.BICUBIC)
    wm_big = wm_small.resize((256, 256), Image.BICUBIC)
    wm_arr = np.array(wm_big).astype(np.float32) / 255.0
    wm_input = np.expand_dims(wm_arr, axis=0)

    # Encoding
    enc_input = np.concatenate([host_input, wm_input], axis=3)
    watermarked = encoder(enc_input)[0].numpy()
    watermarked = np.clip(watermarked, 0, 1)

    # Save watermarked image
    Image.fromarray((watermarked * 255).astype(np.uint8)).save(OUTPUT_WATERMARKED)

    # Decoding
    rec_inp = np.expand_dims(watermarked, axis=0)
    recovered = decoder(rec_inp)[0].numpy()
    recovered = np.clip(recovered, 0, 1)

    # Save recovered watermark
    Image.fromarray((recovered * 255).astype(np.uint8)).save(OUTPUT_RECOVERED)

    # Metrics
    psnr_val = psnr(host_arr, watermarked, data_range=1.0)
    ssim_val = ssim(host_arr, watermarked, channel_axis=2, data_range=1.0)

    print("\n========================")
    print(" FINAL RESULTS")
    print("========================")
    print("Watermarked saved as →", OUTPUT_WATERMARKED)
    print("Recovered watermark saved as →", OUTPUT_RECOVERED)
    print(f"PSNR : {psnr_val:.3f} dB  (Higher → More Imperceptible)")
    print(f"SSIM : {ssim_val:.4f}    (Closer to 1 → Better Quality)")
    print("========================\n")


# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":
    watermark_image(HOST_IMAGE_PATH, WATERMARK_IMAGE_PATH)
