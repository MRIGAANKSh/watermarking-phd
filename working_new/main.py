import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import io

# ---------------------------------------------------
# 🔥 EDIT THESE PATHS (LOCAL FILES)
# ---------------------------------------------------
HOST_IMAGE_PATH = "host.jpg"
WATERMARK_IMAGE_PATH = "watermark.jpg"

OUTPUT_WATERMARKED = "watermarked_output.png"
OUTPUT_RECOVERED = "recovered_output.png"

# ---------------------------------------------------
# Load Saved TF Models (SavedModel folders)
# ---------------------------------------------------
encoder = TFSMLayer("encoder_model", call_endpoint="serve")
decoder = TFSMLayer("decoder_model", call_endpoint="serve")


# ---------------------------------------------------
# Matching augmentation used in training
# ---------------------------------------------------
def add_small_noise_and_jpeg(x):
    x = np.clip(x + np.random.normal(0, 0.003, x.shape), 0, 1)

    img = Image.fromarray((x * 255).astype(np.uint8))
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=92)
    bio.seek(0)

    return np.array(Image.open(bio)).astype(np.float32) / 255.0


# ---------------------------------------------------
# Watermark Function
# ---------------------------------------------------
def watermark_image(host_path, wm_path, out_path, rec_path):

    # Load HOST
    host_img = Image.open(host_path).convert("RGB").resize((256, 256))
    host_array = np.array(host_img).astype(np.float32) / 255.0

    host_aug = add_small_noise_and_jpeg(host_array)
    host_input = np.expand_dims(host_aug, axis=0)

    # Load WATERMARK (64 → 256 upscale exactly like training)
    wm_img = Image.open(wm_path).convert("RGB")
    wm_small = wm_img.resize((64, 64), Image.BICUBIC)
    wm_big = wm_small.resize((256, 256), Image.BICUBIC)
    wm_array = np.array(wm_big).astype(np.float32) / 255.0
    wm_input = np.expand_dims(wm_array, axis=0)

    # ENCODE
    enc_input = np.concatenate([host_input, wm_input], axis=3)
    watermarked = encoder(enc_input)[0].numpy()
    watermarked = np.clip(watermarked, 0, 1)

    watermarked_img = (watermarked * 255).astype(np.uint8)
    Image.fromarray(watermarked_img).save(out_path)

    # DECODE
    dec_input = np.expand_dims(watermarked, axis=0)
    recovered = decoder(dec_input)[0].numpy()
    recovered = np.clip(recovered, 0, 1)

    recovered_img = (recovered * 255).astype(np.uint8)
    Image.fromarray(recovered_img).save(rec_path)

    # METRICS (between original HOST and WATERMARKED)
    psnr_val = psnr(host_array, watermarked, data_range=1.0)
    ssim_val = ssim(host_array, watermarked, channel_axis=2, data_range=1.0)

    print("\n============================")
    print("       FINAL OUTPUT")
    print("============================")
    print(f"Watermarked image saved: {out_path}")
    print(f"Recovered watermark saved: {rec_path}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print("============================\n")


# ---------------------------------------------------
# RUN IT
# ---------------------------------------------------
watermark_image(
    HOST_IMAGE_PATH,
    WATERMARK_IMAGE_PATH,
    OUTPUT_WATERMARKED,
    OUTPUT_RECOVERED
)
