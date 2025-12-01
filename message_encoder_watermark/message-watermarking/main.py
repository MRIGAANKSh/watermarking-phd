import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, render_template, send_file
from PIL import Image
import torchvision.transforms as T

# -----------------------------
#  MODEL DEFINITIONS
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class Embedder(nn.Module):
    def __init__(self, wm_len=32):
        super().__init__()
        self.down = ConvBlock(2, 32)
        self.mid  = ConvBlock(32, 64)
        self.up   = ConvBlock(64, 32)
        self.out  = nn.Conv2d(32, 1, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, img, wm_map):
        x = torch.cat([img, wm_map], dim=1)
        x = self.down(x)
        x = self.mid(x)
        x = self.up(x)

        residual = (self.sig(self.out(x)) - 0.5) * 0.03
        watermarked = torch.clamp(img + residual, 0., 1.)
        return watermarked

class Extractor(nn.Module):
    def __init__(self, wm_len=32):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, wm_len)

    def forward(self, x):
        f = self.cnn(x).view(x.size(0), -1)
        return self.fc(f)


# -----------------------------
#  LOAD MODELS
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = Embedder(32).to(device)
embedder.load_state_dict(torch.load("embedder.pth", map_location=device))
embedder.eval()

# -----------------------------
#  UTILITIES
# -----------------------------

transform = T.Compose([
    T.Grayscale(),
    T.Resize((256,256)),
    T.ToTensor()
])

def text_to_bits(text, length=32):
    bits = ''.join(format(ord(c), '08b') for c in text)
    bits = bits[:length]
    bits += '0' * (length - len(bits))
    return np.array(list(bits)).astype(np.float32)

def make_wm_map(bits, H=256, W=256):
    bits = np.array(bits).astype(np.float32)
    tiled = np.tile(bits, (H * W // len(bits) + 1))[:H*W]
    wm_map = tiled.reshape(1, 1, H, W)
    return torch.tensor(wm_map, dtype=torch.float32)

def load_img_pil(pil_img):
    return transform(pil_img).unsqueeze(0)

def tensor_to_pil(t):
    arr = t.squeeze().cpu().numpy() * 255
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


# -----------------------------
#  FLASK APP
# -----------------------------

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return '''
        <h2>Deep Invisible Watermarking</h2>
        <form action="/embed" method="post" enctype="multipart/form-data">
            Message: <input type="text" name="message" required><br><br>
            Upload Image: <input type="file" name="image" required><br><br>
            <input type="submit" value="Embed Watermark">
        </form>
    '''

@app.route("/embed", methods=["POST"])
def embed():
    message = request.form["message"]
    img_file = request.files["image"]

    pil_img = Image.open(img_file.stream).convert("RGB")
    img = load_img_pil(pil_img).to(device)

    wm_bits = text_to_bits(message, 32)
    wm_map = make_wm_map(wm_bits).to(device)

    with torch.no_grad():
        wm_img = embedder(img, wm_map)

    out_img = tensor_to_pil(wm_img)

    output_path = "watermarked_output.png"
    out_img.save(output_path)

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
