"""
gradcam.py
Grad-CAM cho model dạng: [augmentation] -> base CNN lồng nhau -> GAP -> Dropout -> Dense...
(đúng cấu trúc của species_model.keras và model2_6classes.keras).

Ý tưởng: lấy đầu ra của base CNN (feature map cuối), chạy tay phần "head" phía sau
để tính gradient của lớp dự đoán theo feature map đó. Không cần biết tên lớp conv.
Chỉ dùng numpy + Pillow, không thêm thư viện mới.
"""

import base64
import io

import numpy as np
from PIL import Image


def _call(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def _split_model(model):
    import tensorflow as tf

    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    nested = [(i, l) for i, l in enumerate(layers) if isinstance(l, tf.keras.Model)]
    if not nested:
        raise ValueError("Không tìm thấy base CNN lồng nhau trong model.")
    base_i, base = max(nested, key=lambda t: len(t[1].layers))
    return layers[:base_i], base, layers[base_i + 1:]


def compute_cam(model, x: np.ndarray, class_idx: int) -> np.ndarray | None:
    """Trả về heatmap 2D chuẩn hóa 0..1 (kích thước = feature map), hoặc None."""
    import tensorflow as tf

    pre, base, head = _split_model(model)
    h = tf.convert_to_tensor(x)
    for layer in pre:
        h = _call(layer, h)

    with tf.GradientTape() as tape:
        fmap = _call(base, h)
        tape.watch(fmap)
        out = fmap
        for layer in head:
            out = _call(layer, out)
        score = out[:, class_idx]

    grads = tape.gradient(score, fmap)
    if grads is None:
        return None
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1))[0].numpy()
    peak = float(cam.max())
    if peak <= 1e-8:
        return None
    return cam / peak


def _jet(t: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255.0


def overlay_to_data_url(cam: np.ndarray, img: Image.Image, max_side: int = 640) -> str:
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
    t = np.asarray(cam_img, dtype=np.float32) / 255.0
    alpha = (0.65 * t)[..., None]  # chỗ nóng đậm hơn, chỗ nguội gần như trong suốt
    out = np.asarray(img, dtype=np.float32) * (1 - alpha) + _jet(t) * alpha
    buf = io.BytesIO()
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def gradcam_overlay(model, x: np.ndarray, class_idx: int, img: Image.Image) -> str | None:
    cam = compute_cam(model, x, class_idx)
    return None if cam is None else overlay_to_data_url(cam, img)
