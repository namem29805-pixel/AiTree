# Dán vào 1 cell MỚI trong Colab, chạy SAU cell 30 (khi test_ds, class_names
# vẫn còn trong bộ nhớ). So sánh model hiện có trong biến `model` (bản đã
# lưu thành model2_6classes.keras) với best_model2_6classes.keras trên
# ĐÚNG cùng một test_ds để biết bản nào thực sự tốt hơn.

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate(model, name):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(labels.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = (y_true == y_pred).mean()
    print(f"\n=== {name} — test accuracy: {acc:.4f} ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"confusion_{name}.png", dpi=150)
    plt.show()
    return acc, cm

# Bản hiện tại đang lưu ra model2_6classes.keras (kết quả cuối phase 2)
acc_current, cm_current = evaluate(model, "model_hien_tai_phase2")

# Bản checkpoint tốt nhất theo val_accuracy
best_model = tf.keras.models.load_model("best_model2_6classes.keras")
acc_best, cm_best = evaluate(best_model, "best_checkpoint")

print(f"\nChênh lệch test accuracy: {acc_best - acc_current:+.4f}")

# Xem riêng các ảnh healthy bị đoán nhầm thành leaf_spot_fungal,
# để kiểm tra giả thuyết "lá Calathea có vân tự nhiên".
healthy_idx = class_names.index("healthy")
spot_idx = class_names.index("leaf_spot_fungal")

file_paths = test_ds_raw.file_paths  # cần test_ds_raw còn giữ nguyên thứ tự
y_true_list, y_pred_list = [], []
for images, labels in test_ds:
    probs = best_model.predict(images, verbose=0)
    y_pred_list.extend(np.argmax(probs, axis=1))
    y_true_list.extend(labels.numpy())

wrong_healthy_as_spot = [
    fp for fp, yt, yp in zip(file_paths, y_true_list, y_pred_list)
    if yt == healthy_idx and yp == spot_idx
]

print(f"\nSố ảnh healthy bị đoán nhầm thành leaf_spot_fungal (best_checkpoint): "
      f"{len(wrong_healthy_as_spot)}")
for fp in wrong_healthy_as_spot:
    print(" -", fp)

# Hiển thị thử để soi bằng mắt
cols = 4
rows = (len(wrong_healthy_as_spot) + cols - 1) // cols
if wrong_healthy_as_spot:
    plt.figure(figsize=(14, max(3, rows * 3)))
    for i, fp in enumerate(wrong_healthy_as_spot):
        img = tf.keras.utils.load_img(fp, target_size=(224, 224))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(fp.split("/")[-1], fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("healthy_misclassified_as_spot.png", dpi=150)
    plt.show()
