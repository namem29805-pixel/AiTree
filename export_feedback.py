"""
export_feedback.py
Xuất ảnh phản hồi từ SQLite (bảng feedback) thành thư mục theo nhãn để train lại trên Colab:

    feedback_export/condition/<nhãn tình trạng>/*.jpg
    feedback_export/species/<nhãn loài>/*.jpg

Chạy:
    python export_feedback.py                    # chỉ ảnh người dùng báo SAI (đã có nhãn đúng)
    python export_feedback.py --include-correct  # thêm cả ảnh báo ĐÚNG
    python export_feedback.py --only-unreviewed  # bỏ ảnh đã đánh dấu reviewed=1
    python export_feedback.py --mark-reviewed    # đánh dấu các ảnh vừa xuất là đã duyệt

LƯU Ý: nhãn người dùng có thể sai. Hãy mở từng thư mục và duyệt tay trước khi train.
"""

import argparse
import shutil
from pathlib import Path

import db

BASE = Path(__file__).resolve().parent
IMAGES = BASE / "feedback" / "images"
OUT = BASE / "feedback_export"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-correct", action="store_true")
    ap.add_argument("--only-unreviewed", action="store_true")
    ap.add_argument("--mark-reviewed", action="store_true")
    args = ap.parse_args()

    rows = db.feedback_rows(include_correct=args.include_correct, only_unreviewed=args.only_unreviewed)
    counts, exported = {}, []
    for r in rows:
        src = IMAGES / r["image_name"]
        if not src.exists():
            continue
        wrote = False
        for kind in ("condition", "species"):
            label, predicted = r["label_" + kind], r["predicted_" + kind]
            if not label or (r["verdict"] == "wrong" and label == predicted):
                continue  # không có nhãn mới cho mô hình này
            dst = OUT / kind / label
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
            counts[(kind, label)] = counts.get((kind, label), 0) + 1
            wrote = True
        if wrote:
            exported.append(r["id"])

    if not counts:
        print("Không có ảnh nào phù hợp để xuất.")
        return
    for (kind, label), n in sorted(counts.items()):
        print(f"{kind:9s} {label:22s} {n} ảnh")
    print(f"\nĐã xuất vào: {OUT}. Hãy duyệt tay trước khi train.")
    if args.mark_reviewed:
        with db.connect() as con:
            con.executemany("UPDATE feedback SET reviewed = 1 WHERE id = ?", [(i,) for i in exported])
        print(f"Đã đánh dấu {len(exported)} phản hồi là đã duyệt.")


if __name__ == "__main__":
    main()
