# File: convert_fen_to_yolo.py

import os
import argparse
import random
import shutil

# supported image extensions
IMG_EXTS = ('.jpg', '.jpeg', '.png')

# FEN char → YOLO class idx (0–11)
PIECE_TO_CLASS = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def parse_fen_positions(fen_str):
    rows = fen_str.split('/')
    boxes = []
    for r_idx, row in enumerate(rows):
        f_idx = 0
        for ch in row:
            if ch.isdigit():
                f_idx += int(ch)
            else:
                cls = PIECE_TO_CLASS[ch]
                x_c = (f_idx + 0.5) / 8
                y_c = (r_idx + 0.5) / 8
                boxes.append((cls, x_c, y_c, 1/8, 1/8))
                f_idx += 1
    return boxes

def write_labels_for_split(img_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
    for fname in imgs:
        base, _ = os.path.splitext(fname)
        # dashed-FEN → proper piece-placement
        placement = base.replace('-', '/').split()[0]
        boxes = parse_fen_positions(placement)
        out_txt = os.path.join(label_dir, base + '.txt')
        with open(out_txt, 'w') as fw:
            for cls, x, y, w, h in boxes:
                fw.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def main():
    p = argparse.ArgumentParser(
        description="Split test→val (10k) and generate YOLO labels for train/val/test"
    )
    p.add_argument('--dataset-root', required=True,
                   help="Path to `dataset/`")
    args = p.parse_args()

    ds = args.dataset_root
    train_dir = os.path.join(ds, 'train')
    test_dir = os.path.join(ds, 'test')
    val_dir = os.path.join(ds, 'val')
    labels_root = os.path.join(ds, 'labels')
    labels_train = os.path.join(labels_root, 'train')
    labels_val   = os.path.join(labels_root, 'val')
    labels_test  = os.path.join(labels_root, 'test')

    # 1) Create val/ by moving 10k random images from test/
    os.makedirs(val_dir, exist_ok=True)
    all_test = [f for f in os.listdir(test_dir) if f.lower().endswith(IMG_EXTS)]
    random.shuffle(all_test)
    to_val = all_test[:10000]
    for img in to_val:
        shutil.move(os.path.join(test_dir, img),
                    os.path.join(val_dir, img))
    print(f"Moved {len(to_val)} images from test → val")

    # 2) Generate labels for train split
    write_labels_for_split(train_dir, labels_train)
    print(f"Wrote {len(os.listdir(labels_train))} train labels → {labels_train}")

    # 3) Generate labels for val split
    write_labels_for_split(val_dir, labels_val)
    print(f"Wrote {len(os.listdir(labels_val))} val labels   → {labels_val}")

    # 4) Generate labels for remaining test split
    write_labels_for_split(test_dir, labels_test)
    print(f"Wrote {len(os.listdir(labels_test))} test labels  → {labels_test}")

if __name__ == '__main__':
    main()


# python dataset.py --dataset-root dataset
