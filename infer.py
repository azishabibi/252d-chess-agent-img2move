#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Union
import re
import cv2
import torch
from ultralytics import YOLO
import chess
import chess.svg
import cairosvg
from collections import defaultdict

# Fixed dimensions for square‑based board division
IMAGE_HEIGHTS = 448
IMAGE_WIDTHS  = 448
N_RANKS       = 8

# Mapping from piece type to FEN symbol
PIECE_MAP = {
    "pawn":   "p",
    "knight": "n",
    "bishop": "b",
    "rook":   "r",
    "queen":  "q",
    "king":   "k",
}

def labels2fen(labels: list[str]) -> str:
    """
    Convert a flat list of 64 FEN chars (or '0') into a FEN string.
    """
    fen_rows = []
    for i in range(0, 64, N_RANKS):
        row = labels[i : i + N_RANKS]
        row_str = "".join(row)
        # collapse runs of '0' into digits
        row_str = re.sub(r"0+", lambda m: str(len(m.group(0))), row_str)
        fen_rows.append(row_str)
    return "/".join(fen_rows)

def fen_to_human(fen):
    piece_names = {
        'p': 'pawn', 'r': 'rook', 'n': 'knight',
        'b': 'bishop','q': 'queen','k': 'king'
    }

    placement = fen.split()[0]
    ranks = placement.split('/')
    descriptions = []

    for rank_index, rank in enumerate(ranks):
        rank_number = 8 - rank_index
        file = 0
        for c in rank:
            if c.isdigit():
                file += int(c)
            else:
                color = 'White' if c.isupper() else 'Black'
                piece = piece_names[c.lower()]
                file_letter = chr(ord('a') + file)
                square = f"{file_letter}{rank_number}"
                descriptions.append(f"{color} {piece} is at {square}")
                file += 1

    return descriptions

def infer_image_yolo(
    image_path: Union[str, Path],
    model: YOLO,
) -> str:
    """
    Run YOLO detection on `image_path` and return the board FEN.
    """
    # 1. Load and resize to square grid size for mapping only
    orig = cv2.imread(str(image_path))
    H, W = orig.shape[:2]
    # We'll use the model's own resizing for detection, then map coords
    # But for grid computation we base on orig shape:
    square_h = H / N_RANKS
    square_w = W / N_RANKS

    # 2. Predict
    res = model.predict(source=str(image_path), imgsz=(448, 448), 
                        device=model.device, save=True, name="inference", 
                        conf=0.9,
    exist_ok=False)[0]

    # 3. Initialize empty board
    labels = ["0"] * 64

    # 4. For each detection, compute which square it falls into
    for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
        x1, y1, x2, y2 = box.cpu().numpy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # Map center back to original image coords:
        # NOTE: if model resized internally, you may need to scale cx,cy to orig W,H.
        # Here we assume boxes are in orig-pixel coords.
        row = int(cy // square_h)
        col = int(cx // square_w)
        idx = row * N_RANKS + col

        # Map class name to FEN char
        name = res.names[int(cls)]  # e.g. "white_pawn"
        color, ptype = name.split("_", 1)
        c = PIECE_MAP[ptype]
        if color == "white":
            c = c.upper()
        labels[idx] = c

    # 5. Build FEN
    fen = labels2fen(labels)
    return fen

def main():
    p = argparse.ArgumentParser(description="Chessboard image → FEN via YOLO")
    p.add_argument(
        "--model_path", "-m", required=True,
        help="Path to your YOLO weights (e.g. runs/train/exp/weights/best.pt)"
    )
    p.add_argument(
        "--input", "-i", required=False, default="test1.png",
        help="Path to input board image"
    )
    p.add_argument(
        "--output", "-o", required=False, default="board.png",
        help="Path to output PNG of rendered board"
    )
    args = p.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YOLO model
    model = YOLO(args.model_path, task="detect")
    model.to(device)

    # Inference
    fen = infer_image_yolo(args.input, model)
    print(f"Predicted FEN: {fen}")

    # Render board
    board = chess.Board(fen)
    svg   = chess.svg.board(board=board)
    cairosvg.svg2png(bytestring=svg, write_to=args.output)
    print(f"Board image saved to {args.output}")
    
    grouped = defaultdict(list)
    for desc in fen_to_human(fen):
        tokens = desc.split()           # e.g. ["Black","bishop","is","at","f5"]
        color, piece = tokens[0], tokens[1]
        square       = tokens[-1]       # last token
        grouped[(color, piece)].append(square)


    for (color, piece), squares in grouped.items():
        locs = " and ".join(squares) if len(squares)==2 else ", ".join(squares)
        verb = "are" if len(squares)>1 else "is"
        print(f"{color} {piece}s {verb} at {locs}")
if __name__ == "__main__":
    main()
