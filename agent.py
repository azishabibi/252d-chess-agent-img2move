import os
import io
import json
import re
from pathlib import Path
from ultralytics import YOLO
import torch
import chess
import chess.svg
import cairosvg
from openai import OpenAI, OpenAIError

img_size = (448, 448)
model_name = "gpt-4o"
class ChessPlayAgent:
    # PIECE_REGEX = re.compile(
    #     r"\b(move\s+the\s+)?(?P<piece>pawn|knight|bishop|rook|queen|king)"
    #     r"\s+(from\s+)?(?P<from>[a-h][1-8])\s+(to\s+)?(?P<to>[a-h][1-8])",
    #     re.IGNORECASE
    # )

    # SAN_OR_UGCI_REGEX = re.compile(r"^[PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8][+#]?$", re.IGNORECASE)

    def __init__(self, yolo_weights: str, openai_api_key: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(yolo_weights, task="detect").to(self.device)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai = OpenAI()
        self.reset()

    def reset(self):
        """Clear the game state."""
        self.board = None
        self.history = []

    def infer_fen_from_image(self, image_path: Path, conf_thresh: float = 0.9) -> str:
        res = self.model.predict(
            source=str(image_path),
            imgsz=img_size,
            device=self.device,
            conf=conf_thresh
        )[0]
        H, W = res.orig_shape
        sq_h, sq_w = H / 8, W / 8
        labels = ["0"] * 64
        for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
            x1,y1,x2,y2 = box.cpu().numpy()
            cx, cy = (x1+x2)/2, (y1+y2)/2
            row = int(cy // sq_h)
            col = int(cx // sq_w)
            idx = row*8 + col
            name = res.names[int(cls)]
            color, ptype = name.split("_",1)
            c = ptype[0] if ptype!="knight" else "n"
            c = c.lower() if color=="black" else c.upper()
            labels[idx] = c
        
        rows = []
        for i in range(0,64,8):
            r = "".join(labels[i:i+8])
            rows.append(re.sub(r"0+", lambda m: str(len(m.group(0))), r))
        return "/".join(rows) + " w KQkq - 0 1"

    def describe_board(self) -> str:
        descs = []
        for sq, piece in self.board.piece_map().items():
            color = "White" if piece.color else "Black"
            name  = chess.piece_name(piece.piece_type)
            descs.append(f"{color} {name} on {chess.square_name(sq)}")
        return "; ".join(descs) or "Empty board"

    def decide_next_move(self, image_bytes: bytes = None, max_retries: int = 3) -> dict:
        fen  = self.board.fen()
        desc = self.describe_board()
        system = (
            "You are a top‑tier chess engine. "
            "Given the board state, choose the best next move. "
            "Reply ONLY as JSON: {\"uci\": \"e2e4\", \"san\": \"e4\"}."
        )
        messages = [
            {"role":"system", "content": system},
            {"role":"user",   "content": f"FEN: {fen}\nDesc: {desc}"}
        ]
        if image_bytes:
            messages.append({
                "role": "user",
                "name": "board_image",
                "content": image_bytes
            })
        for attempt in range(max_retries):
            try:
                resp = self.openai.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0
                )
                choice = json.loads(resp.choices[0].message.content)
                uci = choice.get("uci")
                move = chess.Move.from_uci(uci)
                if move in self.board.legal_moves:
                    return choice
                # invalid → ask for retry
                messages.append({
                    "role":"system",
                    "content": "That move was illegal; please provide a valid move."
                })
            except (OpenAIError, json.JSONDecodeError, ValueError):
                messages.append({
                    "role":"system",
                    "content": "I didn’t understand; please reply with a valid move JSON."
                })
        raise RuntimeError("Failed to get a legal move from the model after retries")

    def apply_move(self, uci: str):
        self.board.push(chess.Move.from_uci(uci))

    def render_board_image(self, out_path: str, size: int = 400, orientation: bool = True):
        svg = chess.svg.board(self.board, size=size, orientation=orientation)
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=out_path)

    def parse_user_move_with_llm(self, user_input: str, max_retries: int = 3) -> str:
        """
        Ask the LLM to convert free‑form input (English, SAN, UCI) into a valid UCI move.
        Retries if the move is illegal.
        Returns the UCI string (e.g. "g1f3").
        """
        fen = self.board.fen()
        prompt_sys = (
            "You are a chess move parser. "
            "Given the current board FEN and a user command, "
            "reply ONLY with a JSON {\"uci\": \"g1f3\"}, "
            "where the UCI move is legal on this position."
        )
        prompt_usr = (
            f"FEN: {fen}\n"
            f"User says: \"{user_input}\""
        )

        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user",   "content": prompt_usr}
        ]

        for _ in range(max_retries):
            resp = self.openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0
            )
            try:
                choice = json.loads(resp.choices[0].message.content)
                uci = choice["uci"]
                move = chess.Move.from_uci(uci)
                if move in self.board.legal_moves:
                    return uci
            except Exception:
                pass
            # ask to retry
            messages.append({
                "role": "system",
                "content": "That move was invalid or unparseable. Please give me a legal UCI move in JSON {\"uci\": \"...\"}."
            })

        raise ValueError(f"Unable to parse a legal move from: {user_input}")

    def handle_input(
        self,
        user_input: str,
        side_to_move: str,
        user_side: str,
        output_png: str = "next.png"
    ) -> dict:
        """
        Multi‐round handler that:
         1) restart/reset
         2) image init
         3) auto‐init to standard start if no board yet
         4) if side_to_move != user_side, make engine move immediately
         5) otherwise parse/apply user move via LLM, then engine move
        """
        # 1) restart
        cmd = user_input.strip().lower()
        if cmd in ("restart","reset","start over"):
            self.reset()
            return {"status": "restarted"}

        # 2) image initialization
        if user_input and Path(user_input).is_file():
            image_path = Path(user_input)
            img_bytes = image_path.read_bytes()
            fen = self.infer_fen_from_image(image_path)
            self.board = chess.Board(fen)
            self.render_board_image(out_path=output_png, size=img_size[0])
            return {
                "mode":        "image_init",
                "fen":         fen,
                "description": self.describe_board(),
                "png":         output_png
            }

        # 3) default‐start if no board
        if self.board is None:
            self.board = chess.Board()
            if side_to_move == "black":            # honour the radio **once**
                self.board.turn = chess.BLACK
        orientation = chess.WHITE if user_side == "white" else chess.BLACK
        current_turn = "white" if self.board.turn == chess.WHITE else "black"
        # 4) if it's engine's turn, skip waiting for user_input
        if current_turn != user_side:
            # engine moves
            choice = self.decide_next_move(image_bytes=None)
            self.apply_move(choice["uci"])
            self.render_board_image(output_png,orientation=orientation)
            return {
                "mode":       "engine_move",
                "agent_move": choice["san"],
                "new_fen":    self.board.fen(),
                "png":        output_png
            }

        # 5) it's the user's turn: parse & validate their move
        uci = self.parse_user_move_with_llm(user_input)
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            # reject illegal moves outright
            return {
                "mode":        "illegal_move",
                "description": f"Illegal move: {user_input!r}"
            }

        # OK—apply user move and then engine reply
        self.board.push(move)
        choice = self.decide_next_move(image_bytes=None)
        self.apply_move(choice["uci"])
        self.render_board_image(output_png,orientation=orientation)
        return {
            "mode":       "move",
            "user_move":  uci,
            "agent_move": choice["san"],
            "new_fen":    self.board.fen(),
            "png":        output_png
        }


