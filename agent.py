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
import random
from openai import OpenAI, OpenAIError

img_size = (448, 448)

VALID_LLM_MODELS = {
    # GPT-4o family
    "gpt-4o", "gpt-4o-mini", "gpt-4o-turbo",
    # GPT-4.1 family
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    # o-series (reasoning)
    "o3", "o3-mini", "o4-mini", "o4-mini-high",
}

# o-series models don’t support a temperature parameter
NO_TEMPERATURE_MODELS = {"o3", "o3-mini", "o4-mini", "o4-mini-high"}

# Schema for engine‐move selection (uci + san)
MOVE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "uci": {"type": "string", "description": "best move in UCI"},
        "san": {"type": "string", "description": "best move in SAN"}
    },
    "required": ["uci", "san"],
    "additionalProperties": False
}

# Schema for parsing user input (just uci)
MOVE_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "uci": {"type": "string", "description": "legal move in UCI"}
    },
    "required": ["uci"],
    "additionalProperties": False
}


class ChessPlayAgent:

    def __init__(self, yolo_weights: str, openai_api_key: str, model_name: str = "gpt-4.1",temperature: float = 0.0,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(yolo_weights, task="detect").to(self.device)
        self.model_name = self._validate_model(model_name)
        self.temperature = max(0.0, min(2.0, temperature))
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai = OpenAI()
        self.reset()

    @staticmethod
    def _validate_model(name: str) -> str:
        if name not in VALID_LLM_MODELS:
            raise ValueError(
                f"Unknown / unsupported model '{name}'. Choose from: {sorted(VALID_LLM_MODELS)}"
            )
        return name
    def set_model(self, model_name: str):
        """Change the backing LLM at runtime (e.g. "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4-turbo", "o3")."""
        self.model_name = model_name

    def reset(self):
        """Clear the game state."""
        self.board = None
        self.history = []

    def infer_fen_from_image(self, image_path: Path, conf_thresh: float = 0.9, white_bottom: bool = True) -> str:
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
        if not white_bottom:
            labels = labels[::-1]
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

    def describe_legal_moves(self) -> str:
        descs = []
        for move in self.board.legal_moves:
            piece = self.board.piece_at(move.from_square)
            color = "White" if piece.color else "Black"
            name  = chess.piece_name(piece.piece_type)
            frm   = chess.square_name(move.from_square)
            to    = chess.square_name(move.to_square)
            descs.append(f"{color} {name} from {frm} to {to}")
        return "; ".join(descs) or "No legal moves"

    def decide_next_move(self, image_bytes: bytes = None, max_retries: int = 3) -> dict:
        fen  = self.board.fen()
        desc = self.describe_board()
        # 1) Compute the UCI list:
        legal_moves = [m.uci() for m in self.board.legal_moves]
        legal_moves = ", ".join(legal_moves)
        legal_desc  = self.describe_legal_moves()
        print(f"Legal moves: {legal_desc}")
        system = (
            # ─── Identity ───────────────────────────────────────────────────────────────
            "You are a world-class chess engine and grand-master coach.\n"
            "\n"
            # ─── Input format ───────────────────────────────────────────────────────────
            "You will receive:\n"
            "  • A FEN string describing the current position.\n"
            "  • A natural-language description of every piece on every square.\n"
            "  • The complete list of legal moves, first in UCI format, then in English.\n"
            "\n"
            # ─── Internal reasoning protocol ────────────────────────────────────────────
            "Think silently and follow this private checklist **without revealing it**:\n"
            "  1. Evaluate the position for the side to move (material, king safety, "
            "centre control, piece activity, tactics).\n"
            "  2. Enumerate the most promising candidate moves (at least three).\n"
            "  3. For each candidate, reason step-by-step about its consequences.\n"
            "  4. Reflect on your analysis; if you detect a blunder, reconsider.\n"
            "  5. Select the single strongest move **that appears in the legal-move list**.\n"
            "  6. Double-check the chosen move is legal and does not leave your king "
            "   in check.\n"
            "\n"
            # ─── Output contract ────────────────────────────────────────────────────────
            "Respond ONLY with valid JSON **exactly matching** this schema, "
            "using lower-case keys and no additional text:\n"
            "  {\"uci\": \"<best_move_in_uci>\", \"san\": \"<best_move_in_san>\"}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"FEN: {fen}\nPiece locations: {desc}"},
            {"role": "user",   "content": f"Legal moves (UCI): {legal_moves}"},
            {"role": "user",   "content": f"Move descriptions: {legal_desc}"}
        ]

        if image_bytes:
            messages.append({
                "role": "user",
                "name": "board_image",
                "content": image_bytes
            })
        call_kwargs = {"model": self.model_name, "messages": messages}
        
        call_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name":        "move_selection",                    # required
                "description": "Best chess move in UCI and SAN",    # optional but helpful
                "strict":      True,                                # enforce exact match
                "schema":      MOVE_SELECTION_SCHEMA               # <— here
            }
        }
        if self.model_name not in NO_TEMPERATURE_MODELS:
            call_kwargs["temperature"] = self.temperature
        for attempt in range(max_retries):
            try:
                resp = self.openai.chat.completions.create(
                    **call_kwargs,
                    # model=self.model_name,
                    # messages=messages,
                    # temperature=self.temperature,
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
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                messages.append({
                    "role":"system",
                    "content": e,
                })
        raise RuntimeError("Failed to get a legal move from the model after retries")

    def apply_move(self, uci: str):
        self.board.push(chess.Move.from_uci(uci))

    def render_board_image(self, out_path: str, size: int = 400, orientation: bool = True):
        svg = chess.svg.board(self.board, size=size, orientation=orientation)
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=out_path)

    def parse_user_move_with_llm(self, user_input: str, max_retries: int = 3) -> str:
        """
        Ask the LLM to convert free-form input (English, SAN, UCI) into a valid UCI move.
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
        # 0) Normalize any lowercase‐piece SAN (e.g., "nf3"→"Nf3", "ra8"→"Ra8", "qd6"→"Qd6")
        san = user_input.strip()
        if re.fullmatch(r"[nbrqk][a-h][1-8]", san, flags=re.IGNORECASE):
            # only uppercase the first char if it's a piece letter
            if san[0].islower():
                san = san[0].upper() + san[1:]
        # 1) Try local SAN parsing (e.g., "Nf3")
        try:
            move = self.board.parse_san(san)
            return move.uci()
        except ValueError:
            pass
        # 2) Try direct UCI parsing (e.g., "g1f3")
        try:
            move = chess.Move.from_uci(user_input)
            if move in self.board.legal_moves:
                return user_input
        except ValueError:
            pass
         # 3) Fallback to LLM-based parsing
        call_kwargs = {"model": self.model_name, "messages": messages}
        call_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name":        "move_parse",
                "description": "Parsed legal UCI move",
                "strict":      True,
                "schema":      MOVE_PARSE_SCHEMA
            }
        }
        if self.model_name not in NO_TEMPERATURE_MODELS:
            call_kwargs["temperature"] = self.temperature
        for _ in range(max_retries):
            resp = self.openai.chat.completions.create(
                **call_kwargs,
                # model=self.model_name,
                # messages=messages,
                # temperature=self.temperature,
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
        white_bottom: bool = True,
        chess960: bool = False,
        output_png: str = "next.png",
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

        # chess 960 (Fisher Random Chess) mode
        if chess960 and self.board is None:
            self.reset()
            idx = random.randint(0, 959)
            board = chess.Board.from_chess960_pos(idx)
            board.turn = chess.WHITE if side_to_move.lower() == "white" else chess.BLACK
            self.board = board
            orientation = chess.WHITE if user_side.lower() == "white" else chess.BLACK
            self.render_board_image(output_png, orientation=orientation)
            return {
                "mode": "chess960_init",
                "fen":     self.board.fen(),
                "description": self.describe_board(),
                "png":     output_png
            }

        # 2) image initialization
        if user_input and Path(user_input).is_file():
            image_path = Path(user_input)
            img_bytes = image_path.read_bytes()
            fen = self.infer_fen_from_image(image_path, white_bottom=white_bottom)
            self.board = chess.Board(fen)
            self.board.turn = chess.WHITE if side_to_move.lower() == "white" else chess.BLACK
            fen = self.board.fen()
            self.render_board_image(out_path=output_png, size=img_size[0])
            return {
                "mode":        "image_init",
                "fen":         fen,
                "description": self.describe_board(),
                "png":         output_png
            }

        # 3) default‐start if no board
        if self.board is None:
            if chess960:
                idx = random.randint(0, 959)
                board = chess.Board.from_chess960_pos(idx)
                board.turn = chess.WHITE if side_to_move.lower() == "white" else chess.BLACK
                self.board = board
            else:
                self.board = chess.Board()
                if side_to_move.lower() == "black":
                    self.board.turn = chess.BLACK
        orientation = chess.WHITE if user_side == "white" else chess.BLACK
        current_turn = "white" if self.board.turn == chess.WHITE else "black"
        # 4) if it's engine's turn, skip waiting for user_input
        if current_turn != user_side:
            # engine moves
            choice = self.decide_next_move(image_bytes=None)
            self.apply_move(choice["uci"])
            self.render_board_image(output_png,orientation=orientation)
            # ↳ check for game over right after engine move
            if self.board.is_checkmate():
                winner = "White" if self.board.turn == chess.BLACK else "Black"
                return {
                    "mode":      "game_over",
                    "result":    f"Checkmate! {winner} wins.",
                    "final_fen": self.board.fen(),
                    "png":       output_png
                }
            if self.board.is_stalemate():
                return {
                    "mode":      "game_over",
                    "result":    "Stalemate. Draw.",
                    "final_fen": self.board.fen(),
                    "png":       output_png
                }
            return {
                "mode":       "engine_move",
                "agent_move": choice["san"],
                "new_fen":    self.board.fen(),
                "png":        output_png
            }

        # 5) it's the user's turn: parse & validate their move
        # Attempt SAN/UCI parsing
        try:
            uci = self.parse_user_move_with_llm(user_input)
        except ValueError as err:                 # couldn’t parse the text
            self.render_board_image(output_png, orientation=orientation)
            return {
                "mode":        "parse_error",     # any name is fine; app.py falls back to description
                "description": str(err),          # e.g. 'Unable to parse a legal move from: Qf3'
                "png":         output_png
            }
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            # Illegal move: render board and report
            self.render_board_image(output_png, orientation=orientation)
            return {
                "mode":        "illegal_move",
                "description": f"Illegal move: {user_input!r}",
                "png":         output_png
            }

        # OK—apply user move and then engine reply
        # apply user's move and then engine reply
        self.board.push(move)
        # ↳ check for game over right after engine move
        if self.board.is_checkmate():
            self.render_board_image(output_png, orientation=orientation)
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return {
                "mode":      "game_over",
                "description":    f"Checkmate! {winner} wins.",
                "final_fen": self.board.fen(),
                "png":       output_png
            }
        if self.board.is_stalemate():
            self.render_board_image(output_png, orientation=orientation)
            return {
                "mode":      "game_over",
                "description":    "Stalemate. Draw.",
                "final_fen": self.board.fen(),
                "png":       output_png
            }
        try:
            choice = self.decide_next_move()
            self.apply_move(choice["uci"])
            self.render_board_image(output_png, orientation=orientation)
            return {
                "mode":       "move",
                "user_move":  uci,
                "agent_move": choice["san"],
                "new_fen":    self.board.fen(),
                "png":        output_png
            }
        except RuntimeError:
            self.render_board_image(output_png, orientation=orientation)
            return {
                "mode":        "engine_failed",
                "description": "Failed to get a legal engine move after user move due to runtime error",
                "png":         output_png
            }



