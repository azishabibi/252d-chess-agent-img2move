import os
import tempfile
from pathlib import Path

from fastapi import FastAPI
import gradio as gr
from gradio.routes import mount_gradio_app
from dotenv import load_dotenv
from PIL import Image
from agent import ChessPlayAgent,  VALID_LLM_MODELS

load_dotenv()

YOLO_WEIGHTS   = "yolov5-chess5/weights/best.pt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"
app = FastAPI()
agent = ChessPlayAgent(
    yolo_weights=YOLO_WEIGHTS,
    openai_api_key=OPENAI_API_KEY,
    model_name=DEFAULT_MODEL,
)

def gradio_play(image, move_text, side_to_move, user_side, white_bottom, game_mode, model_choice,restart_flag):
    
    try:
        agent.set_model(model_choice)
    except ValueError as e:
        return None, f"Unsupported model: {e}", gr.update(value=False)
    
    # 1) Restart
    if restart_flag:
        agent.reset()
        return None, "Game restarted - upload a board or enter a move.", gr.update(value=False)

    # chess 960 initialization
    chess960 = (game_mode.lower() == "chess960")
    if agent.board is None and chess960:
            init_out = agent.handle_input(
                user_input="",
                side_to_move=side_to_move,
                user_side=user_side,
                white_bottom=white_bottom,
                chess960=True,
                output_png="out.png"
            )
            board_img = Image.open(init_out["png"])
            msg       = f"Chess960 board initialized. {init_out.get('description','')}"
            return board_img, msg, gr.update(value=False)
    # 2) Initialize from image 
    if image is not None and agent.board is None:
        # — save the uploaded image to disk
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        # — first, initialize the board from the image
        init_out = agent.handle_input(tmp_path, side_to_move, user_side, white_bottom, output_png="out.png")

        # — if the user also typed a move, play it now
        if move_text:
            move_out = agent.handle_input(
                user_input=move_text,
                side_to_move=side_to_move,
                user_side=user_side,
                white_bottom=white_bottom,
                output_png="out.png"
            )
            # build feedback
            if move_out["mode"] == "move":
                engine_side = "white" if user_side == "black" else "black"
                msg = f"You played {move_out['user_move']}. {engine_side.title()} plays: {move_out['agent_move']}."
            elif move_out["mode"] == "engine_move":
                msg = f"{side_to_move.title()} plays: {move_out['agent_move']}."
            else:
                msg = move_out.get("description", "")
            board_img = Image.open(move_out["png"])
            return board_img, msg, gr.update(value=False)

        # — if no move was provided, just return the initialized board
        board_img = Image.open(init_out["png"])
        return board_img, f"Board initialized. {init_out.get('description','')}", gr.update(value=False)
    # 3) Otherwise, handle a move (either user or engine depending on turn)
    try:
        out = agent.handle_input(
            user_input=move_text or "",
            side_to_move=side_to_move,
            user_side=user_side,
            white_bottom=white_bottom,
            output_png="out.png"
        )
        if out["mode"] == "engine_move":
            # engine is moving first (side_to_move != user_side)
            engine_side = side_to_move  # "white" or "black"
            msg = f"{engine_side.title()} plays: {out['agent_move']}."
        elif out["mode"] == "move":
            # user just moved, now engine (the opposite side) plays
            engine_side = "white" if user_side == "black" else "black"
            msg = f"You played {out['user_move']}. {engine_side.title()} plays: {out['agent_move']}."
        else:
            msg = out.get("description", "")
            
        return out.get("png"), msg, gr.update(value=False)

    except Exception as e:
        return None, f"Error: {e}", gr.update(value=False)

def refresh_chess960(side_to_move, user_side, white_bottom, model_choice):
    try:
        agent.set_model(model_choice)
    except ValueError:
        pass
    
    agent.reset()
    init_out = agent.handle_input(
        user_input="",
        side_to_move=side_to_move,
        user_side=user_side,
        white_bottom=white_bottom,
        chess960=True,
        output_png="out.png"
    )
    board_img = Image.open(init_out["png"])
    msg       = f"Chess960 board refreshed. {init_out.get('description','')}"
    return board_img, msg, gr.update(value=False)

def gradio_restart():
    agent.reset()
    return None, "Game restarted - upload a board or enter a move."

PREVIEW_SIZE = 400
MODEL_LIST   = sorted(VALID_LLM_MODELS)

with gr.Blocks() as demo:
    gr.Markdown("## Chess Agent Playground")

    with gr.Row():
        with gr.Column():
            board_in   = gr.Image(type="pil", label="Board Image",height=PREVIEW_SIZE)
            with gr.Row():
                move_in    = gr.Textbox(label="Move", placeholder="e4 or move knight from g1 to f3")
                with gr.Column():
                    play_btn   = gr.Button("Submit")
                    restart_btn = gr.Button("Restart Game")
                    
            with gr.Row():
                side_to    = gr.Radio(["white","black"], value="white", label="Side to move")
                user_side  = gr.Radio(["white","black"], value="white", label="Your side")
                game_mode = gr.Radio(["Standard","Chess960"], value="Standard", label="Game Mode")
            with gr.Row():
                model_dd = gr.Dropdown(MODEL_LIST, value=DEFAULT_MODEL, label="LLM Model")
                with gr.Column():
                    white_bot  = gr.Checkbox(label="White side at bottom",value=True)
                    refresh_btn = gr.Button("Refresh Board(chess960)")           
                    
        with gr.Column():
            board_out  = gr.Image(label="Updated Board",height=PREVIEW_SIZE)
            feedback   = gr.Textbox(label="Status", interactive=False)

    play_btn.click(
        fn=gradio_play,
        inputs=[board_in, move_in, side_to, user_side, white_bot, game_mode, model_dd],
        outputs=[board_out, feedback],
    )
    restart_btn.click(
        fn=gradio_restart,
        inputs=[],
        outputs=[board_out, feedback]
    )

    refresh_btn.click(
        fn=refresh_chess960,
        inputs=[side_to, user_side, white_bot],
        outputs=[board_out, feedback]
    )


# Mount Gradio onto FastAPI
app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
