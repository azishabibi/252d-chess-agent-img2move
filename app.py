import os
import tempfile
from pathlib import Path

from fastapi import FastAPI
import gradio as gr
from gradio.routes import mount_gradio_app
from dotenv import load_dotenv

from agent import ChessPlayAgent

load_dotenv()

YOLO_WEIGHTS   = "yolov5-chess5/weights/best.pt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()
agent = ChessPlayAgent(
    yolo_weights=YOLO_WEIGHTS,
    openai_api_key=OPENAI_API_KEY
)

def gradio_play(image, move_text, side_to_move, user_side, restart_flag):
    # 1) Restart
    if restart_flag:
        agent.reset()
        return None, "Game restarted – upload a board or enter a move.", gr.update(value=False)

    # 2) Initialize from image *only if* no move_text provided
    if image is not None and (not move_text or move_text.strip() == ""):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name)
        out  = agent.handle_input(tmp.name, side_to_move, user_side, output_png="out.png")
        return None, f"Board initialized. {out.get('description','')}", gr.update(value=False)

    # 3) Otherwise, handle a move (either user or engine depending on turn)
    try:
        out = agent.handle_input(
            user_input=move_text or "",
            side_to_move=side_to_move,
            user_side=user_side,
            output_png="out.png"
        )
        if out["mode"] == "engine_move":
            # engine is moving first (side_to_move != user_side)
            engine_side = side_to_move  # "white" or "black"
            msg = f"{engine_side} plays: {out['agent_move']}"
        elif out["mode"] == "move":
            # user just moved, now engine (the opposite side) plays
            engine_side = "white" if user_side == "black" else "black"
            msg = f"You played {out['user_move']}. {engine_side} plays: {out['agent_move']}."
        else:
            msg = out.get("description", "")
        return out.get("png"), msg, gr.update(value=False)

    except Exception as e:
        return None, f"Error: {e}", gr.update(value=False)


with gr.Blocks() as demo:
    gr.Markdown("## Chess Agent Playground")

    with gr.Row():
        with gr.Column():
            board_in   = gr.Image(type="pil", label="Board Image")
            move_in    = gr.Textbox(label="Move", placeholder="e4 or move knight from g1 to f3")
            side_to    = gr.Radio(["white","black"], value="white", label="Side to move")
            user_side  = gr.Radio(["white","black"], value="white", label="Your side")
            restart    = gr.Checkbox(label="Restart game")
            play_btn   = gr.Button("Submit")

        with gr.Column():
            board_out  = gr.Image(label="Updated Board")
            feedback   = gr.Textbox(label="Status", interactive=False)

    play_btn.click(
        fn=gradio_play,
        inputs=[board_in, move_in, side_to, user_side, restart],
        outputs=[board_out, feedback, restart],
    )

# Mount Gradio onto FastAPI
app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
