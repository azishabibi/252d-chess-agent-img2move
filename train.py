from ultralytics import YOLO
import torch
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available—check drivers and PyTorch build."
    # load a YOLOv5 model (you can also specify a custom .pt checkpoint here)
    #model = YOLO('yolov5x6u.pt')
    #model = YOLO('yolov5m6u.pt')  # or yolov5m.pt, yolov5l.pt, yolov5x.pt
    #model = YOLO('yolov5s6u.pt')  # or yolov5s.pt, yolov5l.pt, yolov5x.pt
    model = YOLO('yolov5n6u.pt')  # or yolov5s.pt, yolov5l.pt, yolov5x.pt
    # train on your custom chess dataset
    model.train(
        data='dataset.yaml',      # our config
        epochs=3,                   # how long to train 2 or 3 epochs should be enough
        imgsz=400,                   # image size
        batch=16,                    # batch size (adjust for GPU memory)
        device='cuda:0',                  # device ID (0 for first GPU, 'cpu' for CPU)
        project='runs/chess',        # where to save results
        name='yolov5-chess',         # experiment name
        exist_ok=False,               # overwrite existing runs
        augment=True,                # built-in YOLOv5 augment
    )
