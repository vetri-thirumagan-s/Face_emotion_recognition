from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
results = model.train(data="<dataset_dir>/data.yaml", epochs=150, batch=128)