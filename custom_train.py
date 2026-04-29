from ultralytics import YOLO

# Load a pretrained YOLO26 model
model = YOLO("yolo26n.pt")

# Train on custom dataset
results = model.train(
   data="data/cup.yaml",
   epochs=100,
   imgsz=640,
   batch=16,
   device='cpu',
   optimizer="auto",
   pretrained=True
)