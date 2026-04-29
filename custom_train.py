from ultralytics import YOLO

# Load a pretrained YOLO26 model
model = YOLO("yolo26n.pt")

# Train on custom dataset
results = model.train(
   data="coco8.yaml",
   epochs=100,
   imgsz=640,
   batch=16,
   device='cpu',
   optimizer="auto",
   pretrained=True,
   project="runs/detect",  # Speichert im aktuellen Verzeichnis unter runs/detect/
   name="train"  # Name des Trainings-Ordners (wird automatisch nummeriert)
)