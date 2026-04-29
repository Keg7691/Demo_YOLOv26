import cv2
from ultralytics import YOLO

# demo for detection and tracking with YOLOv26n 

# models
""" yolo26n.pt 
yolo26s.pt 
yolo26m.pt 
yolo26l.pt 
yolo26x.pt """

# Modell laden
model = YOLO("yolo26n.pt")

# Webcam öffnen
cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Inferenz auf aktuellem Frame
    # results = model(frame)  # live detections
    results = model.track(frame, persist=True)  # tracking 

    # Ergebnisse zeichnen
    annotated_frame = results[0].plot()

    # Anzeigen
    cv2.imshow("YOLOv26 Live Demo", annotated_frame)

    # Beenden mit q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()