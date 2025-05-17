import torch
# print(torch.backends.mps.is_available())  # Should return True
import cv2
from ultralytics import YOLO
import ollama

cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")  # or yolov8n.pt for faster performance

# Move to MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Run inference
# results = model("bus.jpg", device=device)

# Show result

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    results[0].show()

    # Draw results on the frame
    annotated_frame = results[0].plot()

    cv2.imshow("Real-Time LLM OD", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()