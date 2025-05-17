from deepface import DeepFace

import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    object_names = [model.names[int(cls)] for cls in results.boxes.cls]
    # Draw results on the frame
    annotated_frame = results[0].plot()

    if "person" in object_names:
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                emotion = analysis[0]["dominant_emotion"]
            else:
                emotion = analysis["dominant_emotion"]
        except Exception as e:
            print("Emotion detection error:", e)
            emotion = None

    if emotion:
        cv2.putText(frame, f"Emotion: {emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    cv2.imshow("Real-Time LLM OD", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
