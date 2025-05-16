import cv2
from ultralytics import YOLO
import ollama

model = YOLO("yolov8s.pt")
llmModel = "artifish/llama3.2-uncensored"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    cv2.imshow("Real-Time LLM OD", annotated_frame)

    prompt = (
        f"You are an intelligent assistant. "
        f"The following objects are visible in a live camera feed: {object_list}. "
        f"Describe what might be happening in this scene."
    )

    response = ollama.chat(
        # model='llama3.2',
        model=llmModel,
        messages=prompt,
        stream=True
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
