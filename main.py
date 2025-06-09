import time

import cv2
import torch
from ultralytics import YOLO
import ollama

model = YOLO("yolov8x.pt")
llmModel = "artifish/llama3.2-uncensored"

cap = cv2.VideoCapture(0)
last_prompt_time = 0.0
cooldown = 5.0

# Move to MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, device=device)

    boxes = results[0].boxes
    object_names = []

    if boxes is not None and boxes.cls is not None:
        for cls_id in boxes.cls:
            cls_int = int(cls_id.item())
            object_names.append(model.names[cls_int])

    # Draw results on the frame
    annotated_frame = results[0].plot()

    current_time = time.time()
    if (current_time - last_prompt_time) > cooldown and object_names:
        unique_objects = sorted(set(object_names))
        object_list = ", ".join(unique_objects)

        prompt = f"""From the list of detected objects: {object_list}, describe in one short sentence what is most likely happening in the scene. Be vivid, clear, and avoid guessing or uncertainty. Do not explain — just describe the scene."""

        response = ollama.chat(
            model=llmModel,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # stream=True,
            options={"num_predict": 50}
        )

        llama_response = response["message"]["content"].strip()
        print("Llama: ", llama_response)

        # for chunk in response:
        #     print(chunk['message']['content'], end='', flush=True)

        last_prompt_time = current_time

    cv2.imshow("Real-Time LLM OD", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
