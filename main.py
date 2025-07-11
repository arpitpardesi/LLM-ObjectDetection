import time

import cv2
import torch
from deepface import DeepFace
from ultralytics import YOLO
import ollama

model = YOLO("yolov8m.pt")
llmModel = "llama3.2"

cap = cv2.VideoCapture(0)
last_prompt_time = 0.0
cooldown = 5.0

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, device=device)

    boxes = results[0].boxes
    object_names = []

    if boxes is not None and boxes.cls is not None and len(boxes.cls) > 0:
        for cls_id in boxes.cls:
            cls_int = int(cls_id.item())
            object_names.append(model.names[cls_int])

    # Draw results on the frame
    annotated_frame = results[0].plot()

    try:
        face_analysis = DeepFace.analyze(
            img_path=frame,  # Pass full frame
            actions=["age", "gender"],
            enforce_detection=False
        )

        age_gender_info = []
        for face in face_analysis:
            age = face["age"]
            gender = face["dominant_gender"]
            emotion = face.get("dominant_emotion", "?")
            age_gender_info.append(f"{gender}, {age} yrs, {emotion}")

            object_names.append(gender)
            # print(age, gender, emotion)

            # Draw bounding box and label
            region = face["region"]
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"{gender}, {age} yrs", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    except Exception as e:
        print("Face analysis failed:", e)
        age_gender_info = []

    current_time = time.time()
    if (current_time - last_prompt_time) > cooldown and object_names:
        unique_objects = sorted(set(object_names))
        object_list = ", ".join(unique_objects)

        # prompt = f"""From the list of detected objects: {object_list}, describe in one short sentence what is most likely happening in the scene. Be vivid, clear, and avoid guessing or uncertainty. Do not explain — just describe the scene."""

        prompt = (
            f"Given these detected objects: {object_list}. "
            "Briefly describe what is most likely happening in the scene in one vivid sentence. "
            "Avoid guessing, speculation, or explanation. Just describe."
        )

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
