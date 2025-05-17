import time

import cv2
from ultralytics import YOLO
import ollama

model = YOLO("yolov8m.pt")
llmModel = "artifish/llama3.2-uncensored"

cap = cv2.VideoCapture(0)
last_prompt_time = 0
cooldown = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    # boxes = results[0].boxes
    # object_names = [model.names[int(cls)] for cls in boxes.cls] if boxes.id is not None else []

    boxes = results[0].boxes
    object_names = []

    if boxes is not None and boxes.cls is not None:
        for cls_id in boxes.cls:
            cls_int = int(cls_id.item())
            object_names.append(model.names[cls_int])

    # print("Detected objects:", set(object_names))

    # Draw results on the frame
    annotated_frame = results[0].plot()



    current_time = time.time()
    if current_time - last_prompt_time > cooldown and object_names:
        unique_objects = sorted(set(object_names))
        object_list = ", ".join(unique_objects)

        # prompt = (
        #     f"You are an intelligent assistant. "
        #     f"The following objects are visible in a live camera feed: {object_list}. "
        #     f"Describe what might be happening in this scene."
        # )

        # prompt = f"""You are an intelligent assistant. The following objects are visible in a live camera feed: {object_list}. Describe what might be happening in this scene."""
        prompt = (
            f"From the list of detected objects: {object_list}, describe in one short sentence what is most likely happening in the scene. "
            f"Be vivid, clear, and avoid guessing or uncertainty. Do not explain — just describe the scene."
        )
        # print(prompt)

        response = ollama.chat(
            model=llmModel,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # stream=True,
            options={"num_predict": 50}
        )

        llama_response = response["message"]["content"].strip()
        print(llama_response)
        # for chunk in response:
        #     print(chunk['message']['content'], end='', flush=True)

        # print("\n🧠 Prompting LLaMA 3.2...\n")
        # response = llm(prompt, max_tokens=150)
        # text = response["choices"][0]["text"].strip()
        # print(f"📋 LLaMA 3.2: {response}\n")

        last_prompt_time = current_time

    cv2.imshow("Real-Time LLM OD", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
