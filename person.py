import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze face using DeepFace with age, gender, and emotion
        face_analysis = DeepFace.analyze(
            img_path=frame,
            actions=["age", "gender", "emotion"],
            enforce_detection=False
        )

        # Normalize to list format
        if isinstance(face_analysis, dict):
            face_analysis = [face_analysis]

        for face in face_analysis:
            age = face.get("age", "?")
            gender = face.get("dominant_gender", "?")
            emotion = face.get("dominant_emotion", "?")
            region = face.get("region", {})

            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label: Gender, Age, Emotion
            label = f"{gender}, {age} yrs, {emotion}"
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    except Exception as e:
        print("Face analysis failed:", e)

    # Display result
    cv2.imshow("Real-Time Age, Gender & Emotion Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()