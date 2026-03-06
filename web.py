import cv2
import time
from glasses_detector import GlassesClassifier, GlassesDetector

# Initialize models
classifier = GlassesClassifier()
detector = GlassesDetector()

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. First classify: are glasses present?
    result = classifier(rgb)   # returns "present" or "absent"

    has_glasses = (result == "present")

    # Overlay classification result
    label_text = f"Glasses: {result.upper()}"
    color = (0, 255, 0) if has_glasses else (0, 0, 255)

    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 2. Only run detector if glasses are present
    predictions = []
    if has_glasses:
        predictions = detector(
            image=rgb,
            format="float"
        )

        # Draw bounding boxes
        h, w = frame.shape[:2]
        for box in predictions:
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Glasses Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
