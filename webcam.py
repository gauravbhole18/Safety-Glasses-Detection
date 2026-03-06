import cv2
import time
from glasses_detector import GlassesDetector

detector = GlassesDetector()

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detector
    predictions = detector(
        image=rgb,
        format="float"
    )

    # Draw boxes
    h, w = frame.shape[:2]

    for i, box in enumerate(predictions):
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label per detection
        cv2.putText(
            frame,
            f"Glasses {i+1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Top overlay text
    cv2.putText(frame, f"Detections: {len(predictions)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Glasses Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
