"""
Face Mask Detection AI
Author/Developer: Vikas Yadav

Run:
    pip install -r requirements.txt
    python app.py

The app uses OpenCV's built-in Haar Cascade face detector and a small
color/feature heuristic for a lightweight demo. Replace the detector
logic with a trained model in models/ when you have one.
"""
import cv2
import os
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def detect_mask(face):
    """Lightweight demo classifier.

    This is intentionally dependency-light. It estimates whether the lower
    face is visually covered; for production/academic accuracy, train and
    load a dedicated CNN model.
    """
    h, w = face.shape[:2]
    if h < 20 or w < 20:
        return "UNKNOWN", 0.0

    # Lower-face region; use saturation/value distribution as a simple demo signal.
    lower = face[int(h * 0.45):int(h * 0.95), :]
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    val = float(np.mean(hsv[:, :, 2]))

    # Conservative demo threshold. The UI labels this as a demo estimate.
    covered = sat < 95 and val < 190
    label = "MASK" if covered else "NO MASK"
    confidence = 0.78 if covered else 0.74
    return label, confidence

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened. Check camera permissions.")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    print("Face Mask Detection AI started.")
    print("Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            label, confidence = detect_mask(face)

            text = f"{label} ({confidence*100:.0f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, text, (x, max(30, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        cv2.putText(
            frame, "Face Mask Detection AI | Vikas Yadav",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        cv2.putText(
            frame, "Q = Quit | Demo classifier",
            (15, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )

        cv2.imshow("Face Mask Detection AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
