from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from config import MODEL_PATH, INPUT_SIZE, DEFAULT_THRESHOLD

class MaskDetector:
    def __init__(self, model_path=MODEL_PATH, threshold=DEFAULT_THRESHOLD):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found: {model_file}. Run: python download_model.py"
            )

        self.model = tf.keras.models.load_model(model_file, compile=False)
        self.threshold = float(threshold)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)

        if self.face_detector.empty():
            raise RuntimeError("OpenCV Haar face detector could not be loaded.")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=5,
            minSize=(60, 60),
        )

    def predict_face(self, face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, INPUT_SIZE).astype("float32") / 255.0
        batch = np.expand_dims(image, axis=0)

        prediction = self.model.predict(batch, verbose=0)
        values = np.asarray(prediction).reshape(-1)

        if values.size == 1:
            p_mask = float(values[0])
        elif values.size >= 2:
            # Common binary Keras models return [no_mask, mask] or [mask, no_mask].
            # Default assumes the second output is mask probability.
            p_mask = float(values[-1])
        else:
            raise RuntimeError("The model returned an empty prediction.")

        p_mask = max(0.0, min(1.0, p_mask))
        is_mask = p_mask >= self.threshold
        return is_mask, p_mask

    def annotate(self, frame, swap_labels=False):
        result = frame.copy()
        mask_count = 0
        no_mask_count = 0

        for (x, y, w, h) in self.detect_faces(frame):
            face = frame[max(0,y):y+h, max(0,x):x+w]
            if face.size == 0:
                continue

            is_mask, p_mask = self.predict_face(face)

            if swap_labels:
                is_mask = not is_mask
                confidence = 1.0 - p_mask
            else:
                confidence = p_mask if is_mask else 1.0 - p_mask

            if is_mask:
                label = f"MASK  {confidence*100:.0f}%"
                mask_count += 1
            else:
                label = f"NO MASK  {confidence*100:.0f}%"
                no_mask_count += 1

            # Green for mask, red for no-mask.
            color = (40, 200, 40) if is_mask else (40, 40, 220)

            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(result, (x, max(0, y-34)), (x+w, y), color, -1)
            cv2.putText(
                result, label, (x+6, max(22, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2,
                cv2.LINE_AA
            )

        cv2.rectangle(result, (10, 10), (245, 78), (20,20,20), -1)
        cv2.putText(result, f"MASK: {mask_count}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (80,220,80), 2)
        cv2.putText(result, f"NO MASK: {no_mask_count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (80,80,230), 2)

        return result, mask_count, no_mask_count
