import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Face Mask Detection AI",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# Face detector
# -----------------------------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# -----------------------------
# Mask detection demo
# -----------------------------
def detect_mask(face):
    h, w = face.shape[:2]

    if h < 20 or w < 20:
        return "UNKNOWN", 0.0

    # Lower part of face
    lower = face[int(h * 0.45):int(h * 0.95), :]

    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

    saturation = float(np.mean(hsv[:, :, 1]))
    value = float(np.mean(hsv[:, :, 2]))

    # Lightweight demo estimation
    covered = saturation < 95 and value < 190

    if covered:
        return "MASK", 0.78
    else:
        return "NO MASK", 0.74


# -----------------------------
# UI
# -----------------------------
st.title("😷 Face Mask Detection AI")

st.write(
    "Take a photo using your camera and the AI will detect the face "
    "and estimate whether a mask is present."
)

st.info("📷 Click 'Take Photo' and allow camera permission.")


# -----------------------------
# Camera input
# -----------------------------
picture = st.camera_input("Take a photo")


if picture is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(picture.getvalue()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("❌ Could not read the camera image.")
        st.stop()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    result_count = 0

    # -----------------------------
    # Process faces
    # -----------------------------
    for (x, y, w, h) in faces:

        face = frame[y:y + h, x:x + w]

        label, confidence = detect_mask(face)

        result_count += 1

        # Box color
        if label == "MASK":
            box_color = (0, 200, 0)
        elif label == "NO MASK":
            box_color = (0, 0, 255)
        else:
            box_color = (255, 165, 0)

        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )

        # Label
        text = f"{label} - {confidence * 100:.0f}%"

        cv2.rectangle(
            frame,
            (x, max(0, y - 40)),
            (x + w, y),
            box_color,
            -1
        )

        cv2.putText(
            frame,
            text,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # Convert BGR → RGB
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show result
    st.image(
        result,
        caption="Face Mask Detection Result",
        use_container_width=True
    )

    # Result message
    if result_count == 0:

        st.warning("⚠️ No face detected. Please take another photo.")

    else:

        st.success(
            f"✅ {result_count} face(s) detected."
        )


# -----------------------------
# Information
# -----------------------------
st.divider()

st.caption(
    "Face Mask Detection AI | Developed by Vikas Yadav"
)

st.caption(
    "Note: This is a lightweight demo classifier based on visual features. "
    "For production-level accuracy, use a trained mask-detection model."
)
