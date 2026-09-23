MODEL_URL = (
    "https://huggingface.co/Sooraj-jain/face-mask-detector/"
    "resolve/main/mask_detector.keras"
)

MODEL_PATH = "models/mask_detector.keras"

# Face detector settings
FACE_SCALE_FACTOR = 1.10
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (60, 60)

# Classification settings
DEFAULT_THRESHOLD = 0.50
INPUT_SIZE = (224, 224)

# If your downloaded model gives reversed labels, use --swap-labels.
LABEL_MASK = "MASK"
LABEL_NO_MASK = "NO MASK"
