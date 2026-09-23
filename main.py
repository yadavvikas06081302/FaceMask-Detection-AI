import argparse
from pathlib import Path
import cv2

from mask_detector import MaskDetector
from config import DEFAULT_THRESHOLD

def save_image(frame, name="outputs/result.jpg"):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(name, frame)
    print(f"Saved: {name}")

def run_webcam(detector, camera=0, swap=False):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera}. Try --camera 1 or close other camera apps."
        )

    print("Webcam started. Q = quit, S = save snapshot.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        output, _, _ = detector.annotate(frame, swap_labels=swap)
        cv2.imshow("FaceMask Detection AI", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_image(output, "outputs/webcam_snapshot.jpg")

    cap.release()
    cv2.destroyAllWindows()

def run_image(detector, path, swap=False):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    output, mask_count, no_mask_count = detector.annotate(frame, swap_labels=swap)
    save_image(output, "outputs/image_result.jpg")
    print(f"MASK={mask_count}, NO_MASK={no_mask_count}")

    cv2.imshow("FaceMask Detection AI", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video(detector, path, save=True, swap=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25

    writer = None
    if save:
        out_path = "outputs/video_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving video to {out_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        output, _, _ = detector.annotate(frame, swap_labels=swap)
        if writer:
            writer.write(output)

        cv2.imshow("FaceMask Detection AI", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="FaceMask Detection AI")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--webcam", action="store_true")
    source.add_argument("--image")
    source.add_argument("--video")

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--swap-labels", action="store_true")
    parser.add_argument("--model", default="models/mask_detector.keras")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()
    detector = MaskDetector(args.model, args.threshold)

    if args.webcam:
        run_webcam(detector, args.camera, args.swap_labels)
    elif args.image:
        run_image(detector, args.image, args.swap_labels)
    elif args.video:
        run_video(detector, args.video, not args.no_save, args.swap_labels)

if __name__ == "__main__":
    main()
