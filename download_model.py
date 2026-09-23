from pathlib import Path
import requests
from config import MODEL_URL, MODEL_PATH

def download():
    path = Path(MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Model already exists: {path}")
        return

    print("Downloading face-mask model...")
    print(MODEL_URL)

    with requests.get(MODEL_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\rDownloaded {done / total * 100:.1f}%", end="")

    print(f"\nModel saved to: {path}")

if __name__ == "__main__":
    download()
