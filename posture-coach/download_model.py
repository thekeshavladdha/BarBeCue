"""BarBeCue download_model.py — fetch the MediaPipe pose model (one command).

The .task file (~9 MB) is git-ignored, so every fresh clone needs it once:

    cd posture-coach
    python download_model.py

Saves to models/pose_landmarker_full.task by default. Re-run is a no-op
unless --force is given. Manual fallback: download the URL printed with
--print-url in a browser and place it at models/pose_landmarker_full.task.
"""
from __future__ import annotations
import argparse
import os
import sys
import urllib.request

MODEL_URL_FULL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_URL_LITE = MODEL_URL_FULL.replace("pose_landmarker_full", "pose_landmarker_lite")
MODEL_URL_HEAVY = MODEL_URL_FULL.replace("pose_landmarker_full", "pose_landmarker_heavy")

URLS = {"full": MODEL_URL_FULL, "lite": MODEL_URL_LITE, "heavy": MODEL_URL_HEAVY}
DEFAULT_OUT = os.path.join("models", "pose_landmarker_full.task")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB ({pct}%)", end="", flush=True)
            print()
    except Exception as exc:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(f"download failed: {exc}\nManual fallback: save {url} as {dest}") from exc
    os.replace(tmp, dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the MediaPipe pose model.")
    ap.add_argument("--model", default="full", choices=list(URLS),
                    help="which pose_landmarker variant (default: full)")
    ap.add_argument("--out", default=None, help="destination path")
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    ap.add_argument("--print-url", action="store_true", help="print the URL and exit")
    args = ap.parse_args(argv)

    url = URLS[args.model]
    if args.print_url:
        print(url)
        return 0
    dest = args.out or (DEFAULT_OUT if args.model == "full"
                        else os.path.join("models", f"pose_landmarker_{args.model}.task"))

    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000 and not args.force:
        print(f"exists ({os.path.getsize(dest)/1e6:.1f} MB): {dest} — nothing to do (use --force to re-download)")
        return 0

    print(f"downloading {args.model} model -> {dest}")
    print(f"  {url}")
    _download(url, dest)
    size = os.path.getsize(dest)
    if size < 1_000_000:
        print(f"WARNING: {dest} is only {size} bytes — download looks truncated, retry with --force", file=sys.stderr)
        return 1
    print(f"saved ({size/1e6:.1f} MB): {dest}")
    print("next: python src/app.py --backend static   # smoke test, no model needed")
    print("      python test_camera.py                  # camera check, no model needed")
    print("      python src/app.py                      # live coach, uses the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
