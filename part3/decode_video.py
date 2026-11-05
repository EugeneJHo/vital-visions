#!/usr/bin/env python3
"""Read an iPhone video of the transmitter and score the bits.

  .venv/bin/python part3/decode_video.py path/to/clip.MOV
  .venv/bin/python part3/decode_video.py --summary

The clip is matched to the newest saved transmission unless you pass
--session. Point the phone so the picture fills the frame, and leave the
flash off.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bfsk import decode_samples

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "sessions"
RESULTS = ROOT / "results.csv"
FIELDS = [
    "time",
    "video",
    "session",
    "distance_cm",
    "delta_alpha",
    "bit_duration",
    "n_bits",
    "n_correct",
    "ber",
    "data_rate_bps",
    "sync_score",
    "message",
    "decoded",
]


def read_brightness(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not 5 <= fps <= 240:
        fps = 30.0
    values: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        crop = gray[int(height * 0.25) : int(height * 0.75), int(width * 0.25) : int(width * 0.75)]
        values.append(float(crop.mean()))
    cap.release()
    if len(values) < 10:
        raise SystemExit(f"{path} has too few frames ({len(values)})")
    times = np.arange(len(values), dtype=float) / fps
    return times, np.asarray(values), fps


def load_session(path: Path) -> dict:
    session = json.loads(path.read_text())
    session["bits"] = [int(bit) for bit in session["bits"]]
    return session


def newest_session() -> Path:
    latest = SESSIONS / "latest.json"
    if latest.is_file():
        return latest
    files = sorted(SESSIONS.glob("*.json"), key=lambda item: item.stat().st_mtime)
    if not files:
        raise SystemExit("No saved transmission. Start one from transmitter.html first.")
    return files[-1]


def append_result(row: dict) -> None:
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def print_summary() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"No results yet. Decode a video first. Expected {RESULTS}")
    with RESULTS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("results.csv is empty")
    print(f"{'alpha':>6} {'cm':>5} {'bit s':>6} {'BER':>7} {'bits/s':>7}  message")
    for row in rows:
        print(
            f"{float(row['delta_alpha']):6.1f} {int(float(row['distance_cm'])):5d} "
            f"{float(row['bit_duration']):6.2f} {float(row['ber']):7.3f} "
            f"{float(row['data_rate_bps']):7.2f}  {row['decoded'] or '—'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a screen-to-phone BFSK video")
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary and args.video is None:
        print_summary()
        return
    if args.video is None:
        parser.error("pass a video, or --summary")

    session_path = args.session or newest_session()
    session = load_session(session_path)
    times, values, fps = read_brightness(args.video)
    result = decode_samples(
        times,
        values,
        session["bits"],
        float(session["bit_duration"]),
        float(session["delta_alpha"]),
        float(session.get("f0", 5)),
        float(session.get("f1", 8)),
    )
    from datetime import datetime, timezone

    append_result(
        {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "video": str(args.video),
            "session": session.get("id", session_path.stem),
            "distance_cm": session.get("distance_cm", ""),
            "delta_alpha": session["delta_alpha"],
            "bit_duration": session["bit_duration"],
            "n_bits": result["n_bits"],
            "n_correct": result["n_correct"],
            "ber": f"{result['ber']:.4f}",
            "data_rate_bps": f"{result['data_rate_bps']:.3f}",
            "sync_score": f"{result['sync_score']:.3f}",
            "message": session.get("message", ""),
            "decoded": result["message"],
        }
    )
    print(f"Video: {args.video.name}    {len(values)} frames at {fps:.2f} fps")
    print(
        f"Session {session.get('id', session_path.stem)}: "
        f"\"{session.get('message', '')}\"  "
        f"delta alpha {session['delta_alpha']}  "
        f"{session.get('distance_cm', '?')} cm  "
        f"{session['bit_duration']} s/bit"
    )
    if result["ok"]:
        print(f"Decoded: {result['message']}")
    else:
        print("Decoded: (the picture was not found in this clip)")
    print(
        f"Bits sent {result['n_bits']}, correct {result['n_correct']}, "
        f"BER {result['ber']:.3f}, data rate {result['data_rate_bps']:.2f} bits/s, "
        f"match {result['sync_score']:.2f}"
    )
    print(f"Appended {RESULTS}")
    if args.summary:
        print()
        print_summary()


if __name__ == "__main__":
    main()
