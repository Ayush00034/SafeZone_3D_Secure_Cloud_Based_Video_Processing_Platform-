"""
local_capture.py
────────────────
Run this on your LAPTOP (Windows, connected to both Logitech C270 cameras).
It saves a raw side-by-side stereo video: 2560x720, mp4.
That file is what you upload to the website.

Requirements:
    pip install opencv-python
"""

import cv2
import numpy as np
import time
from datetime import datetime

print("=" * 60)
print(" SAFEZONE 3D — RAW STEREO CAPTURE")
print("=" * 60)

# ── CAMERAS ──────────────────────────────────────────────
cap_left  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("❌  One or both cameras not found!")
    print("    Check USB connections and try camera index 0,1 or 1,2")
    exit()

current_exposure = -6

for cap in [cap_left, cap_right]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,   720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,       1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,    0)   # 0 = manual on DirectShow
    cap.set(cv2.CAP_PROP_AUTO_WB,          0)
    cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)

print("✅  Cameras ready — stabilising sensor...")
time.sleep(2)

# ── OUTPUT FILE ───────────────────────────────────────────
# Timestamped filename so each run produces a unique file.
ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = f"raw_stereo_{ts}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT, fourcc, 15.0, (2560, 720))

print(f"📁  Output file: {OUT}")
print()
print("   R         — start / stop recording")
print("   Q         — quit and save")
print("   [ / ]     — decrease / increase exposure")
print("=" * 60)

is_recording = False
frame_count  = 0

while True:
    cap_left.grab()
    cap_right.grab()
    _, frameL = cap_left.retrieve()
    _, frameR = cap_right.retrieve()

    # Stitch left | right side-by-side → 2560 × 720
    stereo = np.hstack((frameL, frameR))

    if is_recording:
        writer.write(stereo)
        frame_count += 1

    # ── DISPLAY PREVIEW (scaled to fit laptop screen) ─────
    preview = cv2.resize(stereo, (1280, 360))

    if is_recording:
        cv2.circle(preview, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(preview, f"REC  {frame_count}f  EXP:{current_exposure}",
                    (55, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(preview, f"READY  EXP:{current_exposure}  |  Press R to record",
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imshow("Stereo Capture Preview", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):
        is_recording = not is_recording
        if is_recording:
            print(f"🔴  Recording STARTED")
        else:
            print(f"⏸   Recording PAUSED  ({frame_count} frames so far)")

    elif key == ord(']'):
        current_exposure = min(current_exposure + 1, 0)
        for c in [cap_left, cap_right]:
            c.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
        print(f"🔆  Exposure → {current_exposure}")

    elif key == ord('['):
        current_exposure = max(current_exposure - 1, -13)
        for c in [cap_left, cap_right]:
            c.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
        print(f"🔅  Exposure → {current_exposure}")

# ── CLEANUP ───────────────────────────────────────────────
cap_left.release()
cap_right.release()
writer.release()
cv2.destroyAllWindows()

print()
print(f"✅  Saved  →  {OUT}  ({frame_count} frames)")
print(f"    Upload this file to the SafeZone 3D website.")
