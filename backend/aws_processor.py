"""
aws_processor.py
────────────────
Runs on AWS Backend EC2 (VPC-A private subnet).
Called by app.py after the uploaded video lands in /tmp/.

Takes the raw 2560x720 side-by-side stereo mp4 from the website upload,
runs the full SafeZone 3D pipeline (your exact CLAHE + SGBM + WLS + YOLO math),
and writes ONE combined dashboard mp4:

    Left half  (1280×720)  =  detection HUD  (bounding boxes, distance, height)
    Right half  (900×720)  =  height profile graph

The combined file is 2180×720 and plays in any browser as a single video.
It is also uploaded to S3 so the user can download it.

Requirements on EC2:
    pip3 install flask flask-cors boto3 ultralytics opencv-python-headless opencv-contrib-python-headless

Files needed in the same directory:
    stereo_map.xml     (your calibration file — copy from your project)
    best.pt            (your YOLOv8 weights)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os


def process(input_path: str, output_path: str) -> dict:
    """
    Process a raw stereo video and write the combined dashboard output.

    Parameters
    ----------
    input_path  : str   Full path to the uploaded 2560×720 mp4
    output_path : str   Full path where the output dashboard mp4 will be written

    Returns
    -------
    dict  { peak_height_cm, total_frames, humps_detected }
    """

    print(f"[processor] START  input={input_path}")

    # ── 1. CALIBRATION ───────────────────────────────────────
    calib_path = os.path.join(os.path.dirname(__file__), "stereo_map.xml")
    cv_file = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
    stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
    stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
    stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()
    cv_file.release()
    print("[processor] Calibration loaded")

    # ── 2. YOLO ──────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    model = YOLO(model_path)
    print("[processor] YOLO loaded")

    # ── 3. STEREO PIPELINE — YOUR EXACT PARAMS ───────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=256, blockSize=9,
        P1=8 * 3 * 9 ** 2, P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1, uniquenessRatio=5,
        speckleWindowSize=50, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000)
    wls.setSigmaColor(1.5)

    # ── 4. CONSTANTS — YOUR EXACT VALUES ─────────────────────
    BASELINE_CM = 19.7
    FOCAL       = 1463.0
    DISP_MIN    = 20.0
    DISP_MAX    = 400.0

    dist_buffer   = deque(maxlen=30)
    height_buffer = deque(maxlen=30)

    GRAPH_W, GRAPH_H = 900, 500
    graph_heights   = []
    frozen_graph    = None
    graph_frozen    = False
    final_height_cm = None

    # ── 5. GRAPH DRAWING — EXACT COPY FROM YOUR CODE ─────────
    def draw_graph(heights, final_h, frozen=False):
        img = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)
        for i in range(1, 8):
            cv2.line(img, (0, int(i * GRAPH_H / 8)),
                     (GRAPH_W, int(i * GRAPH_H / 8)), (30, 30, 30), 1)

        title = "HEIGHT PROFILE  [Dynamic Zone]"
        if frozen:
            title += "  *** FROZEN ***"
        cv2.putText(img, title, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        if len(heights) < 2:
            cv2.putText(img, "Waiting for hump in measurement zone...",
                        (20, GRAPH_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            return img

        display_h   = final_h if final_h is not None else float(np.median(heights))
        max_scale_h = max(25.0, display_h * 1.5)
        margin      = 120
        curve_w     = GRAPH_W - 2 * margin
        base_y      = GRAPH_H - 80
        pts         = []

        for x in range(margin, margin + curve_w + 1):
            theta  = np.pi * ((x - margin) / curve_w)
            h_at_x = display_h * np.sin(theta)
            y      = int(base_y - (h_at_x / max_scale_h) * (GRAPH_H - 160))
            pts.append((x, y))

        if pts:
            pts_fill = [(margin, base_y)] + pts + [(margin + curve_w, base_y)]
            cv2.fillPoly(img, [np.array(pts_fill, np.int32)], (0, 60, 20))
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], (0, 255, 0), 2)

            peak_y = int(base_y - (display_h / max_scale_h) * (GRAPH_H - 160))
            cv2.line(img, (margin - 30, peak_y), (margin + curve_w + 30, peak_y),
                     (0, 255, 255), 1)
            text      = f"Peak Median: {display_h:.1f} cm"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x    = (GRAPH_W - text_size[0]) // 2
            cv2.putText(img, text, (text_x, peak_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(img, f"{max_scale_h:.0f}cm", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.line(img, (50, 60), (50, base_y), (100, 100, 100), 2)
        cv2.putText(img, "0cm", (15, base_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(img, f"Calculated Height:   {display_h:.2f} cm",
                    (20, GRAPH_H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img

    # ── 6. OPEN INPUT VIDEO ──────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # should be 2560
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # should be 720

    single_w = width // 2   # 1280 — left camera frame width

    # Output: HUD (1280×720) stitched with Graph (900×720) = 2180×720
    OUT_W = single_w + GRAPH_W   # 2180
    OUT_H = 720

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))

    print(f"[processor] Video  {width}×{height}  @{fps}fps")
    print(f"[processor] Output {OUT_W}×{OUT_H}")

    # ── 7. FRAME LOOP ────────────────────────────────────────
    total_frames    = 0
    humps_detected  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Split the 2560-wide frame back into left and right 1280-wide frames
        frameL = frame[:, :single_w]
        frameR = frame[:, single_w:]

        # Rectify
        rectL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR)

        # CLAHE → gray
        grayL = clahe.apply(cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY))
        grayR = clahe.apply(cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY))

        # Disparity
        dispL = left_matcher.compute(grayL, grayR)
        dispR = right_matcher.compute(grayR, grayL)
        disp  = wls.filter(dispL, rectL, None, dispR).astype(np.float32) / 16.0

        # YOLO
        results = model.predict(rectL, conf=0.45, verbose=False)

        smooth_dist   = None
        smooth_height = None

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi   = disp[y1:y2, x1:x2]
                valid = roi[(roi > DISP_MIN) & (roi < DISP_MAX)]

                if len(valid) < 30:
                    cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(rectL, "HUMP - measuring...", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    continue

                humps_detected += 1

                d          = np.median(valid)
                dist_cm    = (FOCAL * BASELINE_CM) / d
                raw_dist_m = dist_cm / 100.0

                top_band    = roi[0:20, :]
                top_band    = top_band[(top_band > DISP_MIN) & (top_band < DISP_MAX)]
                bottom_band = roi[-20:, :]
                bottom_band = bottom_band[(bottom_band > DISP_MIN) & (bottom_band < DISP_MAX)]

                raw_height = None
                if len(top_band) > 5 and len(bottom_band) > 5:
                    Zt         = (FOCAL * BASELINE_CM) / np.median(top_band)
                    Zb         = (FOCAL * BASELINE_CM) / np.median(bottom_band)
                    pixel_h    = (y2 - y1)
                    raw_height = (pixel_h * ((Zt + Zb) / 2)) / FOCAL

                if len(dist_buffer) > 0 and \
                        abs(raw_dist_m - float(np.median(dist_buffer))) > 1.0:
                    dist_buffer.clear()
                    height_buffer.clear()

                dist_buffer.append(raw_dist_m)
                if raw_height is not None:
                    height_buffer.append(raw_height)

                smooth_dist   = np.median(dist_buffer)
                smooth_height = np.median(height_buffer) if height_buffer else None

                if not graph_frozen and smooth_height is not None \
                        and 1.0 <= smooth_dist <= 1.5:
                    graph_heights.append(smooth_height)
                    final_height_cm = float(np.median(graph_heights))

                if not graph_frozen and smooth_dist < 1.0 \
                        and len(graph_heights) > 3:
                    graph_frozen = True
                    frozen_graph = draw_graph(graph_heights, final_height_cm,
                                              frozen=True)

                cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── HUD OVERLAY ──────────────────────────────────────
        if smooth_dist is not None:
            if   smooth_dist <= 1.5: label = f"!! HUMP AHEAD  {smooth_dist:.2f}m"; color = (0, 60, 255)
            elif smooth_dist <= 3.0: label = f"HUMP AHEAD  {smooth_dist:.2f}m";    color = (0, 165, 255)
            else:                    label = f"HUMP DETECTED  {smooth_dist:.2f}m"; color = (0, 255, 0)

            cv2.rectangle(rectL, (8, 8), (280, 105), (0, 0, 0), -1)
            cv2.putText(rectL, label, (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if smooth_height is not None:
                cv2.putText(rectL, f"Height: {smooth_height:.1f} cm", (15, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(rectL, "Scanning...", (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        cv2.putText(rectL, "SafeZone 3D  |  YOLOv8 + StereoSGBM", (10, 710),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ── GRAPH FRAME ──────────────────────────────────────
        raw_graph = frozen_graph if (graph_frozen and frozen_graph is not None) \
            else draw_graph(graph_heights, final_height_cm)

        # The graph is 900×500; pad vertically to 720 so it matches HUD height
        padded_graph           = np.zeros((720, GRAPH_W, 3), dtype=np.uint8)
        graph_y_offset         = (720 - GRAPH_H) // 2   # = 110
        padded_graph[graph_y_offset: graph_y_offset + GRAPH_H, :] = raw_graph

        # ── STITCH HUD + GRAPH → DASHBOARD ───────────────────
        dashboard = np.hstack((rectL, padded_graph))   # 2180 × 720
        writer.write(dashboard)

        if total_frames % 30 == 0:
            print(f"[processor]  frame {total_frames}  dist={smooth_dist}  h={smooth_height}")

    cap.release()
    writer.release()

    result = {
        "peak_height_cm": round(final_height_cm, 2) if final_height_cm else 0.0,
        "total_frames":   total_frames,
        "humps_detected": humps_detected,
    }
    print(f"[processor] DONE  {result}")
    return result
