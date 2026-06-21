# Setup Notes

## ⚠ Before you run this: `stereo_map.xml` will NOT work for you

`stereo_map.xml` is **not included in this repo**, and even if it were, **it would not work on your machine.**

This file is the output of stereo camera calibration done specifically for *this* pair of two Logitech C270 cameras, at *this* exact baseline (19.7 cm apart), at *this* exact resolution (1280×720 per eye), with *these* exact lenses. It contains the rectification maps (`stereoMapL_x`, `stereoMapL_y`, `stereoMapR_x`, `stereoMapR_y`) generated from a chessboard calibration pass — every value in it is tied to the physical geometry of the cameras used to create it.

If you clone this repo and try to run it with your own cameras using my `stereo_map.xml`:
- Disparity values will be wrong
- Distance/height math will be wrong (it depends on calibrated focal length + baseline)
- In the worst case, `cv2.remap()` will throw a shape mismatch error if your camera resolution differs

**You must generate your own calibration file** if you want to run the stereo pipeline with your own hardware:
1. Print a chessboard calibration pattern (9×6 or similar).
2. Capture 20–30 image pairs from both cameras at various angles/distances.
3. Run OpenCV's `cv2.stereoCalibrate()` + `cv2.stereoRectify()` to generate the rectification maps.
4. Save the maps to a `stereo_map.xml` using `cv2.FileStorage`, matching the node names used in `aws_processor.py`:
   - `stereoMapL_x`, `stereoMapL_y`, `stereoMapR_x`, `stereoMapR_y`
5. Update `BASELINE_CM` and `FOCAL` constants in `aws_processor.py` to match your setup.

## `best.pt` (YOLOv8 weights)

Also not included — it's a custom-trained model for detecting speed humps and is too large/specific to commit. Train your own YOLOv8 model on your own hump dataset, or substitute a stock YOLOv8 model and adjust the detection class filtering in `aws_processor.py` accordingly.

## Where these files go

Both files must sit in the **same directory as `app.py` and `aws_processor.py`** on the backend EC2 instance:

```
/app/
├── app.py
├── aws_processor.py
├── stereo_map.xml     ← your calibration file goes here
├── best.pt             ← your trained weights go here
└── requirements.txt
```

The app will fail to start (or throw on first upload) if either file is missing.

## Environment variables

Set these on the backend EC2 (e.g. in `/etc/environment` or your systemd unit):

| Variable | Default | Purpose |
|---|---|---|
| `S3_BUCKET` | `safezone3d-videos-ayush` | Bucket for processed output videos |
| `S3_REGION` | `ap-south-1` | AWS region |
| `DYNAMO_TABLE` | `safezone-detections` | DynamoDB table for detection metadata |

## Install

```bash
pip3 install flask flask-cors boto3 ultralytics \
             opencv-python-headless \
             opencv-contrib-python-headless
```

`opencv-contrib-python-headless` is required — `cv2.ximgproc` (used for the WLS disparity filter) is not in the base `opencv-python-headless` package.

## Full AWS deployment

For step-by-step VPC/EC2/S3/DynamoDB/Lambda/Route53 setup, see the full deployment guide (not included in this public repo — kept private since it contains account-specific naming).