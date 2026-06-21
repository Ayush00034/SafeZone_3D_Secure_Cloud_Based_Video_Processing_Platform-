"""
app.py
──────
Full Flask backend for SafeZone 3D.
Runs on Backend EC2 (VPC-A private subnet 10.0.1.x).

Start:
    python3 app.py
    -- or --
    gunicorn -w 1 -b 0.0.0.0:5000 --timeout 600 app:app

Environment variables (set in /etc/environment or systemd unit):
    S3_BUCKET      safezone3d-videos-<yourname>
    S3_REGION      ap-south-1
    DYNAMO_TABLE   safezone-detections

Required files in same directory:
    stereo_map.xml
    best.pt
    aws_processor.py

Install:
    pip3 install flask flask-cors boto3 ultralytics \
                 opencv-python-headless \
                 opencv-contrib-python-headless
"""

import os
import uuid
import threading
from datetime import datetime
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS

import aws_processor   # our processing module

# ── CONFIG ────────────────────────────────────────────────
S3_BUCKET  = os.environ.get("S3_BUCKET",    "safezone3d-videos-ayush")
S3_REGION  = os.environ.get("S3_REGION",    "ap-south-1")
TABLE_NAME = os.environ.get("DYNAMO_TABLE", "safezone-detections")
TMP_DIR    = "/tmp/safezone"

os.makedirs(TMP_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)   # Allow requests from VPC-B Nginx

s3    = boto3.client("s3",          region_name=S3_REGION)
ddb   = boto3.resource("dynamodb",  region_name=S3_REGION)
table = ddb.Table(TABLE_NAME)

# ── IN-MEMORY JOB TRACKER ─────────────────────────────────
# { job_id: { status, progress_pct, message, output_key, peak_height_cm, error } }
jobs = {}


# ══════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "time": datetime.utcnow().isoformat()})


# ══════════════════════════════════════════════════════════
#  UPLOAD — receive video from browser, kick off processing
# ══════════════════════════════════════════════════════════
@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Receives multipart/form-data with field 'video'.
    Saves to /tmp/safezone/, kicks off background processing thread.
    Returns immediately with { job_id } so the browser can poll /api/job/<id>.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    f       = request.files["video"]
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id  = str(uuid.uuid4())[:8]

    # Save upload to /tmp
    in_path  = os.path.join(TMP_DIR, f"upload_{ts}_{job_id}.mp4")
    out_path = os.path.join(TMP_DIR, f"output_{ts}_{job_id}.mp4")
    f.save(in_path)

    # Register job
    jobs[job_id] = {
        "status":       "queued",
        "progress_pct": 0,
        "message":      "Queued",
        "output_key":   None,
        "peak_height_cm": 0.0,
        "error":        None,
        "timestamp":    ts,
    }

    # Background thread — process then upload to S3
    t = threading.Thread(target=_process_job,
                         args=(job_id, in_path, out_path, ts),
                         daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "message": "Processing started"})


def _process_job(job_id, in_path, out_path, ts):
    """Background worker: process video → upload to S3 → save to DynamoDB."""
    job = jobs[job_id]

    try:
        # ── STEP 1: process ──────────────────────────────
        job["status"]       = "processing"
        job["progress_pct"] = 10
        job["message"]      = "Running YOLO + StereoSGBM pipeline"

        result = aws_processor.process(in_path, out_path)

        job["progress_pct"]    = 70
        job["peak_height_cm"]  = result["peak_height_cm"]
        job["message"]         = "Uploading to S3"

        # ── STEP 2: upload output to S3 ──────────────────
        s3_key = f"outputs/dashboard_{ts}_{job_id}.mp4"
        s3.upload_file(
            out_path, S3_BUCKET, s3_key,
            ExtraArgs={
                "ContentType": "video/mp4",
                "Metadata": {
                    "peak_height_cm": str(result["peak_height_cm"]),
                    "total_frames":   str(result["total_frames"]),
                    "humps_detected": str(result["humps_detected"]),
                }
            }
        )
        job["output_key"]   = s3_key
        job["progress_pct"] = 90
        job["message"]      = "Saving metadata"

        # ── STEP 3: write to DynamoDB ─────────────────────
        table.put_item(Item={
            "detection_id":   f"{ts}_{job_id}",
            "timestamp":      ts,
            "output_key":     s3_key,
            "peak_height_cm": Decimal(str(result["peak_height_cm"])),
            "total_frames":   result["total_frames"],
            "humps_detected": result["humps_detected"],
            "upload_time":    datetime.utcnow().isoformat(),
            "status":         "complete",
        })

        job["status"]       = "done"
        job["progress_pct"] = 100
        job["message"]      = "Complete"

    except Exception as e:
        job["status"]  = "error"
        job["error"]   = str(e)
        job["message"] = f"Error: {e}"
        print(f"[job {job_id}] ERROR: {e}")

    finally:
        # Clean up temp files
        for p in [in_path, out_path]:
            try:
                os.remove(p)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════
#  JOB POLLING
# ══════════════════════════════════════════════════════════
@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id):
    """Poll this endpoint after upload to get progress + output key."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


# ══════════════════════════════════════════════════════════
#  VIDEO STREAMING — with HTTP Range support for seek
# ══════════════════════════════════════════════════════════
@app.route("/api/stream/<path:s3_key>")
def stream_video(s3_key):
    """
    Stream an mp4 from S3 to the browser.
    Supports Range requests so the HTML5 <video> element can seek.
    """
    try:
        head      = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        file_size = head["ContentLength"]
        ctype     = "video/mp4"

        range_header = request.headers.get("Range")

        if range_header:
            parts       = range_header.replace("bytes=", "").split("-")
            start       = int(parts[0])
            end         = int(parts[1]) if parts[1] else file_size - 1
            end         = min(end, file_size - 1)
            chunk_size  = end - start + 1

            s3_resp = s3.get_object(
                Bucket=S3_BUCKET, Key=s3_key,
                Range=f"bytes={start}-{end}"
            )

            def gen():
                for chunk in s3_resp["Body"].iter_chunks(chunk_size=65536):
                    yield chunk

            return Response(
                stream_with_context(gen()), status=206,
                headers={
                    "Content-Range":  f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges":  "bytes",
                    "Content-Length": str(chunk_size),
                    "Content-Type":   ctype,
                }
            )
        else:
            s3_resp = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)

            def gen():
                for chunk in s3_resp["Body"].iter_chunks(chunk_size=65536):
                    yield chunk

            return Response(
                stream_with_context(gen()), status=200,
                headers={
                    "Content-Length": str(file_size),
                    "Accept-Ranges":  "bytes",
                    "Content-Type":   ctype,
                }
            )

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return jsonify({"error": "File not found"}), 404
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════
#  PRESIGNED DOWNLOAD URL
# ══════════════════════════════════════════════════════════
@app.route("/api/download_url", methods=["POST"])
def download_url():
    """
    Body: { "key": "outputs/dashboard_..." }
    Returns a presigned S3 URL valid for 10 minutes for direct download.
    """
    data = request.get_json() or {}
    key  = data.get("key", "")
    if not key:
        return jsonify({"error": "key required"}), 400
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=600
        )
        return jsonify({"url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════
#  RECORDINGS LIST
# ══════════════════════════════════════════════════════════
@app.route("/api/recordings", methods=["GET"])
def list_recordings():
    """Returns the 10 most recent completed recordings from DynamoDB."""
    try:
        resp  = table.scan(Limit=50)
        items = resp.get("Items", [])

        def fix(obj):
            if isinstance(obj, Decimal): return float(obj)
            if isinstance(obj, dict):    return {k: fix(v) for k, v in obj.items()}
            if isinstance(obj, list):    return [fix(i) for i in obj]
            return obj

        items = fix(items)
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jsonify(items[:10])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Development only — use gunicorn in production
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
