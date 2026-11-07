#!/usr/bin/env python3
"""
⚡ FlashVSR+ Production-Grade RunPod Serverless Handler
High-Performance One-Step Streaming Video Super-Resolution (Untiled 4K on 24GB GPUs)
Zero cold-start delay, lossless audio preservation, and multi-fallback uploader.
"""

from __future__ import annotations

import gc
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import requests
import runpod
import torch

# Add current directory and Triton operator directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TRITON_OP_DIR = str(ROOT_DIR / "operators" / "flashvsr_attn")
if TRITON_OP_DIR not in sys.path:
    sys.path.insert(0, TRITON_OP_DIR)

from inference import InferenceRequest, build_inference_pipeline
from inference.presets import model_paths_for_mode, stream_config_for_mode

# ==============================================================================
# Global Pipeline Pre-warming
# ==============================================================================
PIPELINE = None
ACTIVE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def initialize_global_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print("🚀 [FlashVSR+] Initializing global pipeline in GPU memory...", flush=True)
    t0 = time.time()

    if not torch.cuda.is_available():
        print("⚠️ [FlashVSR+] CUDA not detected, pipeline will run on CPU (very slow)", flush=True)

    config = stream_config_for_mode(
        "turbo",
        device=ACTIVE_DEVICE,
        decode_every=28,  # Segmented streaming bounded VRAM (prevents OOM on long/high-res videos)
        hardware_profile="auto",
        attn_kernel="auto",
        temporal_compression=4,
    )
    paths = model_paths_for_mode(ROOT_DIR, "turbo")

    try:
        PIPELINE = build_inference_pipeline(config, paths)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"✅ [FlashVSR+] Global pipeline pre-warmed successfully in {time.time() - t0:.2f}s!", flush=True)
    except Exception as e:
        print(f"❌ [FlashVSR+] Failed to initialize FlashVSR+ pipeline: {e}", file=sys.stderr, flush=True)
        raise

    return PIPELINE


# ==============================================================================
# Multi-Destination Cloud Media Uploader (Backblaze B2 Primary -> Fallbacks)
# ==============================================================================
def upload_b2(file_path: str, folder: str = "flashvsr") -> str:
    """Upload to Backblaze B2 (boto3 S3 API or native B2 REST API)."""
    key_id = os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    app_key = os.environ.get("B2_APPLICATION_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket = os.environ.get("B2_BUCKET_NAME")
    endpoint = os.environ.get("B2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")

    if not (key_id and app_key and bucket):
        raise ValueError("Backblaze B2 credentials (B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME) not set")

    fname = os.path.basename(file_path)
    clean_folder = folder.strip("/")
    target_key = f"{clean_folder}/{int(time.time())}_{uuid.uuid4().hex[:8]}_{fname}"
    content_type, _ = mimetypes.guess_type(str(file_path))
    if not content_type:
        content_type = "video/mp4"

    # Attempt 1: boto3 S3 API if endpoint is provided
    if endpoint:
        try:
            import boto3
            from botocore.config import Config

            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=key_id,
                aws_secret_access_key=app_key,
                config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
            )
            with open(file_path, "rb") as f:
                s3.upload_fileobj(f, bucket, target_key, ExtraArgs={"ContentType": content_type})

            public_prefix = os.environ.get("B2_PUBLIC_URL_PREFIX", "").rstrip("/")
            if public_prefix:
                return f"{public_prefix}/{target_key}"

            try:
                return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": target_key}, ExpiresIn=86400 * 7)
            except Exception:
                parsed = urllib.parse.urlsplit(endpoint)
                return f"{parsed.scheme}://{bucket}.{parsed.netloc}/{target_key}"
        except ImportError:
            pass

    # Attempt 2: B2 Native REST API (Zero extra dependencies)
    auth_resp = requests.get(
        "https://api.backblazeb2.com/b2api/v3/b2_authorize_account",
        auth=(key_id, app_key),
        timeout=30,
    )
    auth_resp.raise_for_status()
    auth_data = auth_resp.json()
    api_url = auth_data["apiUrl"]
    auth_token = auth_data["authorizationToken"]
    download_url = auth_data["downloadUrl"]

    bucket_id = os.environ.get("B2_BUCKET_ID")
    if not bucket_id:
        bucket_id = auth_data.get("allowed", {}).get("bucketId")
    if not bucket_id:
        lb_resp = requests.post(
            f"{api_url}/b2api/v3/b2_list_buckets",
            headers={"Authorization": auth_token},
            json={"accountId": auth_data["accountId"], "bucketName": bucket},
            timeout=30,
        )
        lb_resp.raise_for_status()
        buckets = lb_resp.json().get("buckets", [])
        if buckets:
            bucket_id = buckets[0]["bucketId"]
        else:
            raise ValueError(f"B2 bucket '{bucket}' not found")

    up_resp = requests.post(
        f"{api_url}/b2api/v3/b2_get_upload_url",
        headers={"Authorization": auth_token},
        json={"bucketId": bucket_id},
        timeout=30,
    )
    up_resp.raise_for_status()
    up_data = up_resp.json()
    upload_url = up_data["uploadUrl"]
    upload_token = up_data["authorizationToken"]

    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha1.update(chunk)
    digest = sha1.hexdigest()

    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        res = requests.post(
            upload_url,
            headers={
                "Authorization": upload_token,
                "X-Bz-File-Name": urllib.parse.quote(target_key),
                "Content-Type": content_type,
                "Content-Length": str(file_size),
                "X-Bz-Content-Sha1": digest,
            },
            data=f,
            timeout=180,
        )
        res.raise_for_status()

    public_prefix = os.environ.get("B2_PUBLIC_URL_PREFIX", "").rstrip("/")
    if public_prefix:
        return f"{public_prefix}/{target_key}"
    return f"{download_url}/file/{bucket}/{target_key}"


def upload_file_resilient(file_path: str, destination: str = "auto") -> Tuple[Optional[str], str]:
    """Robust multi-cloud uploader with fallback sequence."""
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return None, "none"

    filename = os.path.basename(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. Backblaze B2 (Primary destination if configured or explicitly requested)
    has_b2 = (
        (os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"))
        and (os.environ.get("B2_APPLICATION_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"))
        and os.environ.get("B2_BUCKET_NAME")
    )
    if destination in ("auto", "b2", "backblaze") and has_b2:
        try:
            url = upload_b2(file_path, folder="flashvsr")
            if url:
                print(f"✅ Uploaded to Backblaze B2: {url}", flush=True)
                return url, "backblaze_b2"
        except Exception as e:
            print(f"⚠️ Backblaze B2 upload failed ({e}), falling back to public hosts...", flush=True)

    # 2. Try Uguu.se (up to 128MB)
    if destination in ("auto", "uguu") and file_size_mb <= 128:
        try:
            with open(file_path, "rb") as f:
                res = requests.post("https://uguu.se/upload", files={"files[]": (filename, f)}, timeout=60)
            if res.status_code == 200:
                data = res.json()
                if data.get("success") and data.get("files"):
                    url = data["files"][0].get("url")
                    if url:
                        print(f"✅ Uploaded to Uguu.se: {url}", flush=True)
                        return url, "uguu"
        except Exception as e:
            print(f"⚠️ Uguu upload failed ({e}), trying fallback...", flush=True)

    # 3. Try Catbox.moe / Litterbox (up to 1GB)
    if destination in ("auto", "catbox", "litterbox") and file_size_mb <= 1024:
        try:
            with open(file_path, "rb") as f:
                res = requests.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data={"reqtype": "fileupload", "time": "72h"},
                    files={"fileToUpload": (filename, f)},
                    timeout=90,
                )
            if res.status_code == 200 and res.text.strip().startswith("http"):
                url = res.text.strip()
                print(f"✅ Uploaded to Litterbox: {url}", flush=True)
                return url, "catbox"
        except Exception as e:
            print(f"⚠️ Litterbox upload failed ({e}), trying fallback...", flush=True)

    # 4. Try 0x0.st
    try:
        with open(file_path, "rb") as f:
            res = requests.post("https://0x0.st", files={"file": (filename, f)}, timeout=60)
        if res.status_code == 200 and res.text.strip().startswith("http"):
            url = res.text.strip()
            print(f"✅ Uploaded to 0x0.st: {url}", flush=True)
            return url, "0x0.st"
    except Exception as e:
        print(f"⚠️ 0x0.st upload failed ({e})", flush=True)

    return None, "none"


# ==============================================================================
# Dimension & Aspect Ratio Calculations (Divisible by 128)
# ==============================================================================
def calculate_target_resolution(in_w: int, in_h: int, target_res: str, scale: Optional[int] = None) -> Tuple[int, int]:
    """
    Computes output dimensions divisible by 128 while faithfully preserving the input aspect ratio.
    FlashVSR+ architectural requirement: height and width must both be divisible by 128.
    """
    clean_res = str(target_res).lower().strip()
    aspect_ratio = in_w / float(in_h)

    # Base target boundary: 720p (1280 max) vs 1080p (1920 max)
    base_bound = 1280 if clean_res in ("720p", "sd") else 1920

    if aspect_ratio >= 1.0:
        # Landscape or square: scale width first, compute height
        target_w = max(256, int(round(base_bound / 128.0) * 128))
        target_h = max(256, int(round((target_w / aspect_ratio) / 128.0) * 128))
        max_h = 768 if clean_res in ("720p", "sd") else 1024
        if target_h > max_h:
            target_h = max_h
            target_w = max(256, int(round((target_h * aspect_ratio) / 128.0) * 128))
    else:
        # Portrait: scale height first, compute width
        target_h = max(256, int(round(base_bound / 128.0) * 128))
        target_w = max(256, int(round((target_h * aspect_ratio) / 128.0) * 128))
        max_w = 768 if clean_res in ("720p", "sd") else 1024
        if target_w > max_w:
            target_w = max_w
            target_h = max(256, int(round((target_w / aspect_ratio) / 128.0) * 128))

    return target_w, target_h


# ==============================================================================
# RunPod Serverless Handler
# ==============================================================================
def handler(job: dict) -> dict:
    job_id = job.get("id", str(uuid.uuid4())[:8])
    job_input = job.get("input", {})

    print(f"\n⚡ [{job_id}] Received FlashVSR+ job request", flush=True)

    # 1. Parse Input Parameters
    model = str(job_input.get("model", job_input.get("engine", "flashvsr"))).strip().lower()
    raw_media = (
        job_input.get("video")
        or job_input.get("url")
        or job_input.get("input")
        or job_input.get("media")
    )
    if not raw_media:
        return {"error": "Missing input: provide 'video' or 'url' in job payload."}

    target_res = str(job_input.get("resolution", job_input.get("target_resolution", "1080p"))).strip()
    scale = int(job_input.get("scale", 2))
    target_fps = int(job_input.get("fps", 30))
    mode = str(job_input.get("mode", "turbo")).strip().lower()
    temporal_compression = int(job_input.get("temporal_compression", 4))
    if temporal_compression not in (1, 2, 4):
        temporal_compression = 4

    clean_target = str(target_res).lower().strip()
    default_offload = "never" if clean_target in ("720p", "sd", "1080p", "fhd", "hd", "2k", "qhd", "1440p") else "auto"
    cache_offload_mode = str(job_input.get("cache_offload_mode", default_offload)).strip().lower()

    decode_every = job_input.get("decode_every", 28)
    color_correction = bool(job_input.get("color_correction", True) or job_input.get("cf", True))
    upload_dest = str(job_input.get("upload_destination", "auto")).strip().lower()
    video_preset = str(job_input.get("video_preset", "fast")).strip().lower()

    # 2. Setup Temporary Workspace
    temp_dir = tempfile.mkdtemp(prefix=f"flashvsr_{job_id}_")
    input_video_path = os.path.join(temp_dir, f"input_{job_id}.mp4")
    output_video_path = os.path.join(temp_dir, f"upscaled_{job_id}.mp4")

    t_start = time.time()

    try:
        # 3. Ingest Media (Download URL or write binary)
        if isinstance(raw_media, str) and (raw_media.startswith("http://") or raw_media.startswith("https://")):
            print(f"[{job_id}] Downloading video from: {raw_media[:80]}...", flush=True)
            res = requests.get(raw_media, timeout=120, stream=True)
            res.raise_for_status()
            with open(input_video_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        else:
            # Handle base64
            import base64
            clean_b64 = raw_media.split(",")[-1] if "," in str(raw_media) else str(raw_media)
            with open(input_video_path, "wb") as f:
                f.write(base64.b64decode(clean_b64))

        # 4. Probe Video Properties
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video file: {input_video_path}"}

        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if in_w <= 0 or in_h <= 0:
            return {"error": "Invalid video stream: could not read dimensions"}

        # 5. Auto-clamp FPS to 30 unless 60fps explicitly requested (saves 50% GPU time)
        if fps > 35.0 and target_fps <= 30:
            clamped_input = os.path.join(temp_dir, f"clamped_30fps_{job_id}.mp4")
            print(f"[{job_id}] Video FPS is {fps:.1f} > 30: clamping to 30 FPS to optimize GPU time...", flush=True)
            import subprocess
            clamp_cmd = [
                "ffmpeg", "-y", "-i", input_video_path,
                "-r", "30",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15",
                "-c:a", "copy",
                clamped_input
            ]
            res = subprocess.run(clamp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.isfile(clamped_input) and os.path.getsize(clamped_input) > 0:
                input_video_path = clamped_input
                fps = 30.0

        out_w, out_h = calculate_target_resolution(in_w, in_h, target_res, scale=scale)

        print(
            f"[{job_id}] FlashVSR Info: {in_w}x{in_h} @ {fps:.2f}fps ({total_frames} frames) -> Target: {out_w}x{out_h} "
            f"(mode={mode}, temporal_comp={temporal_compression})",
            flush=True,
        )

        # 6. Build or Reconfigure Pipeline (Cached across jobs)
        global PIPELINE
        if (
            PIPELINE is None
            or PIPELINE.config.temporal_compression != temporal_compression
            or getattr(PIPELINE.config, "stream_cache_offload_mode", "auto") != cache_offload_mode
            or PIPELINE.config.color_correction.enabled != color_correction
        ):
            print(f"[{job_id}] Initializing FlashVSR+ pipeline (mode={mode}, temporal_comp={temporal_compression}, offload={cache_offload_mode})...", flush=True)
            config = stream_config_for_mode(
                mode,
                device=ACTIVE_DEVICE,
                decode_every=decode_every,
                hardware_profile="auto",
                attn_kernel="auto",
                temporal_compression=temporal_compression,
                cache_offload_mode=cache_offload_mode,
                color_correction=color_correction,
            )
            paths = model_paths_for_mode(ROOT_DIR, mode)
            PIPELINE = build_inference_pipeline(config, paths)
        pipeline = PIPELINE

        # 7. Run FlashVSR Inference
        t_infer = time.time()
        print(f"[{job_id}] Starting FlashVSR+ video super-resolution...", flush=True)

        req = InferenceRequest(
            input_path=input_video_path,
            output_path=output_video_path,
            height=out_h,
            width=out_w,
        )

        def on_progress(msg: str):
            print(f"[{job_id}] {msg}", flush=True)

        result = pipeline.infer(
            req,
            progress=on_progress,
            video_preset=video_preset,
        )

        infer_duration = time.time() - t_infer
        frames_done = result.frames if result.frames > 0 else total_frames
        fps_proc = frames_done / max(0.001, infer_duration)

        print(
            f"[{job_id}] Inference complete: {frames_done} frames processed in {infer_duration:.2f}s ({fps_proc:.2f} FPS)",
            flush=True,
        )

        # 8. Upload Result
        if not os.path.isfile(output_video_path) or os.path.getsize(output_video_path) == 0:
            return {"error": "Output video was not generated or has zero size"}

        file_size = os.path.getsize(output_video_path)
        out_name = f"{model}_{target_res}_{job_id}.mp4"

        print(f"[{job_id}] Uploading output video ({file_size / (1024*1024):.2f} MB)...", flush=True)
        file_url, storage_provider = upload_file_resilient(output_video_path, destination=upload_dest)

        if not file_url:
            # Fallback to base64 if cloud upload fails
            print(f"[{job_id}] Cloud upload failed; encoding to base64 as fallback...", flush=True)
            import base64
            with open(output_video_path, "rb") as vf:
                b64_content = base64.b64encode(vf.read()).decode("utf-8")
            return {
                "status": "COMPLETED",
                "file_name": out_name,
                "input_resolution": f"{in_w}x{in_h}",
                "output_resolution": f"{out_w}x{out_h}",
                "frames_processed": frames_done,
                "fps_processing": round(fps_proc, 2),
                "execution_time_seconds": round(time.time() - t_start, 2),
                "size_bytes": file_size,
                "base64": b64_content,
            }

        return {
            "status": "COMPLETED",
            "url": file_url,
            "storage": storage_provider,
            "file_name": out_name,
            "input_resolution": f"{in_w}x{in_h}",
            "output_resolution": f"{out_w}x{out_h}",
            "frames_processed": frames_done,
            "fps_processing": round(fps_proc, 2),
            "video_fps": round(fps, 2),
            "execution_time_seconds": round(time.time() - t_start, 2),
            "size_bytes": file_size,
        }

    except Exception as e:
        print(f"❌ [{job_id}] FlashVSR+ processing failed: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Cleanup temporary workspace
        shutil.rmtree(temp_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


# ==============================================================================
# Container Entrypoint
# ==============================================================================
if __name__ == "__main__":
    print("⚡ [FlashVSR+] Starting RunPod Serverless Container Worker...", flush=True)
    # Register immediately with RunPod without blocking cold start!
    # Model pipeline loads lazily on the first incoming job and remains cached in VRAM.
    runpod.serverless.start({"handler": handler})
