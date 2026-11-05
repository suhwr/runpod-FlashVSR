#!/usr/bin/env python3
"""
⚡ FlashVSR+ Production-Grade RunPod Serverless Handler
High-Performance One-Step Streaming Video Super-Resolution (Untiled 4K on 24GB GPUs)
Zero cold-start delay, lossless audio preservation, and multi-fallback uploader.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import requests
import runpod
import torch

# Add current directory to sys.path so inference and flashvsr_plus packages import cleanly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
# Multi-Destination Public Cloud Media Uploader
# ==============================================================================
def upload_file_resilient(file_path: str, destination: str = "auto") -> Optional[str]:
    """Robust multi-cloud uploader with fallback sequence."""
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return None

    filename = os.path.basename(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. Try Uguu.se (up to 128MB)
    if destination in ("auto", "uguu") and file_size_mb <= 128:
        try:
            with open(file_path, "rb") as f:
                res = requests.post("https://uguu.se/upload", files={"files[]": (filename, f)}, timeout=60)
            if res.status_code == 200:
                data = res.json()
                if data.get("success") and data.get("files"):
                    url = data["files"][0].get("url")
                    if url:
                        return url
        except Exception as e:
            print(f"⚠️ Uguu upload failed ({e}), trying fallback...", flush=True)

    # 2. Try Catbox.moe / Litterbox (up to 1GB)
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
                return res.text.strip()
        except Exception as e:
            print(f"⚠️ Litterbox upload failed ({e}), trying fallback...", flush=True)

    # 3. Try 0x0.st
    try:
        with open(file_path, "rb") as f:
            res = requests.post("https://0x0.st", files={"file": (filename, f)}, timeout=60)
        if res.status_code == 200 and res.text.strip().startswith("http"):
            return res.text.strip()
    except Exception as e:
        print(f"⚠️ 0x0.st upload failed ({e})", flush=True)

    return None


# ==============================================================================
# Dimension & Aspect Ratio Calculations (Divisible by 128)
# ==============================================================================
def calculate_target_resolution(in_w: int, in_h: int, target_res: str, scale: Optional[int] = None) -> Tuple[int, int]:
    """
    Computes output dimensions divisible by 128 preserving aspect ratio.
    FlashVSR+ architectural requirement: height and width must be divisible by 128.
    """
    clean_res = str(target_res).lower().strip()

    # Predefined presets
    if clean_res in ("720p", "sd"):
        if in_w >= in_h:  # Landscape
            return 1280, 768
        else:  # Portrait
            return 768, 1280

    if clean_res in ("1080p", "fhd", "hd", "default"):
        if in_w >= in_h:
            return 1920, 1024
        else:
            return 1024, 1920

    if clean_res in ("2k", "qhd", "1440p"):
        if in_w >= in_h:
            return 2560, 1408
        else:
            return 1408, 2560

    if clean_res in ("4k", "uhd", "ultra"):
        if in_w >= in_h:
            return 3840, 2048
        else:
            return 2048, 3840

    # Custom scale multiplier (e.g. 2x, 3x, 4x)
    eff_scale = scale if scale in (2, 3, 4) else 2
    raw_w = in_w * eff_scale
    raw_h = in_h * eff_scale

    # Snap to closest multiple of 128 (minimum 256)
    target_w = max(256, int(round(raw_w / 128.0) * 128))
    target_h = max(256, int(round(raw_h / 128.0) * 128))

    return target_w, target_h


# ==============================================================================
# RunPod Serverless Handler
# ==============================================================================
def handler(job: dict) -> dict:
    job_id = job.get("id", str(uuid.uuid4())[:8])
    job_input = job.get("input", {})

    print(f"\n⚡ [{job_id}] Received FlashVSR+ job request", flush=True)

    # 1. Parse Input Parameters
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
    mode = str(job_input.get("mode", "turbo")).strip().lower()
    temporal_compression = int(job_input.get("temporal_compression", 4))
    if temporal_compression not in (1, 2, 4):
        temporal_compression = 4

    decode_every = job_input.get("decode_every", 28)
    color_correction = bool(job_input.get("color_correction", False) or job_input.get("cf", False))
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

        out_w, out_h = calculate_target_resolution(in_w, in_h, target_res, scale=scale)

        print(
            f"[{job_id}] Video Info: {in_w}x{in_h} @ {fps:.2f}fps ({total_frames} frames) -> Target: {out_w}x{out_h} "
            f"(mode={mode}, temporal_comp={temporal_compression})",
            flush=True,
        )

        # 5. Build or Reconfigure Pipeline
        # If parameters differ from pre-warmed default, construct config
        pipeline = PIPELINE
        if (
            pipeline is None
            or pipeline.config.temporal_compression != temporal_compression
            or pipeline.config.color_correction.enabled != color_correction
        ):
            print(f"[{job_id}] Reconfiguring FlashVSR+ pipeline for job parameters...", flush=True)
            config = stream_config_for_mode(
                mode,
                device=ACTIVE_DEVICE,
                decode_every=decode_every,
                hardware_profile="auto",
                attn_kernel="auto",
                temporal_compression=temporal_compression,
                color_correction=color_correction,
            )
            paths = model_paths_for_mode(ROOT_DIR, mode)
            pipeline = build_inference_pipeline(config, paths)

        # 6. Run Inference
        t_infer = time.time()
        print(f"[{job_id}] Starting FlashVSR+ video super-resolution...", flush=True)

        req = InferenceRequest(
            input_path=input_video_path,
            output_path=output_video_path,
            height=out_h,
            width=out_w,
        )

        result = pipeline.infer(
            req,
            video_preset=video_preset,
        )

        infer_duration = time.time() - t_infer
        frames_done = result.frames if result.frames > 0 else total_frames
        fps_proc = frames_done / max(0.001, infer_duration)

        print(
            f"[{job_id}] Inference complete: {frames_done} frames processed in {infer_duration:.2f}s ({fps_proc:.2f} FPS)",
            flush=True,
        )

        # 7. Upload Result
        if not os.path.isfile(output_video_path) or os.path.getsize(output_video_path) == 0:
            return {"error": "Output video was not generated or has zero size"}

        file_size = os.path.getsize(output_video_path)
        out_name = f"flashvsr_{target_res}_{job_id}.mp4"

        print(f"[{job_id}] Uploading output video ({file_size / (1024*1024):.2f} MB)...", flush=True)
        file_url = upload_file_resilient(output_video_path, destination=upload_dest)

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
    # Pre-warm model weights at cold start
    try:
        initialize_global_pipeline()
    except Exception as err:
        print(f"⚠️ [FlashVSR+] Pre-warming warning: {err} (will retry on first request)", flush=True)

    runpod.serverless.start({"handler": handler})
