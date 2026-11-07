#!/usr/bin/env python3
"""
cugan_pipeline.py — Real-CUGAN Anime Video Upscaling Sub-pipeline for FlashVSR Worker
Provides ultra-fast Real-CUGAN inference with batching, auto 30fps clamping, and streaming FFmpeg.
"""

import os
import sys
import time
import uuid
import subprocess
import cv2
import numpy as np
import torch

from upcunet_v3 import RealWaifuUpScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(BASE_DIR, "weights_v3"))

CUGAN_CACHE = {}


def resolve_cugan_weights(scale: int, denoise: str = "conservative") -> str:
    scale = int(scale) if scale in (2, 3, 4) else 2
    denoise = str(denoise).strip().lower()

    if scale == 2:
        if "cons" in denoise:
            fname = "up2x-latest-conservative.pth"
        elif "no" in denoise or "none" in denoise or "0" in denoise:
            fname = "up2x-latest-no-denoise.pth"
        elif "1" in denoise:
            fname = "up2x-latest-denoise1x.pth"
        elif "2" in denoise:
            fname = "up2x-latest-denoise2x.pth"
        elif "3" in denoise:
            fname = "up2x-latest-denoise3x.pth"
        else:
            fname = "up2x-latest-conservative.pth"
    elif scale == 3:
        if "cons" in denoise:
            fname = "up3x-latest-conservative.pth"
        elif "no" in denoise or "none" in denoise or "0" in denoise:
            fname = "up3x-latest-no-denoise.pth"
        else:
            fname = "up3x-latest-denoise3x.pth"
    else:  # scale == 4
        if "cons" in denoise:
            fname = "up4x-latest-conservative.pth"
        elif "no" in denoise or "none" in denoise or "0" in denoise:
            fname = "up4x-latest-no-denoise.pth"
        else:
            fname = "up4x-latest-denoise3x.pth"

    candidates = [
        os.path.join(WEIGHTS_DIR, fname),
        os.path.join(BASE_DIR, "weights_v3", fname),
        os.path.join("/app", "weights_v3", fname),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
            return p

    raise FileNotFoundError(f"Real-CUGAN weights not found for {fname}. Checked: {candidates}")


def get_cugan_upscaler(scale: int, denoise: str = "conservative", half: bool = True, device: str = "cuda:0") -> RealWaifuUpScaler:
    if not torch.cuda.is_available():
        device = "cpu"
        half = False

    wpath = resolve_cugan_weights(scale, denoise)
    key = (scale, wpath, half, device)
    if key in CUGAN_CACHE:
        return CUGAN_CACHE[key]

    print(f"⚡ Loading Real-CUGAN {scale}x ({os.path.basename(wpath)}) [device={device}, half={half}]...", flush=True)
    t0 = time.time()
    upscaler = RealWaifuUpScaler(scale=scale, weight_path=wpath, half=half, device=device)
    print(f"✅ Real-CUGAN loaded in {time.time() - t0:.2f}s", flush=True)
    CUGAN_CACHE[key] = upscaler
    return upscaler


def calculate_cugan_dimensions(in_w: int, in_h: int, target_res: str, user_scale: int = None) -> tuple[int, int, int]:
    clean_res = str(target_res).lower().strip()
    if clean_res in ("720p", "sd"):
        target_h = 720
    elif clean_res in ("1080p", "fhd", "hd", "default"):
        target_h = 1080
    else:
        s = user_scale if user_scale in (2, 3, 4) else 2
        out_w = ((in_w * s) // 2) * 2
        out_h = ((in_h * s) // 2) * 2
        return out_w, out_h, s

    if in_w >= in_h:
        out_h = target_h
        out_w = int(round(in_w * (target_h / in_h)))
    else:
        out_w = target_h
        out_h = int(round(in_h * (target_h / in_w)))

    out_w = (out_w // 2) * 2
    out_h = (out_h // 2) * 2

    ratio = max(out_w / max(1, in_w), out_h / max(1, in_h))
    if ratio <= 2.2:
        chosen_scale = 2
    elif ratio <= 3.2:
        chosen_scale = 3
    else:
        chosen_scale = 4

    return out_w, out_h, chosen_scale


def process_cugan_video(
    input_path: str,
    output_path: str,
    upscaler: RealWaifuUpScaler,
    scale: int,
    out_w: int,
    out_h: int,
    custom_fps: float = None,
    crf: int = 18,
    batch_size: int = 2,
    job_id: str = "cugan-video",
) -> dict:
    t0 = time.time()

    # Probe FPS
    probe_cap = cv2.VideoCapture(input_path)
    if not probe_cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_path}")
    raw_fps = float(probe_cap.get(cv2.CAP_PROP_FPS))
    probe_cap.release()

    target_fps = float(custom_fps) if (custom_fps and float(custom_fps) > 0) else 30.0
    clamped_temp = None

    if raw_fps > 35.0 and target_fps <= 30.0:
        clamped_temp = os.path.join(os.path.dirname(output_path), f"clamped_30fps_{uuid.uuid4().hex[:6]}.mp4")
        print(f"[{job_id}] Video FPS is {raw_fps:.1f} > 30: clamping to 30 FPS to optimize GPU time...", flush=True)
        clamp_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-r", "30",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15",
            "-c:a", "copy",
            clamped_temp,
        ]
        res = subprocess.run(clamp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.isfile(clamped_temp) and os.path.getsize(clamped_temp) > 0:
            input_path = clamped_temp
            fps = 30.0
        else:
            fps = raw_fps
    else:
        fps = target_fps if (custom_fps and float(custom_fps) > 0) else (raw_fps if raw_fps > 0 else 24.0)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[{job_id}] CUGAN Info: {in_w}x{in_h} @ {fps:.2f}fps ({total_frames} frames) -> Target: {out_w}x{out_h} (scale={scale}x, batch={batch_size})", flush=True)

    # Audio extraction
    temp_audio = None
    has_audio = False
    try:
        temp_audio = os.path.join(os.path.dirname(output_path), f"audio_{uuid.uuid4().hex[:6]}.m4a")
        ext_cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-c:a", "copy", temp_audio]
        ext_res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ext_res.returncode == 0 and os.path.isfile(temp_audio) and os.path.getsize(temp_audio) > 0:
            has_audio = True
        else:
            ext_cmd_aac = ["ffmpeg", "-y", "-i", input_path, "-vn", "-c:a", "aac", "-b:a", "256k", temp_audio]
            ext_aac = subprocess.run(ext_cmd_aac, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ext_aac.returncode == 0 and os.path.isfile(temp_audio) and os.path.getsize(temp_audio) > 0:
                has_audio = True
    except Exception as e:
        print(f"[{job_id}] Audio extraction check failed ({e})", flush=True)

    # Encode video stream
    temp_video = os.path.join(os.path.dirname(output_path), f"temp_video_{uuid.uuid4().hex[:6]}.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-movflags", "+faststart",
        temp_video,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_processed = 0
    t_loop = time.time()

    try:
        with torch.inference_mode():
            batch_rgb = []
            while True:
                ret, bgr_frame = cap.read()
                if not ret or bgr_frame is None:
                    break

                batch_rgb.append(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

                if len(batch_rgb) >= batch_size:
                    upscaled_list = upscaler.forward_batch(batch_rgb, tile_mode=0, cache_mode=0, alpha=1.0)
                    for upscaled_rgb in upscaled_list:
                        upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
                        if upscaled_bgr.shape[0] != out_h or upscaled_bgr.shape[1] != out_w:
                            upscaled_bgr = cv2.resize(upscaled_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                        try:
                            proc.stdin.write(upscaled_bgr.tobytes())
                        except (BrokenPipeError, IOError):
                            break
                        frames_processed += 1
                    batch_rgb.clear()

                    if frames_processed % 30 == 0 or frames_processed >= total_frames:
                        elapsed = time.time() - t_loop
                        fps_rate = frames_processed / max(0.001, elapsed)
                        pct = (frames_processed / max(1, total_frames) * 100)
                        print(f"[{job_id}] Processed {frames_processed}/{total_frames} frames ({pct:.1f}%) — {fps_rate:.1f} FPS", flush=True)

            # Flush remainder
            if batch_rgb:
                upscaled_list = upscaler.forward_batch(batch_rgb, tile_mode=0, cache_mode=0, alpha=1.0)
                for upscaled_rgb in upscaled_list:
                    upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
                    if upscaled_bgr.shape[0] != out_h or upscaled_bgr.shape[1] != out_w:
                        upscaled_bgr = cv2.resize(upscaled_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                    try:
                        proc.stdin.write(upscaled_bgr.tobytes())
                    except (BrokenPipeError, IOError):
                        break
                    frames_processed += 1
                batch_rgb.clear()

    finally:
        cap.release()
        if clamped_temp and os.path.isfile(clamped_temp):
            try:
                os.remove(clamped_temp)
            except Exception:
                pass

        try:
            _, stderr_bytes = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr_bytes = proc.communicate()

        if proc.returncode != 0:
            if os.path.isfile(temp_video):
                try:
                    os.remove(temp_video)
                except Exception:
                    pass
            if temp_audio and os.path.isfile(temp_audio):
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass
            raise RuntimeError(f"FFmpeg encoding failed with exit code {proc.returncode}")

    # Remux audio
    try:
        if has_audio and temp_audio and os.path.isfile(temp_audio):
            video_dur = frames_processed / fps if fps > 0 else None
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", temp_audio,
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
            ]
            if video_dur and video_dur > 0:
                mux_cmd.extend(["-t", f"{video_dur:.4f}"])
            mux_cmd.extend(["-movflags", "+faststart", output_path])
            mux_res = subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if mux_res.returncode != 0:
                import shutil
                shutil.move(temp_video, output_path)
        else:
            import shutil
            shutil.move(temp_video, output_path)
    finally:
        if os.path.isfile(temp_video):
            try:
                os.remove(temp_video)
            except Exception:
                pass
        if temp_audio and os.path.isfile(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass

    total_time = time.time() - t0
    avg_fps = frames_processed / max(0.001, total_time)

    return {
        "input_resolution": f"{in_w}x{in_h}",
        "output_resolution": f"{out_w}x{out_h}",
        "frames_processed": frames_processed,
        "fps_processing": round(avg_fps, 2),
        "video_fps": round(fps, 2),
        "execution_time_seconds": round(total_time, 2),
    }
