# ⚡ RunPod Serverless Worker: FlashVSR+ (Video Super-Resolution)

Production-grade, ultra-fast serverless worker powering **FlashVSR+** (One-Step Streaming Video Super-Resolution via Hybrid Attention and Reward-Yield Distribution Matching).

Enables **spatially untiled 4K video super-resolution on a single 24GB GPU (RTX 4090 / A100 / Blackwell)** with 100% flicker-free temporal consistency, lossless audio preservation, and zero cold-start delay.

---

## 🚀 Key Features

* **One-Step Streaming Diffusion (DMD Distillation):** 15–17 FPS throughput without traditional diffusion iteration delays.
* **Flicker-Free Temporal Consistency:** 8-frame causal lookahead KV cache eliminates edge shimmering and fast-action motion blur artifacts.
* **Bounded VRAM Streaming:** Segmented decoding (`decode_every=28`) prevents Out-Of-Memory (OOM) crashes on long or high-res videos.
* **Zero Cold-Start Delay:** Model weights are baked directly into the Docker image layer during build.
* **Lossless Audio Sync:** Automatic audio demuxing and stream remuxing.
* **Universal GPU Support:** Optimized for Ada (RTX 4090 `sm_89`), Hopper (`sm_90`), and Blackwell (`sm_120`).

---

## 📥 API Request Payload

Send a JSON POST request to your RunPod Serverless endpoint:

```json
{
  "input": {
    "video": "https://example.com/input.mp4",
    "resolution": "1080p",
    "mode": "turbo",
    "temporal_compression": 4
  }
}
```

### Parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video` / `url` | `string` | **Required** | Public video URL or base64-encoded video stream |
| `resolution` | `string` | `"1080p"` | Target resolution: `"720p"`, `"1080p"`, `"2k"`, `"4k"` |
| `mode` | `string` | `"turbo"` | Model preset: `"turbo"` (Fastest) or `"base"` |
| `temporal_compression` | `integer` | `4` | `4` (Standard motion) or `2` / `1` (Large/Fast action motion) |
| `color_correction` | `boolean` | `false` | Enable wavelet low-frequency color correction |
| `video_preset` | `string` | `"fast"` | x264 encoder preset (`"ultrafast"`, `"fast"`, `"medium"`) |

---

## 📤 Output Response

```json
{
  "status": "COMPLETED",
  "url": "https://h.uguu.se/upscaled_video.mp4",
  "file_name": "flashvsr_1080p_d7d048ae.mp4",
  "input_resolution": "848x478",
  "output_resolution": "1920x1024",
  "frames_processed": 315,
  "fps_processing": 16.4,
  "video_fps": 30.0,
  "execution_time_seconds": 19.2,
  "size_bytes": 14205810
}
```

---

## 🐳 Docker Build & Deployment

```bash
docker build -t your-dockerhub-user/runpod-flashvsr:latest .
docker push your-dockerhub-user/runpod-flashvsr:latest
```

Configure your RunPod Serverless Endpoint:
* **Container Image:** `your-dockerhub-user/runpod-flashvsr:latest`
* **GPU Requirement:** Minimum 24GB VRAM (e.g. RTX 4090, A5000, A100, or RTX PRO 6000 Blackwell).
