#!/usr/bin/env python3
"""
Model Weights Downloader for FlashVSR+ (RunPod Serverless Worker)
Downloads Turbo model weights and TCDecoder from Hugging Face during Docker build.
Zero cold-start delay in production.
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "ruchangyao/FlashVSR_plus"

REQUIRED_MODELS = [
    {"filename": "turbo/dit.safetensors", "local_dir": "ckpts/turbo", "target_name": "dit.safetensors"},
    {"filename": "turbo/lq_proj_in.safetensors", "local_dir": "ckpts/turbo", "target_name": "lq_proj_in.safetensors"},
    {"filename": "TCDecoder.ckpt", "local_dir": "ckpts", "target_name": "TCDecoder.ckpt"},
]

def download_models(base_dir: Path):
    print(f"📦 [FlashVSR+] Checking and downloading pretrained model weights from '{REPO_ID}'...", flush=True)
    t0 = time.time()

    for item in REQUIRED_MODELS:
        sub_dir = base_dir / item["local_dir"]
        sub_dir.mkdir(parents=True, exist_ok=True)
        target_path = sub_dir / item["target_name"]

        if target_path.is_file() and target_path.stat().st_size > 1000000:
            print(f"  ✅ Already exists: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)
            continue

        print(f"  ⬇️ Downloading {item['filename']} -> {target_path}...", flush=True)
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=item["filename"],
                local_dir=str(base_dir / "ckpts"),
                local_dir_use_symlinks=False,
            )
            # If HF downloaded to ckpts/turbo/dit.safetensors or ckpts/TCDecoder.ckpt
            actual_file = Path(downloaded)
            if actual_file.resolve() != target_path.resolve():
                if target_path.exists():
                    target_path.unlink()
                actual_file.rename(target_path)
            print(f"  ✅ Successfully downloaded: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)
        except Exception as e:
            print(f"  ❌ Failed to download {item['filename']}: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    print(f"🎉 [FlashVSR+] All weights ready in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    app_dir = Path(__file__).resolve().parent
    download_models(app_dir)
