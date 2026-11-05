#!/usr/bin/env python3
"""
Test script for FlashVSR+ RunPod Serverless Worker.
Can test either a live remote endpoint or locally.
"""

import json
import os
import sys
import time
import requests

def test_remote(endpoint_id: str, api_key: str, payload_file: str):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with open(payload_file, "r") as f:
        payload = json.load(f)

    print(f"🚀 Sending FlashVSR+ request to RunPod Endpoint: {endpoint_id}...")
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    elapsed = time.time() - t0

    print(f"⏱️ Finished in {elapsed:.2f}s (Status: {resp.status_code})")
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 test_endpoint.py <ENDPOINT_ID> <API_KEY> [payload_file]")
        sys.exit(1)

    ep_id = sys.argv[1]
    key = sys.argv[2]
    p_file = sys.argv[3] if len(sys.argv) > 3 else "test_input_video.json"
    test_remote(ep_id, key, p_file)
