#!/usr/bin/env python
"""Fetch pinned TimerXL and TimesFM checkpoints onto the AutoDL data disk."""
from huggingface_hub import snapshot_download

MODELS = (
    ("thuml/timer-base-84m", "70077a71acce1b4c00d98332fcaabc694255d8e5",
     "/root/autodl-tmp/pretrained_models/timer-base-84m-84m"),
    ("google/timesfm-2.5-200m-transformers", "5a9806b9b291fad9233b5249d88263f1846304d3",
     "/root/autodl-tmp/pretrained_models/timesfm-2.5-200m-transformers"),
)
for repo_id, revision, local_dir in MODELS:
    print("DOWNLOAD_START", repo_id, revision, local_dir, flush=True)
    path = snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
    print("DOWNLOAD_COMPLETE", repo_id, path, flush=True)
