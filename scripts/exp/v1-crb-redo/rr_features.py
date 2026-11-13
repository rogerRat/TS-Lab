"""Build P1-filtered direction representatives from cached gradient sketches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from assets.gradient.dataset import GradientDataset
from configs.paths import PROJECT_ROOT, output_path
from rage.experiments.development_inputs import load_fragments

ROOT = PROJECT_ROOT
SRC_ROOT = ROOT / "outputs/train/v1-crb-redo/grad-prepare/gradients/timerxl-84m"
REF_ROOT = ROOT / "outputs/train/v1-crb-redo/12tasks-ref-grad-prepare"
REF_INPUTS = ROOT / "data/ref/shard_dataset/v1-crb-redo-12tasks"
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")


def record_anchors(path, method):
    item = json.loads(path.read_text())
    if method == "crb":
        return np.asarray(item["anchor_positions"], dtype=np.int64)
    ids = np.asarray(item["window_ids"], dtype=np.int64)
    return ids[:, 1].astype(np.int64) - 1


def fetch(cache, dataset, fragment_id, anchors):
    ids = np.column_stack((np.full(len(anchors), fragment_id, dtype=np.int64), anchors + 1))
    values = cache.lookup(dataset, ids)
    if values.shape != (len(anchors), 4096) or not np.isfinite(values).all():
        raise ValueError("Gradient sketch identity or values differ")
    return np.asarray(values, dtype=np.float64)


def filtered_units(anchors, gradients, legal_count, *, offset, granularity):
    norms = np.linalg.norm(gradients, axis=1)
    units = np.divide(gradients, norms[:, None], out=np.zeros_like(gradients),
                      where=norms[:, None] > 0)
    spans = np.diff(anchors)
    if len(spans) == 0 or np.any(spans <= 0) or anchors[0] != 0 or anchors[-1] != legal_count - 1:
        raise ValueError("Anchors must span every legal window")
    out_dirs, out_counts, out_starts, out_ends = [], [], [], []
    for j, width in enumerate(spans):
        count = int(width + (1 if j == len(spans)-1 else 0))
        alpha = (float(width)-1.0)/(2.0*float(width))
        if j == len(spans)-1:
            alpha = 0.5
        mean = units[j]*(1.0-alpha)+units[j+1]*alpha
        # P1 is applied after the shared threshold is supplied by the caller.
        out_dirs.append(mean)
        out_counts.append(count)
        left = int(offset + anchors[j])
        out_starts.append(left)
        out_ends.append(left + count)
    return np.asarray(out_dirs), np.asarray(out_counts), np.asarray(out_starts), np.asarray(out_ends), norms


def source_layout():
    config = yaml.safe_load((ROOT / "configs/data/src-96.yaml").read_text())
    lengths = np.asarray([int(row["length"]) for row in config["trajectories"]], dtype=np.int64)
    legal = lengths - 575
    offsets = np.cumsum(np.r_[0, legal[:-1]])
    files, caches = {}, {}
    for worker in range(8):
        base = SRC_ROOT / ("worker-%d" % worker)
        if not (base / ".complete").is_file():
            raise FileNotFoundError("Source gradient worker incomplete")
        cache_path = next((base / "grad_cache").glob("*/cache_manifest.json")).parent
        caches[worker] = GradientDataset(cache_path)
        method_dir = base
        for path in (method_dir / "crb").glob("*.json"):
            fid = int(path.stem)
            files[("crb", fid)] = (path, caches[worker])
        for path in (method_dir / "uniform").glob("*.json"):
            fid = int(path.stem)
            files[("uniform", fid)] = (path, caches[worker])
    if len(legal) != 96 or int(legal.sum()) != 2666312 or len([k for k in files if k[0] == "crb"]) != 96:
        raise ValueError("Source pool layout changed")
    return config, legal, offsets, files


def build_source(method, granularity, out_dir):
    out_dir = Path(out_dir)
    if (out_dir / "source_units.npz").is_file() and (out_dir / "source_manifest.json").is_file():
        return
    config, legal, offsets, files = source_layout()
    records, norms_all = [], []
    dataset = "src-96"
    for fid in range(96):
        path, cache = files[(method, fid)]
        anchors = record_anchors(path, method)
        grads = fetch(cache, dataset, fid, anchors)
        dirs, counts, starts, ends, norms = filtered_units(
            anchors, grads, int(legal[fid]), offset=int(offsets[fid]), granularity=granularity)
        records.append((fid, dirs, counts, starts, ends))
        norms_all.extend(norms.tolist())
    threshold = float(np.quantile(norms_all, .01))
    rep, counts, starts, ends, owners = [], [], [], [], []
    for fid, dirs, widths, left, right in records:
        norms = None
        path, cache = files[(method, fid)]
        anchors = record_anchors(path, method)
        grads = fetch(cache, dataset, fid, anchors)
        norms = np.linalg.norm(grads, axis=1)
        unit = np.divide(grads, norms[:, None], out=np.zeros_like(grads), where=norms[:, None] > 0)
        unit[norms <= threshold] = 0.0
        spans = np.diff(anchors)
        interval_means=[]
        for j, width in enumerate(spans):
            alpha=(float(width)-1.0)/(2.0*float(width))
            if j == len(spans)-1:
                alpha=0.5
            interval_means.append(unit[j]*(1-alpha)+unit[j+1]*alpha)
        if granularity == "traj":
            vector=np.average(np.asarray(interval_means), axis=0, weights=widths)
            vector=vector/np.linalg.norm(vector) if np.linalg.norm(vector)>0 else vector
            rep.append(vector); counts.append(int(widths.sum())); starts.append(int(left[0])); ends.append(int(right[-1])); owners.append(fid)
        elif granularity == "block":
            for vec, n, a, b in zip(interval_means, widths, left, right):
                norm=float(np.linalg.norm(vec))
                rep.append(vec/norm if norm>0 else vec)
                counts.append(int(n)); starts.append(int(a)); ends.append(int(b)); owners.append(fid)
        else:
            raise ValueError("Feature granularity must be traj or block")
    out_dir=output_path("selection", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir/"source_units.npz", directions=np.asarray(rep,dtype=np.float32),
             counts=np.asarray(counts,dtype=np.int64), starts=np.asarray(starts,dtype=np.int64),
             ends=np.asarray(ends,dtype=np.int64), owners=np.asarray(owners,dtype=np.int64))
    (out_dir/"source_manifest.json").write_text(json.dumps(dict(method=method,granularity=granularity,
        threshold=threshold,units=len(rep),legal_windows=int(sum(counts))),indent=2)+"\n",encoding="ascii")


def build_reference(task, method, granularity, out_dir):
    out_dir = Path(out_dir)
    if (out_dir / "reference_units.npz").is_file() and (out_dir / "reference_manifest.json").is_file():
        return
    root=REF_ROOT/task
    manifest=json.loads((root/"manifest.json").read_text())
    cache=GradientDataset(Path(manifest["cache"]))
    fragments=load_fragments(REF_INPUTS/task/"references.npz")
    rows=[]
    all_norms=[]
    for fragment in fragments:
        path=root/method/("%05d.json"%fragment.fragment_id)
        anchors=record_anchors(path,method)
        grads=fetch(cache,"reference:"+task,fragment.fragment_id,anchors)
        rows.append((fragment,anchors,grads))
        all_norms.extend(np.linalg.norm(grads,axis=1).tolist())
    threshold=float(np.quantile(all_norms,.01))
    reps=[]; counts=[]; ids=[]
    for fragment,anchors,grads in rows:
        norms=np.linalg.norm(grads,axis=1)
        unit=np.divide(grads,norms[:,None],out=np.zeros_like(grads),where=norms[:,None]>0)
        unit[norms<=threshold]=0.
        spans=np.diff(anchors)
        means=[]
        for j,width in enumerate(spans):
            alpha=(float(width)-1)/(2*float(width))
            if j==len(spans)-1: alpha=.5
            means.append(unit[j]*(1-alpha)+unit[j+1]*alpha)
        widths=spans.copy(); widths[-1]+=1
        if granularity=="traj":
            vec=np.average(means,axis=0,weights=widths)
            norm=float(np.linalg.norm(vec)); reps.append(vec/norm if norm>0 else vec)
            counts.append(int(widths.sum())); ids.append((fragment.fragment_id,0))
        else:
            for j,(vec,n) in enumerate(zip(means,widths)):
                norm=float(np.linalg.norm(vec)); reps.append(vec/norm if norm>0 else vec)
                counts.append(int(n)); ids.append((fragment.fragment_id,j))
    out=output_path("selection",out_dir)
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/"reference_units.npz",directions=np.asarray(reps,dtype=np.float32),
             counts=np.asarray(counts,dtype=np.int64),ids=np.asarray(ids,dtype=np.int64))
    (out/"reference_manifest.json").write_text(json.dumps(dict(task=task,method=method,
        granularity=granularity,threshold=threshold,units=len(reps),legal_windows=int(sum(counts))),indent=2)+"\n",encoding="ascii")
