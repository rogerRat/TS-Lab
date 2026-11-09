"""Compare exact full-gradient geometry with three cheap window proxies."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from configs import logger as logging_config
from configs.paths import PROJECT_ROOT, output_path
from scripts.exp.v0_grad_mech_test import load_engine, select_trajectories, _gradient
from rage.engines import per_sample_cross_entropy

LOGGER = None


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def association(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "pearson": None, "spearman": None}
    return {"n": len(x), "pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic)}


def gram_matrix(array, bf16, tile, label):
    """Exact coordinates; FP32 GEMM, FP64 accumulation across coordinate tiles."""
    n, d = array.shape
    LOGGER.info("[gram] start name=%s rows=%d coordinates=%d", label, n, d)
    total = torch.zeros((n, n), dtype=torch.float64, device="cuda")
    for begin in tqdm(range(0, d, tile), desc=label, mininterval=10):
        part = torch.from_numpy(np.array(array[:, begin:begin + tile], copy=True))
        if bf16:
            part = part.view(torch.bfloat16)
        part = part.to(device="cuda", dtype=torch.float32)
        total.add_((part @ part.T).double())
    result = total.cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite Gram matrix")
    LOGGER.info("[gram] end name=%s", label)
    return result


def geometry(gram):
    norms = np.sqrt(np.maximum(gram.diagonal(), 0))
    denom = norms[:, None] * norms[None, :]
    cosine = np.full_like(gram, np.nan)
    np.divide(gram, denom, out=cosine, where=denom > 0)
    cosine = np.clip(cosine, -1, 1)
    squared = np.maximum(norms[:, None] ** 2 + norms[None, :] ** 2 - 2 * gram, 0)
    return norms, cosine, squared


def analyze(destination, cfg, track):
    matrices = np.load(destination / "gram_matrices.npz")
    features = np.load(destination / "window_features.npz")
    loss = features["loss"]
    gs = {name: geometry(matrices[name]) for name in ("full", "head", "embedding")}
    rows = []

    def record(scope, metric, proxy, x, y, lag=0):
        rows.append({"track": track, "scope": scope, "lag": lag,
                     "target": metric, "proxy": proxy, **association(x, y)})

    for proxy in ("head", "embedding"):
        record("window", "full_norm", proxy + "_norm", gs["full"][0], gs[proxy][0])
    record("window", "full_norm", "loss", gs["full"][0], loss)
    pair_tables = []
    for lag in cfg["lags"]:
        if lag >= len(loss):
            continue
        i = np.arange(len(loss) - lag)
        j = i + lag
        angular = 1 - gs["full"][1][i, j]
        distance = np.sqrt(gs["full"][2][i, j])
        full_norm_delta = np.abs(gs["full"][0][i] - gs["full"][0][j])
        table = {"lag": np.full(len(i), lag), "start_i": i, "start_j": j,
                 "full_cosine": gs["full"][1][i, j], "full_l2": distance,
                 "full_norm_difference": full_norm_delta,
                 "loss_difference": np.abs(loss[i] - loss[j])}
        for proxy in ("head", "embedding"):
            cosine = gs[proxy][1][i, j]
            dist = np.sqrt(gs[proxy][2][i, j])
            norm_delta = np.abs(gs[proxy][0][i] - gs[proxy][0][j])
            record("pair", "full_cosine", proxy + "_cosine", gs["full"][1][i, j], cosine, lag)
            record("pair", "full_l2", proxy + "_l2", distance, dist, lag)
            record("pair", "full_norm_difference", proxy + "_norm_difference", full_norm_delta, norm_delta, lag)
            table[proxy + "_cosine"] = cosine
            table[proxy + "_l2"] = dist
        record("pair", "full_angular_distance", "loss_difference", angular, table["loss_difference"], lag)
        record("pair", "full_l2", "loss_difference", distance, table["loss_difference"], lag)
        record("pair", "full_norm_difference", "loss_difference", full_norm_delta, table["loss_difference"], lag)
        pair_tables.append(table)
    np.savez_compressed(destination / "pair_metrics.npz",
                        **{key: np.concatenate([table[key] for table in pair_tables]) for key in pair_tables[0]})
    # All within-trajectory pairs, plus lag-conditioned results above.
    i, j = np.triu_indices(len(loss), 1)
    for proxy in ("head", "embedding"):
        record("all_pairs", "full_cosine", proxy + "_cosine", gs["full"][1][i, j], gs[proxy][1][i, j])
    record("all_pairs", "full_angular_distance", "loss_difference", 1-gs["full"][1][i, j], np.abs(loss[i]-loss[j]))
    blocks = []
    for begin in range(0, len(loss), cfg["block_length"]):
        end = min(begin + cfg["block_length"], len(loss))
        row = {"track": track, "start": begin, "count": end-begin,
               "loss_variance": float(np.var(loss[begin:end]))}
        for name in gs:
            gram = matrices[name][begin:end, begin:end]
            row[name + "_variance"] = max(0., float(np.mean(np.diag(gram)) - np.mean(gram)))
            row[name + "_direction_dispersion"] = max(0., float(1-np.mean(gs[name][1][begin:end, begin:end])))
        blocks.append(row)
    # Keep the shorter terminal block in artifacts, exclude it from correlations.
    complete = [b for b in blocks if b["count"] == cfg["block_length"]]
    for metric in ("variance", "direction_dispersion"):
        target = np.array([b["full_" + metric] for b in complete])
        for proxy in ("head", "embedding"):
            record("block32", "full_" + metric, proxy + "_" + metric,
                   target, np.array([b[proxy + "_" + metric] for b in complete]))
        record("block32", "full_" + metric, "loss_variance", target,
               np.array([b["loss_variance"] for b in complete]))
    save_csv(destination / "correlations.csv", rows)
    save_csv(destination / "blocks.csv", blocks)
    return rows, blocks


def extract(engine, values, full, destination, cfg):
    n, dimension = full.shape
    model = engine.model
    head = np.lib.format.open_memmap(destination / "head_gradients.npy", mode="w+",
                                    dtype=np.float32, shape=(n, model.lm_head.weight.numel()))
    losses = np.empty(n, dtype=np.float64)
    embeddings = np.empty((n, model.config.d_model), dtype=np.float32)
    captured = {}
    hook = model.lm_head.register_forward_pre_hook(lambda module, args: captured.update(head_input=args[0].detach()))
    timings = {"forward_seconds": 0., "head_gradient_seconds": 0., "full_bp_seconds": 0.}
    checks = []
    LOGGER.info("[extract] start windows=%d head_parameters=%d", n, head.shape[1])
    try:
        for begin in tqdm(range(0, n, cfg["batch_size"]), desc="proxies", mininterval=10):
            end = min(n, begin+cfg["batch_size"])
            windows = np.stack([values[k:k+576] for k in range(begin, end)])
            tensors = engine.batch_tensors(windows)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.no_grad():
                output = model(**engine._logits_inputs(tensors), output_hidden_states=True, return_dict=True)
                batch_loss = per_sample_cross_entropy(output.logits, tensors["labels"])
                valid = tensors["labels"].ne(-100)
                hidden = output.decoder_hidden_states[-1]
                mean = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            torch.cuda.synchronize()
            timings["forward_seconds"] += time.monotonic()-started
            started = time.monotonic()
            # Use the framework CE backward to avoid softmax-minus-one cancellation.
            local_logits = output.logits.detach().requires_grad_(True)
            local_loss = per_sample_cross_entropy(local_logits, tensors["labels"]).sum()
            residual = torch.autograd.grad(local_loss, local_logits)[0]
            # Exact per-window head derivative; the backbone remains detached.
            with torch.no_grad():
                grad = torch.bmm(residual.transpose(1, 2), captured["head_input"].float())
            torch.cuda.synchronize()
            timings["head_gradient_seconds"] += time.monotonic()-started
            head[begin:end] = grad.flatten(1).cpu().numpy()
            losses[begin:end] = batch_loss.cpu().numpy()
            embeddings[begin:end] = mean.cpu().numpy()
            if not np.isfinite(head[begin:end]).all():
                raise ValueError("nonfinite output-head gradients")
            if begin == 0:
                weight = model.lm_head.weight.detach().clone().requires_grad_(True)
                validation_loss = per_sample_cross_entropy(
                    torch.nn.functional.linear(captured["head_input"], weight), tensors["labels"])[0]
                reference_head = torch.autograd.grad(validation_loss, weight)[0]
                relative = float(torch.linalg.vector_norm(reference_head-grad[0]) / torch.linalg.vector_norm(reference_head))
                # FP32 softmax subtraction and log-softmax backward differ slightly.
                if relative > 1e-4:
                    raise ValueError("analytic head gradient mismatch: relative=%g" % relative)
                checks.append({"head_analytic_relative_error": relative})
        head.flush()
        # Three full FP32 BP checks per trajectory establish cache/window alignment.
        for k in sorted(set([0, n//2, n-1])):
            torch.cuda.synchronize()
            started = time.monotonic()
            reference, _ = _gradient(engine, values[k:k+576])
            torch.cuda.synchronize()
            timings["full_bp_seconds"] += time.monotonic()-started
            cached = torch.from_numpy(np.array(full[k], copy=True)).view(torch.bfloat16).to("cuda").float()
            relative = float(torch.linalg.vector_norm(reference-cached)/torch.linalg.vector_norm(reference))
            cosine = float(torch.nn.functional.cosine_similarity(reference, cached, dim=0))
            if relative > .01 or cosine < .9999:
                raise ValueError("cached gradient does not match current full BP")
            checks.append({"window": k, "full_cache_relative_error": relative, "full_cache_cosine": cosine})
    finally:
        hook.remove()
    if not np.isfinite(embeddings).all() or not np.isfinite(losses).all():
        raise ValueError("nonfinite proxy features")
    np.savez_compressed(destination / "window_features.npz", loss=losses, embedding=embeddings, start=np.arange(n))
    save_json(destination / "validation_and_timing.json", {"checks": checks, "timings": timings,
              "full_bp_count": 3, "forward_batch_size": cfg["batch_size"],
              "note": "Timing is descriptive; batched proxies and single-window BP are not a matched speed benchmark."})
    LOGGER.info("[extract] end windows=%d validation=passed", n)
    return head, embeddings


def main():
    global LOGGER
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rage/v0-gradient-proxy-test.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    logging_config.HEARTBEAT_INTERVAL_SECONDS = cfg["heartbeat_seconds"]
    LOGGER = logging_config.configure_logger("gradient_proxy", cfg["experiment"], "experiment", "proxy")
    for handler in LOGGER.handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s-[gradient_proxy]-[analysis]-[%(message)s]"))
    out = output_path("train", cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    config_record = out / "config.json"
    if config_record.exists() and json.loads(config_record.read_text()) != cfg:
        raise ValueError("output already contains a different experiment config")
    save_json(config_record, cfg)
    LOGGER.info("[setup] start config=%s output=%s", cfg, out)
    started = time.monotonic()
    source = json.loads((PROJECT_ROOT / cfg["source_config"]).read_text())
    selected, rows = select_trajectories(source)
    LOGGER.info("[model] start loading CPU weights and transferring to GPU")
    engine = load_engine(str(PROJECT_ROOT / cfg["checkpoint"]), "float32")
    torch.backends.cuda.matmul.allow_tf32 = False
    LOGGER.info("[model] end loading; model.eval; GPU ready")
    dimension = sum(p.numel() for p in engine._parameters)
    save_json(out / "parameter_layout.json", [{"name": name, "shape": list(p.shape), "count": p.numel()}
              for name, p in engine._named_trainable])
    all_rows, all_blocks = [], []
    for track, ordinal in enumerate(cfg["source_ordinals"]):
        dest = out / ("trajectory-%02d" % track)
        dest.mkdir(exist_ok=True)
        sid, values = selected[ordinal]
        n = len(values)-575
        path = PROJECT_ROOT / cfg["cache_root"] / ("trajectory-%02d.bf16" % track)
        source_meta = json.loads((PROJECT_ROOT / ("outputs/train/v0-crb-reconst-test/batches/trajectory-%02d/config.json" % track)).read_text())
        if source_meta["trajectory_ordinals"] != [ordinal] or source_meta["cache_ordinals"] != [track]:
            raise ValueError("trajectory manifest mismatch")
        if path.stat().st_size != n*dimension*2 or not path.with_suffix(".complete").exists():
            raise ValueError("incomplete or incorrectly sized gradient cache")
        save_json(dest / "manifest.json", {"track": track, "ordinal": ordinal, "series_id": sid,
                  "windows": n, "source_row": rows[ordinal]["source_row"], "full_gradient_dtype": "BF16 from FP32 BP"})
        if not (dest / ".complete").exists():
            full = np.memmap(path, mode="r", dtype=np.uint16, shape=(n, dimension))
            head, embed = extract(engine, values, full, dest, cfg)
            full_gram = gram_matrix(full, True, cfg["coordinate_tile"], "full-gradient")
            head_gram = gram_matrix(head, False, cfg["coordinate_tile"], "head-gradient")
            embed_gram = embed.astype(np.float64) @ embed.astype(np.float64).T
            np.savez_compressed(dest / "gram_matrices.npz", full=full_gram, head=head_gram, embedding=embed_gram)
            del full, head
            current_rows, current_blocks = analyze(dest, cfg, track)
            (dest / ".complete").write_text("complete\n", encoding="ascii")
        else:
            with (dest / "correlations.csv").open() as handle:
                current_rows = list(csv.DictReader(handle))
            with (dest / "blocks.csv").open() as handle:
                current_blocks = list(csv.DictReader(handle))
        all_rows.extend(current_rows)
        all_blocks.extend(current_blocks)
        save_csv(out / "correlations.csv", all_rows)
        save_csv(out / "blocks.csv", all_blocks)
        save_json(out / "progress.json", {"completed_trajectories": track+1, "total": len(cfg["source_ordinals"]),
                  "elapsed_seconds": time.monotonic()-started})
        LOGGER.info("[trajectory] end track=%d series=%s", track, sid)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("[complete] elapsed=%s output=%s", logging_config.format_duration(time.monotonic()-started), out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if LOGGER is not None:
            LOGGER.exception("Experiment failed; completed trajectories are retained")
        raise
