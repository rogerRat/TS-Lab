"""Unified orchestrator for the RAGE experiment suite.

Uses explicit CLI + YAML config, serial execution, stop-on-failure and resumable
steps, and freezes a
launch-time document for every run:

* ``outputs/<run_id>/RUN.md``     - timestamp, run id, batch, git commit,
  host/GPU/package versions, every parameter definition and value (protocol,
  anchor densities, budget ratio, block budget B, N, schedule, seeds) and the
  exact commands executed;
* ``outputs/rage_runs/RUNS.md``- central append-only index of all runs.

Steps (each a resumable ``python -m`` entry):

  main         rage.experiments.main_table   (§2 main table)
  mechanism1   rage.experiments.mechanism1   (§3 gradient/ranking fidelity)
  mechanism2   rage.experiments.mechanism2   (§4 cancellation detection)
  ablation     rage.experiments.ablation     (§5 module ablation)
  data-curve   rage.experiments.curves data-curve
  time-curve   rage.experiments.curves time-curve

Usage:
    python -m rage.experiments.run --config configs/rage/run.yaml
    python -m rage.experiments.run --config ... --steps main ablation
    python -m rage.experiments.run --config ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

UTC8 = timezone(timedelta(hours=8))

from configs.paths import PROJECT_ROOT

STEP_COMMANDS = {
    "main": [sys.executable, "-m", "rage.experiments.main_table"],
    "mechanism1": [sys.executable, "-m", "rage.experiments.mechanism1"],
    "mechanism2": [sys.executable, "-m", "rage.experiments.mechanism2"],
    "ablation": [sys.executable, "-m", "rage.experiments.ablation"],
    "data-curve": [sys.executable, "-m", "rage.experiments.curves", "data-curve"],
    "time-curve": [sys.executable, "-m", "rage.experiments.curves", "time-curve"],
}
STEP_MODULES = {
    "main": "rage.experiments.main_table",
    "mechanism1": "rage.experiments.mechanism1",
    "mechanism2": "rage.experiments.mechanism2",
    "ablation": "rage.experiments.ablation",
    "data-curve": "rage.experiments.curves",
    "time-curve": "rage.experiments.curves",
}
ALL_STEPS = list(STEP_COMMANDS)

# YAML keys eligible to become CLI flags.  Keys a step's parser does not
# accept are detected and dropped automatically (see _step_argv).
SHARED_KEYS = (
    "series_cap", "slice_length", "stride",
    "pool_equivalents", "pool_seed", "pool_stride", "pool_series", "pool_slice",
    "n_windows",
    "selection_gradient_batch", "selection_inference_batch", "selection_memory_fraction", "selection_cpu_threads",
    "selection_generation",
    "context_length", "prediction_length", "projection_dimension",
    "cover_density", "error_density", "split_density", "block_budget",
    "mean_epsilon", "endpoint_epsilon", "weak_epsilon", "curvature_cap",
    "cover_share", "error_share", "split_share",
    "source_epochs", "train_batch", "learning_rate", "seeds", "model_path",
    "students", "student_environments", "student_model_paths", "student_parallelism", "selection_only",
    "datasets", "with_baselines", "variants", "with_grid",
    "window_entry_budget", "dataset", "arms",
    "n_values", "total_budget_seconds",
    "densities", "b_values", "m_values", "rescan", "scan_dir", "density",
    "cancel_ratio",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rage/main.json")
    parser.add_argument("--resume", help="existing run directory; reuse its saved config and cells")
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=None,
                        help="subset of steps (default: config order)")
    parser.add_argument("--batch", default=None,
                        help="override the batch label from the config")
    parser.add_argument("--out-root", default="outputs",
                        help="root directory that receives <run_id>/")
    parser.add_argument("--dry-run", action="store_true",
                        help="freeze RUN.md and print commands without running")
    return parser.parse_args(argv)


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_status_dirty() -> bool:
    try:
        result = subprocess.run(["git", "status", "--porcelain"],
                                cwd=PROJECT_ROOT, capture_output=True,
                                text=True, timeout=15)
        return bool(result.stdout.strip())
    except Exception:
        return False


def _gpu_summary() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                 "--format=csv,noheader"], capture_output=True,
                                text=True, timeout=15)
        return result.stdout.strip()
    except Exception:
        return "n/a"


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "chronos", "numpy"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "not-installed"
    return versions


def _step_argv(step: str, config: dict, run_dir: Path) -> list[str]:
    """Materialise the exact CLI for one step from the shared YAML.

    Shared keys are passed to every step; flags a step's parser rejects are
    detected by validating against the step's own ``parse_args`` and removed.
    ``step_overrides.<step>`` in the YAML merges step-specific values on top
    (values may reference ``{run_dir}``).
    """
    merged = dict(config)
    for key, value in (config.get("step_overrides") or {}).get(step, {}).items():
        merged[key] = value.replace("{run_dir}", str(run_dir)) if isinstance(value, str) else value

    def build(flags: set[str]) -> list[str]:
        command = list(STEP_COMMANDS[step])
        for key, value in merged.items():
            if key in ("out", "steps", "step_overrides", "batch"):
                continue
            if key not in SHARED_KEYS or key not in flags:
                continue
            flag = "--%s" % key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, list):
                if value:
                    command.extend([flag] + [str(item) for item in value])
            elif value is not None:
                command.extend([flag, str(value)])
        command.extend(["--out", str(run_dir / step)])
        return command

    flags = {key for key in merged if key in SHARED_KEYS}
    command = build(flags)
    # drop flags the step's parser does not accept
    import contextlib
    import importlib
    import io

    module = importlib.import_module(STEP_MODULES[step])
    while True:
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                # skip "python -m <module>" (and a mode positional for curves)
                module.parse_args(command[3:])
            return command
        except SystemExit:
            text = err.getvalue()
            if "unrecognized arguments" not in text:
                raise  # a real value/type error, not a flag mismatch
            dropped = False
            for token in text.split("unrecognized arguments:", 1)[1].split():
                flags.discard(token.lstrip("-").replace("-", "_"))
                dropped = True
            if not dropped:
                raise
            rebuilt = build(flags)
            if rebuilt == command:
                raise RuntimeError("flag cleanup made no progress for step %s" % step)
            command = rebuilt


def _flag_map(argv: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            flag = argv[i][2:]
            values: list[str] = []
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                values.append(argv[i])
                i += 1
            rows.append((flag, " ".join(values) if values else "(flag)"))
        else:
            i += 1
    return rows


PARAM_DOC = """## 冻结的方法与参数定义

### 样本协议（所有阶段一致，域 A1）
| 参数 | 值 | 定义 |
|---|---|---|
| C (context_length) | {context_length} | 每个窗口的上下文输入长度 |
| H (prediction_length) | {prediction_length} | 每个窗口的预测目标长度（六数据集统一，协议适配已披露） |
| D (pool_stride / stride) | {pool_stride} | 相邻合法窗口的最小起点间隔（打分分辨率） |

### 数据协议（域 A2–A7）
| 参数 | 值 | 定义 |
|---|---|---|
| 参考数据集 | 6 个 FEV 数据集 | 全部可用序列（100/52/35/21/14/14=236 条）× 完整历史 |
| 考卷 | FEV 原生 | 每序列末尾 20 窗（jena 10 窗），窗长 64、步长 64，首尾相接 |
| N (n_windows) | {n_windows} | 选入训练的不重叠窗口数（起点间隔 576，A6 拍板值） |
| 源池 (pool) | 5N = {pool_equivalents} 窗当量 | 比例制整条抽样（A5，种子 {pool_seed}），不截断不拼接 |
| 参考数据边界 | 只做参考 | 不进入任何方法的训练集，无目标侧适配阶段（A7/D1） |

### CRB 三份锚点预算（按合法起点密度，互不挪用，域 C1 冻结 4:4:1）
| 预算 | 密度 | 定义 |
|---|---|---|
| K_cover (均布) | {cover_density} | 端点 + 最长区间对分，保证无长未探查空档 |
| K_error (误差驱动) | {error_density} | 风险积分 E_I = h^5 (c_I)^2 / 120 最大的区间优先 |
| K_split (抵消检测) | {split_density} | 拮抗区间一次二分复查 |
| 合计密度 | {total_density} | 三者之和（比例 {ratio}） |

### 检测与召回（域 C2–C4）
| 参数 | 值 | 定义 |
|---|---|---|
| B (block_budget) | {block_budget} | 每个参考查询返回的连续源块数（通过检测的段全部进索引） |
| epsilon_mu (mean_epsilon) | {mean_epsilon} | 区间平均梯度模长近零阈值（C3 标定值） |
| epsilon_g (endpoint_epsilon) | {endpoint_epsilon} | 端点无信号阈值（与 epsilon_mu 同值，C3） |
| epsilon 溯源 | {epsilon_provenance} | 开发任务 ETT_1H 全扫描两侧真实梯度模长第 10 百分位 |
| 平局打破 | 预固定随机序（种子 5） | C4 冻结 |
| curvature_cap (c_bar) | {curvature_cap} | 无任何观测时的弯曲程度上限（乐观初值） |
| 装填 | 各方法原生规则 | RAGE/LESS 不相交；LP-Mirror/LTSV 原生 top-k；TSDS 保留重复展开 |

### 后训练日程（域 D，单阶段，所有方法一致）
| 参数 | 值 | 定义 |
|---|---|---|
| source_epochs | {source_epochs} | D2 2026-09-11：每 epoch 保存并评测；取最终 epoch，不挑最佳测试点 |
| train_batch | {train_batch} | 每步有放回均匀抽样 |
| learning_rate | {learning_rate} | AdamW（权重衰减 0.01，无调度无裁剪） |
| seeds | {seeds} | 每格开始前恢复预训练权重 |
| 适配阶段 | 无 | D1/A7：参考数据不进入训练 |

### 模型与梯度探针（域 B）
| 参数 | 值 | 定义 |
|---|---|---|
| model | {model} | Chronos-T5 选择探针，预训练单一状态；三个学生共享选集 |
| projection_dimension | {projection_dimension} | Count-Sketch 哈希投影维度（种子 42，全流程共用） |
| 精度/损失 | float32 分桶交叉熵 | 窗口内预测位置平均；逐窗 vmap 批处理（B6 工程项） |

### 评测（域 E）
- 仅 MASE（Chronos 中位数 / TimesFM 原生预测 / Time-MoE 自回归），季节表严格 FEV 原生：boomlet_1282=1440、boomlet_619=1440、
  boomlet_1676=48、ETT_15T=96、ETT_1H=24、jena=24；尺度取每个预测时点之前的完整历史。
- 汇总：先序列内平均，再 6 数据集等权。
"""


def freeze_run_doc(run_id: str, config: dict, steps: list[str],
                   step_argv: dict[str, list[str]], run_dir: Path) -> None:
    started = datetime.now(UTC8)
    cover = float(config.get("cover_density", 0.02))
    error = float(config.get("error_density", 0.02))
    split = float(config.get("split_density", 0.005))
    total = cover + error + split
    ratio = "%d:%d:%d" % tuple(
        max(1, int(round(10 * part / total)) if total > 0 else 0)
        for part in (cover, error, split))

    lines: list[str] = []
    lines.append("# RAGE Run %s" % run_id)
    lines.append("")
    lines.append("| 字段 | 值 |")
    lines.append("|---|---|")
    lines.append("| Run ID | `%s` |" % run_id)
    lines.append("| 批次 (batch) | %s |" % config.get("batch", "default"))
    lines.append("| 启动时间 | %s |" % started.strftime("%Y-%m-%d %H:%M:%S %z"))
    lines.append("| git commit | `%s`%s |" % (
        _git_commit(), " (dirty)" if _git_status_dirty() else ""))
    lines.append("| host | %s |" % platform.node())
    from .students import provenance, ENVIRONMENTS
    runtime = provenance(include_lock=True)
    lines.append("| environment | `%s` (`%s`) |" % (runtime["environment"], runtime["python"]))
    lines.append("| model integrations lock SHA256 | `%s` |" % runtime["model_integrations_lock_sha256"])
    lines.append("| student environments | `%s` |" % config.get("student_environments", ENVIRONMENTS))
    lines.append("| students / metric / precision | %s / MASE / FP32 |" % config.get("students", ["chronos-small"]))
    lines.append("| GPU | %s |" % _gpu_summary().replace("\n", "; "))
    versions = _package_versions()
    lines.append("| 版本 | python %s, torch %s, transformers %s, chronos %s, numpy %s |" % (
        versions["python"], versions["torch"], versions["transformers"],
        versions["chronos"], versions["numpy"]))
    lines.append("| 步骤 | %s |" % ", ".join(steps))
    lines.append("")
    model = config.get("model_path") or "amazon/chronos-t5-small"
    lines.append(PARAM_DOC.format(
        context_length=config.get("context_length", 512),
        prediction_length=config.get("prediction_length", 64),
        pool_stride=config.get("pool_stride", 1),
        cover_density=cover, error_density=error, split_density=split,
        total_density=total, ratio=ratio,
        block_budget=config.get("block_budget", 8),
        mean_epsilon=config.get("mean_epsilon", 1e-8),
        endpoint_epsilon=config.get("endpoint_epsilon", 1e-8),
        epsilon_provenance=config.get("epsilon_provenance",
                                      "未记录（须填 mechanism1 全扫描 p10 溯源）"),
        n_windows=config.get("n_windows", 4096),
        pool_equivalents=config.get("pool_equivalents", 20480),
        pool_seed=config.get("pool_seed", 0),
        curvature_cap=config.get("curvature_cap", 1.0),
        source_epochs=config.get("source_epochs", 4),
        train_batch=config.get("train_batch", 32),
        learning_rate=config.get("learning_rate", 1e-5),
        seeds=config.get("seeds", [0, 1, 2]),
        model=model,
        projection_dimension=config.get("projection_dimension", 4096),
    ))
    lines.append("### 数据")
    datasets = config.get("datasets") or config.get("dataset")
    lines.append("- 数据集: %s" % datasets)
    lines.append("- 源池: 比例制整条抽样，%s 窗当量（5N，种子 %s），完整轨迹不截断" % (
        config.get("pool_equivalents", 20480), config.get("pool_seed", 0)))
    lines.append("")
    lines.append("### 对比方法（主表，F 域）")
    arms = ["original", "random", "trajectory_level", "rage"]
    arms += list(config.get("with_baselines") or [])
    lines.append("- " + " / ".join(arms) + "（粒度对照与论文基线均为披露的适配实现）")
    lines.append("")
    lines.append("### 曲线（§2.2）")
    lines.append("- 数据量曲线 N=%s（池容量 5%%/10%%/20%%/40%%），统一 4 epoch" % (
        config.get("n_values") or [1024, 2048, 4096, 8192]))
    lines.append("- 总耗时曲线: 每方法相同总 GPU 预算 %ss" % (
        config.get("total_budget_seconds", 1800)))
    lines.append("")
    lines.append("## 执行命令")
    for step in steps:
        lines.append("```bash")
        lines.append("# %s" % step)
        lines.append(" ".join(step_argv[step]))
        lines.append("```")
    lines.append("")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN.md").write_text("\n".join(lines), encoding="utf-8")


def append_index(out_root: Path, run_id: str, config: dict, steps: list[str],
                 run_dir: Path) -> None:
    index_dir = out_root / "rage_runs"
    index_dir.mkdir(parents=True, exist_ok=True)
    index = index_dir / "RUNS.md"
    if not index.exists():
        index.write_text(
            "# RAGE Run 索引\n\n"
            "| Run ID | 批次 | 启动时间 | 步骤 | RUN.md | 状态 |\n"
            "|---|---|---|---|---|---|\n", encoding="utf-8")
    now = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
    rel = run_dir.relative_to(out_root) if run_dir.is_relative_to(out_root) else run_dir
    with index.open("a", encoding="utf-8") as handle:
        handle.write("| `%s` | %s | %s | %s | [%s/RUN.md](../%s/RUN.md) | launched |\n" % (
            run_id, config.get("batch", "default"), now, ", ".join(steps),
            run_id, rel))


def _update_index_status(out_root: Path, run_id: str, status: str) -> None:
    index = out_root / "rage_runs" / "RUNS.md"
    if not index.exists():
        return
    lines = index.read_text(encoding="utf-8").splitlines()
    marker = "| `%s` |" % run_id
    for i, line in enumerate(lines):
        if line.startswith(marker):
            fields = line.strip().strip("|").split("|")
            fields[-1] = " %s " % status
            lines[i] = "|" + "|".join(fields) + "|"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = PROJECT_ROOT / args.resume / "config.yaml" if args.resume else Path(args.config)
    config = (json.loads(config_path.read_text()) if config_path.suffix==".json"
              else yaml.safe_load(config_path.read_text())) or {}
    if "resampling_revision" in config:
        from .full_pool_run import run_suite
        if args.steps and args.steps != ["main"]:
            raise ValueError("Full-pool suite runs the new main matrix; historical steps require their explicit legacy config")
        stamp = datetime.now(UTC8).strftime("%Y%m%d-%H%M%S")
        run_dir = (PROJECT_ROOT / args.resume if args.resume else PROJECT_ROOT / args.out_root /
                   f"rage-{args.batch or config.get('batch', 'full-pool')}-{stamp}")
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if args.dry_run:return run_suite(config,run_dir,dry_run=True)
        from .seed_rounds import run_rounds
        return run_rounds(config,run_dir)
    if "source_steps" in config or "schedule_variants" in config:
        raise ValueError("obsolete step-based config: use source_epochs (2026-09-11 protocol)")
    steps = args.steps or config.get("steps") or ALL_STEPS

    stamp = datetime.now(UTC8).strftime("%Y%m%d-%H%M")
    run_id = "rage-%s-%s" % (args.batch or config.get("batch", "default"), stamp)
    out_root = PROJECT_ROOT / args.out_root
    run_dir = PROJECT_ROOT / args.resume if args.resume else out_root / run_id
    if args.resume:
        run_id, out_root = run_dir.name, run_dir.parent

    step_argv = {step: _step_argv(step, config, run_dir) for step in steps}
    if not args.resume:
        freeze_run_doc(run_id, config, steps, step_argv, run_dir)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        append_index(out_root, run_id, config, steps, run_dir)
    print("run %s frozen -> %s" % (run_id, run_dir / "RUN.md"), flush=True)

    if args.dry_run:
        for step in steps:
            print("[dry-run] %s" % " ".join(step_argv[step]), flush=True)
        return 0

    failures: list[str] = []
    for step in steps:
        command = step_argv[step]
        log_path = run_dir / ("%s.log" % step)
        print("=== step %s -> %s ===" % (step, log_path), flush=True)
        started = time.time()
        with log_path.open("a" if args.resume else "w") as log:
            child_env = dict(os.environ)
            for key, env_key in (("source_load", "RAGE_SOURCE_LOAD"),
                                 ("reference_root", "RAGE_REFERENCE_ROOT")):
                if config.get(key):
                    child_env[env_key] = str((PROJECT_ROOT / config[key]).resolve())
            process = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, env=child_env)
        elapsed = (time.time() - started) / 60
        if process.returncode != 0:
            print("step %s FAILED (rc=%d) after %.1f min; see %s" % (
                step, process.returncode, elapsed, log_path), flush=True)
            failures.append(step)
            break
        print("step %s done in %.1f min" % (step, elapsed), flush=True)

    _update_index_status(out_root, run_id,
                         "failed: %s" % ",".join(failures) if failures else "complete")
    print("RUN %s %s" % (run_id, "FAILED" if failures else "COMPLETE"), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
