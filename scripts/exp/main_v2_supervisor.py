"""Supervise the fresh v2 matrix and publish only v2 results to MAIN-v2.md."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from rags.experiments.full_pool_run import summarize, validate_config
from rags.experiments.students import atomic_json


def refresh(config, run, document):
    summary = summarize(run, config)
    records = []
    for student in config['students']:
        for path in (run/'students'/student/'cells').glob('*.json'):
            record = json.loads(path.read_text())
            if record.get('experiment_version') != 'v2':
                raise ValueError(f'Non-v2 cell found in v2 run: {path}')
            records.append(record)
    # 附录作为 MAIN-v2.md 自身标记段就地保留：巡检结果整理置顶（用户裁定 2026-09-17），
    # 静态附注（ETA/事故/设计记录）排文末；单文档，无附属 notes 文件
    patrol_section, notes_section = [], []
    if document.exists():
        previous = document.read_text()
        for begin, end, bucket in (
                ('<!-- v2-patrol:start -->', '<-- v2-patrol:end -->', patrol_section),
                ('<!-- v2-notes:start -->', '<!-- v2-notes:end -->', notes_section)):
            i, j = previous.find(begin), previous.find(end)
            if i >= 0 and j >= i:
                bucket.append(previous[i:j + len(end)].strip())
    lines = ['# MAIN-v2 — 主实验', '',
        f'- 更新时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}',
        f'- 配置：`configs/rags/main_v2.json`；运行：`{run}`',
        '- 继承最终 v1 的全部裁定：九任务、三模型；八训练臂（含 rags_global 与 rags_traj_global），只跑 seed 0，统一数值 softmax T=0.25；Quito/LTSV 延续移除。',
        '- C512/H64/D1，CRB 2%/1%/1%；覆盖与误差阶段沿用 v1；分裂阶段反复二分当前中点代表模长最小的一批可分裂区间，预算用尽，所有块保留。',
        '- 原始逐窗梯度共享已有字段校验缓存，只补缺失窗口；CRB 审计、选集、训练与评测全部新生，未复制 v1 cells/checkpoints。',
        '- v1 协议与旧结果保存在 [MAIN-v1](MAIN-v1.md)。', '',
        f'完成 {len(records)}/{len(config["students"])*len(config["datasets"])*(1+len(config["arms"])*len(config["seeds"]))} cells。', '',
        '| 模型 | 已完成 cells | 当前状态 |', '|---|---:|---|']
    for student in config['students']:
        state_path = run/'students'/student/'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        status = '/'.join(str(state.get(key,'')) for key in ['stage','dataset','arm','seed','status'])
        lines.append(f'| {student} | {sum(r["student"]==student for r in records)} | {status} |')
    if patrol_section:
        lines += patrol_section
    lines += ['', '| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |', '|---|---|---|---:|---:|---:|']
    for student in config['students']:
        for dataset in config['datasets']:
            for arm in ['original']+config['arms']:
                group = [r['mase'] for r in records if r['student']==student and r['dataset']==dataset and r['arm']==arm]
                if group:
                    lines.append(f'| {student} | {dataset} | {arm} | {len(group)} | {np.mean(group):.6f} | {np.std(group):.6f} |')
    if not records:
        lines += ['', '尚无 v2 评测结果，实验准备中。']
    if notes_section:
        lines += [''] + notes_section
    document.parent.mkdir(parents=True, exist_ok=True)
    temporary = document.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(document)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--document', type=Path, default=Path('docs/MAIN-v2.md'))
    parser.add_argument('--refresh-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if config.get('experiment_version') != 'v2':
        raise ValueError('Supervisor requires v2')
    verification = json.loads((args.run/'warm_cache_verification.json').read_text())
    if not all(r['manifest_fields_match'] and r['cached_windows'] > 0 and r['recomputed_windows']==0 for r in verification['reports']):
        raise ValueError('Warm gradient cache verification failed')
    refresh(config,args.run,args.document)
    if args.refresh_only:
        return 0
    atomic_json(args.run/'readiness.json', dict(ready=True,config=config,experiment_version='v2'))
    command = [sys.executable,'-u','-m','rags.experiments.formal_main_entry',
               '--config',str(args.config.resolve()),'--run',str(args.run.resolve())]
    for attempt in range(1,4):
        print(f'v2 launch attempt {attempt}: {command}', flush=True)
        process = subprocess.Popen(command)
        atomic_json(args.run/'supervisor_state.json', dict(experiment_version='v2',attempt=attempt,pid=process.pid,status='running'))
        while True:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                refresh(config,args.run,args.document)
                continue
            summary = refresh(config,args.run,args.document)
            atomic_json(args.run/'supervisor_state.json', dict(experiment_version='v2',attempt=attempt,pid=process.pid,status='complete' if code==0 else 'failed',exit_code=code))
            if code==0:
                return 0
            print(f'v2 attempt {attempt} failed ({code}); remaining cells={len(summary["missing"])}',flush=True)
            break
    return code


if __name__ == '__main__':
    raise SystemExit(main())
