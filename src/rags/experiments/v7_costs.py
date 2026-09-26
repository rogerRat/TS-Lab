"""Read-only V7 resource accounting; never import models or compute gradients."""
from pathlib import Path
from datetime import datetime, timezone
import io
import json
import math
import os
import tempfile
import zipfile
import stat
import numpy as np
from .v7_display import display_arm

BEGIN = '<!-- V7-COSTS-BEGIN -->'
END = '<!-- V7-COSTS-END -->'
ARRAYS = ('scores', 'probabilities', 'descending_order', 'starts', 'counts')


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def selection_storage(path):
    """Read ZIP directory + NPY headers, not multi-million-element arrays."""
    result = dict(file_bytes=path.stat().st_size, arrays={})
    with zipfile.ZipFile(path) as archive:
        for name in ARRAYS:
            info = archive.getinfo(name+'.npy')
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY header version')
            if dtype.hasobject:
                raise ValueError('Object arrays are not valid selection resources')
            result['arrays'][name] = dict(elements=math.prod(shape),
                payload_bytes=math.prod(shape)*dtype.itemsize,
                compressed_member_bytes=info.compress_size)
        result['cost'] = json.loads(str(np.load(
            io.BytesIO(archive.read('cost.npy')), allow_pickle=False)))
    return result


def endpoint_mase(cell, epoch):
    if cell.get('arm') == 'orig':
        return cell.get('mase')
    for row in cell.get('test_curve', []):
        if row['epoch'] == epoch:
            return row['metrics'].get('mase')
    return cell.get('mase') if cell.get('final_epoch') == epoch else None


def shared_inventory(paths):
    """Stat only manifest-referenced cache namespaces; never follow directory links."""
    seen = set()
    rows = []
    for path in sorted({str(Path(p).resolve()) for p in paths if p}):
        row = dict(path=path, files=0, logical_bytes=0, allocated_bytes=0,
                   skipped_links=0, vanished_files=0, exists=Path(path).is_dir())
        for parent, dirs, files in os.walk(path, followlinks=False):
            row['skipped_links'] += sum(Path(parent,d).is_symlink() for d in dirs)
            for name in files:
                try:
                    info=Path(parent,name).lstat()
                except FileNotFoundError:
                    row['vanished_files']+=1
                    continue
                if stat.S_ISLNK(info.st_mode):
                    row['skipped_links']+=1
                    continue
                if not stat.S_ISREG(info.st_mode) or (info.st_dev,info.st_ino) in seen:
                    continue
                seen.add((info.st_dev,info.st_ino))
                row['files']+=1
                row['logical_bytes']+=info.st_size
                row['allocated_bytes']+=info.st_blocks*512
        rows.append(row)
    return rows


def collect(config, run):
    run = Path(run)
    rows, assets, warnings = [], [], []
    for seed in config['seeds']:
        for model, settings in config['settings'].items():
            root = run/'seeds'/f'seed{seed}'/model
            for task in config['datasets']:
                asset_path = root/'assets'/f'{task}.json'
                audit = read(asset_path).get('audit', {}) if asset_path.exists() else {}
                if audit:
                    assets.append(dict(seed=seed, model=model, task=task,
                        receipt=str(asset_path), **{k:audit.get(k) for k in (
                            'new_bp', 'imported_rows', 'cached_rows', 'elapsed_seconds',
                            'gradient_seconds', 'prepared_union_rows', 'path',
                            'method_costs', 'native_proxy_cost')}))
                marker_path = root/'select'/f'{task}.json'
                marker = read(marker_path) if marker_path.exists() else {}
                selections = {s['arm']:s for s in marker.get('selections', [])}
                rand_path = root/'cells'/f'{task}__rand.json'
                rand = endpoint_mase(read(rand_path), settings['epochs']) if rand_path.exists() else None
                for arm in config['arms']:
                    path = root/'selections'/f'{task}__{arm}.npz'
                    cell_path = root/'cells'/f'{task}__{arm}.json'
                    if not path.exists() and not cell_path.exists():
                        continue
                    row = dict(seed=seed, model=model, task=task, arm=arm,
                        selection_path=str(path) if path.exists() else None,
                        selection_complete=bool(marker.get('complete') and arm in selections),
                        selection_elapsed_s=selections.get(arm, {}).get('elapsed_seconds'),
                        selection_new_bp=selections.get(arm, {}).get('selection_new_bp'),
                        coverage=selections.get(arm, {}).get('coverage'),
                        peak_cpu_rss_bytes=None, peak_cuda_allocated_bytes=None,
                        standalone_cold_seconds=None, gpu_seconds=None)
                    if path.exists():
                        row['storage'] = selection_storage(path)
                    # Budget is the independent requirement, NOT added BP in this run.
                    plans = audit.get('plans', {})
                    if arm.startswith('rags-') or arm in ('grad-dot','grad-cos','grand-static'):
                        row['required_source_bp'] = plans.get('source_audit', {}).get('required_bp')
                        row['required_reference_bp'] = (0 if arm == 'grand-static' else
                            plans.get('reference_audit', {}).get('required_bp'))
                    elif arm == 'less-main':
                        less = audit.get('method_costs', {}).get('less', {})
                        row['required_source_bp'] = less.get('required_source')
                        row['required_reference_bp'] = less.get('required_reference')
                    if cell_path.exists():
                        cell = read(cell_path)
                        source = cell.get('train', {}).get('source', {})
                        row.update(cell_path=str(cell_path),
                            cell_elapsed_s=cell.get('elapsed_seconds'),
                            recorded_cumulative_training_s=source.get('cumulative_training_seconds'),
                            evaluation_elapsed_s=cell.get('evaluation_elapsed_s'),
                            executed_steps=source.get('steps'),
                            resumed_from_step=source.get('resumed_from_step'),
                            executed_final_epoch=cell.get('final_epoch'),
                            report_epoch=settings['epochs'],
                            mase=endpoint_mase(cell, settings['epochs']), rand_mase=rand)
                    rows.append(row)
    return dict(version='v7-resource-accounting-v1',
        generated_at=datetime.now(timezone.utc).isoformat(), run=str(run),
        datasets=config['datasets'], rows=rows, shared_assets=assets, warnings=warnings,
        shared_cache_inventory=shared_inventory(a.get('path') for a in assets),
        limitations=[
            'Selection elapsed includes saving or warm reload; not standalone cold execution.',
            'Shared scoring is charged to the first missing uniform arm (usually grad-dot).',
            'Asset method deltas depend on execution order; never sum RAGS variants or rags_traj aliases.',
            'cached/imported counts are receipt observations, not cross-task unique window counts.',
            'CPU/GPU peaks, GPU device-time and isolated sort/sampling times were not instrumented.',
            'Wall times span different hardware/concurrency; not GPU-hours or fair isolated speed benchmarks.',
            'Raw training timing preserves executed work (including old four-epoch cells); effects use configured epoch.',
            'Failed/unreceipted attempts are not fully accounted; selection bytes exclude shared gradient caches.'])


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def show(value, digits=2):
    return '未测' if value is None else f'{value:.{digits}f}'


def render(report):
    lines = [BEGIN, '## V7 计算与存储成本（实测快照）', '',
        '更新时间：'+report['generated_at'], '',
        '仅统计当前七任务；不重算、不加载梯度数组。按模型分表，效果按同 cell 配对 rand、指定终点统计；缺结果不填零。', '',
        '**口径：选择时间含保存/重载；三梯度臂共享打分通常记在 grad-dot，不能据此声称其独立运行更慢。**', '',
        '**gradcos-legacy** 是原始 `grad-cos` 结果的展示名：源窗口梯度与“先逐参考轨迹平均、再跨轨迹平均”的原始参考梯度质心计算余弦；不是双侧逐窗口余弦平均。原始身份、选集和指标不变。', '',
        '存储是当前实现产物，不是算法存储下界。BP需求为每 cell 源/参考预算均值，不是新增BP；共享梯度资产另列且不重复摊给各臂。', '']
    for model in sorted({r['model'] for r in report['rows']}):
        groups = {}
        for row in report['rows']:
            if row['model'] == model:
                groups.setdefault(row['arm'], []).append(row)
        def improvement(rows):
            pairs = [(r.get('mase'), r.get('rand_mase')) for r in rows
                     if r.get('mase') is not None and r.get('rand_mase') is not None]
            return (100*(1-sum(a for a,b in pairs)/sum(b for a,b in pairs))
                    if pairs and sum(b for a,b in pairs)>0 else None)
        ordered = sorted(groups, key=lambda a: -(improvement(groups[a])
                         if improvement(groups[a]) is not None else -float('inf')))
        lines += ['### '+model, '',
            '| 方法 | vs rand / 配对n | 选择n | 选择均秒 | BP需求 源/参考 | 磁盘总MiB | 分数/概率/排序索引 平均MiB¹ | 训练记录n / cell均秒² |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ordered:
            rows = groups[arm]
            storage = [r['storage'] for r in rows if 'storage' in r]
            elapsed = [r['selection_elapsed_s'] for r in rows if r['selection_elapsed_s'] is not None]
            training = [r['cell_elapsed_s'] for r in rows if r.get('cell_elapsed_s') is not None]
            pair_n = sum(r.get('mase') is not None and r.get('rand_mase') is not None for r in rows)
            parts = '/'.join(show(mean([s['arrays'][k]['payload_bytes']/2**20 for s in storage]))
                             for k in ('scores','probabilities','descending_order'))
            budget = '/'.join(show(mean([r.get(k) for r in rows]), 0)
                              for k in ('required_source_bp','required_reference_bp'))
            disk = sum(s['file_bytes'] for s in storage)/2**20 if storage else None
            lines.append(f'| {display_arm(arm)} | {show(improvement(rows))}% / {pair_n} | {len(elapsed)} | '
                f'{show(mean(elapsed))} | {budget} | {show(disk)} | {parts} | '
                f'{len(training)} / {show(mean(training))} |')
        lines.append('')
    lines += ['¹ 数组逻辑字节，不含NPY头；JSON含每个成员压缩大小与起点/计数索引。磁盘列为整个NPZ实际文件大小。', '',
              '² cell计时是已有日志计时，不是纯GPU训练时间；恢复、硬件迁移及旧4epoch使横比受限，JSON保留步数/恢复位置/实际终点。', '',
              '### 共享资产准备（按 receipt 记录，不分摊到方法）', '',
              '| 模型 | receipts | 新BP | 导入行次 | 命中行次 | 准备累计小时 |',
              '|---|---:|---:|---:|---:|---:|']
    for model in sorted({a['model'] for a in report['shared_assets']}):
        assets = [a for a in report['shared_assets'] if a['model']==model]
        def total(key):
            values = [a[key] for a in assets if a.get(key) is not None]
            return sum(values) if len(values)==len(assets) else None
        seconds = total('elapsed_seconds')
        lines.append(f'| {model} | {len(assets)} | {show(total("new_bp"),0)} | '
                     f'{show(total("imported_rows"),0)} | {show(total("cached_rows"),0)} | '
                     f'{show(seconds/3600 if seconds is not None else None)} |')
    lines += ['', '### 共享梯度缓存物理库存（不分摊）', '',
              '| 命名空间 | 文件数 | 文件长度 GiB | 已分配 GiB |',
              '|---|---:|---:|---:|']
    for cache in report.get('shared_cache_inventory', []):
        lines.append(f'| {Path(cache["path"]).name} | {cache["files"]} | '
                     f'{cache["logical_bytes"]/2**30:.3f} | {cache["allocated_bytes"]/2**30:.3f} |')
    lines += ['', '仅清点 receipts 引用的缓存命名空间，stat元数据且硬链接去重；含该命名空间中跨任务共享资产，不是七任务独占/新增容量；不遍历旧donor池，不跟随内部软链。', '',
              '历史未测：峰值CPU内存/显存、设备GPU时间、独立排序/采样耗时、无缓存独立冷启动总成本。不得填0或用此表宣称FLOPs优势。', '',
              '共享缓存物理占用未分摊；共享准备累计时间不等于并行墙钟时间。排除任务的历史成本不在本七任务快照内。', END]
    return '\n'.join(lines)+'\n'


def update_markdown(path, block):
    path = Path(path)
    old = path.read_text() if path.exists() else ''
    if old.count(BEGIN) != old.count(END) or old.count(BEGIN)>1:
        raise ValueError('Ambiguous cost section boundaries')
    if BEGIN in old:
        start, stop = old.index(BEGIN), old.index(END)+len(END)
        new = old[:start]+block.rstrip()+old[stop:]
    else:
        new = old.rstrip()+'\n\n'+block
    atomic_text(path, new)
