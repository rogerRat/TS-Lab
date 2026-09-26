"""CPU-only raw-gradient linear reconstruction; no interpolated arrays persisted."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from rags.crb import CRBConfig, CRBEngine
from rags.epsilon_protocol import load_p10_calibration
from rags.window_reconstruction import interpolate_window_gradients
from .anchor_budget_linear import window_cosines
from .anchor_budget_simulate import (
    ArrayGradientComputer, FROZEN, atomic_write, check_rows_complete,
    completed_keys, load_entry_fragment, load_gradient_dataset,
    load_sample_manifest, manifest_entries,
)

METHODS = [(2, 0, 0), (2, 2, 0), (2, 2, 0), (2, 2, 2)]
PLACEMENTS = ["risk", "uniform", "risk", "risk"]
LABELS = ["Coverage only", "Extra uniform anchors", "Error-driven anchors", "Error + cancellation split"]
COLORS = ['#999999', '#9470b6', '#d19b27', '#188b86']
GT = '#2879b4'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--before-classification', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_sample_manifest(args.run_root)
    identity = json.loads((Path(manifest['cache'])/'cache_manifest.json').read_text())['model']
    if args.before_classification:
        # Coverage/error allocation precedes classification; no threshold is used.
        frozen = dict(FROZEN, detection='none')
        methods, labels, colors = METHODS[:3], LABELS[:3], COLORS[:3]
    else:
        if args.calibration is None:
            p.error('--calibration is required for classification and split')
        calibration = load_p10_calibration(args.calibration, identity)
        frozen = dict(FROZEN, mean_epsilon=calibration['epsilon'], endpoint_epsilon=calibration['epsilon'])
        methods, labels, colors = METHODS, LABELS, COLORS
    entries = manifest_entries(manifest, {0})
    done = completed_keys(args.run_root)
    assert len(entries) == 5 and all((e['task'], e['fragment_id'], e['wave']) in done for e in entries)
    dataset = load_gradient_dataset(manifest['cache'])
    check_rows_complete(dataset, entries)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    overview, ovs = plt.subplots(5, 1, figsize=(14, 13), constrained_layout=True)
    metadata = dict(metric='mean of per-window cosine in cached 4096-dimensional space',
                    projection='PCA fitted to all raw GT vectors per trajectory; shared by all estimates',
                    detail='central 256 consecutive windows, fixed before viewing results',
                    overview='nonoverlapping bins of 128 windows; last bin may be shorter',
                    stage='before classification; no split evaluated' if args.before_classification else 'including p10 classification and split', frozen={k:v for k,v in frozen.items() if not(args.before_classification and k in ('mean_epsilon','endpoint_epsilon'))}, calibration_path=str(args.calibration), storage="PNG/PDF and scalar metadata only; no interpolated arrays", model_calls=0, tasks=[])
    for task_no, entry in enumerate(entries):
        fragment = load_entry_fragment(entry)
        starts = fragment.valid_starts(512, 64)
        truth = dataset.lookup(entry['dataset'], np.column_stack([
            np.full(len(starts), entry['fragment_id']), starts]))
        n = len(truth)
        norms = np.linalg.norm(truth.astype(np.float64), axis=1)
        pca = PCA(n_components=2, svd_solver='randomized', random_state=0)
        pc = pca.fit(truth)
        # Explicit transform ensures GT and estimate use exactly the same map.
        pc = pca.transform(truth)
        lo, hi = (n-256)//2, (n-256)//2+256
        x = starts[lo:hi]
        fig, axes = plt.subplots(3, len(methods), figsize=(4.5*len(methods), 8), sharex=True,
                                 sharey='row', constrained_layout=True)
        name = entry['task'].removeprefix('time_')
        task_meta = dict(sample=entry, windows=n, dimension=truth.shape[1],
                         detail_physical_starts=[int(x[0]), int(x[-1])],
                         explained_variance_ratio=pca.explained_variance_ratio_.tolist(), methods=[])
        previous_anchors = {}
        computer = ArrayGradientComputer(starts, truth, entry['fragment_id'])
        for col, (budget, color) in enumerate(zip(methods, colors)):
            result = CRBEngine(CRBConfig(cover_density=budget[0]/100,
                error_density=budget[1]/100, split_density=budget[2]/100, **dict(frozen,error_placement=PLACEMENTS[col])),
                computer).process_fragment(fragment)
            # Only the displayed region is projected; estimates never go to disk.
            est = pca.transform(interpolate_window_gradients(result, np.arange(lo,hi)))
            cosine = window_cosines(truth,result)
            error = 1-cosine
            label = '/'.join(map(str, budget))
            anchor_ids = np.array(sorted(result.anchors), dtype=int)
            visible = anchor_ids[(anchor_ids >= lo) & (anchor_ids < hi)]
            for row in range(2):
                ax = axes[row, col]
                ax.plot(x, pc[lo:hi, row], color=GT, lw=1.5, label='True window gradient')
                ax.plot(x, est[:, row], color=color, lw=1.4, label='Linear estimate')
                ax.scatter(starts[visible], pc[visible, row], s=18, facecolors='white',
                           edgecolors='#263b50', linewidths=.7, zorder=4, label='Computed anchors')
                ax.grid(alpha=.15)
            split_note = 'classification not run' if args.before_classification else f'split budget {budget[2]}%, used {result.split_computations} anchors'
            axes[0, col].set_title(f'{LABELS[col]} | {label}%  |  actual anchors {len(anchor_ids)/n:.1%}\nMean cosine = {cosine.mean():.4f}; {split_note}', fontsize=11)
            axes[2, col].plot(x, error[lo:hi], color=color, lw=1.2)
            axes[2, col].axhline(.1, color='#a34c46', ls='--', lw=.9)
            axes[2, col].set_xlabel('Window start (trajectory index)')
            axes[2, col].grid(alpha=.15)
            bins = np.arange(0, n, 128)
            bx = [starts[b:min(b+128,n)].mean() for b in bins]
            by = [error[b:min(b+128,n)].mean() for b in bins]
            ovs[task_no].plot(bx, by, color=color, lw=1.15,
                             label=f'{LABELS[col]} (cos {cosine.mean():.4f})')
            task_meta['methods'].append(dict(condition=LABELS[col], error_placement=PLACEMENTS[col], budget_percent=budget, mean_cosine=float(cosine.mean()),
                anchors=len(anchor_ids), split_used=result.split_computations,
                discarded_blocks=result.discarded_no_signal+result.discarded_antagonistic+result.banned_budget))
            # Highlight only the anchors introduced by the mechanism in this column.
            base = previous_anchors.get(2 if col==3 else 0, set(anchor_ids))
            new_ids = [i for i in visible if i not in base]
            for row in range(2):
                axes[row,col].scatter(starts[new_ids],pc[new_ids,row],s=30,
                    marker='s' if col==3 else 'D',color=color,zorder=5)
            previous_anchors[col] = set(anchor_ids)
        axes[0, 0].set_ylabel('GT PCA component 1')
        axes[1, 0].set_ylabel('GT PCA component 2')
        axes[2, 0].set_ylabel('1 - cosine (4096-D)')
        axes[0, 0].legend(loc='best', fontsize=8)
        variance = pca.explained_variance_ratio_.sum()
        fig.suptitle(f'{name} | True vs linearly reconstructed window gradients\n'
                     f'Central 256 windows; same GT-only PCA axes across budgets; PC1 + PC2 explain {variance:.1%} of GT variance', fontsize=14)
        for extension in ('png', 'pdf'):
            fig.savefig(args.out/f'{name}_detail.{extension}', dpi=180)
        plt.close(fig)
        ov = ovs[task_no]
        ov.axhline(.1, color='#a34c46', ls='--', lw=.9)
        ov.axvspan(x[0], x[-1], color=GT, alpha=.08)
        ov.set_title(name, loc='left')
        ov.set_ylabel('Mean 1 - cosine')
        ov.set_ylim(bottom=0)
        ov.grid(alpha=.15)
        ov.legend(ncol=4, fontsize=9, loc='upper right')
        metadata['tasks'].append(task_meta)
        atomic_write(args.out/'metadata.json', metadata)
        print(json.dumps(task_meta), flush=True)
    ovs[-1].set_xlabel('Window start (trajectory index)')
    overview.suptitle('Full-trajectory reconstruction error | same per-window cosine metric\n'
        'Display: 128-window bin means. Dashed line: error 0.10. Shading: fixed detail region. Budgets: coverage / error / split.', fontsize=13)
    for extension in ('png', 'pdf'):
        overview.savefig(args.out/f'all_tasks_error.{extension}', dpi=180)
    plt.close(overview)


if __name__ == '__main__':
    main()
