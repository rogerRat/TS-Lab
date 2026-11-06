"""Canonical and compatibility package import contracts."""

import importlib
from pathlib import Path
import subprocess
import sys

from configs.paths import PROJECT_ROOT


def test_canonical_and_compatibility_package_imports() -> None:
    module_names = (
        "assets.dataset_processing.shard_dataset",
        "gradient",
        "gradient.chronos",
        "rage.experiments.data",
    )
    for module_name in module_names:
        assert importlib.import_module(module_name) is not None


def test_project_implementations_live_under_src_and_baselines_are_separate():
    for name in ('rage', 'tsl', 'gradient', 'adapters', 'assets', 'evaluate'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(PROJECT_ROOT / 'src')
        assert not (PROJECT_ROOT / name).exists()
    for name in ('baselines.algorithms', 'baselines.contracts', 'baselines.runners'):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(PROJECT_ROOT / 'baselines')


def test_moved_modules_keep_repository_data_roots():
    for name in ('evaluate.fev', 'evaluate.time', 'evaluate.run',
                 'rage.experiments.run', 'rage.experiments.time_bench'):
        assert importlib.import_module(name).PROJECT_ROOT == PROJECT_ROOT


def test_cli_router_dispatches_without_implementing_selection(monkeypatch):
    from scripts.exp import rage
    calls = []
    class Workflow:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 0
    monkeypatch.setattr(rage, 'import_module', lambda name: Workflow)
    assert rage.main(['main_table', '--help']) == 0
    assert calls == [['--help']]


def test_installed_packages_import_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, '-c',
         'import rage, tsl, baselines, configs.paths; '
         'from baselines.runners import run_baseline'],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_selection_stacks_are_removed_but_external_baselines_remain():
    removed = ('src/tsl/core', 'src/tsl/features', 'src/train', 'src/analysis',
               'src/assets/dataset_processing/features',
               'src/assets/dataset_processing/catalog')
    for relative in removed:
        assert not (PROJECT_ROOT / relative).exists()
    from baselines.contracts import BASELINE_CONTRACTS
    assert set(BASELINE_CONTRACTS) == {'less', 'tsds', 'lp_mirror', 'ltsv'}


def test_assets_cli_includes_data_models_and_gradients(monkeypatch):
    from scripts import assets
    assert set(assets.WORKFLOWS) == {
        'fetch-source', 'fetch-reference', 'load-source', 'load-reference',
        'dedup-source', 'dedup-reference',
        'fetch-model', 'gradient',
    }
    calls = []
    class Stage:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 0
    monkeypatch.setattr(assets, 'import_module', lambda name: Stage)
    assert assets.main(['load-source', '--help']) == 0
    assert calls == [['--help']]


def test_every_external_baseline_owns_its_runner_kernels_and_contract():
    from baselines.runners import BASELINE_RUNNERS
    from baselines.contracts import BASELINE_CONTRACTS
    for name, runner in BASELINE_RUNNERS.items():
        package = importlib.import_module('baselines.' + name)
        assert callable(package.select)
        assert package.CONTRACT is BASELINE_CONTRACTS[name]
        assert runner.__module__ == 'baselines.' + name + '.runner'
        assert importlib.import_module('baselines.' + name + '.algorithms')
        for filename in ('__init__.py', 'algorithms.py', 'runner.py', 'contract.py'):
            assert (PROJECT_ROOT / 'baselines' / name / filename).is_file()
