import yaml

from rags.experiments import run as runner


CONFIG = {
    "batch": "test-batch",
    "steps": ["main", "mechanism1", "mechanism2", "ablation",
              "data-curve", "time-curve"],
    "context_length": 512,
    "prediction_length": 64,
    "pool_stride": 1,
    "cover_density": 0.02,
    "error_density": 0.02,
    "split_density": 0.005,
    "block_budget": 8,
    "mean_epsilon": 0.70388913,
    "endpoint_epsilon": 0.70388913,
    "epsilon_provenance": "diag run p10 calibration",
    "datasets": ["ETT_1H"],
    "series_cap": None,
    "n_windows": 4096,
    "pool_equivalents": 20480,
    "pool_seed": 0,
    "projection_dimension": 4096,
    "source_epochs": 4,
    "train_batch": 32,
    "learning_rate": 1e-5,
    "seeds": [0, 1, 2],
    "with_baselines": ["less", "tsds", "lp_mirror", "ltsv"],
    "variants": ["full", "random_cover"],
    "with_grid": True,
    "n_values": [1024, 4096],
    "total_budget_seconds": 1800,
    "densities": [0.01, 0.05],
    "b_values": [1, 8],
    "m_values": [50, 100],
    "density": 0.05,
    "cancel_ratio": 0.5,
    "step_overrides": {
        "mechanism1": {"dataset": "ETT_1H", "stride": 1,
                       "projection_dimension": 1024,
                       "series_cap": 2, "slice_length": 4096,
                       "pool_series": 4, "pool_slice": 4096},
        "mechanism2": {"scan_dir": "{run_dir}/mechanism1", "stride": 1},
        "data-curve": {"dataset": "ETT_1H",
                       "arms": ["rags", "random", "trajectory_level"]},
        "time-curve": {"dataset": "ETT_1H", "arms": ["rags", "random"]},
        "ablation": {"dataset": "ETT_1H"},
    },
}


def test_step_argv_validates_against_every_real_parser(tmp_path):
    run_dir = tmp_path / "run"
    for step in CONFIG["steps"]:
        argv = runner._step_argv(step, CONFIG, run_dir)
        assert argv[-1] == str(run_dir / step)
        module = __import__(runner.STEP_MODULES[step], fromlist=["parse_args"])
        module.parse_args(argv[3:])  # must not raise


def test_step_argv_drops_unrelated_flags_and_applies_overrides(tmp_path):
    run_dir = tmp_path / "run"
    main_argv = runner._step_argv("main", CONFIG, run_dir)
    joined = " ".join(main_argv)
    assert "--datasets ETT_1H" in joined
    assert "--n-windows 4096" in joined
    assert "--pool-equivalents 20480" in joined
    assert "--source-epochs 4" in joined
    assert "--with-grid" not in joined        # ablation-only flag

    mech1 = runner._step_argv("mechanism1", CONFIG, run_dir)
    joined = " ".join(mech1)
    assert "--dataset ETT_1H" in joined       # override applied
    assert "--projection-dimension 1024" in joined
    assert "--series-cap 2" in joined
    assert "--mean-epsilon 0.70388913" in joined   # calibrated epsilon flows
    assert "--with-baselines" not in joined   # unrelated flag dropped

    mech2 = runner._step_argv("mechanism2", CONFIG, run_dir)
    joined = " ".join(mech2)
    assert ("--scan-dir %s" % (run_dir / "mechanism1")) in joined
    assert "--datasets" not in joined

    curve = runner._step_argv("data-curve", CONFIG, run_dir)
    joined = " ".join(curve)
    assert "data-curve" in curve
    assert "--arms rags random trajectory_level" in joined
    assert "--n-values 1024 4096" in joined
    assert "--schedule-variants" not in joined
    assert "--source-epochs 4" in joined


def test_freeze_doc_and_index(tmp_path):
    run_id = "rags-test-20000101-0000"
    run_dir = tmp_path / run_id
    step_argv = {step: runner._step_argv(step, CONFIG, run_dir)
                 for step in CONFIG["steps"]}
    runner.freeze_run_doc(run_id, CONFIG, CONFIG["steps"], step_argv, run_dir)
    runner.append_index(tmp_path, run_id, CONFIG, CONFIG["steps"], run_dir)

    text = (run_dir / "RUN.md").read_text(encoding="utf-8")
    assert run_id in text
    assert "K_cover" in text and "0.02" in text
    assert "4:4:1" in text                     # frozen budget ratio
    assert "--block-budget 8" in text
    assert "rags.experiments.main_table" in text
    assert "0.70388913" in text                # calibrated epsilon frozen
    assert "diag run p10 calibration" in text  # provenance recorded
    assert "source_epochs" in text             # single-stage epoch schedule
    assert "trajectory_level" in text          # granularity arms listed
    assert "target_only" not in text           # removed per A7/D1

    index = (tmp_path / "rags_runs" / "RUNS.md").read_text(encoding="utf-8")
    assert run_id in index and "RUN.md" in index

    runner._update_index_status(tmp_path, run_id, "complete")
    index = (tmp_path / "rags_runs" / "RUNS.md").read_text(encoding="utf-8")
    assert "| complete |" in index
    assert "| launched |" not in index


def test_yaml_config_roundtrip(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    loaded = yaml.safe_load(path.read_text())
    assert loaded["step_overrides"]["mechanism2"]["scan_dir"] == "{run_dir}/mechanism1"
    argv = runner._step_argv("mechanism2", loaded, tmp_path / "run")
    assert "--cancel-ratio 0.5" in " ".join(argv)


def test_resume_reuses_saved_config_without_refreezing_metadata(tmp_path, monkeypatch):
    run_dir=tmp_path/'existing-run'
    run_dir.mkdir()
    (run_dir/'config.yaml').write_text(yaml.safe_dump(CONFIG))
    (run_dir/'RUN.md').write_text('original launch record')
    monkeypatch.setattr(runner,'freeze_run_doc',lambda *a,**k: (_ for _ in ()).throw(AssertionError('refreeze')))
    assert runner.main(['--resume',str(run_dir),'--steps','main','--dry-run'])==0
    assert (run_dir/'RUN.md').read_text()=='original launch record'
