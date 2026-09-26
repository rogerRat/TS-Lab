"""Tests for frozen Fetch Profile configuration and runner expansion."""

import json
from argparse import Namespace

from assets.dataset_processing.configs.profiles import PROFILE_SPECS
from assets.dataset_processing.fetch.src import runner
from assets.dataset_processing.fetch.src.sources import SOURCE_REVISIONS


def test_profile_specs_are_ascii_sorted_and_nested():
    """All frozen selections are ordered, unique, ASCII, and nested by Profile."""
    previous = None
    for profile in ("1b", "4b", "10b"):
        spec = PROFILE_SPECS[profile]
        assert spec["sample_probability"] in (0.11, 0.32, 0.40)
        current = {
            item["source"]: tuple(item.get("subsets", (item.get("utsd_config"),)))
            for item in spec["sources"]
        }
        for item in spec["sources"]:
            if item["source"] != "utsd":
                assert item["subsets"] == tuple(sorted(set(item["subsets"])))
                assert all(value.isascii() for value in item["subsets"])
        if previous is not None:
            assert set(previous).issubset(current)
            for source, selections in previous.items():
                assert set(selections).issubset(current[source])
        previous = current


def test_profile_source_fields_and_revisions_are_valid():
    """Every frozen source mapping follows the documented field contract."""
    for spec in PROFILE_SPECS.values():
        for item in spec["sources"]:
            if item["source"] == "utsd":
                assert set(item) == {"source", "revision", "utsd_config"}
            else:
                assert set(item) == {"source", "revision", "subsets"}
            assert item["revision"] == SOURCE_REVISIONS[item["source"]]


def test_profile_conflicting_options_return_before_dispatch(monkeypatch):
    """Profile mode rejects every documented conflicting option without a child."""
    monkeypatch.setattr(runner, "configure_logger", lambda *args: None)
    cases = (
        ("--sample-probability", "0.1"),
        ("--source", "chronos"),
        ("--subsets", "x"),
        ("--utsd-config", "UTSD-4G"),
        ("--revision", "main"),
    )
    for option, value in cases:
        assert runner.main(["--profile", "4b", option, value]) == 2


def test_profile_expansion_writes_completed_manifest(monkeypatch, tmp_path):
    """Successful child calls receive fixed arguments and publish completed JSON."""
    calls = []

    def fake_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(runner, "main", fake_main)
    args = Namespace(
        profile="1b", output_dir=str(tmp_path), seed=7, num_workers=2, force=True,
    )
    assert runner._run_profile(args) == 0
    assert len(calls) == len(PROFILE_SPECS["1b"]["sources"])
    for argv in calls:
        assert "--profile" not in argv
        assert "--sample-probability" in argv
        assert argv[argv.index("--sample-probability") + 1] == "0.11"
        assert argv[argv.index("--seed") + 1] == "7"
        assert argv[argv.index("--num-workers") + 1] == "2"
        assert "--force" in argv
    manifest = json.loads((tmp_path / "profile_manifest.json").read_text(encoding="ascii"))
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 7
    assert manifest["exit_codes"] == {
        item["source"]: 0 for item in PROFILE_SPECS["1b"]["sources"]
    }


def test_profile_failure_runs_remaining_sources_and_publishes_codes(monkeypatch, tmp_path):
    """A failed child does not prevent later sources and makes the Profile failed."""
    calls = []
    first_source = PROFILE_SPECS["1b"]["sources"][0]["source"]

    def fake_main(argv):
        calls.append(argv)
        return 3 if argv[argv.index("--source") + 1] == first_source else 0

    monkeypatch.setattr(runner, "main", fake_main)
    args = Namespace(
        profile="1b", output_dir=str(tmp_path), seed=42, num_workers=1, force=False,
    )
    assert runner._run_profile(args) == 3
    assert len(calls) == len(PROFILE_SPECS["1b"]["sources"])
    manifest = json.loads((tmp_path / "profile_manifest.json").read_text(encoding="ascii"))
    assert manifest["status"] == "failed"
    assert manifest["exit_codes"][first_source] == 3
