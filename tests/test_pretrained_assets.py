import json
from pathlib import Path
import pytest
from assets.pretrained_models import fetch


@pytest.mark.parametrize("family,key", [("chronos", "chronos-t5-base-200m"),
    ("timesfm", "timesfm-2.5"), ("timemoe", "timemoe-200m")])
def test_all_three_families_have_symmetric_dry_run(family, key, capsys):
    assert fetch.main([family, key, "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["family"] == family
    assert result["revision"] and result["ckpt"]
    assert result["architecture"]


def test_fetch_reuses_model_without_network_and_retains_ckpt_fields(tmp_path, monkeypatch):
    destination = tmp_path / "model"
    destination.mkdir()
    monkeypatch.setattr(fetch, "validate_checkpoint", lambda *args: None)
    def forbidden(*args):
        raise AssertionError("existing checkpoint must not download")
    monkeypatch.setattr(fetch, "download_checkpoint", forbidden)
    assert fetch.main(["timesfm", "timesfm-2.5", "--destination", str(destination)]) == 0
    manifest = json.loads((destination / "model_manifest.json").read_text())
    assert manifest["model_id"] == "google/timesfm-2.5-200m-transformers"
    assert manifest["ckpt"] == str(destination)
