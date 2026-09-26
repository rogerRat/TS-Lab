from pathlib import Path
import pytest
from configs import paths

from assets.dataset_processing.fetch.ref import runner


@pytest.fixture(autouse=True)
def isolated_path_roots(monkeypatch, tmp_path):
    # Exercise the real path policy within test-owned roots.
    monkeypatch.setattr(paths, "CONFIGS", tmp_path)
    monkeypatch.setitem(paths.OUTPUT_ROOTS, "benchmark", tmp_path)


def test_fetch_uses_explicit_output_dir(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    requested_output = tmp_path / "requested-output"
    captured: dict[str, Path] = {}

    class Specification:
        mapping = {
            "upstream": {"data_root": "default-output"},
            "evaluation_declaration": {"path": "declaration.yaml"},
            "benchmark": "fev",
        }

    monkeypatch.setattr(
        runner.ReferenceExtractionSpecification,
        "from_file",
        lambda path: Specification(),
    )
    monkeypatch.setattr(
        runner, "import_module",
        lambda *args, **kwargs: type(
            "Adapter", (), {"fetch": lambda spec, declaration, data_root: captured.update(data_root=data_root)}
        ),
    )

    assert runner.fetch(spec_path, requested_output) == requested_output.resolve()
    assert captured["data_root"] == requested_output.resolve()


def test_fetch_defaults_to_spec_data_root(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    captured: dict[str, Path] = {}

    class Specification:
        mapping = {
            "upstream": {"data_root": "default-output"},
            "evaluation_declaration": {"path": "declaration.yaml"},
            "benchmark": "fev",
        }

    monkeypatch.setattr(
        runner.ReferenceExtractionSpecification,
        "from_file",
        lambda path: Specification(),
    )
    monkeypatch.setattr(
        runner, "import_module",
        lambda *args, **kwargs: type(
            "Adapter", (), {"fetch": lambda spec, declaration, data_root: captured.update(data_root=data_root)}
        ),
    )

    assert runner.fetch(spec_path) == (tmp_path / "default-output").resolve()
    assert captured["data_root"] == (tmp_path / "default-output").resolve()


def test_main_passes_explicit_output_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Path] = {}
    output_dir = tmp_path / "requested-output"
    # This dispatch test must not retain pytest's captured stream in a global
    # CLI logger whose summary is emitted after capture has closed.
    monkeypatch.setattr("configs.logger.configure_logger", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        runner,
        "fetch",
        lambda spec_path, output_dir=None: captured.update(
            spec_path=spec_path, output_dir=output_dir
        ),
    )

    assert runner.main(["--spec", "spec.yaml", "--output-dir", str(output_dir)]) == 0
    assert captured == {"spec_path": Path("spec.yaml"), "output_dir": output_dir}
