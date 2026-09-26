"""Tests for explicit raw-file trajectory sampling."""

import json
import zipfile
from datetime import datetime, timedelta

import pytest

import numpy as np

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from assets.dataset_processing.fetch.src.chronos import _sample_chronos_file
from assets.dataset_processing.configs.chronos import (
    CHRONOS_CONFIG_PATHS,
    CHRONOS_SEQUENCE_COLUMNS,
    CHRONOS_SUPPORTED_SUBSETS,
)
from assets.dataset_processing.fetch.src.ett import _sample_ett_file, _sampled_marker_matches as _ett_marker_matches
from assets.dataset_processing.fetch.src.monash import _sample_monash_zip
from assets.dataset_processing.fetch.src.sampling import FileTrajectorySampler, file_rng
from assets.dataset_processing.fetch.src.utsd import _sample_utsd_file
from assets.dataset_processing.fetch.src.hub_files import (
    _sample_grouped_parquet, _sample_pile_ts_file, _sample_row_file,
    _sample_time300b_files, _sampled_marker_matches as _extended_marker_matches,
    _split_time300b_leaf_files,
)
from assets.dataset_processing.load.src.extended import _write_records
from assets.dataset_processing.load.src.pile import _records as _pile_records
from assets.dataset_processing.load.src.quito import _records as _quito_records
from assets.dataset_processing.load.src.rmisc import _records as _rmisc_records
from assets.dataset_processing.load.src.time300b import _records as _time300b_records, load as load_time300b
from assets.dataset_processing.load.src.utsd import load_utsd
from assets.dataset_processing.load.bundle import BundleLoadWriter
from assets.dataset_processing.shard_dataset import ShardDatasetReader


def _runtime_writer(tmp_path, dataset, subset):
    """Create the Bundle writer used by Load decoder fixtures."""
    return BundleLoadWriter(tmp_path / "bundle", scope="source", shard_bytes=1024), tmp_path


def _runtime_rows(published):
    """Read Bundle Load rows and their immutable stored values."""
    with (published / "load" / "index.arrow").open("rb") as stream:
        rows = ipc.open_file(stream).read_all().to_pylist()
    reader = ShardDatasetReader(published, 1)
    try:
        values = []
        for row in rows:
            result = reader.read(row["trajectory_id"])
            values.append((row["trajectory_id"].split("/")[-1], result.tolist()))
    finally:
        reader.close()
    return values


def test_sampler_keeps_first_complete_record():
    sampler = FileTrajectorySampler(0.0001, file_rng(42, "x"))
    decisions = [sampler.keep() for _ in range(20)]
    assert decisions[0] is True
    assert sampler.selected >= 1


def test_extended_sampled_marker_requires_nonzero_valid_trajectories(tmp_path):
    """A sampled extended selection must not skip from old or empty markers."""
    raw_file = tmp_path / "part.parquet"
    raw_file.touch()
    files = ["part.parquet"]
    base = {
        "files": files,
        "sampling": {"probability": 0.01, "seed": 42},
    }
    assert not _extended_marker_matches(base, files, 0.01, 42, tmp_path)
    base["valid_trajectories"] = {"part.parquet": 0}
    assert not _extended_marker_matches(base, files, 0.01, 42, tmp_path)
    base["valid_trajectories"] = {"part.parquet": 1}
    assert _extended_marker_matches(base, files, 0.01, 42, tmp_path)


def test_ett_sampled_marker_requires_nonzero_valid_trajectories(tmp_path):
    """A sampled ETT CSV must not skip from an old or empty marker."""
    filename = "ETTh1.csv"
    (tmp_path / filename).touch()
    marker = tmp_path / ".complete" / (filename + ".json")
    marker.parent.mkdir()
    marker.write_text(json.dumps({
        "file": filename,
        "sampling": {"probability": 0.01, "seed": 42},
    }), encoding="ascii")
    assert not _ett_marker_matches(tmp_path, filename, 0.01, 42)
    marker.write_text(json.dumps({
        "file": filename,
        "sampling": {"probability": 0.01, "seed": 42},
        "valid_trajectories": 1,
    }), encoding="ascii")
    assert _ett_marker_matches(tmp_path, filename, 0.01, 42)


def test_chronos_sampling_keeps_one_row_and_schema(tmp_path):
    source = tmp_path / "source.parquet"
    destination = tmp_path / "nested" / "source.parquet"
    table = pa.table({
        "id": ["a", "b", "c"],
        "target": [list(range(10)), list(range(10, 20)), list(range(20, 30))],
        "extra": [1, 2, 3],
    })
    pq.write_table(table, source)
    candidates, selected = _sample_chronos_file(source, destination, "m4_daily", 0.0001, 42)
    assert (candidates, selected) == (3, 1)
    result = pq.read_table(destination)
    assert result.schema == table.schema
    assert result.num_rows == 1


def test_all_supported_chronos_configs_have_explicit_paths_and_fields():
    """Every selectable Chronos config has a Fetch path and Load target field."""
    supported = set(CHRONOS_SUPPORTED_SUBSETS)
    assert supported == set(CHRONOS_CONFIG_PATHS)
    assert supported == set(CHRONOS_SEQUENCE_COLUMNS)
    assert CHRONOS_CONFIG_PATHS["training_corpus_tsmixup_10m"] == "training_corpus/tsmixup_10m"
    assert CHRONOS_CONFIG_PATHS["weatherbench_hourly_2m_temperature"] == "weatherbench_hourly/2m_temperature"


def test_chronos_all_expands_the_full_supported_set(monkeypatch, tmp_path):
    """The explicit all selector must not silently fall back to the default subset list."""
    from assets.dataset_processing.fetch.src import runner
    from assets.dataset_processing.fetch.src import chronos

    captured = {}
    monkeypatch.setattr(runner, "configure_logger", lambda *args: None)
    monkeypatch.setattr(
        chronos,
        "fetch_chronos",
        lambda output_dir, **kwargs: captured.update(output_dir=output_dir, **kwargs) or 0,
    )

    assert runner.main([
        "--source", "chronos", "--subsets", "all",
        "--output-dir", str(tmp_path),
    ]) == 0
    assert captured["subsets"] == CHRONOS_SUPPORTED_SUBSETS


def test_utsd_sampling_keeps_one_record_and_arrow_file(tmp_path):
    source = tmp_path / "source.arrow"
    destination = tmp_path / "nested" / "source.arrow"
    schema = pa.schema([("item_id", pa.string()), ("target", pa.list_(pa.float64()))])
    with pa.OSFile(str(source), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_batch(pa.RecordBatch.from_pylist([
            {"item_id": "a", "target": list(range(10))},
            {"item_id": "b", "target": list(range(10, 20))},
        ], schema=schema))
        writer.close()
    candidates, selected = _sample_utsd_file(source, destination, 0.0001, 42)
    assert (candidates, selected) == (2, 1)
    with pa.memory_map(str(destination), "r") as source_map:
        reader = ipc.open_file(source_map)
        assert sum(reader.get_batch(index).num_rows for index in range(reader.num_record_batches)) == 1


def test_bundle_load_writer_skips_duplicate_trajectory_ids(tmp_path):
    '''All Load adapters share a writer that keeps the first duplicate row.'''
    writer, _ = _runtime_writer(tmp_path, "lotsa", "PEMS03")
    writer.write_rows([
        {"series_id": "duplicate", "series": list(range(10)),
         "length": 10, "dataset": "lotsa", "subset": "PEMS03"},
        {"series_id": "duplicate", "series": list(range(10, 20)),
         "length": 10, "dataset": "lotsa", "subset": "PEMS03"},
    ])
    assert writer.rows_written == 1
    assert writer.duplicate_rows == 1
    published = writer.finalize()
    assert _runtime_rows(published) == [
        ("duplicate", [float(value) for value in range(10)]),
    ]


def test_utsd_load_skips_duplicate_item_ids_across_files(tmp_path, caplog):
    """UTSD Load keeps the first raw record when shards repeat an item_id."""
    raw_root = tmp_path / "raw"
    subset_dir = raw_root / "UTSD-4G"
    subset_dir.mkdir(parents=True)
    filenames = ["UTSD-4G/part-0.arrow", "UTSD-4G/part-1.arrow"]
    (subset_dir / ".complete").write_text(json.dumps({"files": filenames}), encoding="ascii")
    schema = pa.schema([("item_id", pa.string()), ("target", pa.list_(pa.float64()))])
    rows_by_file = [
        [{"item_id": "duplicate", "target": list(range(10))},
         {"item_id": "first", "target": list(range(10, 20))}],
        [{"item_id": "duplicate", "target": list(range(20, 30))},
         {"item_id": "second", "target": list(range(30, 40))}],
    ]
    for filename, rows in zip(filenames, rows_by_file):
        with pa.OSFile(str(subset_dir / filename.split("/")[-1]), "wb") as sink:
            arrow_writer = ipc.new_file(sink, schema)
            arrow_writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))
            arrow_writer.close()

    writer, _ = _runtime_writer(tmp_path, "utsd", "UTSD-4G")
    caplog.set_level("WARNING", logger="load")
    assert load_utsd(raw_root, writer, "UTSD-4G") == 3
    published = writer.finalize()

    assert _runtime_rows(published) == [
        ("duplicate", [float(value) for value in range(10)]),
        ("first", [float(value) for value in range(10, 20)]),
        ("second", [float(value) for value in range(30, 40)]),
    ]
    assert "LOAD skipped 1 duplicate UTSD item_id records" in caplog.text



def test_monash_sampling_keeps_one_tsf_record_and_non_tsf_members(tmp_path):
    source = tmp_path / "source.zip"
    destination = tmp_path / "nested" / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data/test.tsf", "# Dataset Information\n@attribute a numeric\n@data\na:0,1,2,3,4,5,6,7,8,9\nb:10,11,12,13,14,15,16,17,18,19\n")
        archive.writestr("data/info.json", "{}")
    candidates, selected = _sample_monash_zip(source, destination, 0.0001, 42, "x")
    assert (candidates, selected) == (2, 1)
    with zipfile.ZipFile(destination) as archive:
        assert "data/info.json" in archive.namelist()
        assert archive.read("data/test.tsf").count(b"\n") == 4


def test_monash_sampling_rejects_header_only_archive(tmp_path):
    source = tmp_path / "source.zip"
    destination = tmp_path / "nested" / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data/test.tsf", "# Dataset Information\n@data\n")
    with pytest.raises(ValueError, match="no valid TSF trajectory"):
        _sample_monash_zip(source, destination, 0.0001, 42, "x")
    assert not destination.exists()


def test_utsd_sampling_rejects_short_records(tmp_path):
    source = tmp_path / "source.arrow"
    destination = tmp_path / "nested" / "source.arrow"
    schema = pa.schema([("item_id", pa.string()), ("target", pa.list_(pa.float64()))])
    with pa.OSFile(str(source), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_batch(pa.RecordBatch.from_pylist([
            {"item_id": "a", "target": [1.0, 2.0]},
        ], schema=schema))
        writer.close()
    with pytest.raises(ValueError, match="no valid trajectory"):
        _sample_utsd_file(source, destination, 0.0001, 42)
    assert not destination.exists()


def test_ett_sampling_keeps_complete_multivariate_trajectory(tmp_path):
    source = tmp_path / "ETTh1.csv"
    destination = tmp_path / "nested" / "ETTh1.csv"
    source.write_text(
        "date,A,B\n" + "".join("2020-01-%02d,%d,%d\n" % (day, day, day + 1) for day in range(1, 11)),
        encoding="utf-8",
    )
    candidates, selected = _sample_ett_file(source, destination, 0.0001, 42, "ETTh1")
    assert (candidates, selected) == (1, 1)
    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_extended_grouped_sampling_keeps_all_rows_and_variables(tmp_path):
    """Quito/RMISC-style grouping samples a complete synchronized trajectory."""
    source = tmp_path / "source.parquet"
    destination = tmp_path / "nested" / "source.parquet"
    table = pa.table({
        "item_id": ["a"] * 10 + ["b"] * 10,
        "date_time": list(range(10)) + list(range(10)),
        "ind_1": [float(value) for value in range(20)],
        "ind_2": [float(value) for value in range(20, 40)],
    })
    pq.write_table(table, source)
    candidates, selected = _sample_grouped_parquet(source, destination, "quito", 0.0001, 42)
    assert (candidates, selected) == (2, 1)
    result = pq.read_table(destination).to_pydict()
    assert set(result["item_id"]) == {"a"}
    assert result["ind_1"] == [float(value) for value in range(10)]
    assert result["ind_2"] == [float(value) for value in range(20, 30)]


def test_quito_load_merges_noncontiguous_item_blocks(tmp_path):
    """Quito Load globally groups repeated item_id blocks in time order."""
    source = tmp_path / "quito.parquet"
    rows = {
        "item_id": [7, 7, 9, 9, 7, 7],
        "date_time": [
            datetime(2020, 1, 1) + timedelta(hours=offset)
            for offset in (4, 5, 1, 2, 3, 6)
        ],
        "ind_1": [40.0, 50.0, 10.0, 20.0, 30.0, 60.0],
        "ind_2": [41.0, 51.0, 11.0, 21.0, 31.0, 61.0],
        "ind_3": [42.0, 52.0, 12.0, 22.0, 32.0, 62.0],
        "ind_4": [43.0, 53.0, 13.0, 23.0, 33.0, 63.0],
        "ind_5": [44.0, 54.0, 14.0, 24.0, 34.0, 64.0],
    }
    pq.write_table(pa.table(rows), source)

    result = list(_quito_records(source))
    assert len(result) == 10
    assert [item_id for item_id, _name, _values in result].count("7") == 5
    assert [item_id for item_id, _name, _values in result].count("9") == 5
    assert result[0] == ("7", "ind_1", [30.0, 40.0, 50.0, 60.0])
    assert result[5] == ("9", "ind_1", [10.0, 20.0])


def test_rmisc_single_series_parquet_uses_file_as_trajectory(tmp_path):
    """RMISC single-series files need no _original_filename column."""
    source = tmp_path / "source.parquet"
    destination = tmp_path / "nested" / "source.parquet"
    pq.write_table(pa.table({"f1": [float(value) for value in range(10)]}), source)
    assert _sample_grouped_parquet(source, destination, "rmisc", 0.0001, 42) == (1, 1)
    assert pq.read_table(destination).to_pydict() == {"f1": [float(value) for value in range(10)]}
    assert list(_rmisc_records(destination)) == [("source", "f1", [float(value) for value in range(10)])]


def test_extended_record_sampling_keeps_multivariate_record(tmp_path):
    """Arrow-record sources never sample synchronized target variables separately."""
    source = tmp_path / "source.parquet"
    destination = tmp_path / "nested" / "source.parquet"
    table = pa.table({
        "item_id": ["a", "b"],
        "target": [[list(range(10)), list(range(10, 20))], [list(range(20, 30)), list(range(30, 40))]],
    })
    pq.write_table(table, source)
    assert _sample_row_file("lotsa", source, destination, 0.0001, 42) == (2, 1)
    assert pq.read_table(destination).num_rows == 1


def test_extended_load_splits_multivariate_target_at_load_stage(tmp_path):
    """Two-dimensional raw targets become separate canonical variable rows."""
    writer, _ = _runtime_writer(tmp_path, "lotsa", "PEMS03")
    _write_records(
        (("trajectory", "target_0", list(range(20))),
         ("trajectory", "target_1", list(range(20, 40)))),
        writer, "lotsa", "PEMS03", float("inf"), 42,
    )
    published = writer.finalize()
    assert _runtime_rows(published) == [
        ("trajectory__target_0", [float(value) for value in range(20)]),
        ("trajectory__target_1", [float(value) for value in range(20, 40)]),
    ]


def test_pile_ts_sampling_preserves_header_and_complete_records(tmp_path):
    """PILE UCR text sampling preserves format and never truncates a record."""
    source = tmp_path / "ACSF1_TRAIN.ts"
    destination = tmp_path / "nested" / "ACSF1_TRAIN.ts"
    source.write_text(
        "@problemName ACSF1\n@classLabel true 0 1\n@data\n0,1,2,3,4,5,6,7,8,9:0\n10,11,12,13,14,15,16,17,18,19:1\n",
        encoding="utf-8",
    )
    assert _sample_pile_ts_file(source, destination, 0.0001, 42) == (2, 1)
    contents = destination.read_text(encoding="utf-8")
    assert "@problemName ACSF1\n@classLabel true 0 1\n@data\n" in contents
    assert contents.count(":") == 1
    assert list(_pile_records(destination)) == [
        ("ACSF1_TRAIN_0", "", [float(value) for value in range(10)]),
    ]


def test_pile_ts_load_converts_multivariate_rows_to_canonical(tmp_path):
    """PILE multivariate .ts dimensions become separate CanonicalParquet rows."""
    raw = tmp_path / "BasicMotions_TRAIN.ts"
    raw.write_text(
        "@data\n0,1,2,3,4,5,6,7,8,9,10,11:12,13,14,15,16,17,18,19,20,21,22,23\n",
        encoding="utf-8",
    )
    writer, _ = _runtime_writer(tmp_path, "pile", "classification")
    _write_records(_pile_records(raw), writer, "pile", "classification", float("inf"), 42)
    published = writer.finalize()
    assert _runtime_rows(published) == [
        ("BasicMotions_TRAIN_0__dimension_0", [float(value) for value in range(12)]),
        ("BasicMotions_TRAIN_0__dimension_1", [float(value) for value in range(12, 24)]),
    ]


def test_time300b_sampling_preserves_bin_metadata_contract_and_loads(tmp_path):
    """Time-300B sampling rewrites offsets and Load reads the original format."""
    relative = "energy/example"
    cache_root = tmp_path / "cache"
    raw_root = tmp_path / "raw"
    cache_dir = cache_root / relative
    cache_dir.mkdir(parents=True)
    np.asarray(list(range(1, 13)) + [20, 21, 22, 23], dtype=np.float32).tofile(cache_dir / "data-1-of-1.bin")
    (cache_dir / "meta.json").write_text(json.dumps({
        "num_sequences": 2,
        "dtype": "float32",
        "files": {"data-1-of-1.bin": 16},
        "scales": [{"offset": 0, "length": 12}, {"offset": 12, "length": 4}],
    }), encoding="ascii")
    files = [relative + "/data-1-of-1.bin", relative + "/meta.json"]
    assert _sample_time300b_files(cache_root, raw_root, files, 0.0001, 42) == (1, 1)
    metadata = json.loads((raw_root / relative / "meta.json").read_text(encoding="ascii"))
    assert metadata["num_sequences"] == 1
    assert metadata["scales"] == [{"offset": 0, "length": 12}]
    assert metadata["files"] == {"data-1-of-1.bin": 12}
    assert np.fromfile(raw_root / relative / "data-1-of-1.bin", dtype=np.float32).tolist() == [float(value) for value in range(1, 13)]
    assert list(_time300b_records(
        raw_root / relative / "data-1-of-1.bin", raw_root / relative / "meta.json",
    )) == [("data-1-of-1_0", "", [float(value) for value in range(1, 13)])]
    marker = raw_root / ".complete" / "energy__example.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"files": files}), encoding="ascii")
    writer, _ = _runtime_writer(tmp_path, "time300b", relative)
    results = load_time300b(raw_root, writer, relative, float("inf"), 42)
    published = writer.finalize()
    assert results == 1
    assert _runtime_rows(published) == [
        ("data-1-of-1_0", [float(value) for value in range(1, 13)]),
    ]


def test_time300b_parent_selection_splits_complete_leaf_pairs():
    """Each Time-300B leaf receives its own BIN/metadata selection."""
    files = [
        "energy/a/meta.json",
        "energy/a/data-1-of-1.bin",
        "energy/b/meta.json",
        "energy/b/data-1-of-1.bin",
    ]
    assert _split_time300b_leaf_files(files, "energy") == [
        ("energy/a", ["energy/a/data-1-of-1.bin", "energy/a/meta.json"]),
        ("energy/b", ["energy/b/data-1-of-1.bin", "energy/b/meta.json"]),
    ]


def test_time300b_parent_selection_rejects_incomplete_leaf_pair():
    """Parent expansion must not silently discard an incomplete leaf."""
    with pytest.raises(ValueError, match=r"one \.bin and one meta\.json"):
        _split_time300b_leaf_files(
            ["energy/a/meta.json", "energy/b/data-1-of-1.bin"],
            "energy",
        )
