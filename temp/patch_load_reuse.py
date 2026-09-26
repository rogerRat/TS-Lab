from pathlib import Path

def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit("expected text not found in " + path)
    p.write_text(text.replace(old, new, 1))

replace(
    "assets/dataset_processing/load/bundle.py",
    "from assets.dataset_processing.shard_dataset import ReferenceShardDatasetWriter, SourceShardDatasetWriter\n",
    "from assets.dataset_processing.load_reuse import write_load_metadata\nfrom assets.dataset_processing.shard_dataset import ReferenceShardDatasetWriter, SourceShardDatasetWriter\n",
)
replace(
    "assets/dataset_processing/load/bundle.py",
    """    def __init__(self, bundle_root: Path, *, scope: str, shard_bytes: int) -> None:
""",
    """    def __init__(self, bundle_root: Path, *, scope: str, shard_bytes: int,
                 load_metadata: Mapping[str, Any] | None = None) -> None:
""",
)
replace(
    "assets/dataset_processing/load/bundle.py",
    """        self.rows_written = 0
        self.duplicate_rows = 0
""",
    """        self.load_metadata = dict(load_metadata) if load_metadata is not None else None
        self.rows_written = 0
        self.duplicate_rows = 0
""",
)
replace(
    "assets/dataset_processing/load/bundle.py",
    """    def finalize(self) -> Path:
        path = self.writer.finish()
        if self.duplicate_rows:
""",
    """    def finalize(self) -> Path:
        path = self.writer.finish()
        if self.load_metadata is not None:
            metadata_path = write_load_metadata(path, self.load_metadata)
            log.info("LOAD identity metadata written: %s", metadata_path)
        else:
            log.warning(
                "LOAD identity metadata unavailable; bin-shard reuse is disabled for %s",
                path,
            )
        if self.duplicate_rows:
""",
)

replace(
    "assets/dataset_processing/load/src/runner.py",
    "import json\nimport logging\n",
    "import json\nimport logging\nimport math\n",
)
replace(
    "assets/dataset_processing/load/src/runner.py",
    "from assets.dataset_processing.load.bundle import BundleLoadWriter\n",
    "from assets.dataset_processing.load.bundle import BundleLoadWriter\nfrom assets.dataset_processing.load_reuse import (\n    dataset_identity_from_raw,\n    find_and_reuse_load_stage,\n    make_load_metadata,\n)\n",
)
replace(
    "assets/dataset_processing/load/src/runner.py",
    """    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    return parser.parse_args(argv)
""",
    """    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument(
        "--reuse-search-root", default=None,
        help="Sibling bundle root to scan for compatible Load bin shards",
    )
    return parser.parse_args(argv)
""",
)
replace(
    "assets/dataset_processing/load/src/runner.py",
    """    datasets = _discover_datasets(input_root) if args.dataset == "all" else [args.dataset]
    datasets = [item for item in datasets if item not in DEPRECATED_SOURCES]
""",
    """    selection = {
        "dataset": args.dataset,
        "subset": args.subset,
        "max_series": "inf" if math.isinf(args.max_series) else int(args.max_series),
        "seed": int(args.seed),
        "preserve_invalid_values": bool(args.preserve_invalid_values),
    }
    load_metadata = None
    try:
        load_metadata = make_load_metadata(
            dataset_identity_from_raw(input_root),
            scope="source",
            selection=selection,
            shard_bytes=args.shard_bytes,
        )
    except ValueError as exc:
        log.warning(
            "BIN SHARD REUSE FALLBACK: source identity unavailable (%s); "
            "rebuilding bins from raw input",
            exc,
        )
    if load_metadata is not None and args.reuse_search_root:
        if find_and_reuse_load_stage(
            bundle_root, Path(args.reuse_search_root), load_metadata, logger=log,
        ):
            log.info("Source Load completed by bin-shard reuse: %s", bundle_root)
            return 0
    datasets = _discover_datasets(input_root) if args.dataset == "all" else [args.dataset]
    datasets = [item for item in datasets if item not in DEPRECATED_SOURCES]
""",
)
replace(
    "assets/dataset_processing/load/src/runner.py",
    "writer = BundleLoadWriter(bundle_root, scope=\"source\", shard_bytes=args.shard_bytes)\n",
    "writer = BundleLoadWriter(\n        bundle_root, scope=\"source\", shard_bytes=args.shard_bytes,\n        load_metadata=load_metadata,\n    )\n",
)

replace(
    "assets/dataset_processing/load/ref/fev.py",
    "import argparse\nimport json\n",
    "import argparse\nimport hashlib\nimport json\n",
)
replace(
    "assets/dataset_processing/load/ref/fev.py",
    "from assets.dataset_processing.load.bundle import BundleLoadWriter\n",
    "from assets.dataset_processing.load.bundle import BundleLoadWriter\nfrom assets.dataset_processing.load_reuse import (\n    dataset_identity_from_raw,\n    make_load_metadata,\n    reuse_load_stage,\n)\n",
)
replace(
    "assets/dataset_processing/load/ref/fev.py",
    """def load_reference(spec_path: Path, fetch_dir: Path | None, bundle_dir: Path,
                   asset_id: str = "all", shard_bytes: int = DEFAULT_SHARD_BYTES,
                   include_future: bool = False) -> tuple[Path, ...]:
""",
    """def _reference_candidates(search_root: Path, asset_id: str):
    root = Path(search_root)
    if not root.is_dir():
        return ()
    result = []
    for protocol_root in sorted(root.iterdir()):
        if not protocol_root.is_dir():
            continue
        for candidate in (
            protocol_root / "fev" / asset_id,
            protocol_root / asset_id,
        ):
            if candidate.is_dir():
                result.append(candidate)
    return tuple(result)


def load_reference(spec_path: Path, fetch_dir: Path | None, bundle_dir: Path,
                   asset_id: str = "all", shard_bytes: int = DEFAULT_SHARD_BYTES,
                   include_future: bool = False,
                   reuse_search_root: Path | None = None) -> tuple[Path, ...]:
""",
)
replace(
    "assets/dataset_processing/load/ref/fev.py",
    """    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    adapter_name = {"fev": "fev", "gifteval": "gift_eval", "time": "time"}[spec.mapping["benchmark"]]
""",
    """    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    identity_batch = hashlib.sha256(
        spec_path.read_bytes() + declaration_path.read_bytes()
    ).hexdigest()
    raw_identity = dataset_identity_from_raw(
        data_root, dataset_id="fev-reference", batch_id=identity_batch,
    )
    adapter_name = {"fev": "fev", "gifteval": "gift_eval", "time": "time"}[spec.mapping["benchmark"]]
""",
)
replace(
    "assets/dataset_processing/load/ref/fev.py",
    """            writer = BundleLoadWriter(Path(bundle_dir) / asset["asset_id"], scope="reference", shard_bytes=shard_bytes)
            writer.write_rows(_asset_rows(temporary / asset["native_output"]["location"], asset["asset_id"]))
""",
    """            target_bundle = Path(bundle_dir) / asset["asset_id"]
            load_metadata = make_load_metadata(
                raw_identity,
                scope="reference",
                selection={
                    "asset_id": asset["asset_id"],
                    "spec_batch": identity_batch,
                    "include_future": bool(include_future),
                },
                shard_bytes=shard_bytes,
            )
            reused = False
            if reuse_search_root is not None:
                for candidate in _reference_candidates(reuse_search_root, asset["asset_id"]):
                    if reuse_load_stage(
                        target_bundle, candidate, load_metadata, logger=log,
                    ):
                        reused = True
                        break
            if reused:
                log.info("Reference Load reused bin shards: %s", target_bundle)
                produced.append(target_bundle)
                continue
            writer = BundleLoadWriter(
                target_bundle, scope="reference", shard_bytes=shard_bytes,
                load_metadata=load_metadata,
            )
            writer.write_rows(_asset_rows(temporary / asset["native_output"]["location"], asset["asset_id"]))
""",
)
replace(
    "assets/dataset_processing/load/ref/fev.py",
    """    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--include-future", action="store_true", help="FEV only: retain non-safe answer history for debug/test analysis; never use this output in training and evaluation")
""",
    """    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument(
        "--reuse-search-root", default=None,
        help="Sibling protocol root to scan for compatible reference bin shards",
    )
    parser.add_argument("--include-future", action="store_true", help="FEV only: retain non-safe answer history for debug/test analysis; never use this output in training and evaluation")
""",
)
replace(
    "assets/dataset_processing/load/ref/fev.py",
    """    produced = load_reference(Path(args.spec), Path(args.fetch_dir) if args.fetch_dir else None,
                              Path(args.bundle_dir), args.asset_id, args.shard_bytes, args.include_future)
""",
    """    produced = load_reference(
        Path(args.spec), Path(args.fetch_dir) if args.fetch_dir else None,
        Path(args.bundle_dir), args.asset_id, args.shard_bytes, args.include_future,
        Path(args.reuse_search_root) if args.reuse_search_root else None,
    )
""",
)

replace(
    "scripts/assets.py",
    """            "--input-dir", str(source_input),
            "--bundle-dir", str(source_bundle),
""",
    """            "--input-dir", str(source_input),
            "--bundle-dir", str(source_bundle),
            "--reuse-search-root", str(source_bundle.parent),
""",
)
replace(
    "scripts/assets.py",
    """            "--bundle-dir", str(reference_bundle_root),
        ],
""",
    """            "--bundle-dir", str(reference_bundle_root),
            "--reuse-search-root", str(reference_bundle_root.parent),
        ],
""",
)

for path in (
    "assets/dataset_processing/load_reuse.py",
    "assets/dataset_processing/load/bundle.py",
    "assets/dataset_processing/load/src/runner.py",
    "assets/dataset_processing/load/ref/fev.py",
    "scripts/assets.py",
):
    print("patched", path)
