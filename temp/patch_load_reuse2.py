from pathlib import Path

def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit("expected text not found in " + path)
    p.write_text(text.replace(old, new, 1))

replace(
    "assets/dataset_processing/load/ref/fev.py",
    '    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)\n    parser.add_argument("--include-future", action="store_true", help="FEV only: retain non-safe answer history for debug/test analysis; never use this output in training and evaluation")\n',
    '    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)\n    parser.add_argument(\n        "--reuse-search-root", default=None,\n        help="Sibling protocol root to scan for compatible reference bin shards",\n    )\n    parser.add_argument("--include-future", action="store_true", help="FEV only: retain non-safe answer history for debug/test analysis; never use this output in training and evaluation")\n',
)
replace(
    "assets/dataset_processing/load/ref/fev.py",
    '    produced = load_reference(Path(args.spec), Path(args.fetch_dir) if args.fetch_dir else None,\n                              Path(args.bundle_dir), args.asset_id, args.shard_bytes, args.include_future)\n',
    '    produced = load_reference(\n        Path(args.spec), Path(args.fetch_dir) if args.fetch_dir else None,\n        Path(args.bundle_dir), args.asset_id, args.shard_bytes, args.include_future,\n        Path(args.reuse_search_root) if args.reuse_search_root else None,\n    )\n',
)

replace(
    "scripts/assets.py",
    '            "--input-dir", str(source_input),\n            "--bundle-dir", str(source_bundle),\n',
    '            "--input-dir", str(source_input),\n            "--bundle-dir", str(source_bundle),\n            "--reuse-search-root", str(source_bundle.parent),\n',
)
replace(
    "scripts/assets.py",
    '            "--bundle-dir", str(reference_bundle_root),\n        ],\n',
    '            "--bundle-dir", str(reference_bundle_root),\n            "--reuse-search-root", str(reference_bundle_root.parent),\n        ],\n',
)
print("patched remaining files")
