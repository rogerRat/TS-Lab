from pathlib import Path

p = Path("assets/dataset_processing/load/ref/fev.py")
s = p.read_text()
needle = '    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)\n'
if '--reuse-search-root' not in s:
    if needle not in s:
        raise SystemExit("ref shard parser anchor missing")
    s = s.replace(needle, needle + '    parser.add_argument("--reuse-search-root", default=None, help="Sibling protocol root to scan for compatible reference bin shards")\n', 1)
start = s.index("    produced = load_reference(", s.index("def main"))
end = s.index("    for bundle_path in produced:", start)
call = '''    produced = load_reference(
        Path(args.spec), Path(args.fetch_dir) if args.fetch_dir else None,
        Path(args.bundle_dir), args.asset_id, args.shard_bytes, args.include_future,
        Path(args.reuse_search_root) if args.reuse_search_root else None,
    )
'''
s = s[:start] + call + s[end:]
p.write_text(s)

p = Path("scripts/assets.py")
s = p.read_text()
anchor = '            "--bundle-dir", str(source_bundle),\n'
if '"--reuse-search-root", str(source_bundle.parent)' not in s:
    if anchor not in s:
        raise SystemExit("assets source anchor missing")
    s = s.replace(anchor, anchor + '            "--reuse-search-root", str(source_bundle.parent),\n', 1)
anchor = '            "--bundle-dir", str(reference_bundle_root),\n'
if '"--reuse-search-root", str(reference_bundle_root.parent)' not in s:
    if anchor not in s:
        raise SystemExit("assets reference anchor missing")
    s = s.replace(anchor, anchor + '            "--reuse-search-root", str(reference_bundle_root.parent),\n', 1)
p.write_text(s)
print("patched")
