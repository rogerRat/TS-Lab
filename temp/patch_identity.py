from pathlib import Path
p = Path("assets/dataset_processing/load_reuse.py")
s = p.read_text()
old = '''        if (
            path.name in _IDENTITY_METADATA_NAMES
            or path.name == ".complete"
            or path.suffix == ".json"
        ):
            digest.update(_file_digest(path).encode("ascii"))
'''
new = '''        if path == manifest and profile_payload is not None:
            digest.update(_canonical_json(profile_payload))
        elif (
            path.name in _IDENTITY_METADATA_NAMES
            or path.name == ".complete"
            or path.suffix == ".json"
        ):
            digest.update(_file_digest(path).encode("ascii"))
'''
if old not in s:
    raise SystemExit("identity hash anchor missing")
p.write_text(s.replace(old, new, 1))
print("patched identity normalization")
