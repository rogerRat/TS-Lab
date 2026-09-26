from pathlib import Path
import yaml

profile_path = Path("/root/TS3/configs/eval/fev/profile-10b-local.yaml")
default_path = Path("/root/TS3/configs/eval/fev/default.yaml")
configs = [item["dataset_config"] for item in yaml.safe_load(default_path.read_text())["tasks"]]
lines = profile_path.read_text().splitlines()
path_lines = [i for i, line in enumerate(lines) if line.startswith("- dataset_path:")]
if len(path_lines) != len(configs):
    raise RuntimeError(f"expected {len(configs)} dataset_path lines, got {len(path_lines)}")
for line_number, config in reversed(list(zip(path_lines, configs))):
    if line_number + 1 >= len(lines) or not lines[line_number + 1].startswith("  dataset_config:"):
        lines.insert(line_number + 1, f"  dataset_config: {config}")
profile_path.write_text("\n".join(lines) + "\n")
print(f"updated {len(configs)} FEV task configs in {profile_path}")
