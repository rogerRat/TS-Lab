# Dependency Management

Last modified: 2026-11-07

## Categories

- `requ/project/` contains dependencies owned by this repository.
- `requ/third_parties/model/` contains model source pins, integration pins, and model runtime dependencies.
- `requ/third_parties/bench/` contains benchmark source pins and benchmark runtime dependencies.
- Top-level `requ/*-requirements.txt` files are stable Conda profile composition entry points.

## Source locks

- The main profile installs the Chronos model from `requ/third_parties/model/chronos.lock` and FEV from `requ/third_parties/bench/fev.lock`.
- The TIME profile installs TIME from `requ/third_parties/bench/time.lock` and the shared Chronos model from `requ/third_parties/model/chronos.lock`.
- `requ/third_parties/model/integrations.lock` records model repositories, checkpoints, and runtime version anchors.
- `requ/third_parties/bench/baselines.lock` records upstream source anchors for the selection baselines.

## Profile composition

- `requirements.txt` combines repository main dependencies, Chronos runtime dependencies, and FEV requirements.
- `timesfm-requirements.txt`, `timemoe-requirements.txt`, `sundial-requirements.txt`, and `timerxl-requirements.txt` combine repository model-profile dependencies with their model runtime.
- `time-requirements.txt` includes the complete pinned TIME environment under `requ/third_parties/bench/`; it remains isolated because it uses an older data stack.

Keep model environments separate where their Transformers versions differ. Split source locks by responsibility; do not combine model, benchmark, and baseline provenance into one lock.
