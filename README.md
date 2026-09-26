# TS3 / RAGS

时间序列数据选择、后训练与评估项目。当前实现以RAGS、独立梯度资源、Chronos T5 Base、
TimesFM 2.5和TimeMoE为主。

从[文档入口](docs/README.md)开始；开发前读[项目约定](AGENTS.md)，安装见
[环境安装](requ/INSTALL.md)，可用命令见[工具目录](scripts/STD.md)。

- [数据准备](docs/DataPreparation.md)
- [梯度资源与BF16](docs/gradient_assets.md)
- [全池重采样](docs/dev/FullPool0914.md)
- [已批准的8个探测候选](docs/dev/ProbeScopeV2.md)
- [参数与任务探测](docs/dev/ParameterTaskProbing.md)
- [探测执行记录](docs/experiments/probe_execution_20260914.md)
- [历史实验与设计](docs/archive/README.md)

核心仓库为 `/root/TS3`，大容量数据和结果放数据盘。本轮探测执行快照位于
`/root/TS3`；运行版本、环境和恢复路径见探测执行记录。
`docs/archive/`中的旧OT/KDE、Feature/Catalog/AnaSeg与训练指令只用于历史追溯。

当前实验协议为 **v2**，统一在 `/root/TS3` 的 `rags` 分支开发与运行。
主表见 [MAIN-v2](docs/MAIN-v2.md)，[MAIN-v1](docs/MAIN-v1.md) 为旧协议存档，不继续追加新结果。
