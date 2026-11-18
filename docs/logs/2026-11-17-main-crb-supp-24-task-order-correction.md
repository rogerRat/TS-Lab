# main-crb-supp 扩展到 24 任务并调整模型顺序

- 操作时间：2026-11-17 02:20–02:23 CST。
- 用户范围：每个模型均覆盖相同 24 个任务；先完成 TimerXL 全部任务，再执行 Chronos-2。
- TimerXL：在原 12 任务基础上追加 12 项：Water_Quality_Darwin/15T、SG_PM25/H、OpenElectricity_NEM/5T、EWELD_Load/15T、Port_Activity/D、current_velocity/10T、Coastal_T_S/5T、epf_electricity_price/H、current_velocity/15T、Coastal_T_S/15T、azure2019_D/5T、hospital_admissions_1D。队列脚本扩展为 24 项，并新增 `--mode extend`：读取现有每任务 `.complete`，跳过原先 12 个完成任务，执行其余任务；全 24 完成后写 `/root/autodl-tmp/rage/outputs/eval/main-crb-supp-24/run/.complete`。
- TimerXL 调度：`scripts/exp/main-crb-supp/run_24.sh` 已在 tmux `main-crb-supp` 中执行；24 任务 prepare 于 02:22:50 完成，TimerXL 验证于 02:22:52 开始。验证通过后依次运行 `--mode extend`。
- Chronos-2：原队列于 02:20:06 收到 SIGINT，使用队列内 `signal_children` 处理器向训练子进程发 SIGINT 并等待退出。停止时处于首任务训练启动阶段，无已完成评估点；已有安全 reference、梯度缓存、选择与训练部分产物保留。其 tmux 依赖 wrapper 已重启，改为等待上述 TimerXL-24 标记后执行 prepare、validate、run。
- 输出索引：TimerXL 的 `docs/exp/main-crb-supp/results.md` 及全局 EXP/RESULTS 已按 24 任务、384 个评估点重新生成；原先 12 任务的 192 个 MASE 结果保留，新增 192 个评估点等待运行。
- 未解决项：TimerXL 新增任务运行时长受 azure2019_D/5T 等大 reference 集影响；Chronos 具体 ETA 需以 TimerXL-24 完成后实际队列进度重新估算。
