# azure2019_D/5T TimerXL 队列恢复

- 检查时间：2026-11-17 12:46 CST。
- 批次：`main-crb-supp` TimerXL 24 任务续跑。
- 进度：22/24 任务、352/384 评估点完成。失败任务 `time-azure2019-d-5t` 已完成 reference 生成与五个方法的选择；训练阶段尚无该任务的评估点。最后一个 `fev-hospital-admissions-1d` 未开始。
- 异常证据：`e5-target-only` 的训练日志记录 `stage=end name=train code=-9`；队列日志显示训练子进程 wrapper 返回 code=247。其余并行训练臂随后被队列清理。系统可读的内核日志未发现 OOM 记录，因此根因暂不能确认。
- 修复：在 `scripts/exp/main-crb-supp/run_queue.py` 中仅对 `time-azure2019-d-5t` 改为逐臂训练，降低大目标任务的并行资源竞争。原有其他任务和算法参数不变。恢复命令会经 `run_24.sh` 重做快速 prepare/validate，然后 `--mode extend` 根据每任务 `.complete` 跳过前 22 个已完成任务，恢复 Azure 任务后继续医院任务。
- Chronos-2：等待 TimerXL-24 标记；不会与 TimerXL 并行占用 GPU。
- 下一步：重启 TimerXL 扩展队列，观察 Azure 逐臂训练和最后一项 FEV 任务完成情况。

## 恢复重启记录

- 2026-11-17 12:49 CST：TimerXL 续跑重新启动，prepare 检查通过，准备再次验证后进入 `--mode extend`；预计按每任务 `.complete` 跳过已完成的 22 项。
- 查明 Chronos 等待脚本原先使用 tmux 前缀匹配，`main-crb-supp` 会误匹配 `main-crb-supp-chronos2-small`。已改为 tmux 精确目标 `=main-crb-supp`，清理旧等待会话并按新依赖标记重新启动等待器，避免 TimerXL 会话结束时误判。
- Azure 训练子进程被 SIGKILL 的根因仍未确认；现对该任务五臂串行执行作为缓解措施，不改模型、数据或训练参数。

## 并行策略调整

- 用户要求尽量提高 GPU 利用率并行执行。已撤销仅针对 `azure2019_D/5T` 的串行训练分支，恢复五个训练臂并发（E0 继续复用已有 checkpoint，仅执行评估）。不修改训练超参数。SIGKILL 根因尚无证据定论；若再次出现，将保留所有已完成产物并按失败子进程日志修复。
- 2026-11-17 12:57 CST：按用户要求将 Azure 任务恢复为五臂并行后，发现旧进程仍载入串行版本；在该次任务产生新评估前向续跑进程发 SIGINT，优雅退出。重新启动 `run_24.sh`，当前重新验证中；通过后新进程将载入并行版本。已完成任务数仍为 22/24，352/384 评估点。
- 2026-11-17 13:04 CST：重试时发现训练脚本因已有非空 checkpoint 输出目录立即退出（`train.py` 明确拒绝覆盖非空目录）。原未完成 Azure 训练目录已移动保存在 `outputs/train/main-crb-supp/run/time-azure2019-d-5t/incomplete-attempt-20261117-081648/`，未删除。随后直接以 `--mode extend` 重启，不重复 prepare/validate；五个训练臂由 `run_train_parallel` 并发。Chronos 精确 tmux 会话同时进入等待。

## 内存回收等待修复

- cgroup 证据：`memory.current=65.15 GiB`、`memory.high=60 GiB`、`memory.max=62 GiB`，`memory.events high` 持续增长；e5 进程 RSS 约 53.8 GiB，其余训练臂的等待通道为 `mem_cgroup_handle_over_high`。内核未记录 OOM kill。
- 已向主队列发送 SIGINT，再对卡在内存分配等待的 e5 未完成进程发 TERM/KILL，训练进程退出；cgroup 用量回落至约 2.0 GiB，GPU 显存归零。未完成输出已保留/归档，无评估点丢失。
- 调度修复：azure2019_D/5T 改为 e1–e4 并行训练，等待其完成后 e5-target-only 单独运行。用户要求尽量并行，其他任务继续五臂并行；超大 e5 不再与四个共享臂争用内存。
- 2026-11-17 13:20 CST GPU 监测：恢复四臂并行后，每 10 秒读取一次，连续三次 GPU 利用率为 99%，显存约 3.9 GiB；对应训练进度 e1=400/4000、e2=400/4000、e3=300/4000、e4=400/4000。达到近满载判断条件后结束本次监测。后续 e5 单臂训练将继续观察内存与 GPU 利用率。
- 2026-11-17 13:55 CST：最后一项 hospital_admissions/1D 续跑时，异常日志定位到非 Azure 通用分支调用 `run_train_parallel` 漏传 logger；修正后五臂训练正常启动。此次失败未启动训练或生成评估点。
- 2026-11-17 13:56 CST GPU 监测：hospital_admissions/1D 五臂并行训练时，13:56:29、13:56:39、13:56:49 连续三次 GPU 利用率 99%，显存约 4.8 GiB；达到近满载条件后结束本轮监测。
