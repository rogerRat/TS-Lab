from pathlib import Path

path = Path('/root/TS3/docs/MAIN-v2.md')
text = path.read_text()
text = text.replace('新 run（v2-global）全程零 CUDA OOM。', '截至本次检查，尚未在新 run（v2-global）日志中检索到 CUDA OutOfMemoryError；不能由此排除未记录的显存事故。')
text = text.replace('**宿主容器 RAM 压力 → 平台 SIGKILL（新 run 主因）**', '**宿主容器 RAM 压力与 SIGKILL（新 run）**')
text = text.replace('cgroup oom_kill=0（平台级杀不计数）', 'cgroup oom_kill=0（未获得内核或平台杀进程记录，不能确认杀进程来源）')
text = text.replace('杀空三 worker 后 memory.current 从 98.8GiB 骤降至 4GiB，坐实。', '停止三 worker 后 memory.current 从 98.8GiB 降至 4GiB，支持并发资产阶段内存压力的判断；-9 与 RAM 压力相关，但具体 SIGKILL 发出方仍未证实。')
note = '''

## 内存热点修复与验证（2026-09-17，Codex 与主控 Claude 协调）

旧 CUDA OOM 的堆栈定位在 Chronos Darwin / rage_traj_gls / seed2 的最终评估：
`train_eval.evaluate -> quantile_forecasts -> T5 attention softmax(scores.float())`。
当次申请 1.51GiB，而 GPU 只剩 921.94MiB；Chronos 报告 PyTorch 已分配 17.56GiB、
保留但未使用 15.42GiB。该记录发生于旧六臂 run，不能归因为新 Global 臂。
阶段缓存释放只能减少可回收保留量，不能保证解决模型实际活跃张量的并发峰值。

代码热点与修复：

- `full_pool.py` 原块匹配将全部方向 `np.stack`，随后 `resampling.py` 再整体转 FP64。
  以约 360774 个 4096 维源块计算，FP32 stack 约 5.5GiB，额外 FP64 副本约 11GiB；
  尚未计入已有 CRB 块、参考方向与缓存。现在传递原生方向列表，4096 行一批转 FP64，
  每批方向工作区约 128MiB。原 RAGE 仍按参考行批次计算同样的 softmax，Global 仍按原定义
  聚合参考方向并进行全局 softmax；GPU 源方向工作区仍受原来的剩余显存 30% 条件约束。
- `full_pool_worker.py` 在模型构建前获取每个 run/student 的独占 `worker.lock`，
  防止 supervisor/watchdog 重复启动同一学生时重复加载模型。该锁不限制不同学生或不同 run。
- 选集转训练、训练 cell 边界执行 GC 和 CUDA `empty_cache`；`train_eval.evaluate`
  入口释放闲置 CUDA 保留缓存。保持训练 batch、评估 batch、预测样本数、全部窗口、
  梯度预算、概率公式与 seed0 配置不变。
- 主控已停止重复 watchdog/supervisor 拉起与旧 worker，改为 TimeMoE 先行串行 Azure
  资产/选集护航。重新启动的 worker 才会加载新的进程锁与分批转换代码。

验证：选择/调度相关测试 32 passed、1 skipped（CUDA 隐藏）；另一次实际开放 CUDA 的
匹配、评估、训练与恢复相关测试 44 passed，其中包含 CPU/GPU 分批概率等价检验。
新增测试禁止整套方向矩阵转换，并验证原 RAGE/Global 相对显式 FP64 矩阵的概率
在 rtol=1e-12 下等价；独立进程持锁时，重复启动不能进入 worker 主体。
`git diff --check` 通过。两组测试有重叠，不应相加为唯一测试数量。

限制：目前正式 worker 仍处于 Azure 资产准备阶段，尚不能宣称完整实验已通过，
也不能由一次 GPU 快照证明以后不会 OOM。主控继续记录串行运行的内存、选集完成与退出状态；
恢复模型并行前应结合实际训练/评估峰值决定并发数。
'''
marker = '<!-- v2-notes:end -->'
if '## 内存热点修复与验证（2026-09-17，Codex 与主控 Claude 协调）' not in text:
    if marker not in text:
        raise RuntimeError('Missing persistent notes marker')
    text = text.replace(marker, note + '\n' + marker)
path.write_text(text)
print('Recorded diagnosis, fixes, checks and uncertainty in MAIN-v2.md')
