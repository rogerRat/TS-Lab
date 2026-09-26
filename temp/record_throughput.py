from pathlib import Path

path = Path('/root/TS3/docs/MAIN-v2.md')
text = path.read_text()
heading = '## 吞吐优先与阶段错开并行（2026-09-17）'
note = '''

## 吞吐优先与阶段错开并行（2026-09-17）

用户明确要求减少 ETA、尽可能利用全部计算资源。串行护航仅用于故障定位，
不应作为整个主实验的长期执行策略。现有方向分批、进程单例锁和阶段缓存释放
减少内存峰值，但不保证 GPU 满载；不同 student 的单例锁彼此独立，允许并行。

14:08:42–53 实测：只有一个 TimeMoE Python worker，正在 Azure / rags_gateless
selection；GPU 利用率 34–62%，显存固定约 15335MiB，功率约 105–118W。
容器 RAM 约 49GiB，CPU 配额为 22 核（cpu.max=2200000/100000），
不能把 nproc 返回的 208 核视为容器可用算力。
当前原 RAGS 匹配每 32 个参考块做 GPU FP64 矩阵乘，随后把评分传回 CPU，
逐参考块做 NumPy softmax、审计和源边缘概率累加。这种阶段交替会产生 GPU 空隙；
低利用率不能简单归因为模型参数少，也不能单靠增加常驻显存解决。
本次未修改该评分算法，后续若优化应保留 FP64 概率与审计等价并测量实际吞吐。

协调决策：主控 Claude 负责调度与监控，生产进程取消 CUDA_LAUNCH_BLOCKING=1；
当前进程须重启才能取消该环境变量，避免在未落盘的耗时选集中随意丢弃计算。
优先补充第二个 worker，以其他较小参考任务填补 GPU 空隙，错开 Azure 等高 RAM
资产/选集阶段；根据 cgroup RAM 与训练/评估实际 GPU 峰值再增加并发。
不盲目恢复三模型同时进入大资产阶段或峰值训练：历史 42.13+33.66+18.22GiB
已逼近 GPU 总容量。约 95GiB 容器 RAM 可作首轮并发观察上界，仍须保留余量；
当前闲置显存不能替代下一阶段峰值估计。吞吐优化不改变 seed0、训练有效 batch、
全部合法窗口、抽样概率或评估口径。

调度应以唯一 Python worker 计数，不能把 bash launcher wrapper 误报为 worker。
持续记录阶段、GPU 利用率/功率、cgroup RAM、选集/cell 完成时间；是否达到加速
以完成任务吞吐和 ETA 为准，不以显存占用率或单帧 GPU 利用率为准。
'''
if heading not in text:
    marker = '<!-- v2-notes:end -->'
    if marker not in text:
        raise RuntimeError('Missing persistent notes marker')
    path.write_text(text.replace(marker, note + '\n' + marker))
print('Recorded throughput measurements and scheduling decision')
