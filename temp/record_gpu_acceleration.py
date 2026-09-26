from pathlib import Path

path = Path('/root/TS3/docs/MAIN-v2.md')
content = path.read_text()
heading = '## RAGS 匹配后处理留在 GPU（2026-09-17）'
note = '''

## RAGS 匹配后处理留在 GPU（2026-09-17）

用户再次报告 GPU 约 60% 震荡并要求实际加速。现状已从单模型变为 TimeMoE
训练 + Chronos Azure 选集，但仍约 55% GPU 利用率；cgroup 约 96–100GiB，
其中 anon 约 64GiB、file 约 31GiB，不能简单再启动第三个高内存 worker。

实现：`resampling.rags_marginal` GPU 路径在 FP64 下完成按参考行的 softmax、
非正质量/审计归约及源边缘概率累加，只把每行六个审计统计量和最终 beta
传回 CPU，避免每批完整 reference×source 分数的主机复制和 NumPy 后处理。
保持每个参考独立预算、gating、underflow、InfeasibleReference 与审计字段语义；
CPU 路径保留。默认参考批次从 32 增至 128，源方向仍 4096 行上传一批；
分配条件保守计入八份评分工作区，并保持剩余显存 30% 上限。
GPU OOM 时从头重算 CPU 边缘分布，不发布部分结果。

梯度准备的 CUDA allocator 可能保留闲置缓存；匹配分配前 empty_cache，
避免将可回收保留量当作长期显存不足。新增 RAGS_MATCHING_BACKEND 日志，
记录 CPU/CUDA、参考/源块数、维度与批次，便于发现整轮 CPU fallback。
旧 Chronos 进程选集时 GPU 常驻仅约 2GiB、CPU 约 368%，与 CPU fallback
相符，但旧版本没有后端日志，具体路径仍应以新进程日志核实。

在线合成基准：256 参考×32768 源×1024 维，包含方向上传、两个正式 worker
并行背景，每种方案一次预热后两次测量。旧 GPU 矩阵乘+CPU 概率后处理中位数
0.254451s；新 GPU 全后处理、32 行批次 0.155027s（1.64×）；
128 行批次 0.143496s（1.77×）。结果在 temp/gpu-marginal-benchmark.json，
脚本 temp/benchmark_gpu_marginal.py。旧基准省略审计构造，不能视为夸大了旧审计开销。
该测量只验证合成几何的匹配阶段加速，不能直接推算整个正式实验的 ETA 或满载率。

验证：新增 tests/test_gpu_marginal.py 对 gating 两种模式、1/7/32/128 批次
比较 CPU/GPU 概率和全部审计，并覆盖不可匹配及低温概率下溢；概率容差
rtol=1e-12、atol=1e-15。主控负责非 blocking 进程轮换，已重启 TimeMoE；
Chronos 未落盘选集是否重启及重启收益由主控协调，须核实新日志和实际完成时间。
全部窗口、梯度定义、概率公式、seed0、训练有效 batch 和评估口径保持。
'''
marker = '<!-- v2-notes:end -->'
if marker not in content:
    raise RuntimeError('Missing persistent notes marker')
if heading not in content:
    path.write_text(content.replace(marker, note + '\n' + marker))
content = path.read_text()
live = '''

本次最终相关测试合计 36 passed（同一次选择/GPU/worker 测试组），git diff --check
通过。TimeMoE 重启为非 blocking 后，远程 14:59:54–15:00:13 共 20 个一秒采样：
GPU 平均利用率 87.3%，多数 81–97%，一次 39%；功率约 332–381W，显存约
21954MiB。此为短时间正式运行证据，不能保证后续所有阶段稳定满载；Chronos
当时仍为旧版本进程，尚未加载本次 GPU 后处理代码，后端须在轮换后核实。
'''
if '远程 14:59:54–15:00:13' not in content:
    path.write_text(content.replace(marker, live + '\n' + marker))
print('Recorded GPU acceleration implementation and evidence limits')
