# Codex 优化任务反馈：CT 运行 CPU 瓶颈（Kimi 运行侧实测）

日期：2026-09-22。来源：main-rank-volume-20260921（chronos-base + timesfm-2.5，tmux main-chronos-timesfm）运行 7 小时实测。

## 结论（一句话）
当前瓶颈是 CPU 单线程路径，不是 GPU：GPU 冲刺后长时间等待 CPU 铺路（CRB/索引/选择），208 核机器 load 仅 ~10。

## 实测证据
1. **Chronos 源池 CRB 重放**：单线程 ~84 分钟（751/1082 trajectories 用时 12087s 仍在中途；最终约 3.4h）。同一 pool TimerXL warm 命中仅 0-14s。Chronos 的 source-crb fields 未命中 warm（identity 校验不通过？）而回放到慢路径——请核查 fields 不匹配原因（v6 timerxl 命中、chronos 不命中）。
2. **梯度计算尾部塌缩**：缺失窗口稀疏后批大小落到 3-6（timing batch_sizes 众数 5），chronos 有效吞吐 ~10 w/s、timesfm ~20-25 w/s。小批量下分片文件爆炸（单窗口分片数千个）。已用 asset_execution_override.json gradient_batch=8 部分缓解，但根本原因（为何批被打到个位数）未查。
3. **选择阶段 CPU-bound**：selection 阶段 GPU 基本闲置，每臂分钟级（~4-8 臂/30min）。
4. **训练单元间开销**：单元训练 ~186s（timesfm）但两单元间 CPU 间隙（加载/清理/权重重置）显著，双模型合计吞吐 ~3-6 cells/30min。
5. 服务器无 CJK 字体、无 py-spy，诊断工具受限。

## 期望方向（不改动科学语义的前提下）
- Chronos source-crb warm 未命中的根因（可能 identity 字段含 chronos 特有项，或 fields-N 目录未正确发布/匹配）
- 梯度批塌缩的 OOM 重试阶梯是否可恢复（现疑似一旦降下来就不回升）
- 稀疏缺失窗口的合并写入（减少碎分片）
- 选择阶段的并行/批量化可行性评估

## 运行侧已做（无需重复）
- 共享梯度池 TS3_GRADIENT_CACHE_ROOT（cache.py 6 行改动，备份 /tmp/cache.py.bak-20260922）
- vjp16（TimerXL，+61%）、CRB 逐轨迹持久化复用（37x）已生效
- 全部 pytest：6 failed 为预存问题（2 外网 + 4 selection/uni），与运行侧改动无关

## 环境
- /root/TS3（新代码已加载，worker 08:33 重启后生效）
- 共享池 /root/autodl-tmp/TS3/cache/grad-pool
- 联系人：Kimi（定时巡检，30min）；正式运行请勿停止/重算缓存。
