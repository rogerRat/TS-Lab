# v2 执行与缓存复用

统一仓库 `/root/TS3`，分支 `rags`；probe 开发历史已合并，旧探测进度保存在 `docs/archive/ProbeProgress-v1.md`。
最终 v1 主表原样保存为 `docs/MAIN-v1.md`。所有新结果仅写 `docs/MAIN-v2.md`，旧队列暂停且 v1 主实验终止。

配置 `configs/rags/main_v2.json`：继承 v1 最新全部裁定，包括九个实际任务（Quito 已移除）、六个训练臂（三种子）和 original，三模型共 513 cells；统一无门数值 softmax T=0.25；各模型训练配方不变；LTSV 继续移除。
CRB split_policy=closest_zero_v2；详见 `docs/crb.md`。新 resampling_revision 为 full-pool-resampling-v2-closest-zero。

新 run `/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-20260916`。其 `grad_cache` 是指向 v1 run 原始缓存的软链接，保持同一模型单写者锁与原子追加；未复制任何旧选集、cells 或训练 checkpoint。
模型/数据池/窗口/投影/精度字段沿用；新分裂位置命中旧缓存直接读，只补缺失窗口。
`gradient_reuse.json` 记录复用来源；`warm_cache_verification.json` 保存真实回放证据（每模型32条源轨迹、5032次 phase 1/2 锚点读取、0重算；已有梯度1614670/1426160/1140576条）。
如模型字段变化导致新空缓存，准备阶段报错停止，禁止隐式全量重算。
CRB 阈值审计改为 run-local `crb_audits/<模型>/crb-thresholds-<任务>.json`，不会覆盖 v1 阈值记录。原始缓存追加不会改变旧梯度值，累计 timing/index 继续更新。

启动入口（仓库根目录，已有三模型环境与权重）：

```bash
PYTHONPATH=src /root/miniconda3/envs/ts3/bin/python -u -m scripts.exp.main_v2_supervisor \
  --config configs/rags/main_v2.json \
  --run /root/autodl-tmp/TS3/outputs/main-shared-nine-v2-20260916
```

监督进程每30秒更新 v2 主表和 run/results.json，失败最多重试三次（仅续跑同一 v2 未完成 cells）；各模型分别写 run/<模型>.log，监督状态为 supervisor_state.json。
三模型并行；梯度 worker 默认1，TimeMoE为2，批量沿用4/4/8，避免争抢97GiB显存；逐窗 loss/梯度保持独立。
旧 main 配置仍存档，不能以新代码续算 v1。重新启动必须使用本 v2 入口与 run。

验证：140项算法、缓存、训练配方、探测接口、全池分布与编排测试通过。合并时修复 LP-Mirror mapping 接口，使 v1 已裁定的 value softmax 实际可用；其 score/proxy 计算不变。旧池缓存命名中的 hash 改为显式 seed/N/points 字段，符合 AGENTS 约定。
