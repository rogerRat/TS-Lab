# main-crb-v0 expand-resample 与 main V7 方法对照

- 检查时间：2026-11-16 22:34 CST
- 当前项目分支：rage；对照基线读取 main 分支提交内容。
- 当前实验：main-crb-v0；用户所称 expand-resample 对应现有文档批次 q16-src256-eight-task-resample。检查过的实验文档、相关日志和入口文件没有发现字面批次 ID expand-resample。
- 本次为只读对照；未修改代码、参数、配置或运行队列。当前工作树原有改动保留。
- 运行状态只引用截至 2026-11-16 21:11:38 CST 的既有日志，不代表 22:34 的实时进程检查。

## main 分支 V7 的可复现身份

- 初始配置 configs/rags/main_v7.json 使用 3 个模型、9 个任务、20 个方法臂和 seed 0/1/2；其后正式配置与结果按修订后的 configs/rags/main_v7_curve4.json、docs/results/v7.md 解释：主结果为 7 个任务、3 个模型、正式 seed 0/1；TimerXL seed2 仅作历史存档。三项扩展任务单列为探索结果。
- 数据按完整原生轨迹构造可行池 65,136 equivalents，每个 seed 精确 16,284 equivalents；使用稳定轨迹 ID、确定性选择和现有 subset-sum witness，不裁窗、不强制 seed 间互斥。三模型、九任务和方法臂共享对应 seed 的 pool manifest。
- V7 窗口是 Chronos/TimesFM 512+64、TimerXL 480+96，stride=1。梯度来自单个预训练检查点的原生损失，Count-Sketch 4096 维、projection seed 42，资源缓存为 BF16。CRB 使用 coverage/risk/split 三项 0.5% 密度配置和 min_kappa 分裂；V7 主配置记录完整预算上限、端点和并列处理规则。
- 训练是模型原生的全参数微调，不是当前实验的 LoRA 配方。effective batch=32、micro-batch=32；TimerXL lr=1e-4、weight decay=0.01、cosine，最多 4 epoch；Chronos lr=1e-3、weight decay=0、linear，正式主表取 epoch 3；TimesFM lr=3e-5、weight decay=0.01、cosine、4 epoch。调度长度按 6 epoch 固定。
- V7 含 20 个臂，覆盖 block/trajectory 粒度、linearSum/zeroClipSum/maxSum/expSum、窗口/块内均匀化，以及 rand、orig、LESS、TSDS、LP-Mirror、grad-dot、grad-cos、GraNd-static。
- V7 的秩概率规则是冻结的 V5 rank-volume：按等对象数排名，精确保留同分组，头部半数在乘 source legal-window-count 前获得约 80% 秩质量；source 长度只在排名后乘一次，不再 softmax。对原生单位抽样后，在该单位拥有的合法起点中均匀抽取。正式七任务主结果中，rags-block-linearSum、rags-block-zeroClipSum 和 rags-block-expSum 对 rand 的跨模型平均改善分别为 +0.02%、+0.04% 和约 0%；未显示大的稳定总平均收益。

## 当前 main-crb-v0 resample 批次

- src-256 为 256 条轨迹、6,382,020 个合法窗口；TimerXL 窗口 480+96、stride=1。CRB/UNI q16 source 锚点和各任务 q8 ref 资源从 main-crb 既有缓存复用；ref 共八个评估任务。
- 当前 CRB 锚点方案是 cover + cancellation split：q=16，默认 coverage 间距 1.5q，按低 kappa 优先拆分一半可拆区间；本批 source CRB/UNI q16 资源记录各 399,044 锚点。不同于 V7 的 0.5% coverage/risk/split 三阶段配额。
- 梯度探针是 TimerXL 84M 的 LoRA q_proj/v_proj（r=8、alpha=16、dropout=0，2,000 步，seed0）。Count-Sketch 4096 维、seed0；CRB/UNI 筛选与插值使用单位方向，幅值只作审计。和 V7 的预训练检查点、seed42 原始投影向量不是同一梯度身份，不能直接混用其梯度/概率缓存。
- 预测模型只用 TimerXL 84M LoRA；4,000 optimizer steps、effective batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、seed0。CRB/UNI 和轨迹方法按任务各训一次；rand、GraNd-static 共享训练后评估八任务。
- 十个方法组：ori、crb、uni、reft、rand、TSDS、LP-Mirror、GraNd-static、CRB-traj、UNI-traj。窗口方法在全池逐窗打分；轨迹方法按轨迹等权聚合 reference cosine，再按 source legal-window count 部署轨迹质量。均不做 top-k 硬截断。CRB/UNI 窗口和轨迹臂统一 ref linearSum。
- 当前概率构造调用与 V7 相同的 src/rage/experiments/ranking.py::rank_mass：V5 冻结秩区间积分、精确保留同分；当前窗口臂每个窗口作一个等权对象，轨迹臂以轨迹作秩对象再乘合法窗口数。共享的是映射规则，rank 对象与得分构造并不相同。
- 指标快照截至 20:47：已有八个方法组在八任务的 64 项评估，42 次模型训练完成。CRB/UNI/CRB-traj/UNI-traj、ori/reft/rand/GraNd-static 的 MASE 已记录；TSDS、LP-Mirror 尚无 MASE。21:11 日志记录这两类原生概率资产均已生成、训练已启动。本次审查没有再次检查实时进程。

## 主要差异与待决策项

1. V7 的强项是跨模型、跨 seed、20 臂的正式对照和完整训练/评估矩阵；当前批次的强项是把新 cover+cancellation CRB、窗口插值和轨迹抽样在完整 src-256 上做较聚焦的机制检验。
2. 两者可直接复用的核心是 V5 rank-volume 映射与“抽单位、单位内均匀抽窗”的训练语义；需要重新定义或计算的是候选池、source/reference 梯度身份、评分粒度和模型训练 recipe。
3. 若后续建立桥接比较，应先选定要回答的是“新锚点规则的效果”还是“全窗口扩展重采样相对旧 block/trajectory 的效果”。建议一次固定模型、pool、seed、训练更新数、评估配置和 rand 定义，只比较少量指定算法臂；否则方法效应会与数据池、模型、LoRA/全参数、训练步数一起变化。
4. 尚未冻结的项目决策包括：保留 V7 的三模型/双 seed 框架还是沿用当前 TimerXL 单模型；V7 完整轨迹 seed-pool 还是 src-256 全窗口池；比较 block、window、trajectory 中哪些粒度；保留哪些聚合算子和 baselines；是否补齐当前 TSDS/LP-Mirror 后再选方法；是否为新梯度身份重新计算缓存。

## 依据

- 当前实验设置与方法：docs/exp/main-crb-v0/exp.md，resample 段落。
- 当前实测矩阵与进度：docs/exp/main-crb-v0/results.md，q16-src256-eight-task-resample 段落。
- 当前概率构造：scripts/exp/main-crb-v0/prepare_resample_methods.py、src/rage/experiments/ranking.py、src/rage/experiments/resampling.py。
- main V7 配置：configs/rags/main_v7.json、configs/rags/main_v7_curve4.json。
- main V7 协议与结果：docs/experiments/v7_engineering_20260923.md、docs/results/v7.md。
