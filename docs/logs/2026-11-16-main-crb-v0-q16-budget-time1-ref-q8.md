# main-crb-v0 q16 source / q8 TIME1 ref 预算核算

- 检查时间：2026-11-16 12:25 CST。
- 实验代号：main-crb-v0；批次：q16-grad-budget-time1-ref-q8。
- 任务状态：预算核算完成；未启动梯度计算，未生成新缓存或 .complete。
- 项目 checkout：rage，工作区已有未提交修改；未切分支、未覆盖其他改动。
- source：复核 main-crb-v0 复用的 main-ads-v0 ordinal 0 全窗口梯度缓存；230,161个合法窗口，q=16时 K=14,386/方法。CRB(1:0:1) cover/error/split=7,194/0/7,192；Uni=14,386。
- ref：复核当前唯一启用任务 TIME1（CPHL/15T）的安全 ref manifest；10段、3,250个合法窗口且不包含答案值。q=8时 K=410/方法；CRB cover/error/split=220/0/190；Uni=410。
- 模型身份沿用 main-ads-v0 的 TimerXL LoRA、480+96窗口、Count-Sketch 4,096维/seed 0。已有 v1-crb-redo ref 梯度缓存未复用，因其检查点与投影身份不同。
- 资源：每方法14,796条梯度记录；4,096维、2 bytes/元素估算约115.55 MiB/方法，两方法分别保存约231.10 MiB。source全窗口梯度已存在，不需重算；TIME1 ref需最多410条/方法，共用缓存可去重。
- 工具检查：scripts/tools/grad_cache.py 要求预制完整窗口 NPZ；scripts/tools/select.py 的CRB配置不对应固定 q 与 1:0:1 预算。保留现有工具边界，后续可用 assets.gradient.GradientCache、FragmentGradients、CRBEngineMulti 实现稀疏准备。
- 进程与资源：检查到 main-reconst-v0 任务进程；最近 GPU 查询为0%利用率。未启动新 GPU 任务。
- 输出与日志路径规划：outputs/grad_cache/main-crb-v0/q16-src-q8-time1-ref/、logs/main-crb-v0/；本次均未创建新产物。
- 待办：若进入实际准备，按上述预算生成锚点清单和 TIME1 ref 梯度缓存，并回填结果文档。
