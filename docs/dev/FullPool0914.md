# 当前运行协议：全池重采样

2026-09-15 更新：梯度作为独立的 assets 数据集管理，见
[梯度资源格式与读取接口](../gradient_assets.md)。每个学生按任务依次准备窗口梯度与静态权重，
并立刻训练、评估和发布该任务结果；共享源池梯度在十个任务之间复用。缓存位于
`outputs/<run>/grad_cache`，模型/训练采用 BF16，
原 FP32 校准不能直接用于新运行。本文以下旧执行记录中的精度说明以本更新为准。

执行进度与恢复点见 [执行记录](../full_pool_execution_0914.md)。
当前开发尚未完成全部验收；不得将模板中的未标定参数补成旧 Small 数值来启动正式批。

默认 `python -m rags.experiments.run` 已使用 `configs/rags/full_pool.yaml`。
`--dry-run` 展示三个原生模型工作器，不下载数据、不启动模型训练。
历史 `configs/rags/run.yaml` / `main_table.py` 属于旧硬交付流程，不是新正式实验入口。

每个学生使用自己的冻结探针，保留七臂原评分单位，输出全池概率而非选中子集。
N为共同源池 `sum(floor(L/576))`，正式精确为16284；完整轨迹不裁剪。
训练有放回抽样，按N记Epoch，每半轮观察，输出实际步数/抽取数/重复率。
RAGS按Ref合法起点份额逐行归一化，再静态求源块边缘概率；无正配对的Ref明确失败。

三个模型为Chronos T5 Base、TimesFM 2.5、TimeMoE 200M。
`settings` 必须填写每个模型的 `epsilon_calibration`、`epochs`、`learning_rate`、
`development_evidence`，以及经开发验证的概率 `temperature`；没有自动回退到旧设置。
阈值文件必须匹配实际模型ID、revision和投影配置。

产物在TMP新run目录：`protocol.json`、`provenance.json`、`inputs/`；
各学生的 `model.json`、`selection_metrics.jsonl`、`selections/`、`cells/`、`state.json`；
`results.json` 汇总完成/缺失格子和相对不训练/Random的结果。
正式入口验证三模型、六任务、七训练臂、三种子和N，不接受偷偷缩小的正式矩阵。

各格保存半轮权重和最新 `resume.pt`（模型、AdamW、RNG、抽样记录）。
`--resume <run目录>` 使用原配置：完整格不重跑，中断格按观察点继续。
用户已授权新批采用 `checkpoint_retention: completed_cell_cleanup`：每格完成全部
评估并原子写入结果后，立即删除该格epoch权重及resume模型/优化器断点；
保留全部曲线、指标、配置、概率、成本和采样记录。结果标记训练权重不再保留，
因此完成格不能直接恢复训练权重，需重训练才能复现权重。未完成格不清理。
探索入口使用同一显式配置；旧批未设置该字段时不自动改变其保留策略。
共享预训练模型及后续方法仍需的评分缓存不随每格删除，避免重算。
额外检查点盘通过 `checkpoint_root` 指定；当前采用 BF16，不删除历史产物。

当前原生模型测试证据是合成小测试，不是正式效果实验。下一阶段在exp进行参数探索，
可使用安全历史训练但必须与开发验证和最终考试按时间隔离；只有冻结开发设置后才跑正式矩阵。
