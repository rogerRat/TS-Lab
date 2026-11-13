# v1-rage-crb-vs-uniform-timerXL-84m

最后修改日期：2026-11-10

## v1-rage-crb-vs-uniform-timerXL-84m

首次提出：2026-11-10。最后修改：2026-11-10。当前状态：单任务优化计时已完成；正式选择与训练尚未开始，等待用户审阅更新后的单 GPU 估计。

固定协议：使用 v1 TimerXL 84M BF16 LoRA 训练与评估配方。从全部合法 576 点窗口中选择 10%，即 floor(0.1*N)。现有 12 个任务依次独立选择；每个任务只使用自身的合格引用。每一轮为该任务随机选择一个合格 ref，然后加入其余弦得分最高且尚未被选中的 src；达到数据预算即停止。余弦越高越好。

梯度仅使用可训练的 TimerXL LoRA 参数，r=8、alpha=16、dropout=0，目标 q_proj/v_proj，基础模型冻结。使用同一个固定 Count-Sketch 投影，跨任务与方法共享，维度 4096、seed 42。LoRA 初始化 seed 为 0。Uniform 与 CRB 使用相同的梯度点预算，q=32、K=ceil(N/32)，包含端点；两者在相邻锚点之间均使用线性重建。CRB 的锚点分配与放置遵循固定 CRB 协议。

当前源池为 N=360,695,805，正式数据预算为 floor(0.1*N)=36,069,580 个不重复合法窗口。每种方法使用 K=ceil(N/32)=11,271,744 个梯度锚点。选择完成后，每个任务按 v1 配方独立训练一个 TimerXL LoRA 批次：4,000 次更新、batch 32、microbatch 8、BF16，检查点位于 0/100/200/500/1000/2000/3000/4000，并在全部 12 个 FEV/TIME 任务上评估。正式数据选择完成后才开始训练与评估。

#### 单任务可行性计时：hospital_admissions_1D

首个计时任务为 FEV hospital_admissions_1D（task index 9）。合格引用来自本地 FEV train 划分，为长度至少 576 点的有限、非常数单变量窗口。计时 pilot 不产生正式选择清单或锚点，也不运行选择或训练。初始配置、入口与输出分别为 `configs/rage/v1-rage-crb-vs-uniform-timerXL-84m-pilot.json`、`scripts/temp/v1_rage_crb_uniform_timerxl_pilot.py` 和 `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/`。共有 9,246 个合格 ref。初始源梯度速率为 22.64 points/s，其 138.3 天源梯度估计已被下面的优化计时取代。初始 ref-src 矩阵计时为 630.35 million pairs/s；在计入分块开销前，一次完整 hospital_admissions_1D 打分扫描估计为 1.47 小时。

#### BF16 梯度优化计时

最后修改：2026-11-10。已将 PEFT 兼容的批量 TimerXL 逐窗口损失修正为复现检查点的损失掩码归约与逐头顺序累加。在验证样本上，其与标量 adapter 损失的最大绝对差为 0.0。

在 96 个源窗口上，eager batch 32 / VJP 16 达到 437.73 gradients/s（峰值已分配 GPU 显存 0.299 GiB）。`torch.compile(mode=reduce-overhead)` 与 `torch.compile(mode=default)` 均能编译前向路径，但在批量 VJP 反向传播时失败，报 `Cannot access data pointer of Tensor that doesn't have storage`；因此不使用编译执行。

后续测试使用 1,024 个源窗口，batch 取 32/64/128，VJP 分块取 16/32/64；最佳为 batch 32 / VJP 32，达到 624.38 gradients/s。额外的一次 VJP 128 扫描未提升吞吐。两次持续 8,192 窗口的运行在 batch 32 / VJP 32 下测得 596.82 与 575.35 gradients/s，峰值已分配显存均为 0.321 GiB。规划时使用较低的持续速率：两个方法与十二个任务共 270,521,856 个源梯度点，需要 5.44 个单 GPU 日。此估计不含引用梯度、完整打分扫描、源 I/O/缓存写入、训练与评估。产物位于 `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/optimization/`、`.../optimization-stable/`、`.../optimization-vjp128/`、`.../optimization-steady-8192/` 和 `.../optimization-steady-repeat/`；每个包含 `.complete`。推荐计时设置：BF16、单次批量前向、batch 32、VJP 32、FP32 投影累加。这仍只是可行性估计；正式选择与训练尚未开始。


#### 同 GPU 进程并行计时

最后修改：2026-11-10。梯度探测每进程约使用 0.321 GiB 峰值 PyTorch 已分配 GPU 显存。在 RTX 4080 上的并发进程计时使用独立的 TimerXL BF16 LoRA 进程，每个进程在 batch 32 / VJP 32 下处理 4,096 个抽样的源窗口。两个进程合计达到 964 points/s（约 1.68 倍单进程持续速率）；四个达到 1,087/s（1.89 倍）；八个达到 1,159/s（2.01 倍）。每个进程保留约 0.321 GiB 峰值已分配显存。八进程运行期间，`nvidia-smi` 显示进程总显存约 5.7 GiB、GPU 利用率 100%。八进程相对四进程仅提升约 6.6%，尽管显存仍有剩余空间，但收益已递减。其模型初始化也偏重 CPU，在八进程 pilot 中耗时约 1.4 分钟。

若该速率能在真实分布式源窗口流水线中保持，则 270,521,856 个源梯度点在 1,159 points/s 下预计约 2.70 天，不含 worker 启动、同步、源 I/O、引用梯度、打分扫描、训练与评估。应将其视为初始并行上界估计，而非正式运行时承诺；在用于最终排期前，须验证其可持续性、唯一工作划分与端到端缓存写入。带 `.complete` 的并行计时输出位于 `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/parallel-two/`、`parallel-four/` 和 `parallel-eight/`。
