# 梯度代理的直接计算成本

最后修改日期：2026-11-08

## 协议

在 v0-gradient-proxy-test 所用的同一个 Chronos-T5 Tiny warm 过的 checkpoint 和十条 M5 轨迹上进行测量。每条轨迹选取八个均匀间隔的窗口（共 80 个），每种方法对每个窗口在 batch size 为 1 下计时三次，并在每种方法三次 warmup 调用之后随机化方法顺序。所有方法处理同一窗口，并包含 CPU tokenization、模型前向、方法相关计算、将结果复制到 CPU 以及 CUDA 同步。Full BP 返回转换为 BF16 用于存储的完整梯度；head 代理返回转换为 BF16 的全部输出权重坐标。Embedding 和 loss 返回各自对应的向量/标量。

硬件：NVIDIA GeForce RTX 4080。这些测量描述的是本实现、精度、设备以及 batch size。

## 结果

| 方法 | 中位 ms/窗口 | 平均 ms/窗口 | P90 ms/窗口 | 相对 full BP 的配对加速比 | 逐窗口加速比范围 | 跨轨迹中位数范围（ms） |
|---|---:|---:|---:|---:|---:|---:|
| Full BP | 42.80 | 42.79 | 47.29 | 1.00x | 1.00x-1.00x | 40.29-44.62 |
| 输出头梯度 | 14.21 | 14.75 | 16.96 | 2.93x | 2.17x-3.83x | 14.02-14.79 |
| 解码器嵌入均值 | 11.35 | 11.65 | 13.03 | 3.72x | 2.92x-4.86x | 11.04-11.76 |
| 前向 loss | 11.28 | 11.59 | 12.55 | 3.71x | 2.84x-4.66x | 11.03-12.05 |

每行有 240 次计时调用：80 个窗口，每个三次重复。中位加速比是将配对的逐窗口中位延迟与 full BP 比较。逐窗口范围和跨轨迹范围是描述性的，不是置信区间。

## 验证

head 代理方向在每条轨迹的一个采样窗口上与已保存的 head-gradient 缓存一致（余弦大于 0.999）。平均嵌入与已保存特征的匹配最大绝对误差至多为 1.9e-06；前向 loss 的差异至多为 4.8e-07。全部十条轨迹检查均通过。

原始重复计时和精确采样起点：outputs/train/v0-gradient-proxy-cost/per_window_timings.csv。机器可读摘要：outputs/train/v0-gradient-proxy-cost/summary.json。

## 解读

在此 GPU 和实现下，输出头梯度耗时约为 full BP 的三分之一。嵌入均值和 loss 各自约为 full BP 的 3.7 倍快，且彼此成本几乎相同，因为二者都需要相同的模型前向；提取嵌入只增加很少开销。head 代理成本更高，因为它还要计算 loss 对 logits 的导数并构造输出权重梯度。

这些时间包含数据准备和结果传输。它们不包含大规模存储吞吐、并发 CRB replay，或不同的 batching 策略。在观察到容器内存压力后，CRB 队列仍在其 STOP 标记下暂停。

## v0-gradient-proxy-cost

于 2026-11-08 完成。在来自十条固定轨迹的 80 个窗口上，每种方法在 batch size 为 1 下重复计时三次。中位端到端延迟：full BP 42.80 ms/窗口，输出头梯度 14.21 ms（配对加速比 2.93x），解码器嵌入均值 11.35 ms（3.72x），前向 loss 11.28 ms（3.71x）。计时包含窗口 tokenization、GPU 计算、CPU 结果传输以及同步。详情：docs/exp/v0-gradient-proxy-cost/results.md；原始重复：outputs/train/v0-gradient-proxy-cost/per_window_timings.csv。
