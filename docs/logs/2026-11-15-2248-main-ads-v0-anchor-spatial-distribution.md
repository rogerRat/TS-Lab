# main-ads-v0 ADS 新增锚点空间分布核查

- 检查时间：2026-11-15 22:48 CST。
- 批次：`main-ads-v0/ads-histgram`；状态：已完成，只读分析。
- 方法：逐 q 读取已保存的 coarse 与 ADS final anchor position 数组；新增锚点取 final 与 coarse 的集合差。分别按 src/ref 轨迹长度切成10个等长区间，比较新增锚点各区间计数与均匀期望。
- 结果：src q=2,4,8,16,31,64 的新增锚点覆盖全轨迹；每个十分位区间相对均匀期望约0.82–1.12倍。ref q=2–31 的新增点覆盖约89.5%–97.4%跨度；q=64新增8点、均匀落在中间8个十分位。q=1无新增点，因为q=1粗采样已覆盖全部窗口。
- 解释：新增锚点的时间位置形状与粗网格基本一致，但ADS增加了整体采样密度；新增点没有集中到少数时间段。ref q=31、64新增样本较少，逐区间频数的波动较大。
- 数据来源：`outputs/selection/main-ads-v0/ads-histgram-bf16-u8/uni-q/*_anchor_positions_q*.npy` 与 `ads-vic/*_anchor_positions_q*.npy`。
