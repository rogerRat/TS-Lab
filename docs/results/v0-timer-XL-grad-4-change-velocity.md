# TimerXL 四轨 LoRA 梯度随窗口滞后的变化

最后修改日期：2026-11-11。

## 方法

输入为 src-96 前四条轨迹的所有合法576点窗口的 TimerXL LoRA 梯度；基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标 q_proj/v_proj，初始化 seed 0；梯度投影4096D Count-Sketch seed 42，以BF16存储。四轨窗口数分别230161、153、24384、230209。

对每轨、每个 k=1..512 和每个有效起点 t，分别计算：

1. 方向余弦：cos(g[t],g[t+k])，各向量先按自身L2模长归一化。
2. 相对L2差：||g[t]-g[t+k]|| / ((||g[t]||+||g[t+k]||)/2)。
3. 相对模长差：abs(||g[t]||-||g[t+k]||) / ((||g[t]||+||g[t+k]||)/2)。

每个k都包含该轨迹全部N-k个有效窗口对，输出逐对指标的均值和总体标准差，不跨轨迹边界。轨迹1只有153个窗口，k=153..512记为0对与未定义值；图仅绘制k=1..152。GPU矩阵乘法使用float32，关闭TF32；输入仍受BF16量化和4096D投影限制。完整脚本 scripts/temp/v0_timerxl_grad4_change_velocity.py，数据 outputs/train/v0-timer-XL-grad-4-change-velocity/lag_metrics.csv。

## 选定滞后下的曲线数值

表中每格依次为方向余弦 / 相对L2差 / 相对模长差的逐对均值。

| 轨迹 | k=1 | k=8 | k=16 | k=64 | k=512 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.931 / 0.409 / 0.180 | 0.386 / 1.212 / 0.527 | 0.243 / 1.357 / 0.656 | 0.184 / 1.410 / 0.703 | 0.153 / 1.432 / 0.701 |
| 1 | 0.895 / 0.463 / 0.115 | 0.687 / 0.806 / 0.190 | 0.552 / 0.973 / 0.244 | 0.285 / 1.269 / 0.490 | 无有效窗口对 |
| 2 | 0.594 / 0.902 / 0.160 | 0.073 / 1.368 / 0.202 | 0.048 / 1.388 / 0.214 | 0.007 / 1.424 / 0.254 | 0.040 / 1.406 / 0.296 |
| 3 | 0.868 / 0.544 / 0.218 | 0.423 / 1.154 / 0.449 | 0.285 / 1.289 / 0.528 | 0.206 / 1.358 / 0.570 | 0.151 / 1.409 / 0.610 |

方向余弦首次低于0.5的k分别为6、21、2、6；相对L2差首次超过1的k分别为5、18、2、5。轨迹2的方向最快失去相似性，但模长相对差始终较低（k=1为0.160，k=512为0.296），其变化主要由方向贡献。轨迹0、3在k约16后方向相似度下降变缓，但仍从约0.24/0.28继续降至k=512的约0.15；相对L2差最终约1.4倍平均模长。轨迹1的方向变化较慢，但k=128只剩25对，不宜与长轨同等精度解读。以上描述都是该四轨的经验曲线，不能推出其它源轨迹的总体规律。

## 分轨迹图

每张图的三个面板自上而下为方向余弦、相对L2差、相对模长差，横轴为窗口滞后k：

- [轨迹0](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-0-change-velocity.png)（[SVG](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-0-change-velocity.svg)）
- [轨迹1](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-1-change-velocity.png)（[SVG](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-1-change-velocity.svg)）
- [轨迹2](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-2-change-velocity.png)（[SVG](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-2-change-velocity.svg)）
- [轨迹3](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-3-change-velocity.png)（[SVG](../../outputs/train/v0-timer-XL-grad-4-change-velocity/figures/trajectory-3-change-velocity.svg)）

原始数据含2048行（4轨×512滞后）、每行有效对数、三指标均值和标准差。输出目录及figures子目录均有 .complete 标记。图形入口 scripts/temp/plot_timerxl_grad4_change_velocity.py。

## v0-timer-XL-grad-4-change-velocity

[四轨逐滞后变化报告与分轨迹图](v0-timer-XL-grad-4-change-velocity.md)：对全484,907个LoRA窗口在各轨内计算k=1..512的所有有效窗口对。方向余弦首次低于0.5的k为6、21、2、6；相对L2差首次超过平均模长的k为5、18、2、5。轨迹2方向变化最快，但相对模长差在k=512仅0.296。完整2048行均值、标准差与对数见输出CSV；四张独立三面板图见报告。
