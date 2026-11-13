# v0-timer-XL-grad-4-curvature

最后修改日期：2026-11-11

## v0-timer-XL-grad-4-curvature

首次提出：2026-11-11。最后修改：2026-11-11。批次完成。读取四条完整 TimerXL LoRA 4096D BF16 投影梯度，按各轨迹物理窗口起点严格排序。对每轨计算一阶差 g[t+1]-g[t] 与二阶差 g[t+2]-2g[t+1]+g[t]，不跨轨迹边界。指标：二阶差 RMS、均值模长/RMS、中心化 RMS/RMS、模长 CV、相邻二阶差余弦均值及16个等宽时间段的二阶差均值模长。恒定曲率要求二阶差向量围绕同一个值的小幅波动，中心化 RMS/RMS 应接近0；另以二阶差 RMS 与一阶差 RMS 比较平滑性。代码 scripts/temp/v0_timerxl_grad4_analysis.py；正式产物 outputs/train/v0-timer-XL-grad-4-curvature/；结论 docs/exp/v0-timer-XL-grad-4-curvature/results.md。该结论只针对4096D投影和BF16表示。

原 v0-timerxl-src96-first4-full-grad-4096d-bf16 批次已于2026-11-11完成，八个worker与整批完成标记核验；调度脚本自动恢复 Chronos Uniform。上文21:41的运行中状态属于历史启动记录。

曲率批次补充（2026-11-11）：按用户要求增加相邻曲率变化 d3[t]=d2[t+1]-d2[t]，报告 d3 RMS/d2 RMS 与 d2 RMS/梯度 RMS；16个时间段另计算 d2 RMS 以检测分段特征，并比较跨轨绝对与归一化曲率。完整定义和结果见 docs/exp/v0-timer-XL-grad-4-curvature/results.md。
