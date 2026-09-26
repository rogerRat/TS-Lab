# 2026-09-15 固定窗口梯度协议

每个梯度对应一个合法窗口起点，由该起点裁剪 context+prediction 窗口，尾部不足以形成窗口的位置不属于重建域。锚点处保留真实梯度；其他合法起点在最近两锚点之间对原始梯度向量线性插值，不先单位化、不广播块均值。最终分块阶段才以均值代表块，块分类不删除先前的逐窗估计。

当前正式分类规则（2026-09-15 修正）：每条轨迹 epsilon_mu=epsilon_g=均布与误差两个阶段不同真实梯度模长的线性 P10，第三阶段前冻结，分裂新增梯度不更新阈值。参考与源分别逐轨迹计算，不混入其他方法或其他轨迹的缓存窗口。零模长始终为弱，NaN/Inf报错。正式配置使用 epsilon_policy=trajectory_p10；历史固定阈值入口及完整扫描诊断保留追溯，不作为当前正式阈值来源。

本轮旧 anchor_budget_simulate/integer/plot 的块常值主结论撤销，旧产物保留追溯，旧推荐入口已禁用。新的 anchor_budget_linear 使用 rags.window_reconstruction，在全部合法窗口上计算唯一指标：逐窗余弦均值，包含锚点、零范数余弦按 0 计；不以块分类后的保留窗口筛选评估集。

现存 TimeMoE ETT 开发 P10 标定为旧 FP32，不能混用于本轮 BF16。新标定使用同一开发输入，输出到数据盘 anchor-budget-20260914/epsilon-bf16-dev-p10，完整结果保存 epsilon.json 和每片段 norms NPZ；不改旧文件，不计算 hash，不提交运行产物。scripts/exp/repair_anchor_protocol.sh 在独立 tmux anchor-protocol-repair 中依次执行标定与五条 rank0 CPU 复核。原全窗口扫描不中断。本轮不启动主实验。
