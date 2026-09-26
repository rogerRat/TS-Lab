# 2026-09-15 线性梯度重建图（取代旧阶跃图）

命令入口：`python -m rags.experiments.anchor_budget_plot --run-root RUN --out OUT --calibration P10_JSON`。分类之前的覆盖/误差阶段可用 `--before-classification` 单独展示，不运行分类或分裂、不使用分类阈值；该模式不能用来判断分裂效果。

对原始 GT 拟合固定二维 PCA，再把原始锚点梯度的线性插值投到同一坐标系。没有逐窗单位化，因此分段直线保留；余弦仍在原高维空间计算。局部图固定中央 256 个合法起点。每条轨迹比较覆盖 2%、继续均布新增 2%、风险驱动新增 2%、最后 P10 分类及分裂预算 2%。前两种新增方式实际加点数相同；菱形表示相对初始覆盖新增的锚点，方形表示相对误差阶段新增的分裂锚点。

插值只按需临时计算：局部绘图只计算显示的起点，全轨迹余弦按 256 起点一批计算。只保存 PNG/PDF 和 scalar metadata.json，**不保存逐窗估计、二维曲线或逐窗余弦 NPZ**；原始真实梯度缓存不变。旧版 `visualizations-rank0/*_curves.npz` 是已撤销阶跃图的历史产物，未作为新算法输入；不再生成该类文件。

RUN/linear-before-classification-figures 为分类前图；RUN/linear-p10-figures 为固定 P10 完成后的全阶段图。`scripts/exp/plot_after_p10.sh` 在 tmux anchor-linear-plots 等待新 BF16 标定结果，校验身份后自动出图。图和缓存产物仅在 TMP，不入 Git。
