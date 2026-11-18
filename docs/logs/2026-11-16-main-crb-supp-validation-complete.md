# 2026-11-16 main-crb-supp 验证完成

- 检查时间：2026-11-16 23:44（Asia/Shanghai）。
- 实验/批次：main-crb-supp，12 任务 × E1–E5 三个训练节点，加 E0 一个节点，共 192 个正式评估点。
- 状态：正式队列尚未启动；验证已完成。
- 第一轮：首任务 Crypto/D 的安全 reference 有 4 个单变量实体、7,976 个合法窗口、1,000 个 q8 锚点。六臂均完成 1 step 训练或既有 Rand checkpoint 评估，生成并读取 TIME rolling MSE。
- 方法修复：改用原生相邻锚点区间划分 source block 归属，避免最近中心规则使窗口落到原 block 之外；六臂验证在此修复后重新通过。
- 第二轮：通用训练入口新增可选 evaluation_steps；首任务 E1 连续训练 3 step，在 step 1/2 保存模型快照、step 3 保存最终权重，三个 checkpoint 都完成独立 TIME rolling 评估。对应正式队列的 1000/2000/4000 step。
- 安全边界：TIME rolling reference 使用完整测试尾部截断，按每条单变量序列只取首个 rolling input；12 个任务逐项检查有限连续段。SG_Weather/D 与 Oil_Price/B 没有合格段，按独立随机种子 1 抽取 current_velocity/5T 和 azure2019_U/5T 替换。12 个正式任务均有合格 reference。
- 文件校验：三个改动源码均能 AST 解析且全部为 ASCII；结果索引在 docs/exp/main-crb-supp/results.md、docs/EXP.md、docs/RESULTS.md 各有 192 行，无正式指标；全部隔离验证数据、缓存、输出和配置目录已确认不存在。
- 验证遗留：验证运行的程序日志保留在 logs/ 中供审计；无验证训练/评估产物残留。
- 下一步：tmux 启动正式串行队列，在第一训练臂启动后检查 GPU 利用率并记录。
