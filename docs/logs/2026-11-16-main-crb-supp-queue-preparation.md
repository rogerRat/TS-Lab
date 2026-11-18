# 2026-11-16 main-crb-supp 队列准备

- 检查时间：2026-11-16 23:25（Asia/Shanghai）。
- 实验/批次：main-crb-supp，12 个新任务，每任务 E1–E5 与 E0。
- 状态：准备中；未启动正式训练。
- 已核验：src-256 共 6,382,020 合法窗口；CRB 与 Uniform q16 各 399,044 次锚点梯度、398,788 个 block；既有 Rand checkpoint 训练 4000 steps、seed 0、batch 32。
- 数据：下载固定版本 FEV ETT/1H、jena_weather/1H；TIME 未做过候选已有本地数据。按固定随机种子 0 选取并排序 12 个任务。
- 重要修复：检查发现旧 reference helper 仅按单窗口 horizon 截断 TIME 数据，与本批 rolling 评估不匹配，可能产生测试区间泄漏；现改为读取实际评估模式的 test_length，并按每个单变量 item_id 只取首个 rolling input。旧调用默认 single_window 时行为保持原约定。
- 开发：新增 scripts/exp/main-crb-supp/run_queue.py；扩展 scripts/exp/v1-crb-redo/ref_inputs.py 的可选配置与目标目录参数；复用旧 q16 source 梯度、q8 梯度计算、reference 窗口物化及 scripts/tools/run.py。
- 一次准备命令因结果索引字符串格式错误中止；已修复并重跑成功，没有启动训练。
- 待办：运行端到端验证，检查结果，清理验证目录；启动正式 tmux 队列；启动期检查 GPU 利用率。没有设置自动巡检或轮询。
