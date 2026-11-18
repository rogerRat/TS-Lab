# 2026-11-16 main-crb-supp 并行队列重启

- 时间：2026-11-16 23:52:58（Asia/Shanghai）。
- 验证：五个训练臂在同一任务内并发，均成功连续训练 3 step 并输出 step 1/2/3 模型；16 个评估点均生成真实 MSE。验证目录已逐项确认清空。
- 正式执行：tmux 会话 main-crb-supp；12 任务仍严格按文档顺序逐个处理；同一任务的 E1–E5 五个训练臂并发，训练完成后按方法和 1000/2000/4000 step 顺序评估；E0 在 4000 step 对照 checkpoint 上评估。
- 复用：首任务 Crypto/D 的正式安全 reference、q8 梯度及已完成的 E1 block 选择资产继续使用；中断的旧 E1 模型保留在 outputs/train/main-crb-supp/interrupted/。
- 日志与输出：logs/main-crb-supp/run_*.log；outputs/{grad_cache,selection,train,eval}/main-crb-supp/run/；192 个正式指标逐项更新 docs/exp/main-crb-supp/results.md、docs/EXP.md、docs/RESULTS.md。
- 待观察：本轮五臂 4000 step 并发训练的显卡利用率、显存和任务稳定性；不设置后续自动巡检。
