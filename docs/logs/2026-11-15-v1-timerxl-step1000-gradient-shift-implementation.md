# TimerXL step-1000 梯度差异实验脚本准备记录

- 时间：2026-11-15 CST。
- 实验代号：`v1-timerxl-step1000-gradient-shift`。
- 状态：执行脚本已创建；GPU 计算仍未启动。
- 文件：`scripts/exp/v1-timerxl-step1000-gradient-shift/compare_gradients.py`。
- 实现：复用项目 TimerXL 适配器、PEFT LoRA 装载、原生 V6 损失与 Count-Sketch 梯度投影；固定抽样和 bootstrap 方案按已确认文档实施。
- 检查：在远程 rage-timerxl Python 环境执行 AST 语法解析成功；源码 ASCII 检查通过。未运行测试或模型计算。
- GPU 约束：ads-v2 当前任务仍需先完成；未抢占 GPU、未创建排队运行。
- 下一步：GPU 空闲后核对检查点与轨迹窗口数，再用 tmux 启动实验；如核验失败先修正并记录。
