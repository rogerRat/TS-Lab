# main-crb-v0 q16 / TIME1 q8 进度检查

- 检查时间：2026-11-16 14:14:27 CST。
- 批次：`q16-src256-time1-q8-four-arm`。
- tmux：`main_crb_q16_time1` 存活；Python 入口进程已运行约 13 分钟。
- 当前阶段：source UNI q16 梯度准备；日志记录已处理请求 311,322/399,044，阶段 ETA 约 3.1 分钟。CRB q16 source gradients 在本轮前已完成并由缓存复用。
- 资源：RTX 4080 利用率 100%，显存 2,866/32,760 MiB；主进程 CPU 约 106%。
- 后续计划顺序：完成 source UNI 梯度与 TIME1 ref q8 梯度；幅值/方向分解、方向块和插值余弦筛选；生成 CRB/UNI 训练窗口；按 CRB、REFT、ORI、UNI 顺序训练/评估。
- ETA（运行估算，非结果）：source UNI/ref 梯度约 3–5 分钟；资源分解、筛选与窗口物化约 15–40 分钟；之后三臂训练与四臂 TIME1 rolling eval 约 1–3 小时。后续总剩余时间粗估约 1.5–3.75 小时，训练和评估吞吐会影响误差。
- 当前无已完成的本批训练/评估指标；未生成结果不作推断。
