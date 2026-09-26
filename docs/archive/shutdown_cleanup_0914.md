# 2026-09-14 停机扩容前清理

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

用户最新要求：停止实验、清理中间产物，等待用户关机扩容。恢复后只做三模型
原生探针的RAGS与Original开发比较，以相对MASE改善选择RAGS学习率和Epoch，
再做独立种子复核。旧Random日程只是候选，不是RAGS最终冻结规则。
不得自动重启其他基线队列或正式矩阵。

## 已停止

exp `/root/TS3`，连接 `ssh -p 26856 root@connect.bjb1.seetacloud.com`。
停止Base stage2 shell423658、LESS t1训练426045、旧清理131333及资源采样
131336，关闭对应三个tmux会话。本地10:40一次性巡检81610已取消（退出130）。
未关机，SSH、平台面板、Jupyter等系统服务保留。

## 保留与删除

保留代码/Git、配置、文档、JSON/JSONL指标与曲线、日志、概率产物、原始源
数据、当前512输入、N16284完整池及六任务输入、标定JSON、三模型初始权重
和Python环境。结果仍在TMP原outputs/outputs_archive；文本结果另备份到
WSL `/home/rogers/Projects/exp-shutdown-0914-results/`。
压缩传输备份已成功退出0，完整文本结果副本在该目录`combined/`（约887MiB）。
最终核实无实验/清理/采样进程，无GPU计算进程；容器可由用户关机。

授权删除：两outputs树中的训练PT/PTH/safetensors及优化器断点，大梯度/CRB/
嵌入缓存，Base/MoE标定梯度NPZ，pip下载缓存，旧256/1024预处理目录中的
非JSON/YAML/TXT文件（约52GiB；元数据和原始数据保留）。并非移入回收站，
不能直接恢复断点和权重，需重算/重训练。停止的LESS t1不完整，不能当完成格。

实测390GiB数据盘空闲从约34GiB增至326GiB，释放约292GiB。
旧记录中“保留全部检查点”和“Base九格后接续MoE”的计划现已失效。

## 已有结论（开发结果，不是正式主表）

新版全池重采样、三模型原生探针已实现测试。TimesFM概率开发独立seed1/2
平均改善约Random0.045%、RAGS0.031%、LTSV0.178%，不支持RAGS稳定胜过
Random。Base实际N Random改善不足0.1%；TimeMoE实际N Random在E4三种子
改善约1.98%至2.17%，不是RAGS调参证据。Base概率Random及RAGS两温度已
完成，其他试验保留实际落盘结果。正式矩阵未启动；来源见执行记录。

用户随后要求训练/探针模型权重与梯度改为BF16。该修改在dev开发测试，exp
继续停机待命；旧FP32结果仅作历史结论，不冒充BF16结果，不复用旧探针缓存。
