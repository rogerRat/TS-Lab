# v0-timerxl-src96-first4-full-grad-4096d-bf16

最后修改日期：2026-11-11

## v0-timerxl-src96-first4-full-grad-4096d-bf16

首次提出：2026-11-11。最后修改：2026-11-11。用户要求插队，对当前固定 `src-96` 的前四条轨迹（ordinal 0、1、2、3）的每一个合法窗口计算 TimerXL 84M LoRA 梯度，固定 4096 维 Count-Sketch 投影并以 BF16 位编码保存。窗口上下文 512、预测 64、步长 1；四条轨迹分别有 230,161、153、24,384、230,209 个合法窗口，合计 484,907。基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标 q_proj/v_proj，初始化 seed 0；投影 seed 42，批量 32、VJP 子批量 32。使用 rage-timerxl 环境、八个 GPU 进程，每进程处理不重叠的连续全局窗口区间。纯 BF16 向量数据约 3.70 GiB，另外包含索引与分片元数据。

配置 `configs/rage/v0-timerxl-src96-first4-full-grad.json`；入口 `scripts/temp/prepare_timerxl_first4_full.py`；调度 `scripts/temp/run_timerxl_first4_then_resume.sh`；正式输出 `outputs/train/v0-timerxl-src96-first4-full-grad-4096d-bf16/formal/`。已用 16 个窗口预演，核实索引、4096 维与 BF16 分片并删除预演输出。暂停原 Chronos Uniform 四进程，保留其缓存；插队批次全部完成后由同一 tmux 会话自动恢复。正式任务于 2026-11-11 21:41 CST 启动，八个 worker 均存活，GPU 已达到 100% 利用率；尚未生成整批完成标记。
