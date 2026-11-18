# main-crb-v0 八任务重选重训批次启动记录

- 检查时间：2026-11-16 16:21:34 CST。
- 实验代号与批次：main-crb-v0 / q16-src256-eight-task。
- 启动时间：2026-11-16 16:21:19 CST。
- 用户目标：对八个登记任务全部重新选择、重新训练并评估；每任务 ORI/CRB/UNI/REFT 四臂，ORI 不训练。
- 远端项目：/root/TS-Lab；tmux：main_crb_q16_eight_tasks；主进程 PID 223606，资源准备子进程 PID 223609。
- 并行参数：parallelism=8；GPU：RTX 4080 32 GiB。启动后检查 GPU 利用率 67%，显存 3,396 MiB。
- 已执行：src-256 ref 清单与字段检查；8 个 ref manifest 均 answer_values_included=false；合计 290,429 个合法窗口、36,331 个 q8 锚点。两入口脚本源码编译通过，八个评估配置存在，目标输出目录启动前均不存在。
- 当前阶段：reference 梯度计算。FEV09 600/600、TIME1 410/410、TIME2 534/534、TIME3 1,670/1,670 已完成；TIME4 520 个锚点正在计算。之后队列自动并行执行八项 CRB/UNI 选择，再并行运行训练和评估。
- 队列命令：rage-timerxl Python -u scripts/exp/main-crb-v0/run_eight_task_queue.py --stage all --parallelism 8。
- 本次实现修复：补齐 ORI eval config/pipeline 与底座 checkpoint view；选择清单写入任务和批次身份。涉及 scripts/exp/main-crb-v0/prepare_eight_task.py、scripts/exp/main-crb-v0/run_eight_task_queue.py。
- 结果：本批尚无训练或评估指标；完成标记尚未生成。
- 下一步：保持 tmux 后台运行，任务完成后由队列写入各臂及批次 .complete；结果文档按实际 metrics 文件更新。
