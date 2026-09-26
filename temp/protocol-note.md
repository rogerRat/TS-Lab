## 2026-09-18 V4正式启动：固定三seed、V3归档与巡检

最新决策覆盖前一版不限seed数的计划：V4固定仅使用seed0、seed1、seed2，总共三轮。轮间串行，当前轮两个模型×九任务×12训练方法全部完成训练与评估后才开始下一seed；轮内Chronos Base和TimesFM 2.5并行，TimeMoE暂停。13方法及聚合公式不变，外层Softmax T=.05；Exp内层T=1.00、先exp再双层加权，Max保留加权后择优。新seed不是新源池，pool_seed0和projection_seed42仍固定。

总规模666个独立结果项=18个orig基线+216训练评估项×3seed。orig首轮在训练前测量，后续引用；静态选择概率和已有字段校验梯度缓存复用，各seed独立训练/抽样与checkpoint，不复用训练成绩。父协议seeds=[0,1,2]，子轮seeds=[K]，seed_schedule.stop=3，默认第三轮结束后停止，禁止自动进入seed3。

V3已完成189/189。已整理全部既有输出目录，归档746个cell JSON及配置、文档、日志、run元数据，生成完整all_cells.jsonl与按各原run/model/task相对rand改进降序的all_results_ranked.csv。归档位于/root/autodl-tmp/TS3/outputs/archives/results-through-v3-20260918，含results-docs-configs.tar.gz及明确字段inventory.json，无hash。原结果、选集、梯度缓存及保留checkpoint留在原路径；大梯度不重复复制。最终V3报告另存docs/archive/MAIN-v3-final-20260918.md。

归档后停止已结束的V3 scheduler/report loop，清理其过期PID文件及可丢弃报告partial，不删除正式成果或V4依赖材料。V3 execution_control切换archived，防止旧守护误续跑。V4已按用户授权真实启动：configs/rags/main_v4.json；run=/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918；父入口rags.experiments.v4_seed_rounds，先推进seed0。恢复仍须用父入口，不能绕过跨模型seed屏障。

MAIN-v4.md直接复用MAIN-v3的报告生成模板，保持任务分节、模型表、状态/巡检、末尾汇总与固定方法说明的版式，只更换内容及版本标识。所有结果表（任务模型表、末尾聚合汇总表）均按相对rand改进降序排序，正值更好，缺数据/缺rand基准列末；不是按绝对MASE或方法名排序。末尾汇总按方法聚合seed，orig仅计一次，不把基线引用算成新实验。三seed汇总包含均值、样本标准差及逐seed胜rand配对信息。

79项相关回归通过，覆盖固定三seed止步、全矩阵屏障、失败/续跑、基线去重、表格降序和同模板报告、旧V3兼容。已建立本任务每30分钟巡检，并维护远程30分钟报告更新；正常进度保持安静，整轮/全批完成或明显异常时通知。本机不关机，已启用保持空闲唤醒，远程独立运行；不执行关机操作。当前实验进行中，尚不对V4效果作结论；未提交/Push。
