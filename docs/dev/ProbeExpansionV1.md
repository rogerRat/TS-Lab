# 安全历史任务扩展 v1（2026-09-14）

用户批准继续：不限 FEV/TIME/Quito，原生单目标优先、后训练空间待验证、避开 ETT，累计候选最多20。
旧8项保留，本次增加7项，总计15。数据可用性筛查不使用考试分数；收益理由是假设，不证明未进入预训练。

| 新增ID | 结构/有效标量序列数 | 预考试长度 | 理由与限制 |
| --- | --- | --- | --- |
| fev_epf_np | 原生单目标，1 | 51936 | 电价市场适配；经典旧数据，作为单目标对照，不能称新颖 |
| fev_proenfo_gfc14 | 原生单目标，1 | 14160 | 负荷领域；已有 NEM 收益仅是线索，不能外推该任务必涨 |
| fev_solar_with_weather_1h | 原生单目标，1 | 49168 | 较长光伏历史；不使用其天气协变量，非多变量输入实验 |
| time_darwin_water_15t | 7实体×6变量=42 | 7222–17545 | 环境水质多物理量，补充领域；多变量拆通道 |
| time_ne_china_wind_h | 1实体×4变量=4 | 8044 | 风场与气象机制；不称单变量原始数据 |
| time_port_activity_d | 99实体×进出口=198 | 1762 | 港口物流跨实体领域；训练前缀短，须报告真实step |
| time_azure_i_cpu_5t | 492实体，每实体取平均CPU列 | 7386–7775 | 云计算CPU负载，补充非能源应用；原始3变量，选1列不等于原生单变量 |

首次准备在Azure列名检查处停止，失败目录保留。item_770 的平均CPU名为 avg_cpu[sp,rw]，不是 avg_cpu；新增显式别名映射并逐实体记录实际别名，保留该实体而非静默丢弃。初查491是只匹配普通列名的数量，不是完整任务可用量。新输入目录使用 probe-inputs-expansion-v1r2，14份运行配置同步指向该目录。旧失败目录不覆盖或删除。

原始 TIME 为已固定 revision 83e3d0b3be28d11c7182bffcc1892d19b36c4da1；FEV 数据为本地固定 f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8。
FEV 新入口读取显式本地Arrow文件和项目声明，从尾部排除 horizon*num_windows 全部考试区间；只支持声明的标量target，不读取协变量用于训练。显式cutoff/stride声明拒绝，避免猜测边界。随后统一划出开发验证，再以更早历史训练。manifest保存源路径/revision/声明/排除长度/原始目标维数；不生成hash。
输入位于 TMP/TS3/outputs/probe-20260914/probe-inputs-expansion-v1，新结果为同根/probe-expansion-v1/<model>/<task>。全部结果及checkpoint保留，不覆盖旧v2。

本批使用 TimeMoE 与 TimesFM 两模型，同一组7任务，每任务3种子，共42条训练轨迹：
TimeMoE 固定LR1e-4/E1，TimesFM 固定LR3e-5/E2.625。其余batch/优化器/BF16/schedule_epochs=6均继承旧v2已选配置；C512/H64，8验证窗。不重开超参搜索，不增设梯度选样，不用外部源池。Chronos不从最终比较中移除，本批不额外跑其筛查以控制预算。
按模型分别报告相对基础模型的配对MASE改善、各seed、真实step、pool_n，审查极端序列主导现象。不因平均正收益自动视作显著；不自动生成模型私有正式任务清单，也不覆盖旧shared_selection。所有任务最终仍需共同任务集与正式考试评估。

排除记录：FEV住院/零售日频及月年宏观数据保留考试后不足1600点；SG_PM25/H与MetroPT-3/5T预考试历史有非有限值，当前协议不静默插补。Azure另一些类型不重复加入。Quito小时级已有聚合极端值预警，保留结果待审计，不拆成更多候选凑数。

字段与入口验证：test_probe_runtime.py/test_probing.py 27 passed，覆盖FEV原生目标导出、完整考试尾部隔离、拒绝多目标及特殊cutoff，以及既有TIME/Quito/搜索协议。真实新增manifest须由zcode准备后核对上述数量，再实际启动。
