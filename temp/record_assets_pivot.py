from datetime import datetime, timezone
from pathlib import Path
import json

run = Path('/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-global-20260917')
protocol = json.loads((run/'protocol.json').read_text())
control = dict(mode='assets_only',training_enabled=False,evaluation_enabled=False,
    reference_windows='current_method_union',datasets=protocol['datasets'],
    students=protocol['students'],recorded_at=datetime.now(timezone.utc).isoformat(),
    reason='User paused training/evaluation pending scoring-scale corrections; prioritize reusable reference gradients')
temporary = run/'execution_control.json.tmp'
temporary.write_text(json.dumps(control,indent=2))
temporary.replace(run/'execution_control.json')

path = Path('/root/TS3/docs/MAIN-v2.md')
content = path.read_text()
heading = '## 执行转向：暂停训练评估，准备九任务参考并集（2026-09-17）'
note = '''

## 执行转向：暂停训练评估，准备九任务参考并集（2026-09-17）

用户明确要求立即暂停训练/评估，把算力转向接下来九个任务的参考准备，
尤其梯度计算和缓存；随后确认计算当前方法所需的窗口并集，不是全部合法窗口。
任务采用当前协议中的九任务：NEM demand、Azure CPU、Darwin water、Finland、
eWELD、CPHL、NEM price、solar、port activity，覆盖Chronos/TimesFM/TimeMoE。
由于发现参考聚合sum/mean与有效温度的重大混杂，现有训练结果保留为历史记录，
后续是否重训/重评待方法修正与新实验决策，不自动恢复旧全流程。

当前旧worker、调度器、supervisor/entry已停止。已有46个cell、选集、梯度缓存
和现存检查点保留，不删除。run/execution_control.json mode=assets_only；
full_pool_worker在启动、取得worker锁、任务及训练cell入口检查暂停标记，
在模型加载前拒绝误启动的全流程。准备进程独立使用assets_state.json。

新增入口：python -m rage.experiments.full_pool_assets --run RUN --student STUDENT
--reference-batch 32 [--gradient-workers 2]。不读取exam，不执行original评估、
概率选集、fine_tune或checkpoint清理。只构建冻结预训练模型参考梯度：
LESS uniform窗口与RAGE CRB自适应窗口并集，供轨迹级与Global各臂共同复用。
选择规则/温度的后续修正不会使这些原生每窗口梯度失效。

沿用当前grad_cache的模型身份、pool字段指纹、BF16编码及reference:DATASET
namespace，不复制成不兼容缓存。参考轨迹分批推进，完成每条轨迹后原子保存
students/STUDENT/reference_assets/DATASET/trajectory-ID.npz，包含合法起点、
锚点索引与向量、块左右索引/所有权起点、原始块均值/方向、线性轨迹聚合、
三阶段预算统计与p10阈值审计；protocol.json验证数据/模型/CRB身份，
complete.json标记任务完成，可逐轨迹续跑。避免全任务方向stack与完整source
CRB块驻留的大RAM副本。原source方法窗口梯度已由早期任务准备，此次不重复
重建source CRB；未来仍能从原source梯度缓存构建选择资源。

准备模式允许不同模型并行，沿用每run/student锁和模型gradient cache锁；
主控Claude负责错峰启动、内存/GPU监控与任务推进。Chronos可以采用已有
independent native-gradient workers=2路径提升吞吐；TimeMoE默认2，TimesFM
沿用当前梯度batch4，梯度定义/独立原生窗口forward不变；执行参数记录于
任务complete.json。避免把历史三模型训练峰值当准备阶段的实际峰值。

测试：assets-only入口无exam文件且训练/评估函数被禁止时仍能完成准备；
参数保持不变；完成产物续跑不再请求梯度；参考分批并集与原production
prepare_gradients的reference窗口ID集合及块方向一致；暂停标记在模型加载
前阻止全流程worker。相关组14 passed，git diff --check通过。
后台准备尚未完成全部27个model-task，完成情况以assets_state/complete标记为准。
'''
marker = '<!-- v2-notes:end -->'
if marker not in content:
    raise RuntimeError('Missing persistent notes marker')
if heading not in content:
    path.write_text(content.replace(marker,note+'\n'+marker))
print('Applied assets-only control and recorded pivot in MAIN-v2.md')
