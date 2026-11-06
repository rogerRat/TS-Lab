from pathlib import Path
import json
p=Path('tests/test_rage_crb_multi.py');s=p.read_text();a=s.index('    replace(CRBConfig');b=s.index('\n]\n',a);s=s[:a]+s[b:];p.write_text(s)
p=Path('tests/test_shared_main_protocol.py');s=p.read_text().replace('configs/rage/main_shared_ten_20260916_211.json','configs/rage/main.json').replace('configs/rage/shared_tasks_10.json','configs/rage/shared_tasks.json');s=s.replace("    for settings in config['settings'].values():\n        settings['temperature']=1.0  # test value, never writes a production config\n",'').replace("assert len(config['datasets'])==10","assert len(config['datasets'])==9").replace("match='recipe mismatch'","match='recipe differs'").replace("match='budget'","match='CRB differs'");s=s.replace("('original',0,4),('random',0,2),('random',1,4),('rage',0,1),('rage',1,2)","('orig',0,4),('rand',0,2),('rand',1,4),('rage-block-maxSum',0,1),('rage-block-maxSum',1,2)").replace("arms=['random','rage']","arms=['rand','rage-block-maxSum']");p.write_text(s)
p=Path('src/rage/experiments/formal_main_entry.py');s=p.read_text();a=s.index('\ndef main():');s=s[:a]+'''
def preflight(config,readiness):
    if not readiness.get('ready') or readiness.get('config')!=config:
        raise ValueError('Queued formal config differs from the reviewed ready configuration')
    validate_config(config)

'''+s[a:];p.write_text(s)
p=Path('src/rage/experiments/full_pool_smoke.py');s=p.read_text().replace('    args = parser.parse_args(argv)','    parser.add_argument("--source-trajectories",type=int,default=4)\n    parser.add_argument("--reference-trajectories",type=int,default=2)\n    parser.add_argument("--trajectory-length",type=int,default=632)\n    args = parser.parse_args(argv)\n    if args.source_trajectories<1 or args.reference_trajectories<1 or args.trajectory_length<576:\n        parser.error("Need positive source/ref counts and length >= C+H")');s=s.replace('np.arange(584)','np.arange(args.trajectory_length)').replace('rng.normal(0, .03, 584)','rng.normal(0, .03, args.trajectory_length)').replace('for i in range(46)','for i in range(args.source_trajectories+args.reference_trajectories+2)');s=s.replace('    pool, refs, exam = fragments[:40], fragments[40:44], fragments[44:]','    cut=args.source_trajectories;end=cut+args.reference_trajectories\n    pool,refs,exam=fragments[:cut],fragments[cut:end],fragments[end:]').replace('manifest = dict(total_equivalents=40, total_points=40 * 584, lengths=[584] * 40,','manifest = dict(total_equivalents=len(pool)*(args.trajectory_length//576), total_points=len(pool)*args.trajectory_length, lengths=[args.trajectory_length]*len(pool),');p.write_text(s)
# Cache identities use explicit fields; no source/model/data hashing.
p=Path('baselines/trajectory.py');s=p.read_text().replace('import hashlib\n','');a=s.index('        key = hashlib.sha1');b=s.index('    info =',a);s=s[:a]+'''        from assets.gradient.cache import readable, atomic_json
        cache = Path(cache_dir) / (readable(kind)+"-trajectory-features.npy")
        manifest_path=cache.with_suffix('.json')
        if cache.exists() and (not manifest_path.exists() or json.loads(manifest_path.read_text())!=identity):
            raise ValueError('Trajectory feature cache fields differ; use a new cache directory')
        cache.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(manifest_path,identity)
'''+s[b:];s=s.replace('        np.save(cache, features)','        temporary=cache.with_suffix(".npy.tmp")\n        with temporary.open("wb") as stream:np.save(stream,features)\n        temporary.replace(cache)');p.write_text(s)
# Keep former protocols in history, with one active method document.
archive=Path('docs/archive/method-decisions');archive.mkdir(parents=True,exist_ok=True)
for pattern in ('v5_*.md','reference_opinion_*.md','prior_clarification_*.md','rank_decisions_*.md'):
 for p in Path('docs/experiments').glob(pattern):p.rename(archive/p.name)
for name in ('MAIN-v3.md','MAIN-v4.md'):
 p=Path('docs')/name
 if p.exists():p.rename(Path('docs/archive')/name)
p=Path('AGENTS.md');s=p.read_text();a=s.index('当前实验协议为 **v3**');s=s[:a]+'''当前实验协议为 **V5**，统一在 `/root/TS3` 的 `rage` 分支开发。
唯一当前配置为 `configs/rage/main.json`；方法见 [当前方法](docs/method.md)，
操作与测试见 [当前实验](docs/MAIN.md)。旧 V1–V4 文档仅供历史追溯，旧可执行入口已清理。
正式 V5 训练尚未启动；开发测试不构成正式实验结果。
''';p.write_text(s)
p=Path('docs/README.md');p.write_text('''# 文档入口

当前冻结实现为 **V5**，代码维护在 `/root/TS3` 的 `rage` 分支。

- [当前方法](method.md)：原始梯度代表、参考评分权重、最小 κ 分裂、冻结排名温度与源窗口量先验。
- [当前实验](MAIN.md)：唯一配置、准备/运行/汇总入口和开发验证范围。
- [项目约定](../AGENTS.md)：存储、字段校验、预算和进度要求。
- [梯度资源](gradient_assets.md)：独立逐窗口梯度及缓存读取。
- [数据准备](DataPreparation.md)、[安装](../requ/INSTALL.md)、[任务队列](TaskQueue.md)。
- [进度流水账](Progress.md)：实际执行与测试记录。
- [历史归档](archive/README.md)：旧实验结果、旧方法决策与失效命令，不能用于运行当前协议。

V4 已按用户授权停止，已落盘结果完整保留；V5 正式实验尚未启动。
''')
with Path('docs/archive/README.md').open('a') as f:f.write('\n2026-09-18：V3/V4 主表及 V5 逐次方法裁定归档至本目录和 `method-decisions/`；现行唯一方法文档是 `../method.md`，旧分数/温度/P10/随机并列方案不再有效。\n')
