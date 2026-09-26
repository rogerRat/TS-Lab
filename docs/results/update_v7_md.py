import json, glob, os, datetime, statistics
R="/root/autodl-tmp/TS3/outputs/main-v7-20260923"
MODELS=["chronos-base","timerxl-84m","timesfm-2.5"]
TASKS=["time_nem_demand_5t","time_azure_i_cpu_5t","time_nem_price_5t","time_cphl_h","time_darwin_water_15t","time_solar_h","time_port_activity_d"]  # 2026-09-24 踢除 time_finland_15t/time_eweld_15t（commit 9f6538e）
CHRONOS_EPOCH=3.0  # Codex 修正：主表 Chronos 取 test_curve epoch=3.0（非 raw mase/非 best）
FORMAL_SEEDS=(0,1)  # 2026-09-25 用户定稿：正式结果只含 seed0/1；seed2 仅末尾存档
PER_SEED_GOAL=140; PER_MODEL_GOAL=280; TOTAL_GOAL=840
# 协议修订记录
AMEND=["2026-09-24 踢除 time_finland_15t 与 time_eweld_15t（V6 证据：Chronos 上 RAGS 分别弱 12.0%/4.3%；用户批准，已生效 commit 9f6538e）",
       "2026-09-24 Chronos max_epochs 4→3（用户批准，已生效 commit 9f6538e；旧 4ep cell 保留，主表 Chronos 取 3.0 非 best）",
       "2026-09-25 扩展探索三任务：Wind/SG_PM25/Coastal（MetroPT/Smart_Mfg exam=0 被门禁拒；探索口径，非主验证集）",
       "2026-09-25 shape 补做（中点+斜率臂，31b8539，严格全局臂序，用户批准）",
       "2026-09-25 用户定稿：正式结果仅 seed0/1；seed2（仅 TimerXL 有）存档保留但不进主表、不在文首披露"]

def reported_mase(m, d):
    mz = d.get("mase")
    if m == "chronos-base":
        tc = d.get("test_curve")
        if tc:
            pts = [(p.get("epoch"), (p.get("metrics") or {}).get("mase")) for p in tc]
            pts = [(e, v) for e, v in pts if e is not None and v is not None and e <= CHRONOS_EPOCH]
            if pts:
                return max(pts, key=lambda x: x[0])[1]
    return mz

def load_cells(pattern, taskfilter=None, seeds=None):
    data={}
    for f in glob.glob(pattern):
        try: d=json.load(open(f))
        except Exception: continue
        arm=d.get("arm"); t=d.get("task"); s=d.get("seed")
        if taskfilter and t not in taskfilter: continue
        if seeds is not None and s not in seeds: continue
        m=f.split("/cells/")[0].split("/")[-1]
        mz=reported_mase(m,d)
        if arm and t and mz is not None:
            data.setdefault((m,t,arm),{})[s]=mz
    return data

DISPLAY_ALIAS={"grad-cos":"gradcos-legacy"}  # 2026-09-26 用户批准：grad-cos 显示为 gradcos-legacy（参考质心余弦，非双侧逐窗余弦）；原始 arm ID 不变
ALIAS_NOTE="命名声明：gradcos-legacy = 原 grad-cos 臂（参考质心余弦，reference-centroid cosine，NOT two-sided per-window cosine）；原始 arm ID/选择/cell/缓存/指标全部不变，仅显示名变更（用户批准 2026-09-26）。"
def agg_table(L, rows_data, label, n_note):
    rand={}
    for (m,t,arm),seeds in rows_data.items():
        if arm=="rand":
            for s,v in seeds.items(): rand[(m,t,s)]=v
    per={}
    for (m,t,arm),seeds in rows_data.items():
        for s,v in seeds.items():
            k=(m,t,s)
            if k in rand and rand[k]:
                per.setdefault((m,arm),[]).append((rand[k]-v)/rand[k]*100)
    arms=sorted({a for (_,a) in per})
    means={(m,a):statistics.mean(v) for (m,a),v in per.items()}
    def overall(a):
        vals=[means[(m,a)] for m in MODELS if (m,a) in means]
        return statistics.mean(vals) if vals else float("-inf")
    arms.sort(key=lambda a:-overall(a))
    L.append("## 正式结果·%s（vs rand 改善%%，越大越好；%s）"%(label,n_note))
    L.append("")
    L.append("| 方法 | Chronos | TimerXL | TimesFM | 平均 |")
    L.append("|---|---:|---:|---:|---:|")
    for a in arms:
        cells=[("%+.2f"%means[(m,a)]) if (m,a) in means else "—" for m in MODELS]
        L.append("| %s | %s | %s | %s | **%+.2f** |"%(DISPLAY_ALIAS.get(a,a),cells[0],cells[1],cells[2],overall(a)))
    L.append("")

def main():
    now=datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    L=[]
    L.append("# V7 main-rank-volume 结果记录（v7.md）")
    L.append("")
    L.append("实验身份：configs/rags/main_v7_curve4.json（v7-seeded-quarter-pools-20260923 + 曲线协议）；run=%s"%R)
    L.append("池协议：可行池65136/每seed精确16284，源+REF CRB 0.5%/0.5%/0.5%，20臂，零BP选择，rolling resume。正式结果仅 seed0/1。")
    L.append("硬件：双 RTX PRO 6000 96G。MASE：Chronos 取 test_curve 3.0；其余取最终 epoch；逐 cell 曲线在 seeds/seedN/<model>/ 与 cells json。")
    L.append("")
    L.append(ALIAS_NOTE)
    L.append("")
    L.append("## 协议修订记录")
    for a in AMEND: L.append("- %s"%a)
    L.append("")
    L.append("## 总进度（seed0/1 为正式口径；Timer 的 seed2 为用户取消前的历史存档，见文末附录，不计入正式完成度）")
    L.append("")
    L.append("| 模型 | seed0 | seed1 | 小计 | 目标(2seed) |")
    L.append("|---|---:|---:|---:|---:|")
    total=0
    for m in MODELS:
        row=[]; t=0
        for s in (0,1):
            ns=[f for f in glob.glob(R+"/seeds/seed%d/%s/cells/*.json"%(s,m)) if json.load(open(f)).get("task") in TASKS]
            row.append(len(ns)); t+=len(ns)
        extra=len([f for f in glob.glob(R+"/seeds/seed2/%s/cells/*.json"%m) if json.load(open(f)).get("task") in TASKS])
        total+=t
        note="（另有 seed2 存档 %d）"%extra if extra else ""
        L.append("| %s | %d | %d | %d | %d |%s"%(m,row[0],row[1],t,PER_MODEL_GOAL,note))
    L.append("| **合计** | | | **%d** | **%d** |"%(total,TOTAL_GOAL))
    L.append("")
    L.append("- 每 seed 目标 %d/模型；全矩阵 2seed×%d×3模型=%d。巡检时间：%s（北京时间）。"%(PER_SEED_GOAL,PER_SEED_GOAL,TOTAL_GOAL,now))
    L.append("")
    main_data=load_cells(R+"/seeds/seed*/*/cells/*.json", taskfilter=TASKS, seeds=FORMAL_SEEDS)
    agg_table(L, main_data, "主实验·七任务", "n=七任务×2seed")
    ext_data=load_cells(R+"/extensions/*/seeds/seed*/*/cells/*.json", seeds=FORMAL_SEEDS)
    if ext_data:
        agg_table(L, ext_data, "扩展探索·三新任务", "n=三任务×2seed；探索性诊断，非主验证集")
    shape_data=load_cells(R+"/shape_backfill_20260925/seeds/seed*/*/cells/*.json", seeds=FORMAL_SEEDS)
    if shape_data:
        agg_table(L, shape_data, "shape 补做·中点+斜率臂", "n=十任务×2seed（进行中则按已完成计）")
    # seed2 存档（仅 TimerXL，文末附录，不进主表/不在文首披露）
    s2=load_cells(R+"/seeds/seed2/*/cells/*.json", taskfilter=TASKS)
    if s2:
        L.append("---")
        L.append("")
        L.append("## 附录：seed2 存档结果（用户已取消 seed2，仅 TimerXL 在取消前完成；此处仅留档，不构成正式结果，不参与任何总结）")
        L.append("")
        agg_table(L, s2, "TimerXL seed2 存档", "n=七任务×1seed（存档口径）")
    out="\n".join(L)+"\n"
    open("/root/TS3/docs/results/v7.md","w").write(out)
    print("v7.md written: %d chars, formal_cells=%d"%(len(out),total))
main()
