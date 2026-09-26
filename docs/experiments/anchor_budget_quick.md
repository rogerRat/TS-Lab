# 五条已完成 Ref 的快速预算结果

2026-09-14，在不中断原梯度扫描（PID275956）的情况下，以CPU只读缓存测试。
使用前五任务各一条rank0轨迹，共105086个合法窗口；原始BF16梯度来自同一缓存，
每条加载一次到内存并按physical_start精确查表，不计算任何新梯度。
内存回放与直接逐行回调的CRB锚点、块均值及窗口身份一致性测试通过。

先固定error_density=.02、split_density=0，扫描总密度6/10/14/18/22/26/30%，
再补测15/16/17%。模型、C512/H64/stride1及非预算CRB参数均未变化。

| 任务 | 总15% | 总16% | 总17% |
| --- | ---: | ---: | ---: |
| NEM demand | .944415 | .946162 | .947739 |
| Azure CPU | .953491 | .955398 | .956526 |
| Darwin water | .899471 | .901530 | .904369 |
| Finland | .942308 | .944136 | .946299 |
| EWELD | .951777 | .953004 | .954138 |

指标为生产块常值对逐窗真值的余弦均值，负值保留、丢弃/零范数窗口按0计入。
总15%已经非常接近0.9，但Darwin仍严格不通过；本轮这条固定分配路径上的最低已测
通过档为cover=.14/error=.02/split=0，即总16%，实际16814/105086锚点。
不声称这是全部预算分配的全局最小值或13条完整样本的最终参数。
split没有触发；此处取0不说明未来任务可以不保留分裂预算。
16%候选已加入完整样本网格以供复核，最终参数仍待用户判断。

## 用户追加 8% / 5% / 3%

同五条rank0轨迹只读缓存，结果为：NEM .939749、Azure .949251、Darwin .885647、
Finland .932601、EWELD .945255。三项名义总预算16%，但所有轨迹的split_used均为0，
实际锚点总数13660/105086，约13%。未消耗的3%分裂预算不会自动转移给覆盖/误差阶段。
Darwin严格不通过0.9，且低于13/2/0的.899471；由于实际耗用也从15%降到13%，
不能仅凭此对比归因于覆盖/误差配比。原始产物 `quick-rank0-8-5-3/`。
运行参数为 `--covers .08 --error .05 --split .03`，扫描进程275956持续运行，未重启。

## 用户追加 4% / 2% / 2%

同五条rank0轨迹只读缓存，生产均值NEM .904387、Azure .920074、Darwin .838817、
Finland .873541、EWELD .912908。名义总预算8%，split均未触发，实际使用约6%锚点。
Darwin和Finland低于.9。原始产物 `quick-rank0-4-2-2/`，参数
`--covers .04 --error .02 --split .02`；梯度扫描保持运行。

## 用户追加 2% / 1% / 1%

同五条rank0轨迹只读缓存，生产均值NEM .839704、Azure .877512、Darwin .789607、
Finland .787061、EWELD .860128。名义总预算4%，split均未触发，实际使用约3%锚点，
五项均低于.9。原始产物 `quick-rank0-2-1-1/`，参数
`--covers .02 --error .01 --split .01`；梯度扫描保持运行。

## 用户指定 222 / 200 / 600 对照（本组允许零预算）

| 样本 | 2/2/2 | 2/0/0 | 6/0/0 |
| --- | ---: | ---: | ---: |
| NEM | .868850 | .783891 | .904574 |
| Azure | .897249 | .845015 | .919520 |
| Darwin | .808375 | .760131 | .851551 |
| Finland | .824082 | .720634 | .874113 |
| EWELD | .881154 | .820242 | .915755 |

指标仍为生产逐窗块常值余弦均值。实际锚点比例分别约4%、2%、6%；
222中的2%分裂预算未触发，不能将222和600称为实际等成本对照。
本组600每项最高，但Darwin和Finland仍低于.9。
原始产物quick-rank0-2-2-2及quick-rank0-cover-only-2-6；扫描PID275956未中断。

原始产物在数据盘 `outputs/anchor-budget-20260914/quick-rank0-30pct/`
与 `quick-rank0-refine/`，均含grid_rows.json、CSV和逐轨迹明细，所有缓存保留。

复现：
```
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
PYTHONPATH=src /root/miniconda3/envs/time/bin/python -m rags.experiments.anchor_budget_quick \
 --run-root /root/autodl-tmp/TS3/outputs/anchor-budget-20260914 \
 --out /root/autodl-tmp/TS3/outputs/anchor-budget-20260914/quick-rank0-refine \
 --covers .13 .14 .15
```
