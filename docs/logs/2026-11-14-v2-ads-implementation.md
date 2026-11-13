# v2-ads 算法接入与验证记录

- 检查时间：2026-11-14 01:02 +0800
- 实验代号：v2-ads
- 批次：implementation-validation
- 状态：开发完成；正式实验未启动

## 操作

1. 阅读 docs/ads.md，按真实梯度锚点实现 coarse anchors、全局 Top-K、候选区间扩展、最长间隔细算、重匹配、source 列聚合与有限线性衰减。
2. 新增 src/rage/ads.py，提供 ADSConfig、ADSPipeline、梯度计数、零范数统计、目标密度 K 搜索及 source 采样概率。
3. 在 src/rage/experiments/arms.py 添加 ads_select，将概率映射到主表合法窗口并支持调用方指定是否放回。
4. 在主表增加 --selector ads 和全部 --ads-* 参数；统一运行器从 YAML/JSON 读取参数并允许 CLI 覆盖。提供 configs/rage/ads.yaml 与 scripts/exp/v2-ads/run_ads.sh。
5. 运行测试：
   /root/autodl-tmp/rage/envs/rage/bin/python -m pytest -q tests/test_ads.py tests/test_rage_run.py tests/test_selection_execution.py tests/test_region_delivery.py tests/test_package_layout.py
6. 测试结果：53 passed，退出码 0。代码编译检查通过。

## 文件

- 算法：src/rage/ads.py
- 选择器和运行配置：src/rage/experiments/arms.py、src/rage/experiments/main_table.py、src/rage/experiments/run.py
- 开发配置和脚本：configs/rage/ads.yaml、scripts/exp/v2-ads/run_ads.sh
- 测试：tests/test_ads.py
- 实验说明与结果：docs/exp/v2-ads/exp.md、docs/exp/v2-ads/results.md

## 未解决事项

- 正式实验的模型、数据范围、q 参数、K/rho/目标密度模式、采样数量与是否放回、训练日程、种子和评估任务仍待冻结。
- 当前配置是 ETT_1H 小规模选择阶段开发模板，selection_only: true，不代表正式实验参数。
- 没有启动训练、GPU 实验或正式评估；没有正式结果可记录。
