# v1-rage-ads 算法与文档更新

- 检查时间：2026-11-14（Asia/Shanghai）。
- 实验代号：v1-rage-ads。
- 批次：ads-api-q-single。
- 状态：算法参数接口与实验文档正在更新；测试未运行。
- 操作：将 ADS 从独立 q_coarse/q_fine 改为单一 q 与必填 eta；q=16、eta=1 派生两侧间隔32；默认 rho=0.2。
- 邻域规则：锚点邻域重叠仍取最大值；在实验文档补充 reference 锚点非均匀导致 Top 配对话语权偏置的说明。
- 重采样：增加紧凑锚点近邻概率描述和按该描述有放回抽样的函数，seed=0 由调用方配置。
- 缓存规则：仅按字段匹配复用，不进行哈希校验。
- 执行环境：远程 `tsl`，项目 `/root/TS-Lab`；Conda 环境 `rage-timerxl`。
- 产物：算法与文档变更留在项目源文件、`configs/` 与 `docs/`；没有启动梯度计算、模型训练或评估。
- 未完成项：新 API 测试待运行；正式八任务配置和专用 rolling runner 尚未实现。

- 测试结果：首次测试暴露了预算换算后粗锚点数量断言仍沿用旧双 q 数值，以及流式运行夹具未提供必填 eta；已修正测试数据。随后在远程 rage-timerxl 环境执行 `python -m pytest -q tests/test_ads.py`，23 项通过。
- 邻域行为按用户最后修正保持 max-overlap；测试验证两个重叠邻域不相加。
- 批次 `ads-api-q-single` 已完成，完成标记写入 outputs/selection/v1-rage-ads/ads-api-q-single/.complete。正式 rolling-eight-task 仍未启动。
