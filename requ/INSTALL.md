# 一键安装

在项目根目录执行 `bash _install.sh`，默认安装并验证全部六个环境：
ts3、ts3-timesfm、ts3-timemoe、ts3-sundial、ts3-timerxl、time。
前置要求：Linux、conda、git、可访问包索引和 GitHub。
AutoDL 的 GitHub 下载自动在子进程中启用 /etc/network_turbo，
不把该代理传给国内包镜像。无需手工 source；如果当前 shell 已启用代理且
国内包下载缓慢，可先 unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy。

```bash
bash _install.sh
bash _install.sh --env timemoe
bash _install.sh --verify-only
bash _install.sh --env main --verify-only
```

scripts/_install_<环境>.sh 支持单独安装，环境参数为 main、timesfm、
timemoe、sundial、timerxl、time。重复的 scripts/_install.sh、scripts/install_all.sh、
scripts/install_time.sh 已删除，统一使用根目录 _install.sh。
安装中断后可重跑，复用已有环境，不删除或迁移环境。

依赖唯一维护位置是 requ/：requirements.txt 对应主环境，
<环境>-requirements.txt 对应隔离环境，native-selection.txt 为学生环境
共用依赖。旧根目录 environment.*.yml 已移除，避免两套版本漂移。
统一 Python 3.10、PyTorch 2.7.1 CUDA 12.8。TIME 的旧 CUDA 12.6
传递依赖改由 CUDA 12.8 PyTorch 解析。

主环境 transformers 4.57.1；Time-MoE/Sundial/Timer-XL 需要 4.40.1；
TimesFM 需要 5.3.0；TIME 保留旧 numpy/datasets/pyarrow 组合。
五个隔离环境分别保留，供运行配置按现有名称调用。
third_parties.lock、time.third_parties.lock 固定可安装上游源码；
model_integrations.lock 固定模型版本及 Time-MoE 源码。
baselines.lock 是算法审计依据，不将各上游整套依赖混装。

默认新环境、下载源码、pip/conda 缓存、解包临时文件、日志和 pip freeze
快照放 /root/autodl-tmp/ts3；其他平台用 TS3_INSTALL_ROOT 指定数据盘。
安装前检查剩余空间。已有环境原地复用，新源码通过 third_parties 下软链接接入。
PYPI_INDEX_URL、PYTORCH_INDEX_URL 可更换下载源。

验收运行 pip check、模块导入及可用 GPU 上的 CUDA 小型运算；
无 GPU 时明确报告跳过 GPU 验证。不下载数据或模型，不启动训练。
TIME 安装不要求检查点；可通过 CHECKPOINT 显式增加模型加载验证。
安装后用 conda activate ts3 等激活对应环境。
