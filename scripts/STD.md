scripts/ 只保留可复用的工具和 CLI；一次性实验脚本不保留在此目录。
实验编排、诊断、基准及验收工具统一放 scripts/exp/，配置和输出路径通过参数指定。
完整安装入口为根目录 _install.sh，单环境入口为 scripts/_install_<环境>.sh。
已替代的入口直接删除，不保留重复转发脚本。

常用入口：

- python -m scripts.assets：数据、模型、梯度准备。
- python -m scripts.exp.rage：run、main_table、curves。
- python -m scripts.exp.full_pool --config CONFIG.json --out OUTPUT：全池运行/恢复；支持 --dry-run。
- python -m scripts.exp.bench_gradient_assets：梯度批量基准。
- python -m scripts.exp.verify_mixed_precision：混合精度与检查点续跑验收。

2026-09-14 清理了旧 CU/SR20/vote 临时驱动、固定日期批跑、.orig 残留、
重复安装/运行入口及无效 tmux_shut.sh。保留参数化的模型、基准、smoke、
部署检查、数据下载、数据盘链接和容器监控工具。
清理前完整脚本备份（包含未跟踪的 temp/.orig）：
/root/autodl-tmp/rage/maintenance/scripts-before-cleanup-20260914-1752.tar.gz。
已跟踪文件亦可通过 Git 历史恢复。历史实验文档中的旧命令仅作记录。
