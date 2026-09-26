# 可追加的任务队列

队列独立于模型和 Agent。使用 Python 3.10+ 标准库、exp 本地 SQLite 和文件锁，
无需数据库服务或第三方 Python 包。Linux 是当前执行端；WSL/Mac 是 SSH 客户端。
源码为 `src/runtime/task_queue.py`，命令入口为 `scripts/queue.py`。

## 工作方式

```
WSL / Mac --主动 SSH--> exp 本地队列 --串行执行--> 命令 / 脚本 / 指令序列
                           |
                           +--needs_agent <--WSL 主动查询、领取、回传结果
```

WSL 和 Mac 不需要固定公网 IP、入站端口或反向 SSH。两端离线不影响 exp 上的普通任务。
遇到 Agent 任务而没有在线消费者时，队列保留等待状态，不会跳过这个前置步骤。
exp 自身关机则不能运行计算；重新启动 worker 后核实持久状态，不自动重跑结果不明的任务。

任务按成功提交数据库的顺序 FIFO 执行。运行中追加不打断当前进程；空队列的 worker
保持待命。默认每秒检查一次，前一步完成后通常一秒内接续，不依赖估算结束时间。
脚本、命令序列和 Agent 提示在追加时存成快照，后续修改原文件不改变已排队内容。

## 最短用法（在 exp）

以下 `QUEUE_DB` 是任务专用变量，改成实际数据盘路径；工作目录按你的项目调整。

```bash
cd /root/TS3
QUEUE_DB=/root/autodl-tmp/TS3/outputs/task-queue/queue.sqlite
python scripts/queue.py --db "$QUEUE_DB" init
python scripts/queue.py --db "$QUEUE_DB" work
```

`work` 为前台常驻命令，建议放在 tmux 中。另一个终端可随时操作同一个数据库：

```bash
python scripts/queue.py --db "$QUEUE_DB" append --title "计算A" --cwd /root/TS3 -- /path/to/python compute_a.py
python scripts/queue.py --db "$QUEUE_DB" append --title "计算B" --cwd /root/TS3 -- /path/to/python compute_b.py
python scripts/queue.py --db "$QUEUE_DB" status
python scripts/queue.py --db "$QUEUE_DB" logs 1 --lines 30
```

直接命令使用 argv，`$`、反引号、空格等保留为参数，不额外经 shell 展开。
`--env NAME=value` 可重复设置环境。GPU任务用各自环境的 Python 绝对路径。

脚本可以包含多条命令，追加后用 `bash -euo pipefail -c` 执行（失败即停）：

```bash
python scripts/queue.py --db "$QUEUE_DB" append --title "预处理与计算" --cwd /root/TS3 --script /path/to/job.sh
```

也可以准备 JSON 命令序列，每项为一组 argv；任一步失败都不继续下一步：

```json
[["/path/to/python", "prepare.py"], ["/path/to/python", "compute.py"]]
```

使用 `append --title 名称 --cwd /root/TS3 --sequence steps.json` 追加。
任务命令必须在前台等待实际计算结束，不能使用 `nohup ... &` 或只提交作业便退出；
否则退出码只代表提交动作。多个独立客户端同时追加时，以数据库提交次序为准。

## 已经在计算，如何排后续任务

新队列先观察现有进程，后续任务追加在它之后，不终止或重启原进程。例如：

```bash
python scripts/queue.py --db "$QUEUE_DB" attach --title "等待现有计算" \
  --pid 12345 --status-file /data/run/status.json \
  --equals '{"event":"COMPLETE","completed":1000}' \
  --error-file /data/run/error.json --stale-seconds 900
python scripts/queue.py --db "$QUEUE_DB" append --title "接下来的计算" --cwd /root/TS3 -- /path/to/python next.py
```

必须同时满足结果字段和原进程已退出，才把观察任务标为成功。PID身份包括系统启动ID
及进程启动时刻，避免PID复用。没有成功证据的进程消失会失败，不能猜测退出码。
如果只能观察产物，可明确用 `--file-only` 代替 `--pid`；它无法判断产物缺失是尚未完成
还是进程死亡。状态文件必须由生产者原子更新；不要引用上一次运行遗留的成功文件。

命令退出码0也可以追加产物检查：`append ... --success-json rule.json -- COMMAND...`，
其中 rule.json 为 `{"path":"/data/result.json","equals":{"complete":true}}`。
科学指标是否达标由明确的验证脚本决定，不让队列猜测日志含义。

## 状态、失败和恢复

- `status` 给人阅读；`list`、`get ID`、`events` 输出 JSON 供脚本使用。
- `pause` 暂停后续调度，不中断当前命令；`resume` 恢复。
- 命令非0退出、产物检查失败：标 `failed` 并阻断后续任务。
- 异常退出无法判明结果：标 `needs_attention`，保留现场，不自动重跑。
- 修复后 `retry ID`，或明确不需要该任务时 `skip ID`。重试从该任务第一步开始，
  计算脚本应自己支持断点恢复；队列不承诺任意外部副作用的 exactly-once。
- 活 runner 或仍持有执行锁的子进程存在时，不允许 retry/skip，避免重复计算。
- worker退出不主动杀死独立runner；重开 `work` 会识别它，不重复启动。
- 若 runner 被杀但子进程还在运行，保留执行锁并要求人工检查。对于自行后台化、
  主动关闭继承文件描述符的外部程序，不能依赖这个保护；必须用前台包装脚本。

## 从 WSL / Mac 操作

```bash
python3 scripts/queue_remote.py --host exp --db /data/queue.sqlite status
python3 scripts/queue_remote.py --host exp --db /data/queue.sqlite append \
  --title "新计算" --cwd /root/TS3 -- /path/to/python next.py
```

`exp` 使用本机SSH配置；也可配置新的主机别名。默认远端repo和Python可用 `--repo`、
`--python` 覆盖。这里的脚本/命令序列/提示文件路径指exp上的文件，先上传再追加。
SQLite始终在exp本地打开；**不要通过SSHFS/NFS直接打开数据库**。

## 可选 Agent 消费者

普通任务完全不启动Agent。显式追加Agent任务时：

```bash
# exp 上的提示文件应写清工作范围、产物和判定要求
python scripts/queue.py --db "$QUEUE_DB" append-agent --title "检查结果并提出修复" \
  --prompt-file /data/agent-task.txt --cwd /home/rogers/Projects
```

`--cwd` 是 Agent 所在WSL的路径；`--doc PATH` 可以重复附上参考文档路径。
WSL运行以下可选消费者（源码目录要在本地或可读挂载中）：

```bash
PYTHONPATH=src python3 -m runtime.queue_agent \
  --host exp --db /data/queue.sqlite --remote-cwd /root/TS3 \
  --remote-python /root/miniconda3/envs/ts3/bin/python \
  --cwd /home/rogers/Projects --state-dir /home/rogers/queue-agent-state
```

每30秒主动查询，`--once` 单次执行，`--interval` 调整周期。远端原子claim防止WSL/Mac
两个消费者重复领取。桥以 `codex exec` 建立独立会话，继承本机模型与登录配置，
不向当前聊天或其他tmux终端注入文本；这不是保证重新唤醒当前这一条UI会话。
消费者按当前授权配置为非交互、可写环境，只有主动加入此队列的Agent任务才会触发它。

桥记录本地报告、JSONL事件和完成回执。网络断开会重试查询或回执；执行过程中崩溃
不会自动重复调用模型。若停在claim中间，需要先核实本地Agent进程和报告：用
`get ID` 取得claim，再以 `finish-agent ID --claim TOKEN --error "已检查中断"` 终结旧尝试，
保留旧本地目录并给恢复后的消费者使用新的 `--state-dir`，之后才 `retry ID`。
若有完整成功报告，可回传 `--success`；相同claim、相同结果的回执可以安全重复。
Agent完成只代表一次调用成功产出报告，后续实验仍须独立结果检查或由人确认参数。

Codex调用接口见[官方非交互执行说明](https://learn.chatgpt.com/docs/non-interactive-mode)。
本轮桥接已通过替身执行测试，没有自动登录、调用模型、配置反向SSH或开启常驻Agent。

## 存储与本次范围

SQLite只存任务快照、状态和事件；日志在同级 `<数据库名>.jobs/<ID>/`，每次尝试独立保留。
源脚本保留在仓库；模型、梯度与实验结果仍由原计算脚本管理，队列不移动或清理它们。
所有状态默认保留，不计算数据hash、不提交数据库或运行日志。备份活SQLite用SQLite备份
接口，不能只拷贝主db而漏掉WAL；也可在确认无writer后整体备份目录。

本轮仅开发和验证可复用工具：没有接管当前探测、没有排入主实验。探测参数由用户亲自确认。

### 当前 exp 部署

已启动空队列，远端 tmux 会话 `task-queue`。数据库为
`/root/autodl-tmp/TS3/outputs/task-queue/queue.sqlite`，执行环境为
`/root/miniconda3/envs/ts3/bin/python`。没有排入任何实际实验，也没有启动Agent消费者。

在WSL的项目目录执行即可查看或追加（目录为 `/home/rogers/Projects/exp0914/TS3`）：

```bash
python3 scripts/queue_remote.py --db /root/autodl-tmp/TS3/outputs/task-queue/queue.sqlite status
python3 scripts/queue_remote.py --db /root/autodl-tmp/TS3/outputs/task-queue/queue.sqlite append \
  --title "下一项计算" --cwd /root/TS3 -- /path/to/python /path/to/next.py
```

第二条中的Python和脚本必须改为实际路径；不需要停止worker。关闭SSH客户端不影响远端队列。
验收为22项针对性测试通过：真实子进程运行中追加、并发追加、SIGKILL恢复、失败阻断、
命令行独立启动、脚本快照、产物门槛，以及用替身测试Agent领取/回执/中断。
Agent桥未进行真实模型调用测试。
