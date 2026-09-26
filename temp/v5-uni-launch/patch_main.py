from pathlib import Path
p = Path('docs/MAIN.md')
s = p.read_text()
old = '正式 V5 现运行于 `/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918`，tmux 会话 `main-rank-volume`。'
new = ('正式 V5 已于 2026-09-20 全部完成（666/666 cells，seeds 0/1/2 三轮屏障通过，supervisor exit_code=0），'
       '产物位于 `/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918`。')
assert old in s
s = s.replace(old, new)
p.write_text(s)
print('stamped')
