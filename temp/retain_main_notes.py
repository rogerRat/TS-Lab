"""Retain analysis notes while the already-running pre-notes supervisor refreshes."""
import os,time
from pathlib import Path
document=Path('/root/TS3/docs/MAIN-v2.md')
notes=document.with_name('MAIN-v2-notes.md')
marker='<!-- MAIN-v2 analysis notes: retained across supervisor refresh -->'
def append():
    stat=document.stat()
    content=document.read_text()
    if marker in content:return
    addition=notes.read_text().rstrip()
    temporary=document.with_name('MAIN-v2.md.notes.tmp')
    temporary.write_text(content.rstrip()+'\n\n'+addition+'\n')
    if document.stat().st_mtime_ns==stat.st_mtime_ns:
        temporary.replace(document)
    else:
        temporary.unlink()
while Path('/proc/403347').exists():
    append()
    time.sleep(3)
append()
