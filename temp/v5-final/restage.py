from pathlib import Path
import json,subprocess
paths=json.loads(Path('temp/v5-final/commit-paths.json').read_text())['paths']
subprocess.run(['git','add','--',*[name for name in paths if Path(name).exists()]],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
print(subprocess.check_output(['git','diff','--cached','--stat'],text=True))
print('Unstaged changes:')
print(subprocess.check_output(['git','diff','--name-only'],text=True))
