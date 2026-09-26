"""Process leases for independent V7 training tasks, never for asset writers."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json


@contextmanager
def model_lease(run, student, phase, *, parallel_train=False):
    if parallel_train and phase != 'train':
        raise ValueError('Shared model admission is training-only')
    with (Path(run)/f'{student}.lock').open('a+') as lock:
        mode = fcntl.LOCK_SH if parallel_train else fcntl.LOCK_EX
        # Legacy exclusive workers and asset writers block new shared workers.
        fcntl.flock(lock, mode | fcntl.LOCK_NB)
        yield


def train_once(run, student, seed, task, marker, identity, compatible, execute, save):
    """Lease whole task (all arms), recheck after blocking; no polling."""
    path = Path(run)/'training_locks'/student/f'seed{seed}'/f'{task}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker=Path(marker)
        if marker.exists():
            row=json.loads(marker.read_text())
            if not row.get('complete') or not compatible(row.get('identity'), identity):
                raise ValueError('Existing training completion differs')
            if row['identity']==identity:
                return False
        execute()
        save(marker, dict(identity=identity, complete=True))
        return True
