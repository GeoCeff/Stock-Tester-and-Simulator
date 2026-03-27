import json
from pathlib import Path


def save_workspace(filepath, state):
    p = Path(filepath)
    p.write_text(json.dumps(state, default=str, indent=2), encoding='utf-8')


def load_workspace(filepath):
    p = Path(filepath)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))
