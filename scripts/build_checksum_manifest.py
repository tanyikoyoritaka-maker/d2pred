#!/usr/bin/env python3
import hashlib
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]
rows=[]
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'checksum_manifest.csv' and '__pycache__' not in path.parts:
        rows.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
pd.DataFrame(rows).to_csv(root/'checksum_manifest.csv',index=False)
print(f'wrote {len(rows)} checksums')
