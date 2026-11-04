#!/usr/bin/env python3
"""Deterministic SHA-256 inventory and read-only verification."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.venv', '.venv-linux'}
EXCLUDED_SUFFIXES = {'.pyc', '.log', '.tmp', '.swp'}
INVALID_LEGACY_PATHS = {
    'results/controlled_ablation_expert_exact_v3',
    'figures/controlled_ablation_v1',
}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def build(root=ROOT, output=None):
    root=Path(root).resolve();output=Path(output or root/'checksum_manifest.csv').resolve()
    excluded_paths=set()
    if root==ROOT:
        state=root/'results/controlled_ablation_candidate/execution_status.json'
        complete=state.exists() and json.loads(state.read_text()).get('status')=='verified_complete'
        if not complete:
            excluded_paths={'results/controlled_ablation_candidate',
                'manifests/active_experiment_registry.json','manifests/final_submission_audit.json',
                'manifests/controlled_ablation_preexecution_audit.json'}
        scope={'scope':'completed_release' if complete else 'post_ablation_contamination_cleanup',
               'excluded_paths':sorted(excluded_paths | INVALID_LEGACY_PATHS),'excluded_parts':sorted(EXCLUDED_PARTS),
               'excluded_suffixes':sorted(EXCLUDED_SUFFIXES),
               'candidate_integrity':'active materials only; INVALID_LEGACY_ABLATION v3 outputs are archived separately'}
        (root/'checksum_scope.json').write_text(json.dumps(scope,indent=2)+'\n')
    rows=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.resolve()==output or any(x in EXCLUDED_PARTS for x in path.relative_to(root).parts):continue
        if path.suffix in EXCLUDED_SUFFIXES:continue
        relative=str(path.relative_to(root))
        if any(relative==item or relative.startswith(item+'/') for item in excluded_paths):continue
        if any(relative==item or relative.startswith(item+'/') for item in INVALID_LEGACY_PATHS):continue
        rows.append(dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','sha256','bytes'],lineterminator='\n');w.writeheader();w.writerows(rows)
    return len(rows)

def verify(root=ROOT, manifest=None):
    root=Path(root).resolve();manifest=Path(manifest or root/'checksum_manifest.csv')
    with manifest.open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        path=(root/row['path']).resolve()
        if root not in path.parents or not path.is_file():raise RuntimeError(f'Missing/invalid path: {path}')
        if path.stat().st_size!=int(row['bytes']) or sha(path)!=row['sha256']:raise RuntimeError(f'Hash mismatch: {path}')
    return len(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(('verified' if a.verify else 'wrote'),(verify if a.verify else build)(a.root),'checksums')
if __name__=='__main__':main()
