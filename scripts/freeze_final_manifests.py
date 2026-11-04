#!/usr/bin/env python3
"""Validate existing frozen inputs without rewriting device/fold manifests."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_ac_nested_gru_rerun import load_ac,validate_predefined_folds
from scripts.build_checksum_manifest import sha

def main():
    raw=pd.read_csv(ROOT/'data/industrial/DI_df.csv')
    groups=pd.read_csv(ROOT/'manifests/group_manifest.csv')
    expected=set(range(1,39))-{1,3,11,35,38}
    assert set(groups.unit)==expected and len(groups)==33
    assert groups.groupby('group').unit.nunique().to_dict()=={'ACSim':17,'ACCom':16}
    assert ((groups.lifetime<1000)==groups.group.eq('ACSim')).all()
    for unit,part in raw.groupby('unit'):
        part=part.sort_values('time_cycles')
        assert float(part.iloc[-1].RUL)==0
        assert ((part.RUL+part.time_cycles)==float(part.iloc[-1].time_cycles)).all()
    frame=load_ac(group_manifest=ROOT/'manifests/group_manifest.csv')
    validate_predefined_folds(frame,json.loads((ROOT/'manifests/folds.json').read_text()))
    print(json.dumps(dict(status='PASS',eligible_devices=33,group_counts={'ACSim':17,'ACCom':16},
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/industrial/DI_df.csv',ROOT/'manifests/group_manifest.csv',ROOT/'manifests/folds.json']}),indent=2))
if __name__=='__main__':main()
