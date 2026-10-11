"""Generate ACAll Unit 12 mixed ablation RUL case figure."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/acall_preprocess_ablation_final/seed_ensemble_predictions_with_intervals.csv'
OUT=ROOT/'figures/acall_ablation_rul_cases'
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(SRC); d=d[(d.unit==12)&d.model.isin(['d2pred_expert_exact_v1','ablation_preprocess_no_edc','ablation_preprocess_no_decomposition'])]
styles={'d2pred_expert_exact_v1':('Full model','#2ca02c','-'),'ablation_preprocess_no_edc':('Without EDC','#d62728','--'),'ablation_preprocess_no_decomposition':('Without decomposition','#9467bd',':')}
fig,ax=plt.subplots(figsize=(8.2,5.3))
for model,(label,color,ls) in styles.items():
 p=d[d.model.eq(model)].sort_values('endpoint').drop_duplicates('endpoint')
 x=(p.endpoint-p.endpoint.min())/max(p.endpoint.max()-p.endpoint.min(),1.0); initial=float(p.true_rul_cycles.iloc[0]); pred=p.ensemble_prediction.to_numpy(float)/initial
 if model=='d2pred_expert_exact_v1': ax.plot(x,1-x,color='#1f77b4',lw=1.8,label='Original RUL (slope -1)')
 ax.plot(x,pred,color=color,lw=1.5,ls=ls,label=label)
ax.set_title('ACAll frozen mixed ablation case: anonymized Unit 12',fontweight='bold'); ax.set_xlabel('Normalized operation cycle'); ax.set_ylabel('Normalized RUL'); ax.set_xlim(0,1); ax.grid(alpha=.3); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'acall_unit12_mixed_ablation_rul.pdf',bbox_inches='tight'); fig.savefig(OUT/'acall_unit12_mixed_ablation_rul.png',dpi=300,bbox_inches='tight'); plt.close(fig)
