from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/final_ac_nested/seed_ensemble_predictions_with_intervals.csv'
OUT=ROOT/'figures/final_acsim_cases'; OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(SRC); d=d[(d.model=='d2pred_expert_exact_v1')&(d.group=='ACSim')].copy(); units=sorted(d.unit.unique())
rows=[]
with PdfPages(OUT/'final_ac_nested_acsim_all_cases.pdf') as pdf:
 for u in units:
  p=d[d.unit.eq(u)].sort_values('endpoint').drop_duplicates('endpoint'); x=(p.endpoint-p.endpoint.min())/max(p.endpoint.max()-p.endpoint.min(),1.0); init=max(float(p.true_rul_cycles.iloc[0]),1.0); true=1-x.to_numpy(float); pred=p.ensemble_prediction.to_numpy(float)/init; ae=np.abs(pred-true)
  fig,ax=plt.subplots(figsize=(7.5,4.8)); ax.plot(x,true,color='#1f77b4',lw=1.8,label='Original RUL (slope -1)'); ax.plot(x,pred,color='#ff7f0e',ls='--',lw=1.5,label='Predicted RUL'); ax.set_title(f'final_ac_nested ACSim: anonymized Unit {int(u)}',fontweight='bold'); ax.set_xlabel('Normalized operation cycle'); ax.set_ylabel('Normalized RUL'); ax.set_xlim(0,1); ax.grid(alpha=.3); ax.legend(frameon=False); fig.tight_layout(); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
  rows.extend({'unit':int(u),'endpoint':float(e),'original_rul_normalized':float(t),'predicted_rul_normalized':float(pr),'vertical_distance_percent':float(100*(pr-t))} for e,t,pr in zip(p.endpoint,true,pred))
pd.DataFrame(rows).to_csv(OUT/'final_acsim_case_source_data.csv',index=False)
q=d.assign(abs_error=(d.ensemble_prediction-d.true_rul_cycles).abs()).groupby('unit').agg(MAE=('abs_error','mean'),initial_RUL=('true_rul_cycles','first'),n=('abs_error','size')).reset_index().sort_values('MAE'); q.to_csv(OUT/'final_acsim_unit_metrics.csv',index=False)
fig,axes=plt.subplots(len(units),1,figsize=(8,max(2*len(units),8)),sharex=True,sharey=True,squeeze=False)
for ax,u in zip(axes.ravel(),units):
 p=d[d.unit.eq(u)].sort_values('endpoint').drop_duplicates('endpoint'); x=(p.endpoint-p.endpoint.min())/max(p.endpoint.max()-p.endpoint.min(),1.0); init=max(float(p.true_rul_cycles.iloc[0]),1.0); ax.plot(x,1-x.to_numpy(float),color='#1f77b4',lw=.7); ax.plot(x,p.ensemble_prediction.to_numpy(float)/init,color='#ff7f0e',ls='--',lw=.7); ax.set_title(f'Unit {int(u)}',fontsize=8); ax.grid(alpha=.2); ax.set_xlim(0,1)
fig.suptitle('All final_ac_nested ACSim Full-model cases',y=.998,fontweight='bold'); fig.tight_layout(rect=(0,0,1,.995)); fig.savefig(OUT/'final_acsim_all_cases_contact_sheet.png',dpi=220,bbox_inches='tight'); plt.close(fig)
print({'units':len(units),'pdf':str(OUT/'final_ac_nested_acsim_all_cases.pdf')})
