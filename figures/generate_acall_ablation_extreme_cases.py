from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/acall_preprocess_ablation_final/seed_ensemble_predictions_with_intervals.csv'
OUT=ROOT/'figures/acall_ablation_rul_cases'
VARIANTS={'d2pred_expert_exact_v1':'Full D$^2$Pred','ablation_preprocess_no_edc':'Without EDC','ablation_preprocess_no_decomposition':'Without decomposition'}

def norm(p):
 p=p.sort_values('endpoint').drop_duplicates('endpoint'); x=(p.endpoint-p.endpoint.min())/max(p.endpoint.max()-p.endpoint.min(),1.0); init=max(float(p.true_rul_cycles.iloc[0]),1.0); return x.to_numpy(float),1-x.to_numpy(float),p.ensemble_prediction.to_numpy(float)/init

def main():
 d=pd.read_csv(SRC); d=d[d.model.isin(VARIANTS)]
 units=[10,25] # smallest and largest unit-level MAE for Without decomposition
 with PdfPages(OUT/'acall_ablation_extreme_cases_unit10_unit25.pdf') as pdf:
  fig,axs=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey='row')
  for i,u in enumerate(units):
   for j,(m,label) in enumerate(VARIANTS.items()):
    ax=axs[i,j]; x,t,p=norm(d[(d.unit==u)&d.model.eq(m)])
    ax.plot(x,t,color='#1f77b4',lw=1.8,label='Original RUL (slope -1)')
    ax.plot(x,p,color='#ff7f0e',ls='--',lw=1.5,label='Predicted RUL')
    ax.set_title(label,fontweight='bold',fontsize=11); ax.set_xlim(0,1); ax.grid(alpha=.3)
    if i == 1: ax.set_xlabel('Normalized operation cycle')
    if j == 0: ax.set_ylabel(f'Unit {u}\nNormalized RUL')
  h,l=axs[0,0].get_legend_handles_labels()
  fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
  fig.suptitle('ACAll ablation comparison: full model and module removals',y=.995,fontweight='bold',fontsize=14)
  fig.tight_layout(rect=(0,0,1,.93),h_pad=2.0,w_pad=1.0)
  pdf.savefig(fig,bbox_inches='tight',pad_inches=.08); plt.close(fig)
 print('written',OUT/'acall_ablation_extreme_cases_unit10_unit25.pdf')

if __name__=='__main__': main()
