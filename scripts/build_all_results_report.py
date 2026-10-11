from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'; OUT = ROOT / 'figures'; OUT.mkdir(exist_ok=True)
PDF = OUT / 'all_final_experiments_report.pdf'; PNG = OUT / 'all_final_experiments_dashboard.png'

def summary(path, label):
    f = path / 'summary_metrics.csv'
    if not f.exists(): return pd.DataFrame()
    d = pd.read_csv(f); d['experiment'] = label; return d

base = pd.concat([summary(RES/'final_ac_nested','AC comparison'), summary(RES/'acall_nested','ACAll comparison'), summary(RES/'public_benchmark_final','Public benchmark')], ignore_index=True)
abl = pd.concat([summary(RES/k,l) for k,l in [('acsim_preprocess_ablation_final','ACSim ablation'),('accom_preprocess_ablation_final','ACCom ablation'),('acall_preprocess_ablation_final','ACAll ablation')]], ignore_index=True)
sens=[]; pat=re.compile(r'^sensitivity_(acsim|accom|acall)_([a-z_]+)_(.+)$')
for p in sorted(RES.glob('sensitivity_*')):
    m=pat.match(p.name)
    if m and (p/'summary_metrics.csv').exists():
        d=pd.read_csv(p/'summary_metrics.csv')
        if len(d):
            r=d.iloc[0].to_dict(); r.update(dataset=m.group(1).upper(), factor=m.group(2), value=float(m.group(3))); sens.append(r)
sens=pd.DataFrame(sens)
protocol=json.load(open(ROOT/'manifests/complete_experiment_freeze_protocol_v1.json'))

with PdfPages(PDF) as pdf:
    fig,ax=plt.subplots(figsize=(16,9)); ax.axis('off'); ax.text(.03,.92,'D2Pred final experiment report',fontsize=22,weight='bold'); ax.text(.03,.84,'Verified comparison, preprocessing ablation and seven-factor sensitivity results',fontsize=12); ax.text(.03,.76,f'Sensitivity conditions complete: {len(sens)} / 72',fontsize=14); pdf.savefig(fig); fig.savefig(PNG,dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,10)); ax.axis('off'); ax.set_title('Frozen D2Pred method and AEP configuration',fontsize=18,weight='bold',pad=18)
    common=protocol['common_protocol']; aep=protocol['ablation_experiments']; rows=[
        ['Input / target','Causal DI prefix','RUL = endpoint cycle − current cycle'],
        ['Decomposition','DWC + EDC','EDC = DI − DWC; DWC from Wiener/STL trend fusion'],
        ['AEP causality','Prefix-only','No future samples used for change-point detection or token construction'],
        ['AEP token mode','Statistical descriptors','Each accepted residual segment becomes one token'],
        ['Raw residual descriptor dimension','12','Position, length, mean, std, min, max, slope, energy, area, residual error, jump'],
        ['Default sensitivity γ','2.5','Robust MAD/noise-dependent change-point threshold'],
        ['Minimum segment length','8 cycles','Causal event segment constraint'],
        ['Maximum change points','15','Top-down accepted change-point limit'],
        ['Maximum tokens','16','Most recent tokens retained; shorter lists zero-padded and masked'],
        ['Event encoder','3 masked self-attention blocks','4 attention heads; token projection 12→64→64'],
        ['Wear encoder','Unidirectional LSTM','Hidden size 64'],
        ['Fusion / output','Attention-guided DWC–EDC fusion','Softplus non-negative RUL output'],
        ['Joint loss','RUL + 0.1 gradual + 0.1 event + 0.1 trend-fusion','Training-only auxiliary supervision'],
        ['Evaluation','3 device folds × seeds 17/42/73','Training-only normalization; test once after validation selection'],
    ]
    tbl=ax.table(cellText=rows,colLabels=['Component','Frozen setting','Implementation / purpose'],loc='center',cellLoc='left',colWidths=[.24,.27,.49]); tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.55)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor('#1F4E78'); cell.set_text_props(color='white',weight='bold')
        elif r%2==0: cell.set_facecolor('#EAF2F8')
    pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,8)); ax.axis('off'); ax.set_title('AEP processing flow',fontsize=18,weight='bold'); flow='Observed DI prefix → causal DWC/EDC decomposition → EDC residual change-point proposal → robust threshold acceptance → 12-dimensional statistical token construction → recent-token truncation / zero-padding mask → masked self-attention → attention-guided fusion → Softplus RUL'; ax.text(.05,.52,flow,fontsize=16,wrap=True,va='center',ha='left',bbox=dict(boxstyle='round,pad=.8',facecolor='#EAF2F8',edgecolor='#4472C4')); ax.text(.05,.18,'AEP sensitivity factors evaluated separately: γ, minimum segment length, maximum change points, maximum tokens, prefix start, prefix stride and maximum prefixes/unit.',fontsize=11); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
    if len(base):
        base = base.copy()
        base['dataset'] = base['group'].replace({'ACSim':'ACSim','ACCom':'ACCom','ACAll':'ACAll','ims':'IMS','pronostia':'PRONOSTIA','xjtu_sy':'XJTU-SY'})
        for dataset, sub in base.groupby('dataset', sort=True):
            for metric,title in [('device_mean_RMSE_cycles','RMSE (cycles)'),('device_mean_MAE_cycles','MAE (cycles)'),('device_mean_P90AE_cycles','P90AE (cycles)')]:
                q=sub.sort_values(metric).copy(); labels=q.model.str.replace('d2pred_expert_exact_v1','Expert')
                fig,ax=plt.subplots(figsize=(14,7)); ax.bar(range(len(q)),q[metric]); ax.set_xticks(range(len(q))); ax.set_xticklabels(labels,rotation=65,ha='right',fontsize=8); ax.set_ylabel('cycles'); ax.set_title(f'{dataset} comparison — {title}'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    if len(abl):
        fig,ax=plt.subplots(figsize=(16,9));
        for exp,g in abl.groupby('experiment'):
            q=g.sort_values('device_mean_RMSE_cycles'); ax.plot(q.model.str.replace('d2pred_expert_exact_v1','Expert'),q.device_mean_RMSE_cycles,'o-',label=exp)
        ax.set_title('Preprocessing ablation RMSE'); ax.set_ylabel('cycles'); ax.tick_params(axis='x',rotation=70); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    for factor,g in sens.groupby('factor'):
        fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True); fig.suptitle('Sensitivity: '+factor)
        for ax,(ds,q) in zip(axes,g.groupby('dataset')):
            q=q.sort_values('value'); ax.plot(q.value,q.device_mean_RMSE_cycles,'o-'); ax.set_title(ds); ax.set_xlabel('registered value'); ax.grid(alpha=.25)
        axes[0].set_ylabel('RMSE (cycles)'); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    if len(abl):
        t=abl[['experiment','group','model','device_mean_RMSE_cycles','device_mean_MAE_cycles','device_mean_P90AE_cycles']].round(3); fig,ax=plt.subplots(figsize=(16,10)); ax.axis('off'); ax.set_title('Verified preprocessing ablation table'); ax.table(cellText=t.values,colLabels=t.columns,loc='center',cellLoc='center').scale(1,1.4); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
print(PDF); print(PNG)
