#!/usr/bin/env python3
"""Cache-first comparison for XJTU-SY, PRONOSTIA, IMS and NASA batteries.

This runner deliberately has no imports from or writes to AC/FD experiment paths.
Feature extraction, DI construction and DWC/EDC construction happen once before
model training. Dataset.__getitem__ only reads already materialized arrays.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import signal, stats
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'analysis'))
from run_fd_comparison_ac_ablation_main_independent import (make_tokens, resize, train_eval, fd_metric, rank_best)
from analysis.causal_decomposition import MissingHealthStateError

OUT=ROOT/'new_expriments/public_datasets_di_comparison'
DIR={n:OUT/n for n in ['scripts','protocols','diagnostics','di_cache','dwc_edc_cache','comparison_results','combined_tables','logs','checkpoints']}
DATASETS=['xjtu_sy','pronostia','ims','nasa_battery']
MODELS=['CNN','LSTM','BiLSTM','DAG_network','IMDSSN','PM2FN','Transformer','TCN_BiGRU','KGHM','ATCN','MDFA','SBi_Transformer','Attn_ResDSCNN','cpm_stats_tokens_edc_independent_fusion']
SEED=42; TOPK=14
PATHS={'xjtu_sy':ROOT/'XJTU-SY_Bearing_Datasets','pronostia':ROOT/'PHM_IEEE_2012/phm-ieee-2012-data-challenge-dataset-master','ims':ROOT/'IMSDataset/IMS','nasa_battery':ROOT/'BatteryDataSet/DataSet/BatteryAgingARC_unzipped'}

def jwrite(p,obj): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,default=str)+'\n')
def phash(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:16]
def natural(p): return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)',str(p))]
def safe_parquet(df,p,missing):
    try: df.to_parquet(p,index=False); return True
    except Exception as e: missing.append({'dataset':'','model':'','missing_item':str(p),'reason_if_known':f'Parquet unavailable: {e}','suggested_action':'Use adjacent CSV or install pyarrow'}); return False

def discover():
    rows=[]
    for name,p in PATHS.items():
        files=[x for x in p.rglob('*') if x.is_file()] if p.exists() else []
        size=sum(x.stat().st_size for x in files)
        ext=sorted({x.suffix.lower() or 'extensionless' for x in files})
        rows.append({'dataset_name':name,'discovered_path':str(p) if p.exists() else '', 'file_count':len(files),'size':size,'detected_format':','.join(ext[:12]),'status':'found' if files else 'missing','notes':''})
    jwrite(DIR['protocols']/'dataset_path_manifest.json',rows); return rows

def feats(x,fs=25600.0,prefix=''):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<8: return {prefix+k:np.nan for k in FEATURE_NAMES}
    mean=x.mean(); std=x.std(); rms=np.sqrt(np.mean(x*x)); peak=np.max(np.abs(x)); am=np.mean(np.abs(x)); eps=1e-12
    f=np.fft.rfftfreq(len(x),1/fs); power=np.abs(np.fft.rfft(x-x.mean()))**2; ps=power/(power.sum()+eps)
    env=np.abs(signal.hilbert(x-x.mean())); ep=np.abs(np.fft.rfft(env-env.mean()))**2
    vals=[mean,std,rms,peak,np.ptp(x),stats.skew(x),stats.kurtosis(x),peak/(rms+eps),peak/(am+eps),rms/(am+eps),peak/(np.mean(np.sqrt(np.abs(x)))**2+eps),np.sum(x*x),am,np.var(x),np.sum(f*ps),np.sum(f*ps),np.sqrt(np.sum(f*f*ps)),np.sum((f-np.sum(f*ps))**2*ps),np.sqrt(np.sum((f-np.sum(f*ps))**2*ps)),-np.sum(ps*np.log(ps+eps)),np.sqrt(power.max()),float(np.sum(power[(f>=f.max()*.2)&(f<=f.max()*.5)])),np.sqrt(np.mean(env*env)),stats.kurtosis(env),np.sum(env*env),np.sqrt(ep.max())]
    return {prefix+k:float(v) if np.isfinite(v) else 0.0 for k,v in zip(FEATURE_NAMES,vals)}
FEATURE_NAMES=['mean','std','rms','peak','peak_to_peak','skewness','kurtosis','crest_factor','impulse_factor','shape_factor','margin_factor','energy','absolute_mean','variance','spectral_centroid','mean_frequency','rms_frequency','frequency_variance','frequency_std','spectral_entropy','dominant_frequency_amplitude','band_energy','envelope_rms','envelope_kurtosis','envelope_energy','envelope_spectrum_peak']

def unit_split(units,conditions=None):
    rng=random.Random(SEED); groups={}
    for u in units: groups.setdefault((conditions or {}).get(u,'all'),[]).append(u)
    out={}
    for _,us in sorted(groups.items()):
        us=sorted(us); rng.shuffle(us); n=len(us)
        if n>=3: nt=max(1,round(.2*n)); nv=max(1,round(.2*n)); nt=min(nt,n-2); nv=min(nv,n-nt-1)
        elif n==2: nt,nv=1,0
        else: nt,nv=1,0
        for u in us[:nt]: out[u]='test'
        for u in us[nt:nt+nv]: out[u]='val'
        for u in us[nt+nv:]: out[u]='train'
    # tiny condition groups can produce no global train/val; repair deterministically.
    if 'train' not in out.values() and len(units)>1: out[sorted(units)[0]]='train'
    if 'val' not in out.values() and len(units)>2: out[sorted(units)[1]]='val'
    return out

def bearing_files(root):
    units=[]
    for d in sorted([x for x in root.rglob('*') if x.is_dir()]):
        fs=sorted(d.glob('*.csv'),key=natural)
        if fs: units.append((d.name,str(d.parent.name),fs))
    return units

def read_bearing_csv(p,kind):
    if kind=='xjtu_sy':
        d=pd.read_csv(p); return [d.iloc[:,0].to_numpy(float),d.iloc[:,1].to_numpy(float)]
    d=pd.read_csv(p,header=None)
    # PHM columns: hour, minute, second, microsecond, horizontal, vertical.
    return [pd.to_numeric(d.iloc[:,4],errors='coerce').to_numpy(float),pd.to_numeric(d.iloc[:,5],errors='coerce').to_numpy(float)]

def extract_bearing(kind,smoke=False):
    root=PATHS[kind]; units=bearing_files(root)
    if kind=='pronostia':
        # Full_Test_Set contains complete test trajectories. Learning_set provides complete training trajectories.
        units=[u for u in units if '/Test_set/' not in str(u[2][0])]
    if smoke and kind=='xjtu_sy': units=units[:4]
    rec=[]
    for uid,cond,files in units:
        if smoke: files=files[:min(len(files),96)]
        T=len(files)-1
        for i,p in enumerate(files):
            try:
                ch=read_bearing_csv(p,kind); row={}
                for ci,x in enumerate(ch): row.update(feats(x,25600.0,('h_' if ci==0 else 'v_')))
                row.update(unit_id=uid,cycle=i+1,RUL=T-i,normalized_RUL=(T-i)/max(T,1),subset_or_condition=cond,source_file=str(p),endpoint_policy='complete_run_to_failure_last_record')
                rec.append(row)
            except Exception as e: print(f'[WARN] {kind} {p}: {e}',flush=True)
    return pd.DataFrame(rec)

def extract_ims(smoke=False):
    # Only failures explicitly documented by bundled IMS PDF.
    specs=[('1st_test','Bearing3',4,'documented_failed_bearing_inner_race'),('1st_test','Bearing4',6,'documented_failed_bearing_roller'),('2nd_test','Bearing1',0,'documented_failed_bearing_outer_race'),('4th_test','Bearing3',2,'documented_failed_bearing_outer_race')]
    rec=[]; endpoint=[]
    for test,uid,col,policy in specs:
        files=sorted([x for x in (PATHS['ims']/test).iterdir() if x.is_file()],key=natural)
        if smoke: files=files[:96]
        endpoint.append({'unit_id':f'{test}_{uid}','test':test,'endpoint_policy':policy,'usable':True,'records':len(files),'source':'Readme Document for IMS Bearing Data.pdf'})
        T=len(files)-1
        for i,p in enumerate(files):
            try:
                a=np.loadtxt(p,usecols=[col]); row=feats(a,20000.0,'')
                row.update(unit_id=f'{test}_{uid}',cycle=i+1,RUL=T-i,normalized_RUL=(T-i)/max(T,1),subset_or_condition=test,source_file=str(p),endpoint_policy=policy); rec.append(row)
            except Exception as e: print(f'[WARN] IMS {p}: {e}',flush=True)
    pd.DataFrame(endpoint).to_csv(DIR['diagnostics']/'ims_endpoint_policy.csv',index=False)
    return pd.DataFrame(rec)

def mat_field(o,name,default=None): return getattr(o,name,default)
def extract_battery(smoke=False):
    mats=sorted(PATHS['nasa_battery'].rglob('B*.mat'))
    # Duplicate battery IDs occur across archives; prefer the largest MAT file.
    chosen={}
    for p in mats:
        if p.stem not in chosen or p.stat().st_size>chosen[p.stem].stat().st_size: chosen[p.stem]=p
    if smoke: chosen=dict(list(sorted(chosen.items()))[:4])
    rec=[]; endpoints=[]
    for uid,p in sorted(chosen.items()):
        try:
            root=loadmat(p,squeeze_me=True,struct_as_record=False)[uid]; cycles=np.atleast_1d(mat_field(root,'cycle',[])); caps=[]
            for c in cycles:
                if str(mat_field(c,'type','')).lower()!='discharge': continue
                cap=mat_field(mat_field(c,'data',None),'Capacity',np.nan)
                try: cap=float(np.asarray(cap).squeeze())
                except Exception: continue
                if np.isfinite(cap) and cap>0: caps.append(cap)
            if not caps: continue
            c0=float(np.mean(caps[:min(5,len(caps))])); threshold=.70*c0; cross=next((i for i,c in enumerate(caps) if c<=threshold),None)
            observed=cross is not None; end=cross if observed else len(caps)-1; policy='threshold_0.70_C0' if observed else 'excluded_censored_no_threshold_crossing'
            endpoints.append({'unit_id':uid,'C0':c0,'failure_threshold':threshold,'eol_cycle':cross,'eol_observed':observed,'endpoint_policy':policy,'valid_discharge_cycles':len(caps),'source_file':str(p)})
            if not observed: continue
            for i,cap in enumerate(caps[:end+1]): rec.append({'unit_id':uid,'cycle':i+1,'RUL':end-i,'normalized_RUL':(end-i)/max(end,1),'DI_raw':1-cap/c0,'capacity':cap,'SOH':cap/c0,'failure_threshold':threshold,'eol_observed':True,'subset_or_condition':'threshold_EoL','source_file':str(p),'endpoint_policy':policy})
        except Exception as e: print(f'[WARN] battery {p}: {e}',flush=True)
    pd.DataFrame(endpoints).to_csv(DIR['diagnostics']/'nasa_battery_endpoint_policy.csv',index=False)
    return pd.DataFrame(rec)

def make_di(kind,raw,missing):
    if raw.empty: raise ValueError(f'No usable rows for {kind}')
    cond={u:g.subset_or_condition.iloc[0] for u,g in raw.groupby('unit_id')}; split=unit_split(sorted(raw.unit_id.unique()),cond if kind in ('xjtu_sy','pronostia') else None)
    if kind=='pronostia':
        # The extracted Full_Test_Set is complete-to-failure and is the official test population.
        official_test={u for u,g in raw.groupby('unit_id') if g.source_file.astype(str).str.contains('/Full_Test_Set/').any()}
        learning=sorted(set(raw.unit_id.unique())-official_test); rng=random.Random(SEED); rng.shuffle(learning)
        nv=max(1,round(.2*len(learning))) if len(learning)>1 else 0
        split={u:('val' if u in learning[:nv] else 'train') for u in learning}; split.update({u:'test' for u in official_test})
    raw['split']=raw.unit_id.map(split)
    if kind=='nasa_battery':
        tr=raw[raw.split=='train']; sc=StandardScaler().fit(tr[['DI_raw']]); raw['DI']=sc.transform(raw[['DI_raw']]).ravel(); selected=[]; method='capacity_DI_1_minus_capacity_over_C0'
    else:
        feature=[c for c in raw if c in FEATURE_NAMES or c.startswith(('h_','v_'))]
        tr=raw[raw.split=='train']; ranking=[]
        for f in feature:
            vals=[]
            for _,g in tr.groupby('unit_id'):
                r=stats.spearmanr(g[f],g.RUL).statistic
                if np.isfinite(r): vals.append(abs(r))
            ranking.append((f,float(np.mean(vals)) if vals else 0.0))
        ranking=sorted(ranking,key=lambda x:x[1],reverse=True); selected=[x[0] for x in ranking[:min(TOPK,len(ranking))]]
        pd.DataFrame([{'dataset':kind,'feature':f,'mean_abs_unit_spearman':v,'rank':i+1,'selected_flag':i<TOPK} for i,(f,v) in enumerate(ranking)]).to_csv(DIR['diagnostics']/f'{kind}_selected_features.csv',index=False)
        sc=StandardScaler().fit(tr[selected]); pca=PCA(n_components=2,random_state=SEED).fit(sc.transform(tr[selected])); z=pca.transform(sc.transform(raw[selected]))[:,0]; ztr=z[raw.split.eq('train')]
        corr=np.corrcoef(ztr,tr.RUL)[0,1]; z=-z if np.isfinite(corr) and corr>0 else z; raw['DI_raw']=z
        dsc=StandardScaler().fit(raw.loc[raw.split.eq('train'),['DI_raw']]); raw['DI']=dsc.transform(raw[['DI_raw']]).ravel(); method='per_unit_spearman_avg_abs_topK_StandardScaler_PCA2_z1'
    raw.insert(0,'dataset',kind); raw['DI_method']=method; raw['selected_features']=','.join(selected)
    keep=['dataset','subset_or_condition','unit_id','cycle','split','RUL','normalized_RUL','DI_raw','DI','DI_method','source_file','endpoint_policy']
    for c in ['capacity','SOH','failure_threshold','eol_observed','selected_features']:
        if c in raw: keep.append(c)
    out=raw[keep].copy(); out.to_csv(DIR['di_cache']/f'{kind}_di.csv',index=False); safe_parquet(out,DIR['di_cache']/f'{kind}_di.parquet',missing)
    pd.DataFrame([{'dataset':kind,'unit_id':u,'split':g.split.iloc[0],'records':len(g),'DI_min':g.DI.min(),'DI_max':g.DI.max(),'DI_std':g.DI.std(),'endpoint_policy':g.endpoint_policy.iloc[0]} for u,g in out.groupby('unit_id')]).to_csv(DIR['diagnostics']/f'{kind}_di_units.csv',index=False)
    pd.DataFrame([{'unit_id':u,'split':s,'subset_or_condition':cond.get(u,'')} for u,s in split.items()]).to_csv(DIR['protocols']/f'{kind}_split.csv',index=False)
    return out,{'dataset':kind,'DI_method':method,'selected_features':selected,'K':TOPK if selected else None,'split':split,'protocol_hash':phash({'kind':kind,'method':method,'selected':selected,'seed':SEED})}

def build_di(kind,smoke,missing,force=False):
    cp=DIR['di_cache']/f'{kind}_di.csv'; mp=DIR['di_cache']/f'{kind}_meta.json'
    if cp.exists() and mp.exists() and not force and not smoke:
        print(f'[PUBLIC_DI] Loading existing DI cache for {kind}',flush=True); return pd.read_csv(cp),json.loads(mp.read_text())
    print(f'[PUBLIC_DI] Building DI cache for {kind}',flush=True)
    raw=extract_bearing(kind,smoke) if kind in ('xjtu_sy','pronostia') else extract_ims(smoke) if kind=='ims' else extract_battery(smoke)
    di,meta=make_di(kind,raw,missing); jwrite(mp,meta); return di,meta

def build_dwc(kind,di,device,missing,force=False):
    del di, device, missing, force
    fields = ["documented SoH", "Q_actual", "Q_nominal"] if kind != "nasa_battery" else []
    detail = ("No valid physical health quantity is present. " if fields else
              "NASA battery is supported only by the new prefix-keyed causal runner. ")
    raise MissingHealthStateError(
        kind, fields or ["causal temporal fusion checkpoint"],
        detail + "Historical full-lifetime DWC caches are disabled; use run_causal_retraining.py."
    )

def choose_window(cache):
    m=int(cache.groupby('unit_id').size().min())
    return next((w for w in [128,64,32] if m>=w),max(8,min(32,m)))
def rows_from_cache(cache,smoke=False):
    del cache, smoke
    raise RuntimeError("Full-lifetime DWC/EDC caches are disabled; use prefix-keyed causal rows.")

def metrics(pred):
    # train_eval assumes C-MAPSS scale=125; recover normalized predictions then dataset-unit RUL scale.
    pred=pred.copy(); pred['pred_rul']=pred['pred_rul']/125.0*pred['unit_id'].map({}) if False else pred['pred_rul']
    return fd_metric(pred)

def run_models(kind,cache,args,summary,units,preds,missing):
    rows,win=rows_from_cache(cache,args.smoke_test)
    if args.smoke_test: models=['LSTM','cpm_stats_tokens_edc_independent_fusion']
    else: models=args.models or MODELS
    for model in models:
        try:
            print(f'[PUBLIC_DI] Reusing DI cache for model: {model} dataset: {kind}',flush=True)
            # Existing trainer outputs normalized prediction multiplied by 125. Convert both truth and predictions back to normalized percentage scale for a consistent public benchmark.
            pred,ck,best=train_eval(model,rows,'PUBLIC',kind,DIR['checkpoints']/kind,args.epochs,args.device)
            pred['true_rul']=pred['true_rul']/125.0*100.0; pred['pred_rul']=pred['pred_rul']/125.0*100.0
            met,pu=fd_metric(pred); pred['dataset']=kind; pu['dataset']=kind; preds.append(pred); units.append(pu)
            summary.append({'dataset':kind,'subset_or_condition':'all','model':model,'RMSE':met['sample_rmse'],'MAE':met['sample_mae'],'P90AE':met['sample_p90ae'],'Score':met['sample_score'],'sample_rmse':met['sample_rmse'],'sample_mae':met['sample_mae'],'sample_p90ae':met['sample_p90ae'],'sample_score':met['sample_score'],'best_epoch':best,'num_units':met['num_units'],'num_predictions':met['num_predictions'],'input_type':'cached_DWC_EDC_CPM_tokens' if model.endswith('independent_fusion') else 'cached_DI_scalar_prefix' if model=='KGHM' else 'cached_DI','di_protocol':cache.dataset.iloc[0]+'_cached_DI','decomposition_source':'Wiener_STL_autoencoder_DWC_EDC' if model.endswith('independent_fusion') else 'causal_KGHM_adapter_level_difference' if model=='KGHM' else 'not_applicable','window_length':win,'notes':f'checkpoint={ck}; metrics on normalized RUL percentage scale'})
        except Exception as e:
            print(f'[ERROR] {kind} {model}: {e}',flush=True); missing.append({'dataset':kind,'model':model,'missing_item':'comparison_result','reason_if_known':repr(e),'suggested_action':'Inspect log; do not fabricate result'})

def outputs(summary,units,preds,missing,metas,smoke):
    R=DIR['comparison_results']; s=pd.DataFrame(summary); u=pd.concat(units,ignore_index=True) if units else pd.DataFrame(); p=pd.concat(preds,ignore_index=True) if preds else pd.DataFrame()
    s.to_csv(R/'public_comparison_summary.csv',index=False); u.to_csv(R/'public_per_unit_metrics.csv',index=False); p.to_csv(R/'public_predictions.csv',index=False)
    if s.empty:
        wide=pd.DataFrame(columns=['model'])
    else:
        wide=s.pivot_table(index='model',columns='dataset',values=['RMSE','MAE','P90AE','Score'],aggfunc='first'); wide.columns=[f'{d}_{m}' for m,d in wide.columns]; wide=wide.reset_index()
    wide.to_csv(R/'public_comparison_wide.csv',index=False); wide.to_csv(R/'public_main_paper_table.csv',index=False); wide.to_csv(DIR['combined_tables']/'public_comparison_main_table.csv',index=False)
    if not s.empty:
        rr=[]; bb=[]
        for ds,g in s.groupby('dataset'):
            for m in ['RMSE','MAE','P90AE','Score']:
                q=g.dropna(subset=[m]).sort_values(m)
                for rank,(_,x) in enumerate(q.iterrows(),1): rr.append({'dataset':ds,'metric':m,'rank':rank,'model':x.model,'value':x[m]})
                if len(q): bb.append({'dataset':ds,'metric':m,'best_model':q.iloc[0].model,'best_value':q.iloc[0][m],'second_best_model':q.iloc[1].model if len(q)>1 else '','second_best_value':q.iloc[1][m] if len(q)>1 else np.nan})
        pd.DataFrame(rr).to_csv(R/'public_ranking_by_metric.csv',index=False); pd.DataFrame(bb).to_csv(R/'public_best_model_by_metric.csv',index=False); pd.DataFrame(bb).to_csv(DIR['combined_tables']/'public_best_models_summary.csv',index=False)
    md=pd.DataFrame(missing); md.to_csv(R/'public_dataset_missing_results_report.csv',index=False); md.to_csv(DIR['combined_tables']/'public_missing_results_report.csv',index=False)
    proto={'scope':'four public datasets comparison only','smoke_test':smoke,'models':args.models or MODELS,'excluded_models':['cpm_stats_tokens_dwc_cross_attn'],'main_model':'cpm_stats_tokens_edc_independent_fusion','DI_protocols':metas,'DWC':'AutoEncoderFusion(Wiener early-health trend, STL trend)','EDC':'DI-DWC','training':{'optimizer':'Adam','lr':.001,'epochs':1 if smoke else 50,'batch_size':8,'seed':SEED},'no_AC_experiment':True,'no_FD_experiment':True,'no_ablation':True}
    jwrite(R/'public_comparison_protocol_used.json',proto); jwrite(DIR['di_cache']/'di_cache_protocol.json',proto); jwrite(DIR['dwc_edc_cache']/'dwc_edc_protocol.json',{'DWC':proto['DWC'],'EDC':proto['EDC'],'moving_average_21':False})
    (DIR['protocols']/'public_dataset_di_comparison_protocol_summary.md').write_text('# Public dataset DI comparison protocol\n\n- Dataset paths: see `dataset_path_manifest.json`.\n- Bearing DI: vibration features, per-unit Spearman averaged absolute correlation, top-14, training StandardScaler, PCA(2), z1.\n- Battery DI: `1-capacity/C0`, C0 from first five valid discharge cycles, strict 70% threshold endpoint.\n- IMS: only four failures documented in the bundled PDF are used.\n- Splits are unit-level 60/20/20 where feasible; IMS uses a tiny-unit split.\n- DWC = autoencoder fusion of Wiener and STL trends; EDC = DI-DWC.\n- `cpm_stats_tokens_dwc_cross_attn` is excluded. No FD, AC, or ablation experiment is run.\n')

def main():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True); g.add_argument('--smoke-test',action='store_true'); g.add_argument('--full-run',action='store_true'); ap.add_argument('--device',default='cuda'); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--models',default='',help='Comma-separated model subset for a full run'); ap.add_argument('--force-rebuild-cache',action='store_true'); args=ap.parse_args(); args.models=[x.strip() for x in args.models.split(',') if x.strip()] or None
    if args.smoke_test: args.epochs=1
    for p in DIR.values(): p.mkdir(parents=True,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); start=time.time(); manifest=discover(); missing=[]; metas={}; caches={}
    targets=['xjtu_sy'] if args.smoke_test else DATASETS
    for kind in targets:
        if not PATHS[kind].exists(): missing.append({'dataset':kind,'model':'','missing_item':'dataset','reason_if_known':'path not found','suggested_action':'install/extract dataset'}); continue
        try:
            di,meta=build_di(kind,args.smoke_test,missing,args.force_rebuild_cache); metas[kind]=meta; caches[kind]=build_dwc(kind,di,args.device,missing,args.force_rebuild_cache)
        except Exception as e: missing.append({'dataset':kind,'model':'','missing_item':'DI_or_DWC_cache','reason_if_known':repr(e),'suggested_action':'inspect parser and source metadata'}); print(f'[ERROR] cache {kind}: {e}',flush=True)
    pd.DataFrame([{'dataset':k,'rows':len(v),'units':v.unit_id.nunique(),'DI_min':v.DI.min(),'DI_max':v.DI.max()} for k,v in caches.items()]).to_csv(DIR['di_cache']/'di_summary.csv',index=False)
    jwrite(DIR['di_cache']/'di_cache_manifest.json',{'datasets':metas}); jwrite(DIR['dwc_edc_cache']/'dwc_edc_cache_manifest.json',{'datasets':list(caches),'validation':'EDC == DI-DWC, final DWC != moving_average(DI,21)'})
    summary=[]; units=[]; preds=[]
    for kind,c in caches.items(): run_models(kind,c,args,summary,units,preds,missing)
    outputs(summary,units,preds,missing,metas,args.smoke_test)
    elapsed=time.time()-start; jwrite(DIR['logs']/('smoke_test_runtime.json' if args.smoke_test else 'full_run_runtime.json'),{'seconds':elapsed,'passed':bool(summary),'datasets':list(caches),'models':sorted({x['model'] for x in summary})})
    if args.smoke_test and not {'LSTM','cpm_stats_tokens_edc_independent_fusion'}.issubset({x['model'] for x in summary}): raise RuntimeError('Smoke test did not pass both required models')
    print(f'[DONE] mode={"smoke" if args.smoke_test else "full"} runtime_seconds={elapsed:.1f} output={OUT}',flush=True)
if __name__=='__main__': main()
