"""Stage 2 GNSS NAV-PVT Random Forest: exploratory labels (UTC-2h hypothesis).
Run in Codespaces GNSS project root: python build_stage2_nav_rf.py
Does not change original dataset or first-stage models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

ROOT=Path('GNSS_DATASET_JAMMING_SPOOFING/Meaconing')
OUT=Path('stage2_nav_rf')
BASE=['iTOW','fixType','numSV','lat','lon','height','hAcc','vAcc','velN','velE','velD','gSpeed','headMot','sAcc','headAcc','pDOP']
# Don't learn raw absolute coordinates or absolute time; generalize via changes.
FEATURES=['fixType','numSV','hAcc','vAcc','sAcc','headAcc','pDOP','gSpeed']
for sec in [1,3,5]:
    FEATURES += [f'{x}_delta_{sec}s' for x in ['distance_m','speed_mps','hAcc','pDOP','numSV']]

def prepare(folder):
    df=pd.read_csv(folder/'nav_pvt.csv',low_memory=False)
    needed=['real_time',*BASE]
    missing=[x for x in needed if x not in df.columns]
    if missing: raise ValueError(f'{folder}: missing {missing}')
    df=df[needed].copy()
    df['time']=pd.to_datetime(df.pop('real_time'),errors='coerce',utc=True)
    for c in BASE: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['time']).sort_values('time').drop_duplicates('time')
    # Use 1-second bins to avoid duplicate NAV messages per epoch.
    df['second']=df['time'].dt.floor('s')
    df=df.groupby('second',as_index=False).last().sort_values('second').reset_index(drop=True)
    if len(df)<10: raise ValueError(f'{folder}: too few timestamps')
    # NAV-PVT units: raw UBX is integer scaled, but dataset CSV may already be converted.
    # Determine units conservatively, report the chosen scale.
    lat_med=df['lat'].abs().median(); lon_med=df['lon'].abs().median()
    coord_scale=1e-7 if max(lat_med,lon_med)>180 else 1.0
    lat=df['lat']*coord_scale; lon=df['lon']*coord_scale
    if (lat.abs()>90).any() or (lon.abs()>180).any(): raise ValueError(f'{folder}: invalid coordinates')
    # UBX gSpeed typically mm/s; CSV may already use m/s.
    speed_scale=0.001 if df['gSpeed'].abs().quantile(.95)>100 else 1.0
    df['speed_mps']=df['gSpeed']*speed_scale
    # hAcc UBX mm, pDOP UBX scale 0.01; preserve raw but record scale.
    hacc_scale=0.001 if df['hAcc'].median()>100 else 1.0
    pdop_scale=0.01 if df['pDOP'].median()>30 else 1.0
    df['hAcc']=df['hAcc']*hacc_scale
    df['pDOP']=df['pDOP']*pdop_scale
    # Project lat/lon to local meters (approximation over small rover movements).
    lat0=np.deg2rad(lat.median())
    east=(lon-lon.iloc[0])*111320*np.cos(lat0)
    north=(lat-lat.iloc[0])*111320
    for sec in [1,3,5]:
        past=pd.DataFrame({'second':df['second']+pd.Timedelta(seconds=sec),
                           'east_old':east,'north_old':north,
                           'speed_old':df['speed_mps'],
                           'hacc_old':df['hAcc'],'pdop_old':df['pDOP'],
                           'numsv_old':df['numSV']})
        merged=df[['second']].merge(past,on='second',how='left')
        df[f'distance_m_delta_{sec}s']=np.sqrt((east-merged['east_old'])**2+(north-merged['north_old'])**2)
        df[f'speed_mps_delta_{sec}s']=(df['speed_mps']-merged['speed_old']).abs()
        for col,old in [('hAcc','hacc_old'),('pDOP','pdop_old'),('numSV','numsv_old')]:
            df[f'{col}_delta_{sec}s']=(df[col]-merged[old]).abs()
    # Exclude boundary observations so uncertainty in log timing doesn't corrupt labels.
    sc=json.loads((folder/'scenario.json').read_text(encoding='utf-8'))
    events=sc.get('attack_log',[])
    intervals=[]; start=None
    for event in events:
        t=pd.to_datetime(event['timestamp_utc'],utc=True)-pd.Timedelta(hours=2)
        ev=event.get('event','').lower()
        if 'started' in ev or 'start' in ev: start=t
        elif 'end' in ev and start is not None:
            intervals.append((start,t));start=None
    if start is not None: raise ValueError(f'{folder}: unmatched start')
    if not intervals: raise ValueError(f'{folder}: no intervals')
    df['label']=0
    for a,b in intervals:
        df.loc[(df['second']>=a)&(df['second']<b),'label']=1
        df.loc[(df['second']>=a-pd.Timedelta(seconds=10))&(df['second']<a+pd.Timedelta(seconds=10)),'label']=-1
        df.loc[(df['second']>=b-pd.Timedelta(seconds=10))&(df['second']<b+pd.Timedelta(seconds=10)),'label']=-1
    df['experiment']=str(folder.relative_to(ROOT))
    audit={'experiment':df['experiment'].iloc[0], 'rows':len(df),
           'normal':int((df.label==0).sum()),'attack':int((df.label==1).sum()),
           'excluded_boundary':int((df.label==-1).sum()),
           'time_start':str(df.second.min()),'time_end':str(df.second.max()),
           'intervals_assuming_local_UTCplus2':[[str(a),str(b)] for a,b in intervals],
           'scales':{'lat_lon':coord_scale,'gSpeed':speed_scale,'hAcc':hacc_scale,'pDOP':pdop_scale}}
    return df,audit

def main():
    folders=sorted(p.parent for p in ROOT.rglob('scenario.json') if (p.parent/'nav_pvt.csv').exists())
    if len(folders)<3: raise SystemExit('Need at least 3 experiment folders')
    OUT.mkdir(exist_ok=True)
    frames=[]; audits=[]
    for folder in folders:
        df,audit=prepare(folder);frames.append(df);audits.append(audit)
        print(audit['experiment'], 'normal',audit['normal'],'attack',audit['attack'])
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(OUT/'nav_features_provisional.csv',index=False,encoding='utf-8-sig')
    (OUT/'audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    usable=data[data.label.isin([0,1])].copy()
    exps=sorted(usable.experiment.unique())
    # Hold out an entire moving experiment; train includes moving and stationary.
    moving=[x for x in exps if x.startswith('dynamic/')]
    test_exp=moving[-1] if moving else exps[-1]
    train=usable[usable.experiment!=test_exp].copy();test=usable[usable.experiment==test_exp].copy()
    if train.label.nunique()!=2 or test.label.nunique()!=2: raise SystemExit('Missing a class in train/test; no model trained')
    for col in FEATURES:
        train[col]=pd.to_numeric(train[col],errors='coerce')
        test[col]=pd.to_numeric(test[col],errors='coerce')
    missing=[c for c in FEATURES if train[c].notna().sum()==0]
    if missing: raise SystemExit(f'All missing features: {missing}')
    model=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1))
    model.fit(train[FEATURES],train.label.astype(int))
    pred=model.predict(test[FEATURES]); prob=model.predict_proba(test[FEATURES])[:,1]
    report=classification_report(test.label.astype(int),pred,labels=[0,1],target_names=['normal','meaconing'],output_dict=True,zero_division=0)
    result=test[['second','experiment','label']].copy();result['prediction']=pred;result['meaconing_probability']=prob
    result.to_csv(OUT/'test_predictions.csv',index=False,encoding='utf-8-sig')
    joblib.dump({'pipeline':model,'features':FEATURES,'warning':'PROVISIONAL UTC+2 label hypothesis, not validated on M8N'},OUT/'meaconing_rf_provisional.joblib')
    summary={'train_experiments':sorted(train.experiment.unique().tolist()),'test_experiment':test_exp,'train_rows':len(train),'test_rows':len(test),'confusion_matrix':confusion_matrix(test.label,pred,labels=[0,1]).tolist(),'classification_report':report,'features':FEATURES,'warning':'EXPLORATORY ONLY: attack logs interpreted as local Norway time despite UTC field; independently verify. Units inferred heuristically. Receiver transfer not validated.'}
    (OUT/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PROVISIONAL model trained. Held out:',test_exp)
    print('Confusion matrix:',summary['confusion_matrix'])
    print('Files:',OUT)

if __name__=='__main__': main()
