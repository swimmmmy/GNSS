"""Audit and prepare Jammertest meaconing observations. Does not train or assume time offsets."""
from pathlib import Path
import argparse
import json
import pandas as pd
GNSS_DATASET_JAMMING_SPOOFING/Meaconing/stationaryimport numpy as np

ROOT = Path('GNSS_DATASET_JAMMING_SPOOFING/Meaconing')


def process(folder, output, root):
    rinex = pd.read_csv(folder / 'rinex.csv', usecols=['time','satellite','snr_L1','doppler_L1'])
    rinex['time'] = pd.to_datetime(rinex['time'], errors='coerce', utc=True)
    rinex['snr_L1'] = pd.to_numeric(rinex['snr_L1'], errors='coerce')
    rinex['doppler_L1'] = pd.to_numeric(rinex['doppler_L1'], errors='coerce')
    rinex = rinex.dropna(subset=['time']).sort_values('time')
    # Bin to whole seconds; aggregate satellite observations in each second.
    rinex['second'] = rinex['time'].dt.floor('s')
    good = rinex.dropna(subset=['snr_L1'])
    features = good.groupby('second').agg(
        l1_mean=('snr_L1','mean'),
        l1_max=('snr_L1','max'),
        l1_std=('snr_L1',lambda s: s.std(ddof=0)),
        l1_count=('snr_L1','count'),
        l1_satellites=('satellite','nunique'),
        l1_doppler_abs_mean=('doppler_L1',lambda s: s.abs().mean()),
    ).reset_index().sort_values('second')
    features['l1_mean_delta'] = features['l1_mean'].diff().abs()
    nav = pd.read_csv(folder / 'nav_pvt.csv', low_memory=False)
    nav['real_time'] = pd.to_datetime(nav['real_time'], errors='coerce', utc=True)
    nav = nav.dropna(subset=['real_time']).sort_values('real_time')
    for col in ['gSpeed','hAcc','numSV','pDOP','fixType','lat','lon']:
        if col in nav.columns:
            nav[col] = pd.to_numeric(nav[col], errors='coerce')
    keep = ['real_time'] + [c for c in ['gSpeed','hAcc','numSV','pDOP','fixType','lat','lon'] if c in nav.columns]
    nav = nav[keep].drop_duplicates('real_time')
    features = pd.merge_asof(features, nav, left_on='second', right_on='real_time', direction='nearest', tolerance=pd.Timedelta('1s'))
    scenario = json.loads((folder/'scenario.json').read_text(encoding='utf-8'))
    events = scenario.get('attack_log',[])
    starts = [pd.to_datetime(x['timestamp_utc'],utc=True) for x in events if 'start' in x.get('event','').lower()]
    ends = [pd.to_datetime(x['timestamp_utc'],utc=True) for x in events if 'end' in x.get('event','').lower()]
    intervals = list(zip(starts,ends))
    t0,t1=features['second'].min(),features['second'].max()
    overlap = sum(max(0,(min(t1,b)-max(t0,a)).total_seconds()) for a,b in intervals)
    # Explicitly do not label if clocks cannot be aligned.
    features['label'] = -1
    if overlap > 0:
        features['label'] = 0
        for a,b in intervals:
            features.loc[features['second'].between(a,b,inclusive='left'),'label'] = 1
    features['experiment'] = str(folder.relative_to(root))
    features['second'] = features['second'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    features['real_time'] = features['real_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    output.mkdir(parents=True,exist_ok=True)
    filename = 'meaconing_features_'+str(folder.relative_to(root)).replace('/','__').replace(' ','_').replace(',','_')+'.csv'
    features.to_csv(output/filename,index=False,encoding='utf-8-sig')
    return {'experiment':str(folder.relative_to(root)),'rows':len(features), 'first_measurement_utc':str(t0), 'last_measurement_utc':str(t1), 'attack_intervals_utc':[[str(a),str(b)] for a,b in intervals], 'attack_overlap_seconds':overlap,'label_status':'PROVISIONAL - verify clock and actual units' if overlap else 'UNLABELED - no time overlap', 'file':filename,'missing_l1':int(rinex['snr_L1'].isna().sum())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=Path('stage2_output'))
    args=p.parse_args()
    folders=sorted({p.parent for p in args.root.rglob('scenario.json') if (p.parent/'rinex.csv').exists() and (p.parent/'nav_pvt.csv').exists()})
    if not folders: raise SystemExit('Meaconing 폴더에서 CSV를 찾지 못했습니다.')
    reports=[]
    for folder in folders:
        r=process(folder,args.out,args.root); reports.append(r)
        print(f"{r['experiment']}: {r['rows']}초, {r['label_status']}")
    (args.out/'audit_report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    print('완료:',args.out/'audit_report.json')
    print('주의: 라벨은 시간 검증 전 학습에 사용하지 마세요.')

if __name__=='__main__':
    main()
