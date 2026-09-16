# Воспроизведение ревью N0 и проб N1 — 2026-09-15

Исследовательский скрипт; реализацию и исходный пакет N0 не изменяет. Сохранить основной скрипт из последнего блока Python как artifacts/n0_research_review_20260915/review_probe.txt и запустить установленным Python из корня проекта. Первый запуск проверяет пакет и демпфирование; --pitch проверяет частоты резонаторов; --edit — повторные перемещения поля. Для проверки хешей источника нужен локальный checkout commit efa4737af31febf09bd746a45bd9cd57c88f74b1; путь SOURCE заменить своим.

JVM-сверка: python -m demos.network_reference_n0.verify_node --source <checkout> --jdk <jdk/bin> --out artifacts/n0_research_review_20260915/verification_node.json --work artifacts/n0_research_review_20260915/java. Исходный Java-класс исполняется заново; Max не запускается. Общая регрессия: python check.py.

Все пробы частотного управления изменяют только экземпляры модели. R[5]=0 — проверяемое исправление маршрута источника. Временные окна и спектральные показатели — измерения, не слуховые оценки.

Дополнительный показатель temporal flux из отчёта рассчитывается на готовых файлах следующим кодом (окно 4–12 с, нормированные амплитуды STFT):

```python
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
p = Path('artifacts/n0_research_review_20260915')
result = {}
for name in ['pitch_frozen', 'pitch_live', 'pitch_moved']:
    sr, y = wavfile.read(p / (name + '.f32.wav'))
    x = y.astype(float).mean(axis=1)
    f, t, z = stft(x, fs=sr, nperseg=2048, noverlap=1536, boundary=None)
    m = np.abs(z)
    m /= m.sum(axis=0, keepdims=True) + 1e-30
    flux = np.sqrt(np.sum(np.diff(m, axis=1)**2, axis=0))
    result[name] = dict(flux_mean_4to12=float(flux[t[1:] >= 4].mean()),
        rms_db_4to12=float(20*np.log10(np.sqrt(np.mean(y[4*sr:].astype(float)**2)))))
(p / 'pitch_comparison.json').write_text(json.dumps(result, indent=2))
```

Основной скрипт:

```python
"""Research diagnostics only; does not modify implementation or N0 package."""
import sys, json, hashlib, time, copy, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demos.network_reference_n0 import gutter_network as gn, source_manifest as sm
OUT=ROOT/'artifacts/n0_research_review_20260915'
PKG=ROOT/'demos/results/network_reference_n0'
SR=44100
SOURCE=Path(r'C:\Users\Astro\AppData\Local\Temp\claude\C--Users-Astro-Documents-Projects-CASynth-2\1b7c109d-8ed7-47e1-8c94-2719d4c9351c\scratchpad\guttersynthesis')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(y):
    y=np.asarray(y,dtype=float); x=y.mean(axis=1)
    f,p=welch(x,SR,nperseg=8192)
    bands=[(20,100),(100,500),(500,2000),(2000,16000)]
    total=p[(f>=20)&(f<16000)].sum()
    return dict(rms_db=float(20*np.log10(max(np.sqrt(np.mean(y*y)),1e-30))),
        centroid=float(np.sum(f[(f>=20)&(f<16000)]*p[(f>=20)&(f<16000)])/max(total,1e-30)),
        band_fraction=[float(p[(f>=lo)&(f<hi)].sum()/max(total,1e-30)) for lo,hi in bands],
        peak=float(np.abs(y).max()),finite=bool(np.isfinite(y).all()))
def grid_initial():
    g=np.zeros((32,32),dtype=np.uint8)
    for c in [4,12,20,28]: g[14:17,c]=1
    return g
def gol(g):
    n=sum(np.roll(np.roll(g,r,0),c,1) for r in [-1,0,1] for c in [-1,0,1] if r or c)
    return ((n==3)|((g!=0)&(n==2))).astype(np.uint8)
def counts(g):
    H,W=g.shape
    return np.array([g[(r*H)//2:((r+1)*H)//2,(c*W)//4:((c+1)*W)//4].sum() for r in range(2) for c in range(4)],float)
def damping(g): 
    n=counts(g); return 190-150*n/(n+2)
def set_damping(net,raw):
    # Atomic 8-channel target, identical mapping and ramp to the source setters.
    net.r_damp.set((np.array(raw)/256)**2,net._ramp_n['damp'])
def make(cfg,fix_right=True,qscale=1):
    net=gn.GutterNetwork(cfg)
    if fix_right: net.panR[5]=0
    net.svf1.q*=qscale;net.svf2.q*=qscale
    return net
def worker(job):
    name,kind,extra,qscale=job
    cfg=json.loads((PKG/'config.json').read_text())
    cfg['extra_feedback_samples']=extra
    net=make(cfg,name!='stock',qscale)
    start=time.perf_counter(); pieces=[]; schedule=[]
    g=grid_initial()
    if kind.startswith('pitch'):
        original=np.array(cfg['filters_hz'])
        def tune(ratios):
            for i in range(8): net.set_filters(i,original[i]*ratios[i])
        if kind=='pitch_edit':
            for k,offset in enumerate([0,8,0,8,0]):
                g=np.roll(grid_initial(),offset,0);n=counts(g);ratios=2**(2*n/(n+2)-1)
                tune(ratios);schedule.append(dict(t=k*4,counts=n.tolist(),ratios=ratios.tolist()))
                pieces.append(net.render(4*SR)[0])
        elif kind=='pitch_cycle':
            for k,ratio in enumerate([1,0.5,1,2,1,0.5,1]):
                tune(np.full(8,ratio));schedule.append(dict(t=k*4,ratios=[ratio]*8))
                pieces.append(net.render(4*SR)[0])
        else:
            if kind=='pitch_moved': g=np.roll(g,8,0)
            for k in range(24):
                if kind=='pitch_live' and k: g=gol(g)
                n=counts(g);ratios=2**(2*n/(n+2)-1)
                if k==0 or kind=='pitch_live': tune(ratios)
                schedule.append(dict(t=k*.5,counts=n.tolist(),ratios=ratios.tolist()))
                pieces.append(net.render(SR//2)[0])
    elif kind.startswith('field'):
        if kind=='field_moved': g=np.roll(g,8,0)
        set_damping(net,damping(g))
        for k in range(24):
            if kind=='field_live' and k: g=gol(g);set_damping(net,damping(g))
            schedule.append(dict(t=k*.5,counts=counts(g).tolist(),damp_raw=damping(g).tolist()))
            pieces.append(net.render(SR//2)[0])
    elif kind=='cycle':
        for i,v in enumerate([138,40,138,40,138,40,138]):
            net.set_slider('damp',v)
            schedule.append(dict(t=i*4,damp_raw=v))
            pieces.append(net.render(4*SR)[0])
    elif kind=='edit':
        for i,n in enumerate([0,4,0,4,0]):
            raw=np.full(8,190-150*n/(n+2))
            set_damping(net,raw)
            schedule.append(dict(t=i*4,counts=[n]*8,damp_raw=raw.tolist()))
            pieces.append(net.render(4*SR)[0])
    else: pieces=[net.render(12*SR)[0]]
    y=np.concatenate(pieces)
    wavfile.write(OUT/(name+'.f32.wav'),SR,y.astype(np.float32))
    report=dict(name=name,kind=kind,fix_right=name!='stock',extra_feedback=extra,svf_qscale=qscale,
        seconds=len(y)/SR,schedule=schedule,all=stats(y),resets=int(net.resets.sum()),
        windows_2s=[stats(y[k:k+2*SR]) for k in range(0,len(y)-2*SR+1,2*SR)],
        render_seconds=time.perf_counter()-start)
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
    return report
def audit():
    rep={}
    m=json.loads((PKG/'manifest.json').read_text())
    rep['source_hashes']=sm.hash_source(str(SOURCE))
    rep['source_hashes_match']=rep['source_hashes']==sm.SOURCE_HASHES
    rep['files']={}
    for name,v in m['files'].items():
        sr,y=wavfile.read(PKG/v['raw_file'])
        rep['files'][name]=dict(pcm_hash=sha(PKG/v['file'])==v['sha256'],raw_hash=sha(PKG/v['raw_file'])==v['raw_sha256'],sr=sr,stats=stats(y))
    _,a=wavfile.read(PKG/'raw/02_control.f32.wav')
    _,b=wavfile.read(PKG/'raw/02_control_reference.f32.wav')
    rep['control_first4_exact']=bool(np.array_equal(a[:4*SR],b[:4*SR]))
    rep['control_8to12']=dict(changed=stats(a[8*SR:12*SR]),reference=stats(b[8*SR:12*SR]))
    rep['routes']=[]
    for D in [3,64,2064]:
        cfg=gn.default_config();cfg['matrix_delay_samples']=D;cfg['extra_feedback_samples']=0
        cfg['matrix']=np.zeros((8,8)).tolist();cfg['matrix'][0][3]=0.75
        net=gn.GutterNetwork(cfg)
        for s,v in [('mod',0),('damp',128),('rate',39),('gain',162),('interaction',128)]:net.set_slider(s,v,ramp=False)
        net.duffX[0]=0.3
        hist=[];errors=[];write_errors=[];first=None
        for k in range(D+30):
            L,R,o=net.step()
            expected=np.full(8,.25)
            if k>=D: expected+=hist[k-D]*1.25
            expected=np.clip(expected,.0001,1).astype(np.float32).astype(float)
            errors.append(float(np.abs(net.last_inlets[2]-expected).max()))
            mvec=np.zeros(8);mvec[3]=.75*.5*(L[0]+R[0]);hist.append(mvec)
            write_errors.append(float(np.abs(net.ring[:,k%(D+1)]-mvec).max()))
            if net.last_inlets[2][3]!=.25 and first is None:first=k
        rep['routes'].append(dict(delay=D,first_receiver_change=first,max_read_error=max(errors),max_write_error=max(write_errors)))
    cfg=json.loads((PKG/'config.json').read_text())
    ff=np.array(cfg['filters_hz'])
    rep['frequencies_above_5000']=[dict(node=int(i)+1,filter=int(j)+1,hz=float(ff[i,j])) for i,j in np.argwhere(ff>5000)]
    src=json.loads((SOURCE/'Gutter Synth.maxpat').read_text())['patcher']
    ip=next(x['box']['patcher'] for x in src['boxes'] if x['box'].get('text')=='p individual_controls')
    rep['source_6R_incoming']=[x for x in ip['lines'] if x['patchline']['destination'][0]=='obj-92']
    rep['source_node6_outgoing']=[x for x in ip['lines'] if x['patchline']['source'][0]=='obj-17']
    g=grid_initial()
    rep['mapping_cycle']=[dict(grid=g.tolist(),counts=counts(g).tolist(),raw=damping(g).tolist()),
        dict(grid=gol(g).tolist(),counts=counts(gol(g)).tolist(),raw=damping(gol(g)).tolist())]
    (OUT/'audit.json').write_text(json.dumps(rep,indent=2))
    print(json.dumps({k:v for k,v in rep.items() if k not in ['files','mapping_cycle','source_hashes']},indent=2),flush=True)
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if '--edit' in sys.argv:
        jobs=[('pitch_edit','pitch_edit',64,1)]
    elif '--pitch' in sys.argv:
        jobs=[('pitch_cycle','pitch_cycle',64,1),('pitch_frozen','pitch_frozen',64,1),
            ('pitch_live','pitch_live',64,1),('pitch_moved','pitch_moved',64,1),
            ('pitch_live_delay0','pitch_live',0,1),('pitch_live_svf2','pitch_live',64,2)]
    else:
        audit()
        jobs=[('stock','base',64,1),('right6_fixed','base',64,1),('damp_cycle','cycle',64,1),
        ('field_frozen','field_frozen',64,1),('field_live','field_live',64,1),('field_moved','field_moved',64,1),
        ('field_edit','edit',64,1),('field_live_delay0','field_live',0,1),
        ('field_live_delay128','field_live',128,1),('field_live_svf2','field_live',64,2)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(worker,jobs):
            print(r['name'],r['render_seconds'],r['all'],flush=True)

```
