# S/N: воспроизводимые расчёты дизайн-аудита

2026-09-15. **Исследовательская проверка, не изменение движков.** Код ниже выполняется из корня проекта установленным Python с NumPy/SciPy. Читает исходные записи, ничего в них не меняет; пишет результаты в `memory/research/audit-sn-2026-09-15-evidence.json`. Самостоятельные формулы Gaussian/Distance, путей, DFT и PM используются как эталоны. Вызовы действующего движка присутствуют только на проверяемой стороне сравнения.

```python
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json, hashlib, math, wave, collections
from pathlib import Path
import numpy as np
from scipy import signal
from casynth_lab import scan_surface as S, pm_network as N
from casynth_lab import registry, DemoRunner, scene_from_doc
from casynth_lab.engine_api import EngineContext
from casynth_lab.catalog import Catalog
from casynth_lab.provenance import manifest_of_tree
from casynth_config import SR, CHUNK_S

ROOT = Path.cwd()
OUT = ROOT / 'memory/research/audit-sn-2026-09-15-evidence.json'
BLOCK = int(SR * CHUNK_S)
result = dict(date='2026-09-15', scope='saved records, independent numerical references, current runner interventions', sr=SR, block=BLOCK, records=[], checks={})
def save():
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def ratio_db(x, ref):
    return float(20*np.log10(max(rms(x),1e-30)/max(rms(ref),1e-30)))
def wav(path):
    with wave.open(str(path),'rb') as f:
        assert f.getframerate()==SR and f.getnchannels()==2 and f.getsampwidth()==2
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').reshape(-1,2).copy()
def spectrum_stats(pcm, f0=110):
    y=np.asarray(pcm[SR:2*SR,0],float)/32767
    F=np.fft.rfft(y); power=np.abs(F)**2
    total=float(power[1:].sum())
    bins=np.round(np.arange(1,int((SR/2)/f0)+1)*f0).astype(int)
    amps=np.abs(F[bins])
    maxamp=max(float(amps.max()),1e-30)
    top=np.argsort(power[1:])[::-1][:10]+1
    return dict(rms_dbfs=ratio_db(y,np.ones_like(y)), peak=float(np.max(np.abs(pcm)))/32767,
                fundamental_power_fraction=float(power[int(f0)]/max(total,1e-30)),
                first4_power_fraction=float(power[bins[:4]].sum()/max(total,1e-30)),
                harmonics_1_12_db_from_strongest=[float(20*np.log10(max(a,1e-30)/maxamp)) for a in amps[:12]],
                top_frequencies_hz=[int(k) for k in top],
                peak_bin_hz=int(np.argmax(power[1:])+1))

manifest=manifest_of_tree(str(ROOT))
files=list((ROOT/'lab_catalog/sn_demos_2026_09_14').glob('*/*/record.json'))
loaded={}
for f in sorted(files):
    m=json.loads(f.read_text(encoding='utf-8')); cat=Catalog(str(f.parent.parent)); rec=cat.load(m['id'])
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in f.parent.iterdir() if p.is_file()}
    old=m['provenance']['manifest']
    changed=[p for p,h in manifest.items() if old.get(p)!=h]
    pcm={o:wav(f.parent/m['audio'][o]['file']) for o in ('A','B','monitor')}
    check_hash={o:hashlib.sha256(pcm[o].tobytes()).hexdigest()==m['audio'][o]['sha256'] for o in pcm}
    replay=cat.recompute(m['id'],yield_cpu=False)
    assert replay.status=='ok', replay.reason
    exact={o:bool(np.array_equal(pcm[o],replay.pcm[o])) for o in pcm}
    runner,st,_=cat.continue_runner(m['id'])
    runner2=DemoRunner.from_state(st)
    continued=[]
    for _ in range(12):
        a=runner.next_block(); b=runner2.next_block()
        continued.append(all(np.array_equal(a.get(o),b.get(o)) for o in pcm))
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in f.parent.iterdir() if p.is_file()}
    yA=pcm['A'][SR:,0].astype(float); yB=pcm['B'][SR:,0].astype(float)
    item=dict(id=m['id'],catalog=f.parent.parent.name,title=m['title'],scene_id=m['scene']['id'],
              commit=m['provenance']['commit'],sound_files_changed=changed,pcm_hash_matches=check_hash,
              replay_exact=exact,continue_repeat_exact=all(continued),source_files_unchanged=(before==after),
              notes=rec.notes,scene=m['scene'],settings_at_end=m['settings_at_end'],
              journal_counts=dict(collections.Counter(v['kind'] for v in m['journal'])),
              snapshot=dict(running=st['running'],paused=st['paused'],gen=st['gen']),
              rms_B_minus_A_db_relative_A=ratio_db(yB-yA,yA),
              stats={o:spectrum_stats(pcm[o],m['scene']['audio']['f0_hz']) for o in ('A','B')})
    for side in ('A','B'):
        es=st['sides'][side]['engine']
        if st['sides'][side]['engine_id']=='pm_network':
            item.setdefault('network_end',{})[side]=dict(W=es['W'].tolist(),W_target=es['W_target'].tolist(),beta=es['beta'],frozen=es['frozen'])
    result['records'].append(item); loaded[m['scene']['id']]=(m,pcm,st)
    save(); print('record',m['scene']['id'],'exact',exact,'unchanged',before==after,flush=True)

# Independent spatial and audio-rate mathematical references.
def grid_doc(doc):
    g=np.zeros((doc['grid']['rows'],doc['grid']['cols']),np.uint8)
    for r,c in doc['cells']: g[r,c]=1
    return g
def gaussian_ref(g,width):
    n=int(4*width+0.5); k=np.arange(-n,n+1); w=np.exp(-k*k/(2*width*width)); w/=w.sum()
    a=sum(v*np.roll(g.astype(float),int(j),axis=0) for j,v in zip(k,w))
    a=sum(v*np.roll(a,int(j),axis=1) for j,v in zip(k,w))
    return 2*a-1
def dist_ref(g,width):
    if not g.any(): return np.full(g.shape,-1.)
    if g.all(): return np.ones(g.shape)
    h,w=g.shape; xy=np.indices(g.shape).reshape(2,-1).T
    def nearest(value):
        targets=np.argwhere(g==value); d=np.abs(xy[:,None,:]-targets[None,:,:])
        d=np.minimum(d,np.array([h,w])-d)
        return np.sqrt((d*d).sum(axis=2).min(axis=1)).reshape(h,w)
    return np.tanh((nearest(0)-nearest(1))/width)
def points_ref(path,h,w,rx,ry,p):
    if path!=2:
        phi=2*np.pi*p; cx=(w-1)/2; cy=(h-1)/2
        return cx+rx*cx*np.cos((1 if path==0 else 2)*phi), cy+ry*cy*np.sin((1 if path==0 else 3)*phi)
    assert h==32 and w==32
    v=p*h*w; i=np.floor(v).astype(int); t=v-i; r=i//w; c=i%w
    c=np.where(r%2==0,c,w-1-c)
    last=(i%w==w-1)
    return c+np.where(last,0,np.where(r%2==0,1,-1))*t, r+last*t
def bilinear_ref(z,x,y):
    h,w=z.shape; out=np.zeros_like(x); x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    for dx in (0,1):
        for dy in (0,1):
            weight=(1-abs(x-x0-dx))*(1-abs(y-y0-dy))
            out+=weight*z[(y0+dy)%h,(x0+dx)%w]
    return out
def masks_ref(h,w):
    out=[]; rows,cols=np.indices((h,w)); sigma=min(h,w)/6
    for yf in (1/4,3/4):
        for xf in (1/6,1/2,5/6):
            dx=abs(cols-(w-1)*xf); dx=np.minimum(dx,w-dx)
            dy=abs(rows-(h-1)*yf); dy=np.minimum(dy,h-dy)
            a=np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma)); out.append(a/a.sum()/3)
    return np.array(out)
def weights_ref(g): return (masks_ref(*g.shape)*g).sum(axis=(1,2))
edges=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
def network_ref(theta,W,beta):
    x=np.zeros((4,len(theta)))
    for i in range(3,-1,-1):
        drive=np.zeros(len(theta))
        for k,(to,fr) in enumerate(edges):
            if to==i: drive+=W[k]*x[fr]
        x[i]=np.sin(2*np.pi*(i+1)*theta+beta*drive)
    return x.mean(axis=0)

g=grid_doc(loaded['sn_s_path_ellipse_lissajous'][0]['scene'])
p=np.arange(8192)/8192; K=int(.45*SR/110); k=np.arange(1,K+1)
DFT=np.exp(-2j*np.pi*k[:,None]*p[None,:])
scan_checks=[]
for mode in (0,1):
    z=gaussian_ref(g,1.5) if mode==0 else dist_ref(g,1.5)
    z_impl=S.surface(g,mode,1.5)
    for path in (0,1,2):
        params=dict(registry.defaults('scan_surface'),surface=mode,path=path,width=1.5,radius_x=.8,radius_y=.8)
        x,y=points_ref(path,32,32,.8,.8,p); v=bilinear_ref(z,x,y); v-=v.mean()
        coeff=2/len(v)*(DFT@v); actual=S.target_coeffs(g,params,SR,110)
        theta=np.arange(1001)*110/SR
        expected=(coeff.real[:,None]*np.cos(2*np.pi*k[:,None]*theta)-coeff.imag[:,None]*np.sin(2*np.pi*k[:,None]*theta)).sum(axis=0)
        got=S.evaluate(actual,theta)
        scan_checks.append(dict(surface=mode,path=path,surface_max_error=float(abs(z-z_impl).max()),
             coefficient_max_error=float(abs(coeff-actual).max()),evaluation_max_error=float(abs(expected-got).max()),
             raw_rms=rms(v),retained_energy_fraction=float(np.sum(abs(coeff)**2)/2/max(rms(v)**2,1e-30))))
assert max(c['evaluation_max_error'] for c in scan_checks)<1e-10
result['checks']['scan_independent']=scan_checks

net_checks=[]
theta=np.arange(16384)/16384
for sid in ('sn_n_links_top','sn_n_links_bottom','sn_n_field_left','sn_n_field_right','sn_n_frozen'):
    ng=grid_doc(loaded[sid][0]['scene']); w=weights_ref(ng)
    ref=network_ref(theta,w,2); impl=N.pm_network(theta,w,2)[1]
    F=np.fft.rfft(ref)/len(ref); powr=abs(F)**2
    net_checks.append(dict(scene=sid,W=w.tolist(),weight_max_error=float(abs(w-N.field_weights(ng)).max()),
        network_max_error=float(abs(ref-impl).max()),energy_above_fourth=float(powr[5:].sum()/max(powr[1:].sum(),1e-30))))
assert max(c['network_max_error'] for c in net_checks)<1e-10
result['checks']['network_independent']=net_checks

# Independent continuous reference: smoothing and known interventions, then full convolution.
ctx=EngineContext(SR,BLOCK,2,110,1,2); gain=.04*.7
params=dict(registry.defaults('pm_network'),trim_db=0,coupling=2,freeze_links=0)
top=grid_doc(loaded['sn_n_links_top'][0]['scene']); bottom=1-top
e=N.PMNetworkEngine(ctx,params); e.init(top,None,gain)
events={8:('field',bottom),13:('beta',4),18:('freeze',True),23:('field',np.zeros_like(top)),29:('freeze',False),35:('field',top)}
current_grid=top.copy(); w=weights_ref(top); target=w.copy(); beta=beta_target=2.; gate=0.; frozen=False
raw_parts=[]; observed=[]; R=4; n=BLOCK*R; isr=SR*R
a=np.exp(-1/(.03*isr)); j=np.arange(1,n+1)
for b in range(45):
    if b in events:
        kind,value=events[b]
        if kind=='field':
            current_grid=value.copy(); e.update_field(value,None)
            if not frozen: target=weights_ref(value)
        elif kind=='beta':
            beta_target=float(value); params=dict(params,coupling=value); e.set_params(params)
        elif kind=='freeze':
            frozen=value; target=w.copy() if frozen else weights_ref(current_grid)
            params=dict(params,freeze_links=int(value)); e.set_params(params)
    wt=np.array([signal.lfilter([1-a],[1,-a],np.full(n,v),zi=[a*old])[0] for old,v in zip(w,target)])
    bt=signal.lfilter([1-a],[1,-a],np.full(n,beta_target),zi=[a*beta])[0]
    gt=np.clip(gate+(1 if current_grid.any() else -1)*j/(.02*isr),0,1)
    th=np.arange(b*n,(b+1)*n)*110/isr
    raw_parts.append(network_ref(th,wt,bt)*gt)
    w=wt[:,-1]; beta=bt[-1]; gate=gt[-1]
    observed.append(e.render_float(gain)[0])
raw=np.concatenate(raw_parts)
taps=signal.firwin(2*R*32+1,.476*SR,fs=R*SR,window=('kaiser',8.))
y=signal.fftconvolve(raw,taps)[:len(raw):R]
t=math.tan(math.pi*5/SR); a1=(1-t)/(1+t); b0=(1+a1)/2
y=signal.lfilter([b0,-b0],[1,-a1],y)*gain*8
got=np.concatenate(observed); err=float(abs(y-got).max())
result['checks']['network_dynamic_oracle']=dict(max_error=err,residual_db=ratio_db(y-got,y),events=['field','coupling','freeze','empty','unfreeze','refill'])
assert err<1e-9

# Pipeline sensitivity from ACTUAL recorded snapshots: same phase/history, one future intervention.
def future(st, edits=(), blocks=50):
    r=DemoRunner.from_state(st)
    r.post('pause',on=True)
    for row,col,val in edits: r.post('set_cell',r=int(row),c=int(col),v=int(val))
    out=[]
    for _ in range(blocks): out.append(r.next_block())
    return {o:np.concatenate([b.get(o) for b in out]) for o in ('A','B','monitor')},r
scan_st=loaded['sn_s_path_ellipse_lissajous'][2]
base,_=future(scan_st); sg=scan_st['grid']
edit_scan=[]
for r,c in ((16,16),(16,28),(4,16),(0,0)):
    edited,rr=future(scan_st,[(r,c,1-int(sg[r,c]))])
    edit_scan.append(dict(cell=[r,c],delta_db={o:ratio_db(edited[o][-SR//5:,0].astype(float)-base[o][-SR//5:,0].astype(float),base[o][-SR//5:,0]) for o in ('A','B')},
        field_reached=bool(rr.grid[r,c]!=sg[r,c])))
result['checks']['scan_actual_runner_edits']=edit_scan
net_st=loaded['sn_n_links_top'][2]; base,_=future(net_st)
edits=[(r,c,int(bottom[r,c])) for r,c in np.argwhere(net_st['grid']!=bottom)]
changed,rr=future(net_st,edits)
result['checks']['network_actual_runner_top_to_bottom']=dict(
    field_reached=bool(np.array_equal(rr.grid,bottom)),
    A_exact=bool(np.array_equal(base['A'],changed['A'])),
    B_delta_db=ratio_db(changed['B'][-SR//5:,0].astype(float)-base['B'][-SR//5:,0].astype(float),base['B'][-SR//5:,0]))

# Compare saved cross-record tones and spectra, without claiming audibility from a metric.
for a,b,label in [('sn_n_links_top','sn_n_links_bottom','N_top_vs_bottom'),
                  ('sn_n_field_left','sn_n_field_right','N_pond_left_vs_right')]:
    aa=loaded[a][1]['B'][SR:2*SR,0].astype(float); bb=loaded[b][1]['B'][SR:2*SR,0].astype(float)
    result['checks'][label]=dict(residual_db=ratio_db(bb-aa,aa),
        normalized_waveform_cosine=float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb))))

result['checks']['all_saved_records_verified']=all(not r['sound_files_changed'] and all(r['pcm_hash_matches'].values()) and all(r['replay_exact'].values()) and r['source_files_unchanged'] for r in result['records'])
save()
print('RESULT', json.dumps(result['checks'],ensure_ascii=True),flush=True)
```
