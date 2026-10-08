import os, sys
"""Measurement-informed computational replay. All replay episodes are simulations.
This code creates no clinical diagnosis labels and does not evaluate arrhythmia diagnosis.
"""
from pathlib import Path
import numpy as np,csv,json,time
from scipy.signal import butter,sosfiltfilt,find_peaks,welch
from scipy.ndimage import uniform_filter1d
FS=500; N=10000; ROOT=Path(os.environ.get('JMS_ANALYSIS_DIR', str(Path(__file__).resolve().parents[1]/'source_data')))
S5=butter(2,[5,20],btype='bandpass',fs=FS,output='sos')
PRE=100; POST=250

def detect(x,mode='energy'):
 q=sosfiltfilt(S5,x)
 if mode=='energy':
  e=uniform_filter1d(q*q,size=40,mode='nearest'); med=np.median(e);hi=np.quantile(e,.98)
  peaks,_=find_peaks(e,height=med+.30*(hi-med),prominence=.15*(hi-med),distance=125)
 else:
  e=np.abs(q);hi=np.quantile(e,.98)
  peaks,_=find_peaks(e,height=.65*hi,prominence=.45*hi,distance=125)
 aligned=[]
 for r in peaks:
  lo=max(0,r-50); high=min(len(q),r+51)
  aligned.append(lo+int(np.argmax(np.abs(q[lo:high]))))
 keep=[]
 for r in sorted(set(aligned)):
  if not keep or r-keep[-1]>=100:keep.append(r)
  elif abs(q[r])>abs(q[keep[-1]]):keep[-1]=r
 return np.array(keep,dtype=int)

def match(a,b,tol=38):
 a=np.sort(a);b=np.sort(b);i=j=tp=0
 while i<len(a) and j<len(b):
  if abs(a[i]-b[j])<=tol:tp+=1;i+=1;j+=1
  elif a[i]<b[j]:i+=1
  else:j+=1
 return tp

def agreement(a,b):
 return 2*match(a,b)/(len(a)+len(b)) if len(a)+len(b) else 0.

def score(pred,ref):
 pred=pred[(pred>=1000)&(pred<9000)];ref=ref[(ref>=1000)&(ref<9000)]
 tp=match(pred,ref);fp=len(pred)-tp;fn=len(ref)-tp
 return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

def template(x,peaks):
 b=np.array([x[r-PRE:r+POST] for r in peaks if r>=PRE and r+POST<len(x)])
 if len(b)<5:return None
 t=np.median(b,axis=0);t=t-np.linspace(t[0],t[-1],len(t))
 w=np.ones(len(t));w[:30]=np.sin(np.linspace(0,np.pi/2,30))**2;w[-30:]=w[:30][::-1]
 return t*w[:,None]

def reconstruct(t,peaks,n=N):
 out=np.zeros((n,3))
 for r in peaks:
  a,b=r-PRE,r+POST
  if a>=0 and b<=n:out[a:b]+=t
 return out


def save_banks(path, targets, noise):
    nmax=max(len(t['events']) for t in targets)
    events=np.full((len(targets),nmax),-1,dtype=np.int64)
    for i,t in enumerate(targets): events[i,:len(t['events'])]=t['events']
    np.savez_compressed(path,target_ids=np.array([t['id'] for t in targets]),target_groups=np.array([t['group'] for t in targets]),patterns=np.array([t['pattern'] for t in targets]),clean=np.stack([t['clean'] for t in targets]),events=events,templates=np.stack([t['template'] for t in targets]),residual_ids=np.array([n['id'] for n in noise]),residual_groups=np.array([n['group'] for n in noise]),residual_stages=np.array([n['stage'] for n in noise]),residuals=np.stack([n['residual'] for n in noise]))

def load_banks(path):
    with np.load(path,allow_pickle=False) as b:
        targets=[dict(id=str(i),group=str(g),pattern=str(p),clean=x,events=e[e>=0],template=t) for i,g,p,x,e,t in zip(b['target_ids'],b['target_groups'],b['patterns'],b['clean'],b['events'],b['templates'])]
        noise=[dict(id=str(i),group=str(g),stage=int(s),residual=r) for i,g,s,r in zip(b['residual_ids'],b['residual_groups'],b['residual_stages'],b['residuals'])]
    return targets,noise

def make_banks():
 targets=[];noise=[];meta=[]
 for gi,g in enumerate(['G1','G2','G3','G4']):
  z=np.load(ROOT/f'{g}_processed.npz');xf=z['xf'][:,[0,1,6]];phase=z['phase']
  for stage in [0,5]:
   xx=xf[phase==stage]
   for wi,offset in enumerate([20,60,100,140]):
    x=xx[offset*FS:(offset+20)*FS];r=detect(x[:,1]);t=template(x,r)
    if t is None:meta.append(dict(kind='template_excluded',group=g,stage=stage,offset=offset,reason='fewer_than_5_candidates'));continue
    rr=np.clip(np.median(np.diff(r))/FS,.55,1.2)
    for pattern in ['regular','variable']:
     seed=1000+gi*100+stage*10+wi;rng=np.random.default_rng(seed);events=[];pos=FS
     while pos<(N-POST-1):
      events.append(int(pos));pos+=int(round(FS*rr*(1 if pattern=='regular' else np.clip(rng.normal(1,.18),.55,1.45))))
     events=np.array(events);clean=reconstruct(t,events);ds=score(detect(clean[:,1]),events);as_=score(detect(clean[:,1],'amplitude'),events)
     keep=ds['f1']>=.99 and as_['f1']>=.99
     meta.append(dict(kind='target',group=g,stage=stage,offset=offset,pattern=pattern,clean_f1=ds['f1'],clean_f1_amplitude=as_['f1'],eligible=bool(keep)))
     if keep:targets.append(dict(id=f'{g}_p{stage}_w{wi}_{pattern}',group=g,pattern=pattern,clean=clean,events=events,template=t))
  for stage in [1,2,3,4]:
   xx=xf[phase==stage];x=xx[20*FS:40*FS];r=detect(x[:,1]);t=template(x,r)
   if t is None:raise ValueError(f'insufficient candidate beats in residual {g} {stage}')
   residual=x-reconstruct(t,r);residual-=residual.mean(axis=0);residual*=np.hanning(N)[:,None];residual-=residual.mean(axis=0)
   noise.append(dict(id=f'{g}_p{stage}',group=g,stage=stage,residual=residual))
 json.dump(meta,open(ROOT/'bank_eligibility.json','w'),indent=2)
 save_banks(ROOT/'replay_banks.npz', targets, noise)
 return targets,noise

def features(x):
 f,p=welch(x,fs=FS,nperseg=1000,noverlap=500,axis=0)
 tot=p[(f>=.5)&(f<=40)].sum(axis=0)+1e-12;ans=[];names=[]
 for j,l in enumerate(['I','II','V1']):
  for lo,hi,tag in [(.5,5,'slow'),(5,15,'qrs'),(15,40.01,'fast')]:
   ans.append(p[(f>=lo)&(f<hi),j].sum()/tot[j]);names.append(f'{l}_{tag}_fraction')
  v=x[:,j];s=np.std(v)+1e-9;ans.extend([np.log(s),np.mean((v-v.mean())**4)/(s**4),np.quantile(np.abs(np.diff(v)),.99)/(s)])
  names.extend([f'{l}_log_sd',f'{l}_kurtosis',f'{l}_diff99'])
 cc=np.corrcoef(x.T);ans.extend(cc[np.triu_indices(3,1)]);names.extend(['corr_I_II','corr_I_V1','corr_II_V1']);global_n=len(ans)
 r=[detect(x[:,j]) for j in range(3)];ra=detect(x[:,1],'amplitude')
 ans.extend([agreement(r[1],r[0]),agreement(r[1],r[2]),agreement(r[1],ra)]);names.extend(['beat_agreement_II_I','beat_agreement_II_V1','detector_agreement_II'])
 for j,l in enumerate(['I','II','V1']):
  beats=np.array([x[k-60:k+100,j] for k in r[1] if k>=60 and k+100<len(x)])
  if len(beats)>2:
   beats-=np.mean(beats,axis=1,keepdims=True);b=beats/(np.linalg.norm(beats,axis=1,keepdims=True)+1e-12);t=np.median(b,axis=0);t/=np.linalg.norm(t)+1e-12;cs=b@t
   ans.extend([float(np.median(cs)),float(np.quantile(cs,.1))])
  else:ans.extend([0.,0.])
  names.extend([f'{l}_beat_cos_median',f'{l}_beat_cos_q10'])
 return np.nan_to_num(ans),names,global_n,r[1],ra

def main():
 start=time.time();targets,noise=(load_banks(ROOT/'replay_banks.npz') if '--use-bank' in sys.argv else make_banks());print('banks',len(targets),len(noise),flush=True)
 rows=[];feat=[];shifts=[0,69,196,367];snrs=[20.,10.,0.,-5.]
 for ti,t in enumerate(targets):
  clean=t['clean'];scale=np.sqrt(np.mean(clean**2))
  for no in noise:
   residual=no['residual'];residual=residual/(np.sqrt(np.mean(residual**2))+1e-12)
   for snr in snrs:
    for shi in shifts:
     perturb=np.roll(residual,shi,axis=0)*scale*10**(-snr/20);x=clean+perturb
     ft,names,gn,rs,ra=features(x);s=score(rs,t['events']);sa=score(ra,t['events'])
     rows.append(dict(target=t['id'],target_group=t['group'],pattern=t['pattern'],residual=no['id'],residual_group=no['group'],activity_phase=no['stage'],snr_db=snr,shift_samples=shi,noise_rms_mV=float(np.sqrt(np.mean(perturb**2))),f1=s['f1'],f1_amplitude=sa['f1'],tp=s['tp'],fp=s['fp'],fn=s['fn'],failure=int(s['f1']<.95)))
     feat.append(ft)
  if ti%4==0:print('target',ti,'of',len(targets),'time',round(time.time()-start,1),flush=True)
 with open(ROOT/'replay_outcomes.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(ROOT/'reliability_features.npz',X=np.array(feat),names=names,global_n=gn)
 print('DONE',len(rows),time.time()-start,flush=True)
if __name__=='__main__':main()
