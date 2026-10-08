"""Render each scientific panel separately, then assemble the seven figure plates.
No new recordings are created. All plotted values come from saved measurements,
explicit replay reconstruction, or deterministic summaries of saved outcomes.
"""
from pathlib import Path
import os, sys, json, math, string, shutil
import numpy as np, pandas as pd
from scipy.signal import welch
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import fitz

O=Path(os.environ.get('JMS_PACKAGE_DIR',str(Path(__file__).resolve().parents[1])))
D=Path(os.environ.get('JMS_TABLE_DIR',str(O/'source_data')))
F=Path(os.environ.get('JMS_FIGURE_OUTPUT',str(O/'figures')))
P=F/'panels';P.mkdir(parents=True,exist_ok=True)
RAW=Path(os.environ.get('JMS_RAW_CACHE',str(O/'author_only/raw_cache')))
BANK=Path(os.environ.get('JMS_REPLAY_BANK',str(O/'author_only/replay_banks.npz')))
sys.path.insert(0,str(O/'analysis_code'))
import run_stress as rs
r=pd.read_csv(D/'replay_outcomes.csv');h=pd.read_csv(D/'heldout_predictions.csv');q=pd.read_csv(D/'quartet_details.csv')
spec=pd.read_csv(D/'spectral_profiles.csv');fold=pd.read_csv(D/'fold_metrics.csv');ss=pd.read_csv(D/'sensitivity_analyses.csv')
ret=pd.read_csv(D/'retention_full_grid.csv'); pat=pd.read_csv(D/'pattern_metrics.csv'); ps=pd.read_csv(D/'pattern_retention.csv')
coef=json.load(open(D/'fitted_reliability_models.json'));z=np.load(BANK,allow_pickle=False)
T,N=rs.load_banks(BANK);td={x['id']:x for x in T};nd={x['id']:x for x in N}
sel=json.load(open(D/'illustrative_case_selection.json'))
raw={g:dict(np.load(RAW/f'{g}_processed.npz')) for g in ['G1','G2','G3','G4']}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.7,'axes.linewidth':.7,'lines.linewidth':1.0,'lines.markersize':3,'xtick.direction':'in','ytick.direction':'in','xtick.major.width':.7,'ytick.major.width':.7,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
SN=[20,10,0,-5];G=['G1','G2','G3','G4'];MOD=['Global','Event','Combined','Agreement']; manifest=[]
# Widths are final publication widths; no reduction after assembly.
NARROW=2.30;WIDE=3.47;HEIGHT=2.18
layouts={1:[2,3,3],2:[3,3,3],3:[2,3,3],4:[3,3,3],5:[3,3,3],6:[2,3,3],7:[2,3,3]}

def panel(num,letter,width=None):
 idx=string.ascii_lowercase.index(letter);counts=layouts[num];row=next(i for i in range(len(counts)) if idx<sum(counts[:i+1]));w=(WIDE if counts[row]==2 else NARROW) if width is None else width
 fig=plt.figure(figsize=(w,HEIGHT));ax=fig.add_axes([.19 if w==NARROW else .135,.22,.76 if w==NARROW else .82,.71])
 fig.text(.018,.98,letter,fontweight='bold',fontsize=11,va='top')
 ax.tick_params(length=2.7,pad=2);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 return fig,ax

def done(fig,num,letter,description,source):
 # Keep every tick, label and legend inside its standalone panel before assembly.
 for _pass in range(5):
  fig.canvas.draw(); ren=fig.canvas.get_renderer(); boxes=[ax.get_tightbbox(ren) for ax in fig.axes]
  left=min(b.x0 for b in boxes)/fig.bbox.width;right=max(b.x1 for b in boxes)/fig.bbox.width
  bottom=min(b.y0 for b in boxes)/fig.bbox.height;top=max(b.y1 for b in boxes)/fig.bbox.height
  dl=max(0,.025-left);dr=max(0,right-.985);db=max(0,.035-bottom);dt=max(0,top-.985)
  if max(dl,dr,db,dt)<.001:break
  poses=[ax.get_position().frozen() for ax in fig.axes];xmin=min(p.x0 for p in poses);xmax=max(p.x1 for p in poses);ymin=min(p.y0 for p in poses);ymax=max(p.y1 for p in poses)
  sx=max(.45,(xmax-xmin-dl-dr)/(xmax-xmin));sy=max(.45,(ymax-ymin-db-dt)/(ymax-ymin))
  for ax,pos in zip(fig.axes,poses):ax.set_position([xmin+dl+(pos.x0-xmin)*sx,ymin+db+(pos.y0-ymin)*sy,pos.width*sx,pos.height*sy])
 stem=f'Fig{num}{letter}';fig.savefig(P/(stem+'.pdf'));fig.savefig(P/(stem+'.svg'));fig.savefig(P/(stem+'.png'),dpi=300);plt.close(fig)
 manifest.append({'figure':num,'panel':letter,'description':description,'source':source,'panel_file':stem+'.svg'})

def xy(ax,x,y):ax.set_xlabel(x,labelpad=2);ax.set_ylabel(y,labelpad=2)
def legend(ax,**kwargs):ax.legend(frameon=False,handlelength=1.5,handletextpad=.4,borderaxespad=.2,labelspacing=.25,**kwargs)
def heat(fig,ax,a,xlabels,ylabels,fmt=None,cbar=True,vmin=None,vmax=None):
 im=ax.imshow(a,aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax)
 ax.set_xticks(range(len(xlabels)),xlabels);ax.set_yticks(range(len(ylabels)),ylabels)
 if fmt:
  for i in range(len(ylabels)):
   for j in range(len(xlabels)):ax.text(j,i,format(a[i,j],fmt),ha='center',va='center',fontsize=7,bbox=dict(boxstyle='square,pad=.05',facecolor='white',edgecolor='none',alpha=.65))
 if cbar:
  cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.025);cb.ax.tick_params(labelsize=6.5,length=2,pad=1)
 return im

def build_episode(row):
 t=td[row['target']];n=nd[row['residual']]['residual'];pert=np.roll(n/(np.sqrt(np.mean(n*n))+1e-12),int(row['shift_samples']),axis=0)*np.sqrt(np.mean(t['clean']**2))*10**(-float(row['snr_db'])/20)
 return t['clean'],pert,t['clean']+pert,t['events']
def marker_trace(ax,row,interval=(6,10),showtext=True):
 clean,pert,x,events=build_episode(row);pred=rs.detect(x[:,1]);a,b=np.array(interval)*500; a=int(a);b=int(b);tt=np.arange(a,b)/500
 ax.plot(tt,x[a:b,1],label='Replay');ev=events[(events>=a)&(events<b)];pr=pred[(pred>=a)&(pred<b)]
 ax.plot(ev/500,x[ev,1],linestyle='none',marker='o',mfc='none',ms=5,label='Reference');ax.plot(pr/500,x[pr,1],linestyle='none',marker='x',ms=4,label='Detected')
 xy(ax,'Time (s)','Lead II (mV)');legend(ax,loc='upper right',fontsize=6)
 if showtext:ax.text(.02,.02,f"F1 = {row['f1']:.3f}",transform=ax.transAxes,fontsize=7,va='bottom')

def cov_matrix(v):return np.cov(v.T,bias=True)
# FIGURE 1: measurements and acquired-channel representation.
f,a=panel(1,'a'); dur=[180,60,60,60,60,180];left=0
for j,dt in enumerate(dur):
 a.barh(0,dt,left=left,height=.44,label=['Rest','Breathing','Arm raise','Turning','Stepping','Recovery'][j]);a.text(left+dt/2,0,str(j+1),ha='center',va='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.6,pad=.6));left+=dt
a.set_xlim(0,600);a.set_ylim(-.48,.65);a.set_yticks([]);a.text(0,.44,'Four sequential recordings · 500 Hz',fontsize=8);xy(a,'Scheduled time (s)','');legend(a,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=6.4)
done(f,1,'a','Recorded six-stage protocol and source count','source_manifest.json; acquisition phase metadata')
f,a=panel(1,'b');A=np.array([[1,0,0],[0,1,0],[-1,1,0],[-.5,-.5,0],[1,-.5,0],[-.5,1,0],[0,0,1]])
heat(f,a,A,['I','II','V1'],['I','II','III','aVR','aVL','aVF','V1'],fmt='.1f',cbar=False);xy(a,'Acquired signals','Displayed leads');a.text(.98,1.02,'rank = 3',transform=a.transAxes,ha='right',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
done(f,1,'b','Exact three-signal to seven-lead linear map','Methods lead definition; code matrix A')
for letter,phase,offset in [('c',0,20),('d',3,20)]:
 f,a=panel(1,letter)
 vals=[raw[g]['x'][raw[g]['phase']==phase][offset*500:(offset+5)*500,1] for g in G]
 # Subtract within-segment median and stack by a common 99% range; unchanged physical scaling.
 step=max(np.quantile(v,.99)-np.quantile(v,.01) for v in vals)*1.12
 for j,(g,v) in enumerate(zip(G,vals)):a.plot(np.arange(len(v))/500,v-np.median(v)+j*step,label=g)
 a.set_yticks(np.arange(4)*step,G);xy(a,'Time within excerpt (s)','Centered lead II');a.text(.98,.96,f'Offset = {step:.2f} mV',transform=a.transAxes,ha='right',va='top',fontsize=6.8)
 done(f,1,letter,'Raw lead-II excerpts, '+('initial rest' if phase==0 else 'body turning'),'raw CSV; phase '+str(phase)+'; 20–25 s; centered/offset for display only')
for letter,kind in [('e','x'),('f','xf')]:
 f,a=panel(1,letter)
 for g in G:
  xx=raw[g][kind][raw[g]['phase']==3,1];freq,pp=welch(xx,fs=500,nperseg=10000,noverlap=5000);m=(freq>=.05)&(freq<=100 if kind=='x' else freq<=40);a.loglog(freq[m],pp[m]*1e6,label=g)
 xy(a,'Frequency (Hz)',r'PSD ($\mu$V$^2$/Hz)');legend(a,loc='lower left',ncol=2);a.set_ylim(bottom=1e-4 if kind=='x' else 1e-6)
 done(f,1,letter,('Raw' if kind=='x' else 'Filtered')+' turning-phase Welch spectra','raw CSV; phase 3; full stage; existing preprocessing')
for letter,state,xcol,ycol,xlab,ylab in [('g','raw','line_rms_uV','low_rms_uV',r'49–51 Hz RMS ($\mu$V)',r'0.05–0.5 Hz RMS ($\mu$V)'),('h','filtered','qrs_rms_uV','slow_rms_uV',r'5–15 Hz RMS ($\mu$V)',r'0.5–5 Hz RMS ($\mu$V)')]:
 f,a=panel(1,letter)
 for g in G:
  s=spec[(spec.group==g)&(spec.lead=='II')&(spec.preprocessing==state)].sort_values('phase');a.loglog(s[xcol],s[ycol],'o',ms=4,label=g)
  for _,v in s.iterrows():a.annotate(str(int(v.phase)+1),(v[xcol],v[ycol]),xytext=(2,2),textcoords='offset points',fontsize=5.8)
 xy(a,xlab,ylab);legend(a,loc='upper left',ncol=2)
 done(f,1,letter,'All-stage lead-II component comparison, '+state,'spectral_profiles.csv; all six stages, lead II')
print('Figure 1 panels complete')
# FIGURE 2: measured templates, subtraction and source-bank diversity.
source=raw['G2'];rest=source['xf'][source['phase']==0][:,[0,1,6]][20*500:40*500];beats=rs.detect(rest[:,1]);template=rs.template(rest,beats)
f,a=panel(2,'a');t=np.arange(2500)/500;a.plot(t,rest[:2500,1]);b=beats[beats<2500];a.plot(b/500,rest[b,1],'o',mfc='none',ms=4);xy(a,'Time within source window (s)','Lead II (mV)')
done(f,2,'a','Resting source window with extraction candidates','G2 phase 0, 20–25 s')
f,a=panel(2,'b');snips=np.array([rest[k-100:k+250,1] for k in beats if k>=100 and k+250<len(rest)]);tt=np.arange(-100,250)/500
for v in snips:a.plot(tt,v,lw=.55,alpha=.28)
a.plot(tt,np.median(snips,axis=0),lw=1.5,label='Median');xy(a,'Candidate-relative time (s)','Lead II (mV)');legend(a)
done(f,2,'b','Aligned snippets and measured median','G2 phase 0 source window; candidate-centered snippets')
f,a=panel(2,'c')
for j,l in enumerate(['I','II','V1']):a.plot(tt,template[:,j],label=l)
xy(a,'Candidate-relative time (s)','Template (mV)');legend(a)
done(f,2,'c','Three-signal morphology template after endpoint processing','G2 p0 w0 template; saved bank')
active=source['xf'][source['phase']==3][:,[0,1,6]][20*500:40*500];bt=rs.detect(active[:,1]);tm=rs.template(active,bt);recon=rs.reconstruct(tm,bt)
f,a=panel(2,'d');sl=slice(3000,5000);t=np.arange(3000,5000)/500;a.plot(t,active[sl,1],label='Measured');a.plot(t,recon[sl,1],label='Reconstructed',ls='--');xy(a,'Time within source window (s)','Lead II (mV)');legend(a,fontsize=6)
done(f,2,'d','Activity trace and local template reconstruction','G2 phase 3, source-relative 6–10 s')
f,a=panel(2,'e');res=nd['G2_p3']['residual']
for j,l in enumerate(['I','II','V1']):a.plot(np.arange(10000)/500,res[:,j],label=l,lw=.7)
xy(a,'Time within residual (s)','Residual (mV)');legend(a,ncol=3,fontsize=6)
done(f,2,'e','Joint activity-associated residual','G2_p3 final demeaned/tapered residual, saved bank')
f,a=panel(2,'f');tm=z['templates'][::2,:,1];v=tm/(np.linalg.norm(tm,axis=1,keepdims=True)+1e-12);corr=v@v.T;im=a.imshow(corr,aspect='equal',interpolation='nearest',vmin=float(corr.min()),vmax=1);a.set_xticks([3.5,11.5,19.5,27.5],G);a.set_yticks([3.5,11.5,19.5,27.5],G);xy(a,'Template source','Template source');cb=f.colorbar(im,ax=a,fraction=.046,pad=.03);cb.ax.tick_params(labelsize=6)
done(f,2,'f','Cosine similarity of 32 distinct lead-II templates','replay_banks.npz templates; one per source window')
f,a=panel(2,'g');fr=[]
for rr in z['residuals']:
 freq,powr=welch(rr[:,1],fs=500,nperseg=1000,noverlap=500);total=powr[(freq>=.5)&(freq<=40)].sum();fr.append([powr[(freq>=lo)&(freq<hi)].sum()/total for lo,hi in [(.5,5),(5,15),(15,40.01)]])
heat(f,a,np.array(fr),['0.5–5','5–15','15–40'],[f'{g}/{s}' for g in G for s in range(2,6)],cbar=True,vmin=0,vmax=1);xy(a,'Band (Hz)','Residual source / stage');a.tick_params(axis='y',labelsize=5.8)
pd.DataFrame(fr,index=z['residual_ids'],columns=['slow','qrs','fast']).to_csv(D/'residual_spectral_fractions.csv')
done(f,2,'g','Frequency-band fractions in all sixteen residuals','Saved residuals; lead II Welch PSD')
f,a=panel(2,'h');cc=[]
for rr in z['residuals']:cc.append(np.corrcoef(rr.T)[np.triu_indices(3,1)])
heat(f,a,np.array(cc),['I/II','I/V1','II/V1'],[f'{g}/{s}' for g in G for s in range(2,6)],cbar=True,vmin=-1,vmax=1);xy(a,'Signal pair','Residual source / stage');a.tick_params(axis='y',labelsize=5.8)
pd.DataFrame(cc,index=z['residual_ids'],columns=['I_II','I_V1','II_V1']).to_csv(D/'residual_correlation.csv')
done(f,2,'h','Cross-signal correlation in all residuals','Saved residuals; full 20-s segments')
f,a=panel(2,'i');
for j,p in enumerate(['regular','variable']):
 t=td['G2_p0_w0_'+p];a.plot(np.arange(2500)/500,t['clean'][:2500,1]+j*.5,label=p.capitalize())
xy(a,'Generated time (s)','Lead II + offset (mV)');legend(a,fontsize=6.4)
done(f,2,'i','Matched-morphology targets with two generated timing patterns','Saved targets G2_p0_w0_regular/variable; interval patterns are not diagnoses')
print('Figure 2 panels complete')
# FIGURE 3: controlled alignment example and all-bank invariant verification.
rows=sel['matched_quartet'];best=max(rows,key=lambda v:v['f1']);worst=min(rows,key=lambda v:v['f1']);clean,p1,x1,events=build_episode(best);_,p0,x0,_=build_episode(worst)
f,a=panel(3,'a');sl=slice(2500,5000);tt=np.arange(2500,5000)/500;a.plot(tt,clean[sl,1],label='Generated target');ev=events[(events>=2500)&(events<5000)];a.plot(ev/500,clean[ev,1],'o',mfc='none',label='Placed beat',ms=4);xy(a,'Time (s)','Target lead II (mV)');legend(a,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2,fontsize=7)
done(f,3,'a','Fixed clean target with known placed beats','Selected quartet target; illustrative selection rule saved')
f,a=panel(3,'b');a.plot(tt,p0[sl,1],label=f"{int(worst['shift_samples']*2)} ms");a.plot(tt,p1[sl,1],ls='--',label=f"{int(best['shift_samples']*2)} ms");xy(a,'Time (s)','Added residual (mV)');legend(a,ncol=2)
done(f,3,'b','Identical residual samples under two joint rotations','Selected quartet; -5 dB scale fixed')
f,a=panel(3,'c');marker_trace(a,best,interval=(6,10));done(f,3,'c','Nonfailure rotation with reference and detected events','Selected within-source quartet; best rotation; F1 over central 16 s')
f,a=panel(3,'d');marker_trace(a,worst,interval=(6,10));done(f,3,'d','Failure rotation with reference and detected events','Same quartet and time window; worst rotation; F1 over central 16 s')
f,a=panel(3,'e');freq=np.fft.rfftfreq(10000,1/500)
for p,label,ls in [(p0,'0 ms','-'),(p1,'392 ms','--')]:
 powr=np.abs(np.fft.rfft(p[:,1]))**2/len(p);m=(freq>=.5)&(freq<=40);a.semilogy(freq[m],powr[m],label=label,ls=ls,lw=.8)
xy(a,'Frequency (Hz)','Residual Fourier power');legend(a)
done(f,3,'e','Overlapping perturbation periodograms','Explicit full-segment rFFT; pertains to perturbation, not corrupted signal')
f,a=panel(3,'f');c0=cov_matrix(p0);c1=cov_matrix(p1);ix=np.triu_indices(3);xx=c0[ix];yy=c1[ix];a.plot(xx,yy,'o',mfc='none',ms=5);ab=[min(xx.min(),yy.min()),max(xx.max(),yy.max())];a.plot(ab,ab,ls='--',lw=.7);xy(a,r'Covariance at 0 ms (mV$^2$)',r'Covariance at 392 ms (mV$^2$)');a.ticklabel_format(style='sci',scilimits=(0,0),axis='both',useMathText=True)
done(f,3,'f','Invariant zero-lag covariance entries','Three variances and three unique covariances of selected perturbation')
f,a=panel(3,'g');a.plot([v['shift_samples']*2 for v in rows],[v['f1'] for v in rows],'o-',label='Energy');a.plot([v['shift_samples']*2 for v in rows],[v['f1_amplitude'] for v in rows],'s--',label='Amplitude');a.axhline(.95,ls=':',lw=.8);xy(a,'Joint rotation (ms)','Beat F1');a.set_ylim(.7,1.015);legend(a,loc='lower right')
done(f,3,'g','Both detector outcomes across all four rotations','Selected quartet; both scores read from replay_outcomes.csv')
f,a=panel(3,'h');iv=pd.read_csv(D/'rotation_invariance_all_residuals.csv').groupby('residual').max().reindex(z['residual_ids'])
for metric,lab,m in [('periodogram_error','Spectrum','o'),('energy_error','Energy','s'),('covariance_error','Covariance','^')]:a.semilogy(np.arange(1,17),np.maximum(iv[metric].values,1e-18),m+'-',ms=3,lw=.8,label=lab)
a.axhline(1e-12,ls=':',lw=.8);a.set_ylim(8e-19,2e-12);a.set_xticks([1,4,8,12,16]);xy(a,'Residual index','Maximum relative discrepancy');legend(a,loc='upper center',ncol=1,fontsize=6)
done(f,3,'h','Rotation-invariance checks for every residual','rotation_invariance_all_residuals.csv; errors computed from arrays')
print('Figure 3 panels complete')
# FIGURE 4: complete factorial results and invariant-descriptor bound.
f,a=panel(4,'a')
for col,lab,m in [('f1','Energy','o'),('f1_amplitude','Amplitude','s')]:
 vals=[r[r.snr_db==snr][col] for snr in SN];means=np.array([v.mean() for v in vals]);a.plot(range(4),means,m+'-',label=lab);a.fill_between(range(4),[v.quantile(.25) for v in vals],[v.quantile(.75) for v in vals],alpha=.12)
a.set_xticks(range(4),SN);a.set_ylim(.65,1.015);xy(a,'Signal-to-perturbation ratio (dB)','Beat F1');legend(a)
done(f,4,'a','Mean and interquartile beat F1 at all severity levels','All 16,384 replays; shaded regions are episode IQR, not confidence intervals')
f,a=panel(4,'b')
for col,lab,m in [('fp','False positives','o'),('fn','Missed beats','s')]:
 vv=[100*r[r.snr_db==snr][col].sum()/(r[r.snr_db==snr].tp.sum()+r[r.snr_db==snr].fn.sum()) for snr in SN];a.plot(range(4),vv,m+'-',label=lab)
a.set_xticks(range(4),SN);xy(a,'Signal-to-perturbation ratio (dB)','Count per 100 reference beats');legend(a,loc='upper left',fontsize=6)
done(f,4,'b','False and missed beat counts','Primary detector aggregate TP/FP/FN; all replays')
f,a=panel(4,'c');vals=[q[q.snr_db==snr].straddles_failure.mean()*100 for snr in SN];a.bar(range(4),vals,width=.65)
for j,v in enumerate(vals):a.text(j,v+.6,f'{v:.2f}',ha='center',fontsize=7)
a.set_ylim(0,28);a.set_xticks(range(4),SN);xy(a,'Signal-to-perturbation ratio (dB)','Quartets crossing failure (%)')
done(f,4,'c','Crossing of failure boundary under rotation alone','4,096 quartets, 1,024 at each ratio')
f,a=panel(4,'d')
for snr in SN:
 v=np.sort(q[q.snr_db==snr].f1_range.values);a.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{snr} dB')
xy(a,'Within-quartet F1 range','Cumulative fraction');legend(a,loc='lower right',fontsize=6)
done(f,4,'d','Distribution of the alignment response across all quartets','matched_alignment_quartets.csv; no quartet excluded')
f,a=panel(4,'e');m=r.pivot_table(index='target_group',columns='residual_group',values='failure',aggfunc='mean').reindex(index=G,columns=G).values*100
heat(f,a,m,G,G,fmt='.1f',cbar=False,vmin=0,vmax=100);xy(a,'Residual source','Target source')
done(f,4,'e','Failure percentages for all target/residual source pairs','All four ratios and four rotations; 1,024 replays per cell')
f,a=panel(4,'f');m=q.pivot_table(index='activity_phase',columns='snr_db',values='straddles_failure',aggfunc='mean').reindex(index=[1,2,3,4],columns=SN).values*100
heat(f,a,m,[str(s) for s in SN],['Breath','Arm','Turn','Step'],fmt='.1f',cbar=False,vmin=0,vmax=60);xy(a,'Ratio (dB)','Residual activity')
done(f,4,'f','Alignment-crossing percentages by activity and severity','Quartets; 256 per activity-ratio cell')
f,a=panel(4,'g')
for pattern,m in [('regular','o'),('variable','s')]:a.plot(range(4),[100*r[(r.snr_db==s)&(r.pattern==pattern)].failure.mean() for s in SN],m+'-',label=pattern.capitalize())
a.set_xticks(range(4),SN);xy(a,'Signal-to-perturbation ratio (dB)','Failure (%)');legend(a,loc='upper left')
done(f,4,'g','Failure by generated timing pattern','All replay outcomes; patterns are not rhythm diagnoses')
f,a=panel(4,'h');by=q.groupby(['target','residual','snr_db']).first();a.scatter(q.min_f1,q.max_f1,s=5,alpha=.32);a.plot([.35,1],[.35,1],ls='--',lw=.8);a.axhline(.95,ls=':',lw=.7);a.axvline(.95,ls=':',lw=.7);xy(a,'Minimum F1 in quartet','Maximum F1 in quartet');a.set_xlim(0,1.02);a.set_ylim(0,1.02)
done(f,4,'h','Best and worst outcomes of each complete quartet','All 4,096 matched quartets; lines mark F1=0.95')
f,a=panel(4,'i');bb=pd.read_csv(D/'rotation_invariant_bound_summary.csv').iloc[:4];a.bar(range(4),bb.minimum_error_rate.values*100,width=.65)
for j,v in enumerate(bb.minimum_error_rate.values*100):a.text(j,v+.2,f'{v:.2f}',ha='center',fontsize=7)
a.set_xticks(range(4),SN);a.set_ylim(0,9);xy(a,'Signal-to-perturbation ratio (dB)','Invariant-rule error floor (%)')
done(f,4,'i','Finite-grid lower bound for rotation-invariant binary rules','Within-quartet sum min(k,4-k); not a bound on the fitted Global model')
print('Figure 4 panels complete')
# FIGURE 5: reliability prediction, calibration and interpretation.
f,a=panel(5,'a');mask=np.full((4,4),1.);mask[:3,:3]=0;mask[3,3]=2;heat(f,a,mask,G,G,cbar=False,vmin=0,vmax=2)
for i in range(4):
 for j in range(4):a.text(j,i,['Train','Omit','Test'][int(mask[i,j])],ha='center',va='center',fontsize=6.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=.15))
xy(a,'Residual source','Target source');a.text(.02,1.025,'Example fold: G4 held out',transform=a.transAxes,fontsize=7)
done(f,5,'a','Joint source exclusion in the held-out evaluation','Exact train/test masks; nine train cells, six omitted, one test')
f,a=panel(5,'b')
for model in MOD:
 fpr,tpr,_=roc_curve(h.failure,h['risk_'+model]);a.plot(fpr,tpr,label=f'{model} {roc_auc_score(h.failure,h["risk_"+model]):.3f}')
a.plot([0,1],[0,1],ls=':',lw=.7);xy(a,'False-positive rate','True-positive rate');legend(a,loc='lower right',fontsize=5.9)
done(f,5,'b','Pooled held-out ROC curves','heldout_predictions.csv; n=4,096')
f,a=panel(5,'c')
for model in MOD:
 pr,re,_=precision_recall_curve(h.failure,h['risk_'+model]);a.plot(re,pr,label=f'{model} {average_precision_score(h.failure,h["risk_"+model]):.3f}')
a.axhline(h.failure.mean(),ls=':',lw=.7);xy(a,'Recall','Precision');legend(a,loc='upper center',bbox_to_anchor=(.5,1.22),ncol=2,fontsize=5.9)
done(f,5,'c','Pooled precision–recall curves','heldout_predictions.csv; dotted prevalence reference')
f,a=panel(5,'d');cal=[]
for model in MOD[:3]:
 p=h['risk_'+model].values;y=h.failure.values;xs=[];ys=[]
 for j in range(10):
  m=(p>=j/10)&((p<(j+1)/10) if j<9 else p<=1)
  if m.any():xs.append(p[m].mean());ys.append(y[m].mean());cal.append(dict(method=model,bin=j,n=m.sum(),predicted=p[m].mean(),observed=y[m].mean()))
 a.plot(xs,ys,'o-',label=model,ms=3)
a.plot([0,1],[0,1],ls=':',lw=.7);xy(a,'Predicted failure probability','Observed failure fraction');legend(a,loc='lower right',fontsize=6)
pd.DataFrame(cal).to_csv(D/'calibration_bins_v2.csv',index=False)
done(f,5,'d','Reliability diagrams with ten equal-width bins','Held-out predictions; no post-test recalibration')
f,a=panel(5,'e')
for cls,lab in [(0,'Nonfailure'),(1,'Failure')]:a.hist(h.loc[h.failure==cls,'risk_Event'],bins=np.linspace(0,1,21),density=True,histtype='step',lw=1,label=lab)
xy(a,'Event failure probability','Density');legend(a,loc='upper center',fontsize=6)
done(f,5,'e','Distribution of Event probabilities by outcome','All held-out episodes')
for letter,col,ylab in [('f','AUROC','AUROC'),('g','ECE_10bin','Calibration error'),('h','Brier','Brier score')]:
 f,a=panel(5,letter)
 for model in MOD[:3]:
  s=fold[fold.method==model].set_index('heldout_record').reindex(G);a.plot(range(4),s[col],'o-',label=model)
 a.set_xticks(range(4),G);xy(a,'Held-out recording',ylab);legend(a,loc='lower left' if col=='AUROC' else 'upper right',fontsize=5.8)
 if col=='AUROC':a.set_ylim(.92,1.005)
 done(f,5,letter,'Source-specific '+ylab,'fold_metrics.csv; 1,024 test episodes per source')
f,a=panel(5,'i');cc=np.array([next(v for v in coef if v['record']==g and v['method']=='Event')['coefficients'] for g in G]);labels=['Beat I','Beat V1','Detector','I median','I q10','II median','II q10','V1 median','V1 q10'];lim=max(abs(cc.min()),abs(cc.max()));heat(f,a,cc.T,G,labels,cbar=True,vmin=-lim,vmax=lim);xy(a,'Held-out fold','Standardized coefficient');a.tick_params(axis='y',labelsize=5.8)
done(f,5,'i','Event-model coefficients from all four training folds','fitted_reliability_models.json; standardized-feature coefficients, not causal effects')
print('Figure 5 panels complete')
# FIGURE 6: cutoff, endpoint, ablation and timing-pattern checks.
for letter,col,ylab in [('a','AUROC','Failure-prediction AUROC'),('b','Brier','Brier score')]:
 f,a=panel(6,letter)
 for model in ['Global','Event']:
  s=ss[(ss.detector=='energy')&(ss.method==model)].sort_values('failure_threshold');a.plot(s.failure_threshold,s[col],'o-',label=model)
 a.set_xticks([.9,.95,.98]);xy(a,'F1 failure cutoff',ylab);legend(a)
 done(f,6,letter,'Sensitivity to the operational failure cutoff','sensitivity_analyses.csv; fixed source partitions, retrained labels per cutoff')
for letter,col,ylab in [('c','AUROC','Failure-prediction AUROC'),('d','risk_at_80pct','Retained failure at 80% (%)')]:
 f,a=panel(6,letter)
 for j,model in enumerate(['Global','Event']):
  vals=[]
  for endpoint in ['energy','amplitude']:
   vals.append(float(ss[(ss.detector==endpoint)&(ss.failure_threshold==.95)&(ss.method==model)][col].iloc[0])*(100 if col=='risk_at_80pct' else 1))
  a.bar(np.arange(2)+(j-.5)*.3,vals,width=.3,label=model)
 a.set_xticks([0,1],['Energy','Amplitude']);xy(a,'Detector endpoint',ylab);legend(a,loc='upper left' if col=='risk_at_80pct' else 'lower left',fontsize=6)
 if col=='AUROC':a.set_ylim(.9,1.005)
 done(f,6,letter,'Alternative detector endpoint, '+col,'sensitivity_analyses.csv')
for letter,col,ylab in [('e','AUROC','AUROC'),('f','Brier','Brier score')]:
 f,a=panel(6,letter);methods=['Global','Agreement_features','Morphology_features','Event'];v=[ss[(ss.detector=='energy')&(ss.failure_threshold==.95)&(ss.method==m)][col].iloc[0] for m in methods];a.barh(np.arange(4),v);a.set_yticks(range(4),['Global','Agreement','Morphology','Event']);a.invert_yaxis();xy(a,ylab,'')
 if col=='AUROC':a.set_xlim(.9,1.008)
 for j,val in enumerate(v):a.text(val-.001 if col=='AUROC' else val+.002,j,f'{val:.3f}',ha='right' if col=='AUROC' else 'left',va='center',fontsize=6.5)
 if col=='Brier':a.set_xlim(0,.105)
 done(f,6,letter,'Feature-ablation '+col,'sensitivity_analyses.csv; agreement-only here is fitted, not the fixed heuristic')
for letter,col,ylab in [('g','AUROC','AUROC'),('h','ECE','Calibration error')]:
 f,a=panel(6,letter)
 for j,model in enumerate(['Global','Event','Combined']):
  s=pat[pat.method==model].set_index('pattern').reindex(['regular','variable']);a.plot([0,1],s[col],'o-',label=model)
 a.set_xticks([0,1],['Regular','Variable']);xy(a,'Generated interval pattern',ylab);legend(a,fontsize=5.8,loc='lower left' if col=='AUROC' else 'center right')
 if col=='AUROC':a.set_ylim(.92,1.005)
 done(f,6,letter,'Prediction performance by generated interval pattern','pattern_metrics.csv; no refitting; diagnosis is not inferred')
print('Figure 6 panels complete')
# FIGURE 7: retention, finite error floor, source variation and failure cases.
f,a=panel(7,'a')
for model in MOD:
 s=ret[(ret.subset=='pooled')&(ret.method==model)];a.plot(s.coverage*100,s.failure_rate*100,label=model)
s=ret[(ret.subset=='pooled')&(ret.method=='Event')];a.plot(s.coverage*100,s.oracle_minimum*100,ls='--',label='Oracle floor');xy(a,'Retained episodes (%)','Failure among retained (%)');legend(a,loc='upper left',fontsize=6.5)
done(f,7,'a','Full retrospective error–coverage curves','retention_full_grid.csv; identical retained counts per coverage')
f,a=panel(7,'b')
for model in MOD[:3]:
 s=ret[(ret.subset=='pooled')&(ret.method==model)];a.plot(s.discarded_nonfailures,s.retained_failures,label=model)
xy(a,'Nonfailures discarded','Failures retained');legend(a,loc='upper right',fontsize=6.5)
done(f,7,'b','Two costs of exclusion without a misleading accuracy-only summary','retention_full_grid.csv; all coverage points')
f,a=panel(7,'c');s=ret[(ret.subset=='pooled')&(ret.coverage==.7)].set_index('method').reindex(MOD);a.bar(range(4),s.retained_failures.values)
for j,v in enumerate(s.retained_failures):a.text(j,v+8,str(v),ha='center',fontsize=7)
a.set_xticks(range(4),['Global','Event','Comb.','Agree.'],rotation=25);a.set_ylim(0,560);xy(a,'','Failures in 2,868 retained')
done(f,7,'c','Failure counts at the same 70% coverage','44 Event failures versus 175 Global; all four methods')
f,a=panel(7,'d')
for pattern,m in [('regular','o'),('variable','s')]:
 s=ps[ps.pattern==pattern].set_index('method').reindex(MOD);a.plot(range(4),s.retention_rate*100,m+'-',label=pattern.capitalize())
a.set_xticks(range(4),['Global','Event','Comb.','Agree.'],rotation=25);a.set_ylim(79,81);xy(a,'','Pattern retained at 80% (%)');legend(a,fontsize=6)
done(f,7,'d','Retention of the two generated interval patterns','pattern_retention.csv; does not test arrhythmia retention')
f,a=panel(7,'e')
for g in G:
 s=ret[(ret.subset==g)&(ret.method=='Event')];a.plot(s.coverage*100,s.failure_rate*100,label=g)
xy(a,'Within-source retention (%)','Event retained failure (%)');legend(a,loc='upper left',ncol=2,fontsize=6)
done(f,7,'e','Source-specific Event ranking curves','retention_full_grid.csv; retrospective source-specific ranks')
f,a=panel(7,'f');a.scatter(h.risk_Event,h.f1,s=4,alpha=.22);a.axhline(.95,ls=':',lw=.8);thr=h.sort_values('risk_Event',kind='stable').iloc[int(np.ceil(.7*len(h)))-1].risk_Event;a.axvline(thr,ls='--',lw=.8);xy(a,'Event failure probability','Observed beat F1');a.text(.02,.02,f'70% rank cutoff {thr:.3f}',transform=a.transAxes,fontsize=6.4)
done(f,7,'f','Full held-out probability/F1 distribution and retrospective cutoff','All held-out episodes; cutoff derived only to display retrospective 70% rank')
f,a=panel(7,'g');marker_trace(a,sel['retained_failure'],interval=(6,10));a.text(.02,.15,f"Retained · p = {sel['retained_failure']['risk_Event']:.3f}",transform=a.transAxes,fontsize=6.4);done(f,7,'g','A retained failure shown rather than omitted','Worst-F1 false negative at pooled 70% Event coverage; explicit selection record')
f,a=panel(7,'h');marker_trace(a,sel['discarded_nonfailure'],interval=(6,10));a.text(.02,.15,f"Excluded · p = {sel['discarded_nonfailure']['risk_Event']:.4f}",transform=a.transAxes,fontsize=6.4);done(f,7,'h','An excluded episode that meets the beat-F1 criterion','Highest-risk nonfailure excluded at pooled 70% Event coverage; explicit selection record')
print('Figure 7 panels complete')
# Assemble each already-rendered independent panel into a full-width vector PDF/SVG.
from lxml import etree
NS='http://www.w3.org/2000/svg'
for num,rows in layouts.items():
 W=7.08*72;H=(3*HEIGHT+.10)*72;doc=fitz.open();page=doc.new_page(width=W,height=H)
 svg=etree.Element('{%s}svg'%NS,nsmap={None:NS});svg.set('width',f'{W}pt');svg.set('height',f'{H}pt');svg.set('viewBox',f'0 0 {W} {H}')
 k=0;positions=[]
 for rr,count in enumerate(rows):
  ww=(W-8*(count-1))/count; yy=rr*HEIGHT*72
  for col in range(count):
   letter=string.ascii_lowercase[k];xx=col*(ww+8);src=P/f'Fig{num}{letter}.pdf';pd=fitz.open(src);rect=fitz.Rect(xx,yy,xx+ww,yy+HEIGHT*72);page.show_pdf_page(rect,pd,0)
   root=etree.parse(str(P/f'Fig{num}{letter}.svg')).getroot();vb=[float(v) for v in root.attrib['viewBox'].split()];scale=min(ww/vb[2],(HEIGHT*72)/vb[3]);group=etree.SubElement(svg,'{%s}g'%NS);group.set('transform',f'translate({xx} {yy}) scale({scale})')
   prefix=f'f{num}{letter}_'
   for el in root.iter():
    if 'id' in el.attrib:el.attrib['id']=prefix+el.attrib['id']
    for key,val in list(el.attrib.items()):
     if 'url(#' in val:el.attrib[key]=val.replace('url(#','url(#'+prefix)
     elif key.endswith('href') and val.startswith('#'):el.attrib[key]='#'+prefix+val[1:]
   for child in list(root):group.append(child)
   k+=1
 pagepath=F/f'Fig{num}.pdf';doc.save(pagepath,garbage=4,deflate=True)
 pix=page.get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False);pix.save(F/f'Fig{num}.png')
 page.get_pixmap(matrix=fitz.Matrix(150/72,150/72),alpha=False).save(F/f'Fig{num}_preview.png')
 etree.ElementTree(svg).write(str(F/f'Fig{num}.svg'),xml_declaration=True,encoding='UTF-8')
 doc.close()
json.dump(manifest,open(D/'figure_panel_manifest.json','w'),indent=2)
json.dump({'figures':7,'panels':len(manifest),'panel_counts':{str(n):sum(v) for n,v in layouts.items()},'layouts':layouts,'width_inches':7.08,'height_inches':3*HEIGHT+.10,'raster_dpi':600},open(D/'figure_design_summary.json','w'),indent=2)
print('Completed seven composite figures with',len(manifest),'independent panels.')
