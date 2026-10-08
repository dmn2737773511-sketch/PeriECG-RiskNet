import os
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
ROOT=Path(os.environ.get('JMS_ANALYSIS_DIR', str(Path(__file__).resolve().parents[1]/'source_data')))
OUT=Path(os.environ.get('JMS_FIGURE_DIR', str(ROOT/'figures')));OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT;SRC.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10,'lines.linewidth':1.7,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600})
def rd(name):return list(csv.DictReader(open(ROOT/name,encoding='utf-8')))
def wr(name,rows):
 with open(SRC/name,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def base():
 fig=plt.figure(figsize=(6.7,4.1));ax=fig.add_subplot(111);ax.spines[['top','right']].set_visible(False);return fig,ax
def save(fig,name):
 fig.tight_layout(pad=1.2)
 for ext in ['png','eps','pdf']:
  fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)
def phase_labels(ax, q, x, y, state):
 offsets_raw={('G1',0):(-13,-12),('G1',1):(5,7),('G2',1):(-12,-3),('G2',3):(6,2),('G3',0):(7,4),('G3',1):(-12,-2),('G3',2):(-4,-20),('G3',4):(7,2),('G3',5):(-12,7),('G4',1):(12,0),('G4',5):(-14,12)}
 offsets_filtered={('G1',0):(-13,7),('G1',1):(7,-3),('G2',1):(8,0),('G2',5):(-9,-13),('G3',0):(-4,12),('G3',1):(-16,3),('G3',2):(-14,12),('G3',3):(0,-16),('G3',4):(3,-21),('G3',5):(-18,-8),('G4',0):(8,6),('G4',1):(10,0),('G4',5):(9,-7)}
 offsets=offsets_raw if state=='raw' else offsets_filtered
 for a,xx,yy in zip(q,x,y):
  off=offsets.get((a['group'],int(a['phase'])),(5,4))
  kw={'arrowprops':{'arrowstyle':'-','lw':0.45,'shrinkA':1,'shrinkB':4}} if abs(off[0])+abs(off[1])>=15 else {}
  ax.annotate(str(int(a['phase'])+1),(xx,yy),xytext=off,textcoords='offset points',fontsize=8,**kw)

# Main figure 1: all six phases, not a single favorable example.
r=rd('spectral_profiles.csv');rs=[a for a in r if a['lead']=='II' and a['preprocessing']=='raw'];fig,ax=base();markers=['o','s','^','D']
for g,m in zip(['G1','G2','G3','G4'],markers):
 q=sorted([a for a in rs if a['group']==g],key=lambda a:int(a['phase']));x=[float(a['line_rms_uV']) for a in q];y=[float(a['low_rms_uV']) for a in q]
 ax.plot(x,y,marker=m,linestyle='none',markersize=6,label=g)
 phase_labels(ax,q,x,y,'raw')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('49–51 Hz component (µV RMS)');ax.set_ylabel('0.05–0.5 Hz component (µV RMS)');ax.legend(frameon=False,ncol=4,loc='lower left',bbox_to_anchor=(0,1.02));save(fig,'Fig1');wr('Fig1_source.csv',rs)
# Main figure 2: alignment-only failure threshold crossings.
rs=rd('snr_summary.csv');fig,ax=base();x=np.arange(4);vals=[100*float(a['fraction_straddling']) for a in rs]
ax.bar(x,vals,width=.58);ax.set_xticks(x,[f"{float(a['snr_db']):g}" for a in rs]);ax.set_xlabel('Signal-to-perturbation ratio (dB)');ax.set_ylabel('Quartets crossing the failure threshold (%)');ax.set_ylim(0,28)
for xx,yy in zip(x,vals):ax.text(xx,yy+.7,f'{yy:.2f}',ha='center',fontsize=10)
save(fig,'Fig2');wr('Fig2_source.csv',rs)
# Main figure 3: prediction of detector failure, NOT disease diagnosis.
r=rd('heldout_predictions.csv');y=np.array([int(a['failure']) for a in r]);fig,ax=base();roc=[];styles=['-','--','-.',':'];summary={a['method']:a for a in rd('reliability_summary.csv')}
for k,ls in zip(['Global','Event','Combined','Agreement'],styles):
 p=np.array([float(a['risk_'+k]) for a in r]);fpr,tpr,thr=roc_curve(y,p);ax.plot(fpr,tpr,linestyle=ls,label=f"{k}  {float(summary[k]['AUROC']):.3f}")
 roc.extend(dict(method=k,fpr=float(a),tpr=float(b),threshold=float(c)) for a,b,c in zip(fpr,tpr,thr))
ax.plot([0,1],[0,1],linestyle='--',linewidth=.8);ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate for detector failure');ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(frameon=False,title='Method / AUROC',loc='lower right');save(fig,'Fig3');wr('Fig3_source.csv',roc)
# Main figure 4: coverage-risk, with deterministic oracle lower bound.
fig,ax=base();rows=[];N=len(y);nsucc=int((y==0).sum());cs=np.linspace(.25,1,76)
for k,ls in zip(['Global','Event','Combined','Agreement'],styles):
 p=np.array([float(a['risk_'+k]) for a in r]);order=np.argsort(p,kind='stable');v=[]
 for c in cs:
  n=int(np.ceil(c*N));fail=int(y[order[:n]].sum());v.append(100*fail/n);rows.append(dict(method=k,coverage=float(c),n_retained=n,n_fail=fail,failure_rate=fail/n))
 ax.plot(cs*100,v,linestyle=ls,label=k)
lo=[100*max(0,int(np.ceil(c*N))-nsucc)/int(np.ceil(c*N)) for c in cs];ax.plot(cs*100,lo,linestyle=(0,(1,1)),linewidth=2,label='Oracle lower bound')
rows.extend(dict(method='Oracle lower bound',coverage=float(c),n_retained=int(np.ceil(c*N)),n_fail=max(0,int(np.ceil(c*N))-nsucc),failure_rate=l/100) for c,l in zip(cs,lo))
ax.set_xlabel('Episodes retained (%)');ax.set_ylabel('Failure rate among retained episodes (%)');ax.set_xlim(25,100);ax.set_ylim(0,30);ax.legend(frameon=False,loc='upper left');save(fig,'Fig4');wr('Fig4_source.csv',rows)
# Supplementary figure 1: filtered data, identical all-phase representation.
r=rd('spectral_profiles.csv');rs=[a for a in r if a['lead']=='II' and a['preprocessing']=='filtered'];fig,ax=base()
for g,m in zip(['G1','G2','G3','G4'],markers):
 q=sorted([a for a in rs if a['group']==g],key=lambda a:int(a['phase']));x=[float(a['line_rms_uV']) for a in q];yv=[float(a['slow_rms_uV']) for a in q];ax.plot(x,yv,marker=m,linestyle='none',markersize=6,label=g)
 phase_labels(ax,q,x,yv,'filtered')
ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(50,330);ax.set_xlabel('Residual 49–51 Hz component (µV RMS)');ax.set_ylabel('0.5–5 Hz component (µV RMS)');ax.legend(frameon=False,ncol=4,loc='lower left',bbox_to_anchor=(0,1.02));save(fig,'FigS1');wr('FigS1_source.csv',rs)
# S2 calibration for three probabilistic models (heuristic omitted).
r=rd('heldout_predictions.csv');y=np.array([int(a['failure']) for a in r]);fig,ax=base();cal=[]
for k,ls in zip(['Global','Event','Combined'],styles):
 p=np.array([float(a['risk_'+k]) for a in r]);xs=[];ys=[]
 for bi in range(10):
  a,b=bi/10,(bi+1)/10;m=(p>=a)&((p<b) if bi<9 else(p<=b))
  if m.any():xs.append(float(p[m].mean()));ys.append(float(y[m].mean()));cal.append(dict(method=k,bin=bi,n=int(m.sum()),mean_probability=xs[-1],observed_failure_rate=ys[-1]))
 ax.plot(xs,ys,marker='o',markersize=4,linestyle=ls,label=k)
ax.plot([0,1],[0,1],linestyle='--',linewidth=.8);ax.set_xlabel('Mean predicted failure probability');ax.set_ylabel('Observed detector failure proportion');ax.legend(frameon=False);ax.set_xlim(0,1);ax.set_ylim(0,1);save(fig,'FigS2');wr('FigS2_source.csv',cal)
# S3 fold-disjoint AUROCs.
r=rd('fold_metrics.csv');fig,ax=base();gs=['G1','G2','G3','G4']
for k,ls in zip(['Global','Event','Combined'],styles):
 vals=[float(next(a for a in r if a['heldout_record']==g and a['method']==k)['AUROC']) for g in gs];ax.plot(np.arange(4),vals,marker='o',linestyle=ls,label=k)
ax.set_xticks(np.arange(4),gs);ax.set_xlabel('Held-out source recording');ax.set_ylabel('Detector-failure AUROC');ax.set_ylim(.85,1.005);ax.legend(frameon=False);save(fig,'FigS3');wr('FigS3_source.csv',r)
# S4 average precision curves.
r=rd('heldout_predictions.csv');y=np.array([int(a['failure']) for a in r]);fig,ax=base();pr=[]
for k,ls in zip(['Global','Event','Combined','Agreement'],styles):
 p=np.array([float(a['risk_'+k]) for a in r]);prec,rec,thr=precision_recall_curve(y,p);ax.plot(rec,prec,linestyle=ls,label=f"{k}  {float(summary[k]['AUPRC']):.3f}");pr.extend(dict(method=k,recall=float(a),precision=float(b)) for a,b in zip(rec,prec))
ax.axhline(y.mean(),linestyle='--',linewidth=.8,label='Failure prevalence');ax.set_xlabel('Recall for detector failure');ax.set_ylabel('Precision for detector failure');ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0.04,0.43),title='Method / average precision');save(fig,'FigS4');wr('FigS4_source.csv',pr)
print('Created 4 main figures and 4 supplementary figures, with vector and 600-dpi versions.')
