"""Rebuild additional figures from saved rows; no outcome filtering or fitting."""
from pathlib import Path
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'manuscript/figures';DATA=ROOT/'results/control_analysis'
COL=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#333333','#999999']
MARK=['o','s','^','D','v','P','X','h'];STY=['-','--',':','-.'];HATCH=['','///','xx','..','\\\\','++']
RUNS=[f'{f}_batch{b}' for f in ['qwen','smollm'] for b in range(1,5)]
METHODS=['full_update','simple_scaling','minimum_norm','local_prediction','calibrated_prediction','direct_validation']
NAMES=['Full','Scale','Norm','Local','Calibrated','Direct']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
def read(name,folder=DATA):return list(csv.DictReader((folder/name).open()))
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(FIG/(name+'.'+ext))
    plt.close(fig)
def label(run):return run.replace('qwen_batch','Qwen ').replace('smollm_batch','SmolLM ')
def pairedbar(ax,groups,values,names):
    n=len(values);x=np.arange(len(groups));width=.78/n
    for i,(y,name) in enumerate(zip(values,names)):
        ax.bar(x+(i-(n-1)/2)*width,y,width,label=name,color=COL[i],hatch=HATCH[i],edgecolor='white',linewidth=.4)
    ax.set_xticks(x,groups)

def candidate_grid():
    rows=read('all_candidate_strength_outcomes.csv');choices=read('methods.csv')
    fig,axs=plt.subplots(2,4,figsize=(7.3,5.6),layout='constrained',sharex=True,sharey=True)
    for k,(run,ax) in enumerate(zip(RUNS,axs.flat)):
        data=[r for r in rows if r['run']==run]
        for i,c in enumerate(sorted({r['candidate'] for r in data})):
            rr=sorted([r for r in data if r['candidate']==c],key=lambda r:float(r['strength']))
            ax.plot([100*float(r['old_family_loss_rate']) for r in rr],[100*float(r['new_accuracy']) for r in rr],STY[i],color=COL[i],marker=MARK[i],markersize=3,label=c)
            for r in rr:
                if r['audit_benefit_pass']=='False':ax.scatter([100*float(r['old_family_loss_rate'])],[100*float(r['new_accuracy'])],marker='x',s=18,color='black',zorder=5)
        ax.axhline(25,color='.4',ls=':',lw=.7);ax.set_title(chr(97+k)+'  '+label(run));ax.legend(fontsize=6,loc='lower right')
    fig.supxlabel('Protected old-family loss (%)');fig.supylabel('New-task success (%)');save(fig,'R142_ED5_candidate_grid')

def declarations():
    rows=[x for x in read('methods.csv') if x['method']=='calibrated_prediction'];fig,ax=plt.subplots(2,2,figsize=(7.3,5.8),layout='constrained')
    names=['Q1','Q2','Q3','Q4','S1','S2','S3','S4'];by={x['run']:x for x in rows};rr=[by[x] for x in RUNS]
    pairedbar(ax[0,0],names,[[int(x['declared_families']) for x in rr],[int(x['old_family_n']) for x in rr]],['Declared','All protected']);ax[0,0].set(ylabel='Family appearances',title='a  Declaration coverage');ax[0,0].legend(fontsize=7)
    pairedbar(ax[0,1],names,[[int(x['false_declarations']) for x in rr],[int(x['old_family_loss_n']) for x in rr]],['False declarations','All old losses']);ax[0,1].set(ylabel='Family appearances',title='b  Errors inside and outside declarations');ax[0,1].legend(fontsize=7)
    for i,x in enumerate(rr):ax[1,0].scatter(100*int(x['declared_families'])/int(x['old_family_n']),100*float(x['old_family_loss_rate']),marker=MARK[i],color=COL[i],s=35,label=names[i])
    ax[1,0].set(xlabel='Declared families (%)',ylabel='All old-family loss (%)',title='c  Different estimands');ax[1,0].legend(ncol=4,fontsize=6)
    p=np.linspace(0,1,101)
    for i,a in enumerate([.01,.05,.1]):ax[1,1].plot(100*p,100*np.minimum(1,1-p+a),STY[i],color=COL[i],label=f'α = {a:g}')
    ax[1,1].set(xlabel='Population declaration probability (%)',ylabel='Permitted total loss bound (%)',title='d  Sharp probability bound');ax[1,1].legend(fontsize=7)
    save(fig,'R142_ED6_declarations')

def sensitivity():
    rows=read('posthoc_calibration_pilot_sensitivity.csv');fig,axs=plt.subplots(2,2,figsize=(7.3,5.8),layout='constrained')
    ns=sorted({int(x['calibration_families']) for x in rows});pilots=sorted({float(x['pilot_strength']) for x in rows})
    # Full available calibration size is the largest listed size per run.
    for i,run in enumerate(RUNS):
        rr=[x for x in rows if x['run']==run and float(x['alpha'])==.1];n=max(int(x['calibration_families']) for x in rr)
        ss=sorted([x for x in rr if int(x['calibration_families'])==n],key=lambda x:float(x['pilot_strength']))
        for ax,key in [(axs[0,0],'allowance'),(axs[0,1],'any_declared_family_n')]:
            vals=[float(x[key]) if key=='allowance' else 100*float(x[key])/float(x['audit_families']) for x in ss]
            ax.plot([float(x['pilot_strength']) for x in ss],vals,color=COL[i],ls=STY[i%4],marker=MARK[i],ms=4,label=label(run))
    axs[0,0].set(xlabel='Pilot strength',ylabel='Joint residual allowance',title='a  Finite-path approximation');axs[0,0].legend(fontsize=6,ncol=2)
    axs[0,1].set(xlabel='Pilot strength',ylabel='Families declared by any pair (%)',title='b  Exploratory declaration availability')
    for i,a in enumerate([.01,.05,.1,.2]):
        x=np.arange(1,101);y=np.ceil((x+1)*(1-a))<=x
        axs[1,0].step(x,y.astype(int)+i*1.15,where='post',color=COL[i],ls=STY[i],label=f'α={a:g}')
    axs[1,0].set(xlabel='Calibration families',yticks=[0,1.15,2.3,3.45],yticklabels=['.01','.05','.10','.20'],ylabel='Error level (step up = finite)',title='c  Finite-bound thresholds')
    benefit=read('audit_only_benefit_sensitivity.csv');keys=list(benefit[0])
    # Display exact table schema rather than infer a missing column.
    abskey=next(k for k in keys if k in ['absolute_floor','cap_abs','absolute_requirement','absolute_threshold'])
    gainkey=next(k for k in keys if k in ['gain_floor','cap_gain','increment_requirement','gain_requirement','gain_threshold'])
    countkey=next(k for k in keys if k in ['audit_only_admissible_pairs','admissible_pairs','benefit_passing_pairs','eligible_pairs','passing_pairs'])
    levels=sorted({float(x[abskey]) for x in benefit});gains=sorted({float(x[gainkey]) for x in benefit})
    grid=np.array([[np.mean([float(x[countkey]) for x in benefit if float(x[abskey])==a and float(x[gainkey])==g]) for a in levels] for g in gains])
    im=axs[1,1].imshow(grid,origin='lower',aspect='auto',cmap='cividis');axs[1,1].set(xticks=range(len(levels)),xticklabels=[f'{100*x:g}' for x in levels],yticks=range(len(gains)),yticklabels=[f'{100*x:g}' for x in gains],xlabel='Absolute new-task floor (%)',ylabel='Required gain (percentage points)',title='d  Mean audit-eligible pairs / 16');fig.colorbar(im,ax=axs[1,1],shrink=.75)
    save(fig,'R142_ED7_sensitivity')

def costs():
    rows=read('methods.csv');fig,axs=plt.subplots(2,2,figsize=(7.3,5.8),layout='constrained')
    for k,(key,ylabel,title) in enumerate([('selection_record_calls','Additional old-record calls','a  Selection observations'),('selection_seconds','Selection seconds','b  Measured selection time'),('sequence_tokens','Teacher-forced token positions','c  Scoring token positions'),('generated_tokens','Generated tokens','d  Generation token positions')]):
        ax=axs.flat[k]
        values=[sum(float(x[key]) for x in rows if x['method']==m) for m in METHODS]
        b=ax.bar(range(6),values,color=COL[:6],edgecolor='white',linewidth=.4)
        for i,patch in enumerate(b):patch.set_hatch(HATCH[i])
        ax.set(xticks=range(6),xticklabels=NAMES,ylabel=ylabel,title=title);ax.tick_params(axis='x',labelrotation=28,labelsize=7);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,4));ax.margins(y=.16)
    save(fig,'R142_ED8_costs')

def deployment():
    folder=ROOT/'results/deployment';rows=read('workflow_replays.csv',folder);cuts=read('exhaustive_cut_checks.csv',folder)
    fig,axs=plt.subplots(2,2,figsize=(7.3,5.8),layout='constrained');ax=axs.flat
    rr=[x for x in rows if x['topology']=='separate'];x=range(len(rr))
    ax[0].plot(x,[int(r['full_reads']) for r in rr],color=COL[1],ls='--',label='Complete record')
    ax[0].plot(x,[int(r['selective_reads']) for r in rr],color=COL[0],ls='-',marker='o',markevery=12,ms=3,label='Exact stopping')
    ax[0].set(xlabel='Synthetic tape replicate',ylabel='Observations',title='a  Same complete-record decisions');ax[0].legend(fontsize=7)
    tops=['separate','shared','unequal'];vals=[[np.mean([float(r[key]) for r in rows if r['topology']==top]) for top in tops] for key in ['full_operation_work','selective_operation_work']]
    pairedbar(ax[1],['Separate','Shared','Unequal'],vals,['Complete recomputation','Affected operations']);ax[1].set(ylabel='Specified operation work',title='b  Dependencies change work');ax[1].legend(fontsize=7)
    a=[float(r['exhaustive']) for r in cuts];b=[float(r['min_cut']) for r in cuts];ax[2].scatter(a,b,s=8,color=COL[2],alpha=.5,marker='s');lim=max(a+b);ax[2].plot([0,lim],[0,lim],'--',color='black',lw=.7);ax[2].set(xlabel='Exhaustive optimum',ylabel='Minimum-cut cost',title='c  All 600 optima match')
    rr=read('equal_count_different_work.csv',folder)
    ax[3].plot([float(r['heavy_operation_cost']) for r in rr],[float(r['ratio']) for r in rr],'o-',color=COL[0],label='One changed root each');ax[3].set(xscale='log',yscale='log',xlabel='Cost of affected shared operation',ylabel='Work ratio',title='d  Equal counts, unequal costs');ax[3].legend(fontsize=7)
    save(fig,'R142_ED10_deployment')

if __name__=='__main__':
    candidate_grid();declarations();sensitivity();costs();deployment()
