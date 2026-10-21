"""Rebuild manuscript figures from released data, without rerunning experiments."""
from pathlib import Path
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullLocator,ScalarFormatter
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'manuscript/figures'
DATA=ROOT/'manuscript/source_data';SIM=ROOT/'results/theory_simulations'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
BLUE='#326B92';ORANGE='#BC654C';GREEN='#438B74'
def read(name,folder=DATA):return list(csv.DictReader((folder/name).open()))
def save(fig,name):
    path=FIG/name
    fig.savefig(path);fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)
def bars(ax,labels,values,colors=None,ylim=None):
    bb=ax.bar(np.arange(len(labels)),values,color=colors or BLUE,width=.62)
    ax.set_xticks(np.arange(len(labels)),labels)
    if ylim:ax.set_ylim(*ylim)
    fmt='%.3f' if values and max(values)<=1.1 else ('%g' if all(float(v).is_integer() for v in values) else '%.1f')
    ax.bar_label(bb,fmt=fmt,padding=3,fontsize=8)
    return bb

def joint():
    data=json.loads((ROOT/'results/R140_joint_reevaluation_recovered.json').read_text())
    fig=plt.figure(figsize=(7.2,5.9),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.5,1])
    ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    rows=[]
    for candidate,col in [('U1',BLUE),('U4',ORANGE)]:
        vv=data['candidates'][candidate]
        for strength,rec in vv.items():
            rows.append({'candidate':candidate,'strength':strength,'new_success_n':round(rec['new_ast']*60),'new_n':60,'old_loss_n':rec['old_loss_n'],'protected_n':rec['protected_n']})
        pts=[vv[s] for s in ['0.25','1.0']]
        ax[0].plot([p['old_loss_rate']*100 for p in pts],[p['new_ast']*100 for p in pts],'o-',color=col,label=candidate)
        for s,p in zip(['quarter','full'],pts):
            if s=='quarter':continue
            ax[0].annotate(s,(p['old_loss_rate']*100,p['new_ast']*100),xytext=(4,6),textcoords='offset points',fontsize=8,color=col)
    ax[0].axhline(25,color='0.4',ls=':',lw=1)
    ax[0].set(xlabel='Old protected-task loss (%)',ylabel='New-task success (%)',xlim=(-2,67),ylim=(5,45),title='a  Both outcomes on the same model')
    ax[0].legend(loc='lower right',fontsize=8)
    sub=sorted(rows,key=lambda r:(r['candidate'],float(r['strength'])))
    bars(ax[1],['M0','U1\n¼','U1\n1','U4\n¼','U4\n1'],[5]+[r['new_success_n'] for r in sub],['0.65',BLUE,BLUE,ORANGE,ORANGE],(0,28))
    ax[1].axhline(15,color='0.4',ls=':',lw=1)
    ax[1].set(ylabel='New successes / 60 tasks',title='b  Smaller updates lose benefit')
    cats=['both_loss','U1_only_loss','U4_only_loss','neither_loss'];names=['Both lose','U1 only loses','U4 only loses','Neither loses'];colors=[ORANGE,'#D69573','#E4BB99','#9CB9C8']
    left=np.zeros(2)
    for k,name,c in zip(cats,names,colors):
        vv=np.array([data['paired_overlap'][s][k] for s in ['1.0','0.25']]);ax[2].barh([1,0],vv,left=left,color=c,label=name,height=.48)
        for yy,v,ll in zip([1,0],vv,left):
            if v>=2:ax[2].text(ll+v/2,yy,str(v),ha='center',va='center',fontsize=8)
        left+=vv
    ax[2].set(yticks=[0,1],yticklabels=['Quarter strength','Full strength'],xlim=(0,61),xlabel='Paired old tasks / 61 incumbent successes',title='c  The paired ledger identifies shared and different losses')
    ax[2].legend(loc='upper center',bbox_to_anchor=(.5,-.33),ncol=4,fontsize=7.5,frameon=False)
    with (DATA/'R141_figure1_joint.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(fig,'R141_Figure1_joint_outcomes.png')

def theory():
    sub=[x for x in read('conformal_summary.csv',SIM) if float(x['alpha'])==.1]
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.65),layout='constrained');ns=[int(x['n_family']) for x in sub];xx=np.arange(len(ns))
    for off,key,label,col in [(-1,'fixed_candidate_miss_rate','Fixed candidate',BLUE),(0,'uncorrected_any_candidate_miss_rate','Any candidate: unadjusted',ORANGE),(1,'joint_miss_rate','Any candidate: simultaneous',GREEN)]:
        ax[0].bar(xx+.24*off,[float(x[key]) for x in sub],.24,label=label,color=col)
    ax[0].axhline(.1,color='0.3',ls=':',lw=1)
    ax[0].set(xticks=xx,xticklabels=ns,ylim=(0,.52),xlabel='Calibration families',ylabel='Envelope failure probability',title='a  Selection changes the guarantee')
    ax[0].legend(loc='lower left',bbox_to_anchor=(-.06,1.12),fontsize=7,frameon=False)
    # Abstention explanation belongs in the caption, not over a bar.
    t=np.linspace(.01,.25,400);mins=np.ceil((1-t)/t-1e-10)
    ax[1].step(t,mins,where='post',color=BLUE)
    ax[1].scatter([.1],[9],color=ORANGE)
    ax[1].annotate('90%: at least 9',(.1,9),xytext=(.095,45),arrowprops={'arrowstyle':'->'},fontsize=8)
    ax[1].set(xlabel='Allowed marginal error',ylabel='Minimum calibration families',ylim=(0,105),title='b  Finite bounds need evidence')
    save(fig,'R141_Figure2_calibration.png')

    rows=read('repair_feasibility_grid.csv',SIM);bs=sorted({float(x['budget']) for x in rows});ds=sorted({float(x['effect_uncertainty']) for x in rows})
    grid=np.array([{'proved_infeasible':0,'unresolved':1,'guaranteed_feasible':2}[x['status']] for x in rows]).reshape(len(ds),len(bs))
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.55),layout='constrained')
    im=ax[0].imshow(grid,origin='lower',aspect='auto',extent=[bs[0],bs[-1],ds[0],ds[-1]],cmap=ListedColormap(['#C66B5A','#DFC16C','#60A690']),vmin=-.5,vmax=2.5)
    ax[0].set(xlabel='Intervention budget',ylabel='Effect uncertainty',title='a  Three conclusions')
    cb=fig.colorbar(im,ax=ax[0],location='bottom',ticks=[0,1,2],pad=.18,fraction=.06)
    cb.ax.set_xticklabels(['Infeasible','Unresolved','Feasible']);cb.ax.tick_params(labelsize=7)
    x=np.linspace(0,2,301)
    for vals,label,col in [(x-.8,'Old loss',BLUE),(.4-.8*x,'Old survivor',ORANGE),(-.35-.25*x,'New requirement',GREEN)]:ax[1].plot(x,vals,label=label,color=col)
    ax[1].axhline(0,color='0.3',ls=':');ax[1].set(xlabel='First intervention amount',ylabel='Margin above required threshold',title='b  Repair can consume reserves')
    ax[1].legend(fontsize=8,loc='lower left',bbox_to_anchor=(-.07,1.12),frameon=False,ncol=1)
    save(fig,'R141_Figure3_repair_geometry.png')

    sub=[x for x in read('gaussian_summary.csv',SIM) if float(x['alpha'])==.05 and float(x['g'])>0]
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.65),layout='constrained')
    for key,style,label,col in [('mean_n','o-','Simulated rule',BLUE),('lower','s--','Lower bound',ORANGE),('upper','^:','Proved upper bound',GREEN)]:ax[0].plot([float(x['g']) for x in sub],[float(x[key]) for x in sub],style,label=label,color=col)
    ax[0].set(xscale='log',yscale='log',xlabel='Margin / observation s.d.',ylabel='Validation observations',title='a  Evidence rises near failure')
    ax[0].set_xticks([.25,.5,1,2],['0.25','0.5','1','2']);ax[0].xaxis.set_minor_locator(NullLocator());ax[0].legend(fontsize=7.5)
    for c,col in zip([.5,2,8],[BLUE,ORANGE,GREEN]):
        xs=np.linspace(0,3,300);gg=.5;alpha=.05;opt=max((4*math.log(1/alpha)/c)**(1/3)-gg,0)
        ax[1].plot(xs,c*xs+2*math.log(1/alpha)/(gg+xs)**2,color=col,label=f'Unit price {c:g}')
        ax[1].scatter([opt],[c*opt+2*math.log(1/alpha)/(gg+opt)**2],color=col,s=24)
    ax[1].set(xlabel='Intervention amount',ylabel='Leading cost approximation',ylim=(0,28),title='b  Training can reduce testing');ax[1].legend(fontsize=8)
    save(fig,'R141_Figure4_information.png')

def historical():
    fig,ax=plt.subplots(2,2,figsize=(7.2,6.7),layout='constrained');ax=ax.ravel();r=read('figure1_frontier_turnover.csv');xx=np.arange(len(r));labels=[x['update'].replace(' ','\n',1) for x in r]
    for off,k,l,c in [(-1,'old_strict_4of4','Old 4/4',BLUE),(1,'new_strict_4of4','New 4/4',ORANGE)]:ax[0].bar(xx+off*.18,[int(x[k]) for x in r],.36,label=l,color=c)
    ax[0].set(xticks=xx,xticklabels=labels,ylabel='Tasks passing all four trials',title='a  Aggregate counts');ax[0].legend(fontsize=7)
    for off,k,l,c in [(-1,'exits','Old pass → fail',BLUE),(1,'entries','Old fail → pass',ORANGE)]:ax[1].bar(xx+off*.18,[int(x[k]) for x in r],.36,label=l,color=c)
    ax[1].set(xticks=xx,xticklabels=labels,ylabel='Tasks',title='b  Paired transitions');ax[1].legend(fontsize=7)
    r=read('figure1_sorry_judge_canonical.csv');ax[2].plot([x['checkpoint'] for x in r],[int(x['strict_10of10_groups']) for x in r],'o-',color=BLUE)
    ax[2].set(ylabel='Groups with 10/10 label agreement',title='c  Safety-evaluator checkpoints');ax[2].text(.02,.94,'M75 → M100: 25 exits, 9 entries',transform=ax[2].transAxes,va='top',fontsize=7)
    r=read('figure1_sampling_illustration.csv')[0];bars(ax[3],['Next batch\nall 4 pass','Next batch\nnot all 4 pass'],[100*float(r['prob_strict_batch']),100*float(r['prob_exit_next_independent_batch'])],ylim=(0,80));ax[3].set(ylabel='Probability (%)',title='d  No-update sampling illustration')
    for a in ax[:2]:a.tick_params(axis='x',labelsize=6.5)
    save(fig,'R141_ED1_turnover.png')

    r=read('figure2_boundary_summary.csv');fig,ax=plt.subplots(2,2,figsize=(7.2,6.6),layout='constrained');ax=ax.ravel()
    bars(ax[0],['M25→M50','M50→M75','M75→M100'],[float(x['value']) for x in r[:3]],ylim=(.5,1.05));ax[0].set(ylabel='ROC AUC',title='a  Next evaluator changes')
    bars(ax[1],['Margin','Metadata','Both'],[float(x['value']) for x in r[3:6]],ylim=(.5,1.05));ax[1].set(ylabel='Grouped-CV ROC AUC',title='b  Metadata comparison')
    # Values are selected by their published source-row names below.
    for off,metric,label,col in [(-1,'ROC AUC','Endpoint AUC',BLUE),(1,'Spearman rho','Crossing rank',ORANGE)]:
        rr=[x for x in r[6:] if x['metric']==metric]
        if not rr:
            rr=r[6:8] if off==-1 else r[8:10]
        ax[2].bar(np.arange(2)+off*.18,[float(x['value']) for x in rr],.36,label=label,color=col)
    ax[2].set(xticks=[0,1],xticklabels=['Qwen2.5\n3B','SmolLM2\n360M'],ylim=(.5,1.08),ylabel='Score',title='c  Fixed update directions');ax[2].legend(fontsize=7)
    x=np.linspace(0,1,200);ax[3].plot(x,.1-.01*x,label='g₁ = 0.1 − 0.01s',color=BLUE);ax[3].plot(x,1-2*x,'--',label='g₂ = 1 − 2s',color=ORANGE);ax[3].axhline(0,color='0.5',lw=.7);ax[3].axvline(.5,color='0.5',ls=':');ax[3].legend(fontsize=7)
    ax[3].set(xlabel='Update strength',ylabel='Signed margin',title='d  Direction changes the crossing')
    save(fig,'R141_ED2_boundaries.png')

    fig,ax=plt.subplots(2,2,figsize=(7.2,6.6),layout='constrained');ax=ax.ravel();r=read('figure3_single_system.csv');xx=np.arange(2)
    for off,k,l,c in [(-1,'heldout_exit_recovery_rate','Losses recovered',BLUE),(1,'survivor_collateral_rate','Survivors lost',ORANGE)]:ax[0].bar(xx+off*.18,[100*float(x[k]) for x in r],.36,label=l,color=c)
    ax[0].set(xticks=xx,xticklabels=['Repair losses','Protect survivors'],ylabel='Tasks (%)',ylim=(0,120),title='a  One-model repair trade-off');ax[0].legend(fontsize=7)
    r=read('figure3_repair_runs.csv')
    for row in r:
        col=BLUE if row['family'].startswith('Smol') else ORANGE
        ax[1].plot([0,1],[100*float(row['collateral_exit_only_rate']),100*float(row['collateral_replay_rate'])],'o-',color=col,alpha=.7,ms=3)
        ax[2].plot([0,1],[100,100*float(row['replay_recovery_rate'])],'o-',color=col,alpha=.7,ms=3)
    ax[1].set(xticks=[0,1],xticklabels=['Repair losses','Protect survivors'],ylabel='Survivor loss (%)',title='b  Ten paired survivor audits',ylim=(-1,46))
    ax[2].set(xticks=[0,1],xticklabels=['Repair losses','Protect survivors'],ylabel='Old losses recovered (%)',title='c  Recovery under protection',ylim=(70,103))
    rr=read('figure3_sealed_mixture_summary.csv');bars(ax[3],['Affine\nmixtures','Neural\nmixtures'],[100*int(x['sign_correct'])/int(x['sign_total']) for x in rr],ylim=(0,114));ax[3].set(ylabel='Margin signs predicted (%)',title='d  Previously sealed mixtures')
    save(fig,'R141_ED3_historical_repair.png')

    fig,ax=plt.subplots(2,2,figsize=(7.2,6.6),layout='constrained');ax=ax.ravel();r=read('figure4a_capability_selection.csv');xx=np.arange(3)
    for off,k,l,c in [(-1,'c_validation_accuracy','Validation',BLUE),(1,'c_audit_accuracy','Fresh audit',ORANGE)]:ax[0].bar(xx+off*.18,[100*float(x[k]) for x in r],.36,label=l,color=c)
    ax[0].set(xticks=xx,xticklabels=[x['update'] for x in r],ylabel='Routing accuracy (%)',title='a  New routing capability',ylim=(0,108));ax[0].legend(fontsize=7)
    r=read('figure4b_directional_prediction.csv')
    for off,key,l,col in [(-1,'margin','Old margin',BLUE),(0,'direction','Pilot direction',ORANGE),(1,'ratio','Margin / adverse rate',GREEN)]:
        rr=[x for x in r if x['score']==key]
        if key=='ratio' and not rr:rr=[x for x in r if x['score'] not in ['margin','direction']]
        ax[1].bar(np.arange(2)+off*.23,[float(x['endpoint_flip_auc']) for x in rr],.23,label=l,color=col)
    ax[1].set(xticks=[0,1],xticklabels=['A2','B2'],ylabel='Endpoint-change ROC AUC',title='b  Pilot information',ylim=(0,1.08));ax[1].legend(fontsize=6.8)
    # Preserve the original successful follow-up while also showing its earlier
    # failed fifth-direction attempt; both are documented in the current SI.
    ax[2].scatter([0,1,2],[0,0,1],s=65,color=[ORANGE,ORANGE,GREEN]);ax[2].set(xticks=[0,1,2],xticklabels=['Four\ndirections','First added\ndirection','Frozen\nfollow-up'],yticks=[0,1],yticklabels=['Infeasible','Feasible'],ylim=(-.3,1.3),title='c  One witness is insufficient')
    r=read('figure4d_fresh_audit.csv');vals=[float(x['percent']) for x in r]
    bars(ax[3],['Losses\nrecovered','Survivors\nlost','New task\nbefore','New task\nafter','New task\nM0'],vals,ylim=(0,100));ax[3].set(ylabel='Fresh audit (%)',title='d  Repair failed to transfer safely');ax[3].tick_params(axis='x',labelsize=6)
    save(fig,'R141_ED4_historical_prospective.png')

if __name__=='__main__':
    joint();theory();historical();print('Rebuilt four main figures and all sixteen historical panels.')
