"""Reproducible checks of R141 mathematical mechanisms, not LLM evidence."""
from pathlib import Path
import json, hashlib, math, csv, time
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'theory_simulations'
FIG = ROOT/'manuscript'/'figures'
SEED = 1410921
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':220})

def csvwrite(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sequential():
    rng=np.random.default_rng(SEED)
    rows=[]; summary=[]; reps=1200
    for alpha in [.01,.05,.1]:
        for g in [-2.,-1.,-.5,-.25,.25,.5,1.,2.]:
            sums=np.zeros(reps);stops=np.zeros(reps,dtype=int);signs=np.zeros(reps,dtype=int)
            active=np.arange(reps);n=0
            while len(active):
                n+=1;sums[active]+=rng.normal(g,1,len(active))
                b=math.sqrt(2*n*math.log(math.pi**2*n*n/(3*alpha)))
                hit=np.abs(sums[active])>b
                ix=active[hit];stops[ix]=n;signs[ix]=np.sign(sums[ix]);active=active[~hit]
                assert n<200000, 'Unfinished simulation; no censoring permitted'
            kl=(1-2*alpha)*math.log((1-alpha)/alpha)
            lower=2*kl/g**2
            upper=[]
            for eta in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
                nn=1
                while math.sqrt(2*math.log(math.pi**2*nn*nn/(3*alpha))/nn)>(1-eta)*abs(g):nn+=1
                k=eta**2*g**2/2
                upper.append(nn+math.exp(-nn*k)/(-math.expm1(-k)))
            for j in range(reps):
                rows.append(dict(alpha=alpha,g=g,sigma=1,replicate=j,n=int(stops[j]),decision=int(signs[j]),error=bool(signs[j]!=np.sign(g))))
            summary.append(dict(alpha=alpha,g=g,sigma=1,replicates=reps,mean_n=float(stops.mean()),se_n=float(stops.std(ddof=1)/math.sqrt(reps)),error_n=int(sum(signs!=np.sign(g))),lower=lower,upper=min(upper)))
    csvwrite('gaussian_replicates.csv',rows);csvwrite('gaussian_summary.csv',summary)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    sub=[x for x in summary if x['alpha']==.05 and x['g']>0]
    xx=[x['g'] for x in sub]
    ax[0].plot(xx,[x['mean_n'] for x in sub],'o-',label='Simulated stopping rule')
    ax[0].plot(xx,[x['lower'] for x in sub],'s--',label='Information lower bound')
    ax[0].plot(xx,[x['upper'] for x in sub],'^:',label='Proved finite-error upper bound')
    ax[0].set(xscale='log',yscale='log',xlabel='Signed margin / observation s.d.',ylabel='Validation observations',title='a  Evidence rises near the boundary')
    ax[0].legend(fontsize=8,loc='upper right')
    for c in [.5,2,8]:
        xs=np.linspace(0,3,300);g=.5;beta=1.;alpha=.05
        cost=c*xs+2*math.log(1/alpha)/(g+beta*xs)**2
        opt=max((4*beta*math.log(1/alpha)/c)**(1/3)-g,0)/beta
        ax[1].plot(xs,cost,label=f'Intervention unit cost = {c:g}')
        ax[1].scatter([opt],[c*opt+2*math.log(1/alpha)/(g+opt)**2],s=28)
    ax[1].set(xlabel='Intervention amount',ylabel='Leading cost approximation',title='b  Training and testing can substitute',ylim=(0,28))
    ax[1].legend(fontsize=8)
    fig.savefig(FIG/'R141_Figure4_information.png');plt.close(fig)
    return {'replicate_n':len(rows),'max_simulated_error_rate':max(x['error_n']/reps for x in summary)}

def calibration():
    rng=np.random.default_rng(SEED+1);rows=[];reps=3000;K=16
    for n in [8,9,19,49]:
        for alpha in [.01,.05,.1,.2]:
            rank=math.ceil((n+1)*(1-alpha))
            for j in range(reps):
                shared=rng.normal(size=(n+1,1));individual=rng.normal(size=(n+1,K))
                residual=np.abs(np.sqrt(.7)*shared+np.sqrt(.3)*individual)
                joint=residual.max(axis=1)
                q=np.sort(joint[:n])[rank-1] if rank<=n else np.inf
                singleq=np.sort(residual[:n,0])[rank-1] if rank<=n else np.inf
                rows.append(dict(n_family=n,alpha=alpha,replicate=j,rank=rank,finite=bool(np.isfinite(q)),joint_miss=bool(joint[n]>q),fixed_candidate_miss=bool(residual[n,0]>singleq),uncorrected_any_candidate_miss=bool(joint[n]>singleq)))
    csvwrite('conformal_replicates.csv',rows)
    summaries=[]
    for n in [8,9,19,49]:
        for a in [.01,.05,.1,.2]:
            sub=[x for x in rows if x['n_family']==n and x['alpha']==a]
            summaries.append(dict(n_family=n,alpha=a,replicates=reps,finite=sub[0]['finite'],joint_miss_rate=np.mean([x['joint_miss'] for x in sub]),fixed_candidate_miss_rate=np.mean([x['fixed_candidate_miss'] for x in sub]),uncorrected_any_candidate_miss_rate=np.mean([x['uncorrected_any_candidate_miss'] for x in sub])))
    csvwrite('conformal_summary.csv',summaries)
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    selected=[x for x in summaries if x['alpha']==.1]
    ns=np.array([x['n_family'] for x in selected]);xx=np.arange(len(ns));width=.24
    for offset,key,label,color in [(-1,'fixed_candidate_miss_rate','One fixed candidate','#34699a'),(0,'uncorrected_any_candidate_miss_rate','Any candidate, unadjusted','#c56952'),(1,'joint_miss_rate','Any candidate, simultaneous','#409b87')]:
        ax[0].bar(xx+offset*width,[x[key] for x in selected],width,label=label,color=color)
    ax[0].axhline(.1,color='black',ls=':',lw=1);ax[0].set(xticks=xx,xticklabels=ns,xlabel='Independent calibration families',ylabel='Envelope failure probability',title='a  Candidate selection needs joint calibration')
    ax[0].legend(fontsize=7.7,loc='upper left')
    ax[0].text(.02,.7,'At n = 8, the bound is infinite;\nzero declarations are not success.',transform=ax[0].transAxes,fontsize=8)
    tolerances=np.linspace(.01,.25,400);mins=np.ceil((1-tolerances)/tolerances-1e-10).astype(int)
    ax[1].step(tolerances,mins,where='post',color='#34699a')
    ax[1].set(xlabel='Allowed marginal error',ylabel='Minimum calibration families',title='b  A finite bound requires enough evidence',ylim=(0,105))
    ax[1].scatter([.1],[9],color='#c56952');ax[1].annotate('90% coverage: at least 9',(0.1,9),xytext=(.12,42),arrowprops={'arrowstyle':'->'},fontsize=8)
    fig.savefig(FIG/'R141_Figure2_calibration.png');plt.close(fig)
    return {'replicate_n':len(rows),'design':'Exchangeable Gaussian family effects; 16 coordinates; correlation .7. Envelope coverage, not empirical LLM safety.'}

def feasibility():
    A=np.array([[1,.25],[-.8,.35],[-.25,1.]])
    d=np.array([.8,-.4,.35]);c=np.array([1.,1.4]);rows=[]
    bs=np.linspace(.6,3,101);ds=np.linspace(0,.4,81)
    image=np.zeros((len(ds),len(bs)),dtype=int)
    for iy,delta in enumerate(ds):
        eps=.03
        lower=A-delta;upper=A+delta
        for ix,B in enumerate(bs):
            inner=linprog(c,A_ub=np.vstack([-lower,c]),b_ub=np.r_[-d-eps,B],bounds=(0,None),method='highs')
            outer=linprog(c,A_ub=np.vstack([-upper,c]),b_ub=np.r_[-d+eps,B],bounds=(0,None),method='highs')
            category='guaranteed_feasible' if inner.success else ('proved_infeasible' if not outer.success else 'unresolved')
            image[iy,ix]={'proved_infeasible':0,'unresolved':1,'guaranteed_feasible':2}[category]
            witness=None;gap=None
            if not outer.success:
                co=np.r_[-(d-eps),B]
                cert=linprog(co,A_ub=np.column_stack([upper.T,-c]),b_ub=np.zeros(2),A_eq=np.ones((1,4)),b_eq=[1.],bounds=(0,None),method='highs')
                assert cert.success and cert.fun < -1e-9
                witness=cert.x.tolist();gap=-cert.fun
            rows.append(dict(budget=float(B),effect_uncertainty=float(delta),residual_bound=eps,status=category,robust_x=json.dumps(inner.x.tolist()) if inner.success else '',dual_witness=json.dumps(witness),dual_gap=gap))
    csvwrite('repair_feasibility_grid.csv',rows)
    (OUT/'repair_design.json').write_text(json.dumps({'A':A.tolist(),'d':d.tolist(),'c':c.tolist(),'residual_bound':.03,'interpretation':['old loss','old survivor','new requirement']},indent=2))
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    im=ax[0].imshow(image,origin='lower',aspect='auto',extent=[bs[0],bs[-1],ds[0],ds[-1]],cmap=ListedColormap(['#c66b5a','#dfc16c','#60a690']),vmin=-.5,vmax=2.5)
    ax[0].set(xlabel='Intervention budget',ylabel='Uncertainty in each signed effect',title='a  Failure to certify has different causes')
    cb=fig.colorbar(im,ax=ax[0],ticks=[0,1,2]);cb.ax.set_yticklabels(['Infeasible','Unresolved','Feasible']);cb.ax.tick_params(labelsize=8)
    x=np.linspace(0,2,301)
    ax[1].plot(x,x-.8,label='Old loss reserve')
    ax[1].plot(x,.4-.8*x,label='Old survivor reserve')
    ax[1].plot(x,-.35-.25*x,label='New requirement reserve')
    ax[1].axhline(0,color='black',lw=1,ls=':');ax[1].set(xlabel='Amount of first intervention only',ylabel='Margin above required threshold',title='b  More of one intervention can make repair worse')
    ax[1].legend(fontsize=8)
    fig.savefig(FIG/'R141_Figure3_repair_geometry.png');plt.close(fig)
    return {'grid_n':len(rows),'dual_certificates':sum(x['status']=='proved_infeasible' for x in rows)}

if __name__=='__main__':
    start=time.time();meta={'seed':SEED,'scientific_role':'Numerical checks of stated model; not new LLM experiments.'}
    meta['gaussian']=sequential();meta['calibration']=calibration();meta['feasibility']=feasibility();meta['seconds']=time.time()-start
    meta['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='files'},indent=2))
