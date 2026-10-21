"""Analyse all eight fixed batches; no outcome-based selection or retuning.

Additional threshold sweeps are explicitly descriptive sensitivity analyses.
They do not change the choices that were frozen by the experimental runner.
"""
from pathlib import Path
from collections import defaultdict
import json,csv,math,re,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/experimental';OUT=ROOT/'results/control_analysis'
OUT.mkdir(exist_ok=True)
METHODS=['full_update','simple_scaling','minimum_norm','local_prediction','calibrated_prediction','direct_validation']
LABELS=['Full\nupdate','Simple\nscaling','Minimum\nnorm','Local\nprediction','Calibrated\nprediction','Direct\nvalidation']
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def j(path):return json.loads(path.read_text())
def safe_div(a,b):return a/b if b else None
def method_cost(method,events):
    if method in ['full_update','simple_scaling','minimum_norm']:return []
    if method=='direct_validation':return [e for e in events if '_direct_' in e.get('tag','')]
    if method=='local_prediction':return [e for e in events if '_sel_pilot' in e.get('tag','')]
    return [e for e in events if re.match(r'V\d+_(cal_|sel_pilot)',e.get('tag',''))]

def analyse():
    paths=sorted(RAW.glob('*/RESULT.json'))
    assert len(paths)==8, ('Eight complete batches required',len(paths))
    runs=[];methods=[];grid=[];costs=[];sensitivity=[];cal_sensitivity=[];checks=[]
    for p in paths:
        run=p.parent.name;d=j(p);frozen=j(p.parent/'choices_frozen.json')
        assert frozen['fresh_outcomes_opened'] is False
        assert frozen['choices']==d['frozen']['choices']
        assert len(d['candidates'])==8 and len(d['shortlist'])==4 and len(d['fresh_grid'])==16
        split=j(p.parent/'benchmark_rows.json');fam={k:{x['family'] for x in v} for k,v in split.items()}
        for roles in [['source_train','calibration','selection','fresh_old'],['update_train','cap_validation','cap_audit']]:
            for i,k in enumerate(roles):
                for kk in roles[i+1:]:assert not fam[k]&fam[kk],(run,k,kk)
        c=j(p.parent/'calibration.json');q=float(c['q']);ss=sorted(c['scores'].values());rank=math.ceil((len(ss)+1)*.9)
        expected=ss[rank-1] if rank<=len(ss) else math.inf
        assert math.isclose(q,expected) or q==expected
        assert d['frozen']['direct_candidate_record_calls']<=d['frozen']['candidate_record_budget']
        events=j(p.parent/'cost_events.json')
        checks.append({'run':run,'split_families_disjoint':True,'frozen_choice_preserved':True,'quantile_recomputed':True,'fresh_grid_pairs':16,'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        runs.append({'run':run,'model_family':run.split('_')[0],'calibration_families':len(ss),'calibration_records':d['protected_calibration_n'],'fresh_records':d['protected_fresh_n'],'fresh_families':next(iter(d['fresh_grid'].values()))['old_family_n'],'q':q,'rank':rank,'incumbent_new_accuracy':d['incumbent_new_accuracy'],'admissible_pairs':len(d['admissible']),'observation_budget':d['frozen']['candidate_record_budget'],'direct_observations':d['frozen']['direct_candidate_record_calls']})
        for key,out in d['fresh_grid'].items():
            name,s=key.split('@');rows=j(p.parent/f'rows/{name}_fresh_{float(s):.2f}_AUDIT.json');cap=j(p.parent/f'rows/{name}_cap_audit_{float(s):.2f}.json')
            assert out['old_loss_n']==sum(not x['valid'] for x in rows)
            assert out['new_success_n']==sum(x['valid'] for x in cap)
            families=defaultdict(list)
            for row in rows:families[row['family']].append(row['valid'])
            assert out['old_family_loss_n']==sum(not all(v) for v in families.values())
            cert=d['certificates'][key]
            grid.append({'run':run,'candidate':name,'strength':float(s),'new_accuracy':out['new_accuracy'],'new_gain':out['new_gain'],'audit_benefit_pass':out['new_floor_pass'],'old_loss_n':out['old_loss_n'],'old_n':out['old_n'],'old_family_loss_n':out['old_family_loss_n'],'old_family_n':out['old_family_n'],'old_family_loss_rate':safe_div(out['old_family_loss_n'],out['old_family_n']),'declared_families':cert['declared_family_n'],'false_declarations':cert['false_declared_family_n'],'envelope_miss_families':cert['envelope_miss_family_n']})
        for method in METHODS:
            v=d['methods'][method];choice=v['choice'];out=v['outcome'];key=f'{choice[0]}@{choice[1]:.2f}' if choice else None;cert=d['certificates'].get(key,{})
            if method=='calibrated_prediction' and choice:
                assert math.isfinite(q) and frozen['prediction_scores'][key]['calibrated']>0
            ev=method_cost(method,events)
            methods.append({'run':run,'method':method,'choice':key,'abstained':v['abstained'],'audit_benefit_pass':out['new_floor_pass'] if out else False,'new_accuracy':out['new_accuracy'] if out else None,'new_gain':out['new_gain'] if out else None,'old_loss_n':out['old_loss_n'] if out else None,'old_n':out['old_n'] if out else None,'old_family_loss_n':out['old_family_loss_n'] if out else None,'old_family_n':out['old_family_n'] if out else None,'old_family_loss_rate':safe_div(out['old_family_loss_n'],out['old_family_n']) if out else None,'declared_families':cert.get('declared_family_n') if method=='calibrated_prediction' else None,'false_declarations':cert.get('false_declared_family_n') if method=='calibrated_prediction' else None,'selection_record_calls':sum(e.get('record_calls',0) for e in ev),'selection_seconds':sum(e.get('seconds') or 0 for e in ev),'selection_seconds_complete':all(e.get('seconds') is not None for e in ev),'generated_tokens':sum(e.get('generated_tokens',0) for e in ev),'sequence_tokens':sum(e.get('sequence_tokens',0) for e in ev)})
        for e in events:
            tag=e.get('tag','')
            if e['kind']=='training':role='shared_training'
            elif 'AUDIT' in tag or 'audit' in tag or tag=='M0_fresh_old':role='scientific_audit'
            elif '_direct_' in tag:role='direct_selection'
            elif tag.startswith('M0_') or '_capval_' in tag:role='shared_preparation'
            else:role='pilot_calibration_selection'
            costs.append({'run':run,'role':role,'kind':e['kind'],'operation':tag or e.get('name'),'record_calls':e.get('record_calls',0),'training_records':e.get('records',0),'seconds':e.get('seconds'),'seconds_available':e.get('seconds') is not None,'prompt_tokens':e.get('prompt_tokens',0),'generated_tokens':e.get('generated_tokens',0),'sequence_tokens':e.get('sequence_tokens',0),'forward_calls':e.get('batch_forward_calls'),'reused_source_job':e.get('reused_source_job')})
        # Audit-only feasible menus reveal where a requirement changes the
        # available options. They never replace prospective selected outcomes.
        for absolute in [.20,.25,.30,.35,.40]:
            for gain in [.05,.10,.15,.20]:
                pool=[v for v in d['fresh_grid'].values() if v['new_accuracy']>=absolute and v['new_gain']>=gain-1e-12 and v['old_family_n']]
                sensitivity.append({'run':run,'absolute_floor':absolute,'gain_floor':gain,'audit_only_admissible_pairs':len(pool),'best_observed_old_family_loss':min(v['old_family_loss_n']/v['old_family_n'] for v in pool) if pool else None})
        # Post hoc sensitivity of measured error bounds; frozen method choices
        # are untouched. Larger pilots use already saved quarter/half margins.
        cm0=j(p.parent/'margins/M0_calibration.json');fm0=j(p.parent/'margins/M0_fresh_audit.json')
        cg=np.array(cm0['values']);fg=np.array(fm0['values'])
        lookup={row['id']:row['family'] for arr in split.values() for row in arr}
        cf=np.array([lookup[i] for i in cm0['ids']]);ff=np.array([lookup[i] for i in fm0['ids']])
        families=sorted(set(cf),key=lambda f:hashlib.sha256(f.encode()).hexdigest())
        sizes=sorted({n for n in [8,9,19,39,len(families)] if n<=len(families)})
        for pilot in [.05,.25,.5]:
            cr=np.zeros(len(cg));fr=np.zeros(len(fg));predict={}
            for name in d['shortlist']:
                cp=f'{name}_cal_pilot' if pilot==.05 else f'{name}_cal_{pilot:.2f}'
                fp=f'{name}_fresh_pilot_AUDIT' if pilot==.05 else f'{name}_fresh_margin_{pilot:.2f}_AUDIT'
                cd=(np.array(j(p.parent/f'margins/{cp}.json')['values'])-cg)/pilot
                fd=(np.array(j(p.parent/f'margins/{fp}.json')['values'])-fg)/pilot
                for strength in [.25,.5,.75,1.]:
                    ca=np.array(j(p.parent/f'margins/{name}_cal_{strength:.2f}.json')['values'])
                    fa=np.array(j(p.parent/f'margins/{name}_fresh_margin_{strength:.2f}_AUDIT.json')['values'])
                    cr=np.maximum(cr,np.abs(ca-cg-strength*cd));fr=np.maximum(fr,np.abs(fa-fg-strength*fd))
                    predict[f'{name}@{strength:.2f}']=fg+strength*fd
            cs={f:float(cr[cf==f].max()) for f in families}
            fs={f:float(fr[ff==f].max()) for f in set(ff)}
            for n in sizes:
                values=sorted(cs[f] for f in families[:n])
                for alpha in [.01,.05,.1,.2]:
                    rank=math.ceil((n+1)*(1-alpha));allowance=values[rank-1] if rank<=n else math.inf
                    any_declared=set();any_false=set()
                    for key,pre in predict.items():
                        for f in set(ff):
                            if float(pre[ff==f].min())-allowance>.001:
                                any_declared.add(f)
                                if not d['fresh_grid'][key]['family_pass'][f]:any_false.add(f)
                    cal_sensitivity.append({'run':run,'pilot_strength':pilot,'calibration_families':n,'alpha':alpha,'allowance':allowance,'finite':math.isfinite(allowance),'audit_families':len(fs),'joint_envelope_miss_families':sum(v>allowance for v in fs.values()),'any_declared_family_n':len(any_declared),'any_false_declaration_family_n':len(any_false),'analysis_status':'post_hoc_sensitivity_no_choice_retuning'})
    write('batches.csv',runs);write('methods.csv',methods);write('all_candidate_strength_outcomes.csv',grid);write('operation_costs.csv',costs);write('audit_only_benefit_sensitivity.csv',sensitivity)
    write('posthoc_calibration_pilot_sensitivity.csv',cal_sensitivity)
    summary={}
    for method in METHODS:
        rr=[x for x in methods if x['method']==method];chosen=[x for x in rr if not x['abstained']];passed=[x for x in chosen if x['audit_benefit_pass']]
        summary[method]={'batches':len(rr),'choices':len(chosen),'abstentions':len(rr)-len(chosen),'audit_benefit_passes':len(passed),'mean_family_loss_among_chosen':np.mean([x['old_family_loss_rate'] for x in chosen]) if chosen else None,'mean_family_loss_when_benefit_passes':np.mean([x['old_family_loss_rate'] for x in passed]) if passed else None,'selection_observations':sum(x['selection_record_calls'] for x in rr),'selection_seconds':sum(x['selection_seconds'] for x in rr)}
    contrasts=[]
    for comparator in ['full_update','simple_scaling','minimum_norm','direct_validation']:
        for proposed in ['local_prediction','calibrated_prediction']:
            paired=[]
            for run in [x['run'] for x in runs]:
                aa=next(x for x in methods if x['run']==run and x['method']==comparator);bb=next(x for x in methods if x['run']==run and x['method']==proposed)
                if aa['audit_benefit_pass'] and bb['audit_benefit_pass']:paired.append(bb['old_family_loss_rate']-aa['old_family_loss_rate'])
            contrasts.append({'method':proposed,'comparator':comparator,'paired_batches_both_benefit_pass':len(paired),'lower_old_loss_n':sum(x<0 for x in paired),'equal_old_loss_n':sum(x==0 for x in paired),'higher_old_loss_n':sum(x>0 for x in paired),'mean_difference':np.mean(paired) if paired else None})
    write('paired_contrasts.csv',contrasts)
    result={'complete_batches':len(runs),'candidate_training_runs':len(runs)*8,'audited_candidate_strength_pairs':len(grid),'methods':summary,'contrasts':contrasts,'limitations':['Repeated benchmark pools; batch outcomes are descriptive recipe replications, not new task populations.','Candidate-record equality is not FLOP or runtime equality.','Missing original shared-stage elapsed times remain missing.','A calibrated false-declaration guarantee is not a total old-task risk cap.','No faithful MUSCLE/forgetting-training baseline is implemented; no superiority claim over those algorithms.','Only incumbent-protected fresh old cases are candidate-audited; new entries into old correctness are not estimated.'],'verification':checks}
    (OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(3,1,figsize=(7.2,6.8),layout='constrained')
    colors=['#326B92','#BC654C','#438B74','#886697','#C7AA48','#4A8690'];rng=np.random.default_rng(141)
    for i,method in enumerate(METHODS):
        rr=[x for x in methods if x['method']==method]
        for row in rr:
            if row['abstained']:continue
            jitter=rng.uniform(-.13,.13);filled=row['audit_benefit_pass']
            marker='o' if row['run'].startswith('qwen') else '^'
            for k,value in enumerate([100*row['old_family_loss_rate'],100*row['new_accuracy'],row['selection_seconds']]):
                ax[k].scatter(i+jitter,value,s=34,marker=marker,facecolors=colors[i] if filled else 'none',edgecolors=colors[i],linewidths=1.0,zorder=3)
        for k,field in enumerate(['old_family_loss_rate','new_accuracy','selection_seconds']):
            vals=[r[field]*(100 if k<2 else 1) for r in rr if not r['abstained']]
            ax[k].plot([i-.19,i+.19],[np.mean(vals)]*2,color=colors[i],linewidth=1.8)
    ax[0].set(ylabel='Old family loss (%)',title='a  Old-task loss at the selected update',ylim=(-.8,19))
    ax[1].set(ylabel='New-task success (%)',title='b  Benefit at the same selected update',ylim=(8,72))
    ax[1].axhline(25,color='0.5',linestyle='--',linewidth=.8)
    ax[1].text(.02,26.2,'25% floor',transform=ax[1].get_yaxis_transform(),fontsize=7,ha='left',color='0.4')
    ax[2].set(ylabel='Selection time (seconds)',title='c  Measured old-task selection time',ylim=(-18,585))
    ax[0].scatter([],[],c='0.25',marker='o',label='Qwen');ax[0].scatter([],[],c='0.25',marker='^',label='SmolLM')
    ax[0].scatter([],[],facecolors='none',edgecolors='0.25',label='Benefit floor failed')
    ax[0].legend(fontsize=7.2,loc='upper left',ncol=3,frameon=False)
    for k,a in enumerate(ax):
        a.spines[['right','top']].set_visible(False);a.set_xticks(range(6),LABELS if k==2 else ['']*6);a.tick_params(axis='x',labelsize=8)
        a.set_xlim(-.48,5.48);a.tick_params(axis='y',labelsize=8);a.title.set_fontsize(10);a.yaxis.label.set_fontsize(9)
    figure=ROOT/'manuscript/figures/R141_Figure5_control'
    fig.savefig(figure.with_suffix('.png'),dpi=300)
    fig.savefig(figure.with_suffix('.pdf'));fig.savefig(figure.with_suffix('.svg'));plt.close(fig)
    print(json.dumps({'methods':summary,'contrasts':contrasts},indent=2))
    return result
if __name__=='__main__':analyse()
