"""Analyse all eight completed R142 comparisons from raw generated responses."""
from pathlib import Path
from collections import defaultdict
import json,csv,math
import numpy as np
from r142_extended_figures import COL,MARK,STY,HATCH,save,label,RUNS,ROOT,plt,pairedbar
DATA=ROOT/'results/compatibility';OUT=ROOT/'results/compatibility_analysis';OUT.mkdir(parents=True,exist_ok=True)
MAN=ROOT/'manuscript';METHODS=['full_update','muscle_loss','extra_cross_entropy'];NAMES=['Full update','MUSCLE loss','Extra CE']
def load(path):return json.loads(path.read_text())
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyse():
    rows=[];pairs=[];reproduction=[];costs=[];checks=[]
    for run in RUNS:
        p=DATA/run;result=load(p/'RESULT.json');protocol=load(p/'protocol.json')
        assert protocol['protocol']['frozen_before_new_outputs'] and protocol['prior_audit_reused']
        assert result['candidate']==protocol['spec']['candidate']
        base=load(p/'rows/M0_fresh_old.json');new0=load(p/'rows/M0_cap_audit.json');val0=load(p/'rows/M0_cap_validation.json')
        events=load(p/'cost_events.json');candidate=result['candidate'];benchmark=load(p/'benchmark_rows.json')
        assert [x['id'] for x in base]==[x['id'] for x in benchmark['fresh_old']]
        for method in METHODS:
            old=load(p/f'rows/{method}_fresh_old.json');new=load(p/f'rows/{method}_cap_audit.json');val=load(p/f'rows/{method}_cap_validation.json')
            assert [x['id'] for x in old]==[x['id'] for x in base]
            assert [x['id'] for x in new]==[x['id'] for x in new0]
            counts=defaultdict(int);protected=defaultdict(list)
            for a,b in zip(base,old):
                typ=('both_correct' if a['valid'] and b['valid'] else 'old_loss' if a['valid'] else 'old_entry' if b['valid'] else 'both_wrong');counts[typ]+=1
                pairs.append({'run':run,'method':method,'id':a['id'],'family':a['family'],'incumbent_correct':a['valid'],'updated_correct':b['valid'],'transition':typ})
                if a['valid']:protected[a['family']].append(b['valid'])
            n0=sum(x['valid'] for x in base);nf=len(protected);fl=sum(not all(x) for x in protected.values());acc=sum(x['valid'] for x in new)/len(new);bacc=sum(x['valid'] for x in new0)/len(new0)
            vacc=sum(x['valid'] for x in val)/len(val);vbase=sum(x['valid'] for x in val0)/len(val0)
            training=[x for x in events if x['kind']=='training' and x['method']==method]
            tr=training[0] if training else {}
            rows.append({'run':run,'method':method,'candidate':candidate,'all_old_n':len(base),'incumbent_old_correct':n0,'updated_old_correct':sum(x['valid'] for x in old),'both_correct':counts['both_correct'],'old_loss':counts['old_loss'],'old_entry':counts['old_entry'],'both_wrong':counts['both_wrong'],'protected_family_n':nf,'protected_family_loss':fl,'protected_family_loss_rate':fl/nf,'new_success':sum(x['valid'] for x in new),'new_n':len(new),'new_accuracy':acc,'new_gain':acc-bacc,'audit_benefit_pass':acc>=.25 and acc-bacc>=.15,'validation_benefit_pass':vacc>=.25 and vacc-vbase>=.15,'additional_training_seconds':tr.get('seconds',0),'optimizer_steps':tr.get('optimizer_steps',0),'teacher_forward_token_positions':tr.get('teacher_forward_token_positions',0),'old_teacher_target_tokens':tr.get('old_teacher_tokens',0),'training_target_tokens':tr.get('target_tokens',0)})
            assert counts['both_correct']+counts['old_loss']==n0
            assert sum(counts.values())==len(base)
        a=next(x for x in rows if x['run']==run and x['method']=='muscle_loss');b=next(x for x in rows if x['run']==run and x['method']=='extra_cross_entropy')
        mtrain=load(p/'training/muscle_loss.json');ctrain=load(p/'training/extra_cross_entropy.json')
        assert mtrain['order']==ctrain['order'] and mtrain['seed']==ctrain['seed']
        assert a['optimizer_steps']==b['optimizer_steps']
        assert all(math.isfinite(v) for v in mtrain['losses']+ctrain['losses'])
        # Compare the new joint evaluation with original R141 counts at same full update.
        prior=list(csv.DictReader((ROOT/'results/control_analysis/methods.csv').open()));prior=next(x for x in prior if x['run']==run and x['method']=='full_update');now=next(x for x in rows if x['run']==run and x['method']=='full_update')
        reproduction.append({'run':run,'original_candidate':prior['choice'],'recovered_candidate':candidate,'old_protected_n_match':int(prior['old_n'])==now['incumbent_old_correct'],'old_loss_match':int(prior['old_loss_n'])==now['old_loss'],'new_success_match':round(float(prior['new_accuracy'])*60)==now['new_success'],'original_old_loss':int(prior['old_loss_n']),'reevaluated_old_loss':now['old_loss'],'original_new_success':round(float(prior['new_accuracy'])*60),'reevaluated_new_success':now['new_success']})
        for e in events:costs.append({'run':run,'kind':e['kind'],'operation':e.get('tag',e.get('method')),'seconds':e['seconds'],'records':e.get('records',0),'prompt_tokens':e.get('prompt_tokens',0),'generated_tokens':e.get('generated_tokens',0),'teacher_forward_token_positions':e.get('teacher_forward_token_positions',0),'optimizer_steps':e.get('optimizer_steps',0)})
    write('methods.csv',rows);write('paired_old_transitions.csv',pairs);write('original_count_reproduction.csv',reproduction);write('operation_costs.csv',costs)
    summary={'batches':len(RUNS),'old_row_appearances':sum(x['all_old_n'] for x in rows if x['method']=='full_update'),'all_original_counts_reproduced':all(x['old_protected_n_match'] and x['old_loss_match'] and x['new_success_match'] for x in reproduction),'methods':{},'paired_contrasts':[],'scope':'all eight fixed candidates; reused audit pools; unweighted batch summaries; no independent deployment-population inference'}
    for method in METHODS:
        ss=[x for x in rows if x['method']==method];summary['methods'][method]={'audit_benefit_passes':sum(x['audit_benefit_pass'] for x in ss),'validation_benefit_passes':sum(x['validation_benefit_pass'] for x in ss),'mean_old_family_loss':float(np.mean([x['protected_family_loss_rate'] for x in ss])),'mean_all_old_accuracy':float(np.mean([x['updated_old_correct']/x['all_old_n'] for x in ss])),'mean_new_accuracy':float(np.mean([x['new_accuracy'] for x in ss])),'old_loss_total':sum(x['old_loss'] for x in ss),'old_entry_total':sum(x['old_entry'] for x in ss),'additional_training_seconds':sum(x['additional_training_seconds'] for x in ss)}
    for first,second in [('muscle_loss','full_update'),('muscle_loss','extra_cross_entropy'),('extra_cross_entropy','full_update')]:
        diff=[next(x for x in rows if x['run']==r and x['method']==first)['protected_family_loss']-next(x for x in rows if x['run']==r and x['method']==second)['protected_family_loss'] for r in RUNS]
        summary['paired_contrasts'].append({'method':first,'comparator':second,'lower_old_loss':sum(x<0 for x in diff),'equal_old_loss':sum(x==0 for x in diff),'higher_old_loss':sum(x>0 for x in diff)})
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return rows,summary,reproduction

def figures(rows):
    fig,axs=plt.subplots(2,4,figsize=(7.3,5.7),layout='constrained',sharey=True)
    for i,(run,ax) in enumerate(zip(RUNS,axs.flat)):
        rr=[next(x for x in rows if x['run']==run and x['method']==m) for m in METHODS]
        for j,x in enumerate(rr):
            ax.bar(j-.17,x['old_loss'],.32,color=COL[j],edgecolor='white',hatch='///',label='Losses' if j==0 else None)
            ax.bar(j+.17,x['old_entry'],.32,color=COL[j],edgecolor='white',hatch='..',label='Entries' if j==0 else None)
        ax.set(xticks=range(3),xticklabels=['Full','MUSCLE','Extra CE'],title=chr(97+i)+'  '+label(run));ax.tick_params(axis='x',labelrotation=30,labelsize=7)
        if i==0:ax.legend(fontsize=6)
    fig.supylabel('Paired old-task transitions / 100 audit rows');save(fig,'R142_ED9_compatibility')
    # Three old selection panels remain intact in the main figure; the new
    # compatibility comparison is a second separately labelled row.
    from r142_extended_figures import METHODS as SELECTORS,NAMES as SELECTNAMES
    prior=list(csv.DictReader((ROOT/'results/control_analysis/methods.csv').open()))
    fig,axs=plt.subplots(2,3,figsize=(7.3,6.6),layout='constrained')
    for j,(key,mult,title,ylabel) in enumerate([('old_family_loss_rate',100,'a  Frozen selector: old loss','Protected old-family loss (%)'),('new_accuracy',100,'b  Frozen selector: new benefit','New-task success (%)'),('selection_seconds',1,'c  Additional selection work','Measured seconds')]):
        ax=axs[0,j]
        for i,method in enumerate(SELECTORS):
            rr=[x for x in prior if x['method']==method];vals=[float(x[key])*mult for x in rr]
            for b,(x,v) in enumerate(zip(rr,vals)):
                ax.scatter(i+(b-3.5)*.045,v,marker='o' if x['run'].startswith('qwen') else '^',s=17,edgecolor=COL[i],facecolor=COL[i] if x['audit_benefit_pass']=='True' else 'white',linewidth=.8)
            ax.plot([i-.2,i+.2],[np.mean(vals)]*2,color='black',lw=1.1)
        ax.set(xticks=range(6),xticklabels=SELECTNAMES,ylabel=ylabel,title=title);ax.tick_params(axis='x',labelrotation=55,labelsize=6.5)
        if j==1:ax.axhline(25,color='.4',ls=':',lw=.6)
    for j,(key,mult,title,ylabel) in enumerate([('protected_family_loss_rate',100,'d  Training comparison: old loss','Protected old-family loss (%)'),('new_accuracy',100,'e  Training comparison: new benefit','New-task success (%)'),('additional_training_seconds',1,'f  Additional training work','Measured seconds')]):
        ax=axs[1,j]
        for i,run in enumerate(RUNS):
            rr=[next(x for x in rows if x['run']==run and x['method']==m) for m in METHODS]
            vals=[x[key]*mult for x in rr]
            ax.plot(range(3),vals,color=COL[i],ls=STY[i%4],marker=MARK[i],ms=3,alpha=.9,label=label(run))
        ax.set(xticks=range(3),xticklabels=['Full','MUSCLE','Extra CE'],ylabel=ylabel,title=title);ax.tick_params(axis='x',labelrotation=30,labelsize=7)
        if j==1:ax.axhline(25,color='.4',ls=':',lw=.6)
    axs[1,2].set_ylim(-1.2,34)
    axs[1,2].legend(fontsize=5.5,ncol=2,loc='upper right',framealpha=1);save(fig,'R142_Figure5_control')

def manuscripts(rows,summary,reproduction):
    s=summary['methods'];m=s['muscle_loss'];c=s['extra_cross_entropy'];f=s['full_update'];contrast=summary['paired_contrasts'][0]
    tex=r'''\paragraph{A learning comparator tests a different intervention.}
The supplementary comparison implements MUSCLE's published compatibility loss and a paired extra cross-entropy epoch on each of the eight frozen full-update candidates (Algorithm~\ref{alg:muscle}; Fig.~\ref{fig:control}d--f). Both continuations use the same training rows, initialization and number of optimizer steps; compatibility training additionally evaluates two frozen teachers. These previously inspected audit pools are reused transparently. Mean protected-family loss is '''+f"{100*m['mean_old_family_loss']:.2f}\\% with MUSCLE loss, {100*c['mean_old_family_loss']:.2f}\\% with extra cross-entropy and {100*f['mean_old_family_loss']:.2f}\\% for the joint full-update reevaluation. "+f"Their new-benefit audits pass in {m['audit_benefit_passes']}/8, {c['audit_benefit_passes']}/8 and {f['audit_benefit_passes']}/8 batches, respectively. "+r'''The complete old audit now includes entries into correctness as well as losses (Extended Data Fig.~\ref{fig:ed_compatibility}). A higher overall old-task accuracy can coexist with failures on previously correct tasks. The result measures a specific compatibility-training intervention; it does not establish a universal replacement for update selection. \sref{sec:compatibility}{The complete comparison} reports each batch, all transitions, costs and original-count reconstruction checks.
'''
    (MAN/'R142_COMPATIBILITY_MAIN.tex').write_text(tex)
    tex=r'''\section{Compatibility training and complete old-task transitions}
\label{sec:compatibility}
\mref{main:control}{Return to Result 5 and Figure~\ref{fig:control}}; \mref{alg:muscle}{return to Algorithm~\ref{alg:muscle}}.
The protocol fixes eight existing full-strength candidate choices before the new outputs are produced. The old and new teachers are frozen; the student-error token mask and forward KL follow the published MUSCLE objective.\cite{echterhoff2024} The original paper's task setup used different ranks, target layers and training duration. Here both compatibility and ordinary continued training use rank-eight query/value adapters, one epoch, the original update examples plus forty original replay rows, AdamW at $2\times10^{-4}$, accumulation of eight and gradient clipping at one. Both start from the same selected adapter, use the same row order and stop at the final epoch; no checkpoint is chosen on the audit. The compatibility objective uses temperature one and no auxiliary cross-entropy. All prompt and response token positions, stopping tokens, decoded responses, parser outcomes and final adapter hashes are retained.

\begin{equation}
 m_j=\mathbf 1\{\arg\max p_\theta(\cdot\mid y_{<j})\ne y_j\},\qquad
 \mathcal L=\frac{1}{J}\sum_{j=1}^J\left[m_j\mathrm{KL}(p_0\Vert p_\theta)+(1-m_j)\mathrm{KL}(p_1\Vert p_\theta)\right].
 \label{eq:muscle}
\end{equation}
The indicator uses the current student's next-token error. It does not switch teachers according to whether the old model is correct on the whole task. The first KL term pulls an incorrect student token toward the old teacher; the second retains the new teacher's distribution when the student predicts the target token. Ordinary continuation minimizes response-token cross-entropy. Equal optimizer steps do not equalize the additional teacher computation. These are method-level measurements under a common adapter budget, not an original-benchmark reproduction or a prospective test on newly collected tasks.

The entire old audit pool contains one hundred records per batch. This expands the original protected-only audit and distinguishes all four paired correctness transitions. A positive entry count does not cancel a loss on a particular previously working task. Family loss still uses only incumbent-protected members, as in the selection experiment. The same pinned checker and deterministic decoding evaluate all four models in each batch.

\begingroup\small\singlespacing
\begin{longtable}{llrrrrrr}\caption{Complete old transitions and new-task audit. Each row uses 100 old and 60 new records.}\label{tab:compatibility_transitions}\\\toprule
Batch & Model & Both correct & Lost & Entered & Both wrong & New correct & Benefit\\\midrule\endfirsthead
\toprule Batch & Model & Both correct & Lost & Entered & Both wrong & New correct & Benefit\\\midrule\endhead
'''
    for x in rows:
        short=x['run'].replace('qwen_batch','Q').replace('smollm_batch','S');name={'full_update':'Full','muscle_loss':'MUSCLE','extra_cross_entropy':'Extra CE'}[x['method']]
        tex+=f"{short} & {name} & {x['both_correct']} & {x['old_loss']} & {x['old_entry']} & {x['both_wrong']} & {x['new_success']}/60 & {'Pass' if x['audit_benefit_pass'] else 'Fail'}\\\\\n"
    tex+=r'\bottomrule\end{longtable}\endgroup'+'\n'+r'\mref{fig:ed_compatibility}{Return to Extended Data Fig.~\ref{fig:ed_compatibility}}.'+'\n'
    tex+=r'''\begin{table}[H]\centering\small
\caption{Additional training work, with the original full update as the starting checkpoint.}\label{tab:compatibility_cost}
\begin{tabular}{llrrr}\toprule Batch & Method & Steps & Seconds & Teacher input tokens\\\midrule
'''
    for x in rows:
        if x['method']=='full_update':continue
        tex+=f"{x['run'].replace('qwen_batch','Q').replace('smollm_batch','S')} & {'MUSCLE' if x['method']=='muscle_loss' else 'Extra CE'} & {x['optimizer_steps']} & {x['additional_training_seconds']:.1f} & {x['teacher_forward_token_positions']}\\\\\n"
    tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'+r'\mref{fig:control}{Return to Figure~\ref{fig:control}f}. Training times exclude shared checkpoint preparation and scientific evaluation. Teacher input tokens count both frozen forward passes; output scoring and memory costs are not represented by this count alone.'+'\n'
    tex+=r'\paragraph{Reconstruction and scope.} '
    if summary['all_original_counts_reproduced']:tex+='All eight jointly reevaluated incumbents and full updates reproduce the original protected denominators, full-update loss counts and new-audit success counts. '
    else:tex+='Some joint reevaluation counts differ from the original selection record. The per-batch reconstruction table is retained, and the two records are not pooled. '
    tex+=f"MUSCLE loss gives fewer protected-family losses than its paired full update in {contrast['lower_old_loss']} batches, the same number in {contrast['equal_old_loss']} and more in {contrast['higher_old_loss']}. "
    tex+=r'The eight batches reuse benchmark pools, and the added experiments reuse already inspected audit tasks. Counts and unweighted means are descriptive. No independent-population significance claim or equal-total-compute superiority claim is made. Exact row-level results and training losses are released, including benefit failures and errors.'+'\n'
    (MAN/'R142_COMPATIBILITY_SUPPLEMENT.tex').write_text(tex)
    # Add new panels to the main caption while preserving the original a--c.
    p=MAN/'R142_CONTROL_RESULTS.tex'
    if p.exists():
        text=p.read_text().replace('Repeated benchmark pools preclude interpreting the markers as independent deployment populations.}',r'Repeated benchmark pools preclude interpreting the markers as independent deployment populations. \textbf{d--f}, the separate supplementary training comparison shows protected-family loss, new-task success and additional training seconds for the original update, MUSCLE-loss continuation and ordinary cross-entropy continuation. Lines and markers identify the eight batches. The audits are jointly reevaluated; teacher computation is included in training time.}')
        p.write_text(text)
if __name__=='__main__':
    rows,summary,reproduction=analyse();figures(rows);manuscripts(rows,summary,reproduction)
