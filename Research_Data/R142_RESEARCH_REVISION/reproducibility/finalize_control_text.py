"""Integrate the complete, frozen eight-batch result without retuning choices."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript';OUT=ROOT/'results/control_analysis'
rows=list(csv.DictReader((OUT/'methods.csv').open()))
batches=list(csv.DictReader((OUT/'batches.csv').open()))
summary=json.loads((OUT/'SUMMARY.json').read_text())
assert summary['complete_batches']==8
text=r'''The complete experiment contains eight separately trained incumbents, sixty-four candidate updates and 128 finally audited candidate--strength pairs. Each batch has 52--60 protected calibration families and 58--78 protected fresh old families; each new-task audit has sixty records. Six frozen rules choose among the same shortlisted candidates. A useful update must pass both new-benefit requirements at its selected strength. Figure~\ref{fig:control} reports all eight selected outcomes per rule; no candidate is removed after seeing its old-task audit.

Full updates, local prediction, calibrated prediction and direct validation passed the new-benefit audit in all eight batches. Their unweighted mean protected-family loss rates were 9.81\%, 4.22\%, 8.33\% and 2.64\%, respectively. Simple scaling and minimum norm each passed in six batches. Their low losses cannot therefore be evaluated without the two benefit failures. Among their six benefit-passing batches, mean old-family losses were 2.56\% and 2.33\%. These are descriptive averages of complete batches drawn from repeated benchmark pools, not estimates from eight independent deployment populations.

Local prediction lost fewer old families than full update in six batches and the same number in two, with both rules retaining the required benefit. Compared with direct validation, it was better in one batch, equal in four and worse in three. Calibrated prediction was worse than direct validation in seven batches and equal in one. Against simple scaling, it was worse in five of the six batches where both retained the benefit and equal in the other. Thus this experiment supports checking the direction of change before adopting a full update, but does not establish that the calibrated selector minimizes old-task loss. Simple scaling often retained a useful new capability, so the earlier quarter-strength failure is not a universal consequence of reducing update size.

The declaration results explain part of this distinction. The calibrated choices issued 68 preservation declarations over 548 protected family appearances across the eight batches, with three false declarations. These totals are descriptive because families can recur across batches. The rule selected an update in every batch; finite bounds did not imply wide declaration coverage. Its selection objective counts families with positive lower bounds, leaving the remaining families outside that objective. A candidate can therefore receive more declarations while harming more undeclared tasks. The sharp total-loss bound above identifies this limitation without contradicting the scoped false-declaration theorem. Neither three observed errors nor sixty-five observed correct declarations establishes population calibration under unverified sampling assumptions.

The methods also used different computations. Local prediction required 660 additional old-record score evaluations across the eight batches, calibrated prediction 9,560, and direct validation 8,335 endpoint generations within the same per-batch observation cap. Their recorded selection times were 24.0, 347.9 and 3,137.4 seconds, respectively, on the assigned L4 devices. Teacher-forced scoring was cheaper than sequential generation in this implementation. These are measured selection-stage totals, excluding shared training, capability screening and final scientific audit; they are not complete deployment costs. The result exposes a measurable trade-off between observed loss and additional selection work, with direct validation producing the lowest mean loss among rules that passed the benefit audit in every batch.

Finally, saved margins allow sensitivity analyses without changing a primary choice. Increasing the pilot from 5\% to 25\% reduced the joint residual allowance in seven of eight batches; at 50\% it did so in all eight. More accurate path approximation did not produce universal task retention or a prospective improvement of the frozen selector. Confidence and calibration-size sweeps include the infinite-bound cases, while new-benefit sweeps show which audited options remain available as requirements tighten. These analyses identify settings for a future registered comparison; they are not used to replace unfavourable primary results.

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/R141_Figure5_control.pdf}
\caption{\textbf{A complete selection test reveals both useful prediction and its limits.} \textbf{a}, protected old-family loss; \textbf{b}, new-task success on sixty records; \textbf{c}, measured additional old-task selection time. Each marker is one complete batch at its exact selected candidate and strength. Circles denote Qwen and triangles SmolLM; hollow markers fail the audit benefit requirement. Short horizontal lines are unweighted means of the eight plotted outcomes, not uncertainty intervals. The dashed line shows the 25\% absolute floor; every passing point also meets the fifteen-percentage-point gain requirement. All methods selected an update in all batches. Zero additional selection time excludes shared preparation and training. Repeated benchmark pools preclude interpreting the markers as independent deployment populations.}
\label{fig:control}\end{figure}
'''
(MAN/'R141_CONTROL_RESULTS.tex').write_text(text)

abstract='''Language-model updates can introduce useful capabilities while disrupting tasks that previously worked. Average performance alone does not determine whether a replacement meets both requirements or how much evidence adoption needs. We analyse this problem through task-level score changes, uncertain intervention effects and observation costs. A simultaneous calibration rule bounds false preservation declarations across a fixed candidate set and update grid. A sharp probability bound shows why this guarantee can coexist with substantial loss among undeclared tasks. Signed repair constraints distinguish guaranteed feasibility, infeasibility within a stated budget and unresolved evidence. For independent Gaussian observations, we prove finite-error validation bounds and show how increasing a task's margin can reduce subsequent evidence cost. Eight complete update-selection experiments across two open-weight model families audit old loss and new benefit on each selected model. Local prediction reduces mean old-family loss from 9.81% for full updates to 4.22%; calibrated prediction yields 8.33%, compared with 2.64% for direct validation, with all four rules meeting the required new benefit in every batch. The results connect update assessment, repair and verification while exposing the limits of selective guarantees and inexpensive prediction.'''
assert 150<=len(abstract.split())<=200,len(abstract.split())
(MAN/'R141_ABSTRACT.tex').write_text(abstract.replace('%',r'\%')+'\n')

labels={'full_update':'Full update','simple_scaling':'Simple scaling','minimum_norm':'Minimum norm','local_prediction':'Local prediction','calibrated_prediction':'Calibrated prediction','direct_validation':'Direct validation'}
supp=r'''\section{A complete candidate-selection experiment}
\label{sec:control_experiment}
The experiment fixes two model families, four complete batches per family, eight candidate seeds per batch, a shortlist of four candidates and four strengths. It repeats incumbent training, candidate construction, selection and final audit. A candidate seed is not counted as a replicate of the complete procedure. Fresh within-batch audits do not create eight independent deployment populations from the previously studied BFCL pools. Table~\ref{tab:batch_design} gives realized protected units and observation budgets.

The frozen scientific rule requires at least 25\% new-task validation accuracy and a fifteen-percentage-point improvement over the incumbent. Both conditions must also hold on the independent audit. A calibrated choice requires a finite bound and at least one positive selection-family lower bound. There were no calibrated abstentions in these eight batches; this outcome does not remove the required abstention branch at smaller sample sizes. All choices were saved before candidate fresh outcomes were opened.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrr}\toprule
Batch & Cal. families & Cal. records & Audit families & Audit records & Obs. cap\\\midrule
'''
for r in batches:
 name=r['run'].replace('qwen_batch','Qwen ').replace('smollm_batch','SmolLM ')
 supp+=' & '.join([name]+[r[k] for k in ['calibration_families','calibration_records','fresh_families','fresh_records','observation_budget']])+r'\\'+'\n'
supp+=r'''\bottomrule\end{tabular}
\caption{Realized protected units and the common candidate-record cap. Only incumbent successes enter these old-task denominators. Each new audit has sixty records. Families may reappear in different batches, so summing the table does not yield the number of independent benchmark tasks.}
\label{tab:batch_design}\end{table}

\subsection*{Every selected update}
Table~\ref{tab:all_choices} reports each primary choice and its paired endpoints. An old family fails if any protected member fails. New-task counts use sixty audit records. The rule names describe the implemented selectors, not full reproductions of neighbouring learning algorithms. The released CSV also contains record-level old loss, new gains, declaration counts and operation costs.

\begingroup\small\singlespacing
\begin{longtable}{llcrrc}
\caption{Frozen choices and final outcomes in all eight complete batches. The final column records attainment of both audit benefit requirements.}\label{tab:all_choices}\\
\toprule Batch & Rule & Candidate/strength & Old-family loss & New success & Benefit\\\midrule\endfirsthead
\toprule Batch & Rule & Candidate/strength & Old-family loss & New success & Benefit\\\midrule\endhead
\bottomrule\endfoot
'''
for r in rows:
 name=r['run'].replace('qwen_batch','Q').replace('smollm_batch','S')
 choice=r['choice'].replace('@',' / ')
 supp+=' & '.join([name,labels[r['method']],choice,f"{r['old_family_loss_n']}/{r['old_family_n']}",f"{round(float(r['new_accuracy'])*60)}/60",'Yes' if r['audit_benefit_pass']=='True' else 'No'])+r'\\'+'\n'
supp+=r'''\end{longtable}\endgroup

Local prediction improved on full update in six batches and tied in two. It improved on direct validation in one, tied in four and lost in three. Calibrated prediction lost to direct validation in seven and tied in one. These paired comparisons concern old-family loss when both methods pass the benefit audit, not equal new-task accuracy or a universal ordering of preferences. In SmolLM batch four, for example, calibrated prediction achieved 35/60 new successes with 2/64 old-family losses, whereas direct validation achieved 30/60 with no old-family losses. A higher benefit requirement could change the admissible choice, but it cannot be introduced after observing results to alter the registered primary comparison.

\subsection*{Declarations and calibration errors}
The full-family calibration bound was finite in every batch. On the selected calibrated updates there were 68 declarations and three errors among 548 protected family appearances. They are reported as descriptive counts only. An error probability over all future eligible families differs from the conditional error fraction among declared families. The guarantee does not bound failures among undeclared families. Table~\ref{tab:declarations} records these quantities separately. A missed joint envelope need not be a false declaration: the prediction can be inaccurate while still leaving the relevant decision correct or the family undeclared.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrr}\toprule
Batch & Allowance & Audit families & Declared & False & Joint misses\\\midrule
'''
sens=list(csv.DictReader((OUT/'posthoc_calibration_pilot_sensitivity.csv').open()))
for b in batches:
 r=next(r for r in rows if r['run']==b['run'] and r['method']=='calibrated_prediction')
 s=next(s for s in sens if s['run']==b['run'] and s['pilot_strength']=='0.05' and s['alpha']=='0.1' and s['calibration_families']==b['calibration_families'])
 name=b['run'].replace('qwen_batch','Qwen ').replace('smollm_batch','SmolLM ')
 supp+=' & '.join([name,f"{float(b['q']):.3f}",r['old_family_n'],r['declared_families'],r['false_declarations'],s['joint_envelope_miss_families']])+r'\\'+'\n'
supp+=r'''\bottomrule\end{tabular}
\caption{Allowance in each model's logit units, declarations for the selected update, and envelope errors across all sixteen candidate--strength pairs. The maxima missed on 58 of 548 family appearances. This finite realized frequency is not a test of population calibration based on independent deployment families.}
\label{tab:declarations}\end{table}

\subsection*{Observation and runtime accounting}
Table~\ref{tab:selection_cost} sums the additional old-task selection operations, retaining shared preparation separately in the operation ledger. Direct validation uses 8,335 endpoint generations within the calibrated rule's aggregate cap of 9,560 candidate-record observations; its per-pair whole-family prefixes can leave unused budget. Pilot and calibration computations teacher-force accepted incumbent responses, whereas direct validation generates candidate outputs sequentially. Their counts have the same record unit but different computational work. Recorded times characterize this implementation on assigned L4 devices and do not include common training, new-capability screening or final audit.

\begin{table}[H]\centering\small
\begin{tabular}{lrrr}\toprule
Rule & Benefit-passing batches & Additional observations & Selection seconds\\\midrule
'''
for method,v in summary['methods'].items():
 supp+=' & '.join([labels[method],f"{v['audit_benefit_passes']}/8",str(v['selection_observations']),f"{v['selection_seconds']:.1f}"])+r'\\'+'\n'
supp+=r'''\bottomrule\end{tabular}
\caption{Observed selection-stage totals. Zero denotes no additional old-task selection operation, not free preparation or training. All methods made a choice in all eight batches. Missing original shared-stage times remain missing in the complete ledger.}
\label{tab:selection_cost}\end{table}

\subsection*{Sensitivity analyses without primary-choice changes}
The 160 audit-only benefit settings cross absolute requirements of 20\%, 25\%, 30\%, 35\% and 40\% with gains of five, ten, fifteen and twenty percentage points over the incumbent in each batch. Each row reports how many of the sixteen audited pairs remain admissible and the smallest observed old-family loss among them. This is an audit-only attainable menu, not a prospectively available selector. It reports sensitivity to operational requirements without replacing any primary result.

The 480 calibration sensitivity settings cross three pilots (0.05, 0.25 and 0.50), five family counts (8, 9, 19, 39 and the full available count), and four error levels (0.01, 0.05, 0.10 and 0.20) in each batch. Family prefixes follow a fixed hash order. Larger-pilot calculations use already saved quarter- and half-strength margins. Every score still maximizes across the frozen four candidates, four strengths and family members. Files report finite or infinite bounds, joint misses, families with any declaration and families with any false declaration across that grid. These grid-wide unions differ from the selected-candidate counts above.

At the primary 10\% error level and full calibration size, the 25\% pilot lowered the residual allowance in seven batches, while the 50\% pilot lowered it in all eight relative to the 5\% pilot. These analyses were specified for this revision but calculated after primary outcomes were available. They do not validate a retuned selector on an untouched task pool. Lower residual error is useful evidence about approximation quality, not proof of universal retention or an automatic improvement in total old-task loss.

\subsection*{What is and is not recoverable}
All eight final scientific records, split rows, prompts, generated responses, margins, frozen choices and cost events were recovered and their artifact checksums verified. The final analysis recomputes split disjointness, calibration ranks, method choices, row counts, family losses and observation caps from those records. Training adapters have verified hashes and their source-job recovery identifiers; binary checkpoint payloads are omitted from this compact scientific release. The training recipe and authenticated recovery script are included, and the source-job archive is distinguished from locally delivered data. Unavailable pre-crash shared-stage timings are not reconstructed by assumption.

The old audit evaluates only records that the incumbent passed. It therefore does not estimate entries from old-task failure into old-task success or net overall old-task accuracy. The repeated benchmark pool, operational family definition, numerical scoring/generation differences and lack of faithful MUSCLE or forecasting-training baselines remain limitations. These are boundaries of the claim, not additional positive results supplied by the new code.
'''
(MAN/'R141_CONTROL_SUPPLEMENT.tex').write_text(supp)
(MAN/'R141_RESPONSE_RESULTS.tex').write_text(r'''\section*{Results that change the interpretation}
All eight complete experimental batches finished and their scientific artifacts passed checksum and row-level consistency checks. There are sixty-four candidate training runs and 128 audited candidate--strength pairs. Full update, local prediction, calibrated prediction and direct validation passed the predefined new-benefit audit in all eight batches; their mean old-family loss rates were 9.81\%, 4.22\%, 8.33\% and 2.64\%. Simple scaling and minimum norm passed in six batches each. The calibrated selector performed worse than direct validation on old-family loss in seven batches and tied in one, so its statistical guarantee is not presented as empirical superiority.

The selected calibrated updates issued 68 declarations with three errors across 548 protected family appearances. A new sharp probability bound explains why accurate selective declarations need not imply low total old-task loss. All figures, the abstract, Results, Discussion, Methods and Supplement reflect these results. The old stateful-agent endpoint correction also changes the claims: both candidates lose all nine protected complete tasks because required final confirmations are missing. Functional state success alone is no longer called complete task retention.
''')
print(json.dumps({'abstract_words':len(abstract.split()),'complete_batches':8,'generated_primary_rows':len(rows),'sensitivity_rows':len(sens)}))
