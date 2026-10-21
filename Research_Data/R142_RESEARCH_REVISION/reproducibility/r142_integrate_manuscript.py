"""Build R142 text and cross-document links from the preserved R141 text.

This does not regenerate experimental observations. The compatibility result
fragment is generated separately from completed, verified provider artifacts.
"""
from pathlib import Path
import re,json,csv
ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript'
TITLE='Evaluating language-model updates under task and deployment constraints'
def base(name):return (MAN/('R141_'+name+'.tex')).read_text().replace('R141','R142')
def write(name,text):(MAN/(name+'.tex')).write_text(text)
def anchor(label):return '\\phantomsection\\label{'+label+'}\n'

main=base('MAIN_BODY')
heads=[('Introduction','main:introduction'),('Tasks, measurements and study design','main:design'),('Old-task loss and new capability must be measured on the same update','main:joint'),('Direction predicts change only within an adequately measured range','main:direction'),('Uncertain repair effects give three different conclusions','main:repair'),('Distance from failure determines evidence cost in a specified observation model','main:information'),('A complete update-selection test evaluates benefit, loss and cost together','main:control'),('Discussion','main:discussion')]
for h,l in heads:
    pattern=r'(\\(?:sub)?section\*?\{'+re.escape(h)+r'\})';main=re.sub(pattern,lambda m:m[0]+'\n'+anchor(l),main)
main=main.replace('Supplementary Part I is provided as a separate document; its direction section gives the formal statement and proof.',r'The \sref{sec:current_direction}{Supplementary direction theorem} gives the assumptions and proof.')
main=main.replace('Its corrected category definitions and original panels are retained in Extended Data and the Supplement.',r'Its corrected definitions and complete panels are in Extended Data Fig.~\ref{fig:ed_prospective} and \sref{hist:prospective}{the supporting protocol}.')
main=main.replace('The same questions also determine whether evidence can be reused at the next release.',r'The same questions determine whether evidence and dependent work can be reused at the next release. A regenerated report, for example, may depend on several model-assisted decisions, so its cost is shared rather than charged independently to every changed answer.')
main=main.replace('The need to look beyond average performance is established.',r'The need to look beyond average performance is established. Continuous integration for machine learning already formalizes release conditions and their evidence costs.\cite{renggli2019}')
main=main.replace('Three outcomes are kept distinct throughout.',r'Table~\ref{tab:main_support} links the five results to their algorithms, complete records and proofs. Three outcomes are kept distinct throughout.')
main=main.replace('This result changes the deployment interpretation.',r'Extended Data Fig.~\ref{fig:ed_turnover} reports additional finite-record transitions with their separate denominators. This result changes the deployment interpretation.')
main=main.replace('An earlier prospective SmolLM2 routing experiment',r'Extended Data Fig.~\ref{fig:ed_boundaries} gives the broader boundary comparisons. An earlier prospective SmolLM2 routing experiment')
main=main.replace('Those stress tests did not independently protect a new capability.',r'Extended Data Fig.~\ref{fig:ed_repair} retains every paired run. Those stress tests did not independently protect a new capability.')
support_add={
'fig:joint':r'\sref{hist:prospective}{Supporting paired records and historical protocol}; \sref{sec:control_experiment}{current independent-within-batch comparison}.',
'fig:calibration':r'\sref{sec:current_conformal}{Proof and interpretation}; \sref{sec:current_simulations}{simulation design}; Extended Data Figs.~\ref{fig:ed_declarations} and \ref{fig:ed_sensitivity}.',
'fig:repair_geometry':r'\sref{sec:current_robust}{Complete repair alternative and proof}; Algorithm~\ref{alg:repair}; Extended Data Fig.~\ref{fig:ed_repair}.',
'fig:information':r'\sref{sec:current_information}{Gaussian bounds and proofs}; \sref{sec:current_reuse}{information reuse}; \sref{sec:deployment}{finite-record and deployment algorithms}.',
}
for label,text in support_add.items():main=main.replace('\\label{'+label+r'}\end{figure}',r'\label{'+label+r'}\end{figure}'+'\n'+text+'\n')
main=main.replace(r'\subsection*{Direction predicts change only within an adequately measured range}',r'''A separate same-weight check identifies an evaluation sensitivity in the new experiment. On one fixed Qwen candidate, evaluating all one hundred old rows together gives one loss among the seventy-two incumbent successes; batching only those protected rows gives three losses. Repeating the complete pool reproduces the original token outputs exactly (Fig.~\ref{fig:joint}d; \sref{sec:joint_records}{paired records and batch-composition control}). This controlled difference makes batch composition part of the recorded evaluation policy. It does not explain every historical reproduction discrepancy.

\subsection*{Direction predicts change only within an adequately measured range}''')
main=main.replace('The displayed thresholds do not turn this historical record into a prospectively registered test.}',r'The displayed thresholds do not turn this historical record into a prospectively registered test. \textbf{d}, a separate same-weight Qwen batch-composition check holds the candidate and device fixed; protected losses are 1/72 with the complete pool, 3/72 with only protected rows and 1/72 when the complete pool is repeated. Complete-pool token outputs repeat exactly; two protected validity outcomes change between the different batch compositions.}')
main=main.replace(r'\sref{hist:prospective}{Supporting paired records and historical protocol}',r'\sref{sec:joint_records}{Supporting paired records and batch-composition control}')
deployment=r'''
Deployment adds a second kind of work: deciding which existing artifacts must be re-established. For a fixed evaluation record, bounded unread observations give exact lower and upper limits on the complete score. Algorithm~\ref{alg:tape} stops when both limits imply the same decision; it needs no population sampling assumption. For available sound preservation certificates with known prices, the choice between certificate purchase and reopening is then a minimum-cut problem with shared downstream costs. The reduction uses classical closure optimization.\cite{picard1976} Its scope and proof are given in \sref{sec:deployment}{the deployment analysis}.

This distinction restores the operational question behind the update problem. Two updates can each change one decision while requiring arbitrarily different work if those decisions activate different downstream operations. Knowing the location and shared dependencies of failures is therefore necessary in this model. In 360 synthetic workflow replays, exact stopping and dependency reuse reproduced every complete-evaluation decision and artifact. Reading used a mean 1,762.7 of 2,480 observations; all 600 small planning problems matched exhaustive optima (Extended Data Fig.~\ref{fig:ed_deployment}). These checks establish implementation consistency under specified prices and bounded records. They do not measure enterprise savings or solve uncertain adaptive testing and model training jointly.
'''
main=main.replace(r'\subsection*{A complete update-selection test',deployment+'\n'+r'\subsection*{A complete update-selection test')
main=main.replace('Simple selectors alone do not supply that evidence.',r'The supplementary compatibility experiment implements the published MUSCLE loss with a paired extra-training control (Algorithm~\ref{alg:muscle}). It supplies a specific learning comparison, while broader claims still require genuinely new deployment tasks and faithful access-matched comparisons.')
main=main.replace('The present evidence supports the scoped mechanisms and their limits; it should not be represented as a completed general solution to safe GenAI deployment.',r'The reusable contribution is an explicit chain from the task requirement to evidence, feasible intervention and deployment work, with each mathematical assurance tied to an observable endpoint and a stated failure output.')
# Remove repeated cautions after retaining each substantive limit at its claim.
shorten=[
('The decision unit is consequently a complete replacement at a specified strength. Candidate generation supplies possible replacements; the selection rule chooses one using development data; the audit measures that choice on held-out tasks. These stages answer different questions. A good candidate in the final audit does not establish that an organization could have identified it beforehand. Conversely, a poor selection rule does not show that every available candidate is poor. We retain the complete audited menu to distinguish the availability of a useful update from the ability to select it with limited observations. The audit-only best choice serves solely as a descriptive reference.',
'The decision unit is a complete replacement at a specified strength. Candidate generation, selection and audit have separate data roles. We retain the complete audit menu to distinguish an available useful update from a rule capable of identifying it beforehand; the audit-only best choice is descriptive.'),
('The mathematical probability concerns a false declaration for one randomly sampled eligible future family, simultaneously across the frozen candidate grid. It is not the conditional failure rate among declared families, and it does not protect every undeclared task.',
'The guarantee concerns a false declaration for one randomly sampled eligible future family across the frozen grid; it is neither conditional on declaration nor protection of every task.'),
('The two sides have a concrete resource interpretation. The left asks whether the least favourable allowed improvements can cover every deficit. The right assigns nonnegative weights to requirements and shows that their combined demand exceeds what any allowed expenditure can deliver. The weights expose the conflicting requirements. They are sensitivity values for this model, not measured economic valuations.',
'The left asks whether the least favourable allowed improvements cover every deficit. The right weights the requirements to show that their combined demand exceeds what the budget can deliver. Those weights identify the conflict and are model sensitivity values, not measured valuations.'),
('These are heavily repeated task variants, not independent conversational successes. The same extension illustrates an evidence limit. It had eight protected calibration variants; a 90\\% rank bound requires at least nine exchangeable calibration units. The infinite bound therefore reflected insufficient evidence for that rule, not an observed proof that every nonzero update is unsafe. Its 152 new-task variants represented only nineteen base tasks, and equivalent base variants were not fully isolated across historical splits.',
'The 152 variants represent nineteen bases with historical split overlap. Only eight protected calibration variants were available; a finite 90\\% rank bound requires at least nine exchangeable units. The infinite bound therefore indicated insufficient evidence for this rule.'),
('We provide an explicit stopping rule, a proved finite-error upper bound and a matching leading asymptotic constant for this Gaussian class. The Supplement withdraws the previous unproved general iterated-logarithm upper remainder. Simulations at positive and negative margins distinguish the lower bound, achieved observations and finite-error upper bound (Fig.~\\ref{fig:information}). They show the expected increase in evidence near zero while retaining the rule\'s finite-confidence conservativeness. They do not substitute for measuring a suitable observation model in an LLM application.',
'An explicit stopping rule has a proved finite-error upper bound and the matching leading asymptotic constant for this Gaussian class. Simulations separate the lower bound, achieved observations and finite-error upper bound (Fig.~\\ref{fig:information}). The Supplement corrects the earlier unsupported general remainder. Applying these costs to LLMs requires an appropriate observation model.'),
('The study separates three reasons why a model update may fail to receive support. It may not retain the new benefit at a strength that preserves old tasks. Its permitted interventions may be unable to satisfy the combined requirements. Or the available observations may be too weak to distinguish an acceptable update from an unacceptable one. These statements imply different actions and require different evidence. A mathematical certificate is useful only when its task endpoint, intervention scope and statistical coverage are made explicit.',
'An update may fail to retain the new benefit, its allowed interventions may conflict with the combined requirements, or its evidence may be inconclusive. These explanations call for different actions: a different candidate, intervention or budget, or more information. Every mathematical assurance must specify its endpoint, intervention scope and coverage.'),
('Repeated releases introduce a further distinction between recovering eventually and preserving service throughout. An intervention can restore a final task margin after an earlier release has already broken the task. The supporting finite-horizon model therefore distinguishes endpoint recovery from constraints imposed at every release. Similarly, a conditional error budget across releases controls cumulative false declarations only when each release satisfies the stated conditional guarantee. Reusing an old calibration set after adapting candidate construction to its results need not satisfy that requirement. This identifies a concrete research problem for continuing deployment: how to retain useful evidence while keeping the next release\'s inference valid. The present finite-grid and transcript results specify the conditions; they do not demonstrate a general adaptive release policy.',
'Repeated releases distinguish eventual recovery from uninterrupted preservation. The finite-horizon model therefore imposes constraints at each release, not just at the final endpoint. Likewise, cumulative false-declaration control needs valid conditional guarantees at successive releases; repeatedly adapting to an old calibration set need not preserve them. Retaining useful evidence while validating the next release remains an open deployment problem.'),
('Several findings resist a simple interpretation of better model scores as easier deployment. Similar new-task performance coexisted with different old-task losses in one controlled comparison. Shrinking those updates improved retention but missed the stated new-benefit reference. A repair that exactly satisfied calibration constraints lost both survivors and new capability on fresh cases. Moreover, a stronger model margin can reduce later evidence cost in the Gaussian class, so minimal repair need not minimize total expenditure. Each conclusion is supported within its declared experiment or mathematical model; none establishes that all model updates obey one universal scalar law.',
'The observed trade-offs matter for adoption. Shrinking one historical update removed its intended benefit; repairing calibration rows failed to protect fresh survivors and new capability. In the Gaussian model, increasing a margin reduces later testing, so minimum movement need not minimize total cost. These are distinct scoped mechanisms, not one universal law.'),
('Three outcomes are kept distinct throughout. Task loss means an incumbent success becomes a failure under the task evaluator. A preservation declaration is a claim issued by a specified rule about a particular task or family. Envelope coverage measures whether a bound contains the observed score change. A high envelope-coverage rate does not mean most tasks receive declarations, and issuing no declarations does not mean an update has succeeded. Population reliability claims require additional sampling assumptions beyond these finite benchmark outcomes.',
'Task loss is an incumbent success becoming a failure. A preservation declaration is a rule\'s claim about a task or family. Envelope coverage measures whether a score bound contains the observed change. We report all three because a wide envelope may support few declarations, while abstention does not establish successful adoption. Population claims need additional sampling assumptions.'),
('Figure~\\ref{fig:repair_geometry} varies uncertainty and budget in the stated three-requirement model. An intermediate unresolved region separates guaranteed repair from proved infeasibility. Every plotted optimistic infeasibility has a verified dual witness. This analysis identifies where the recommended action changes and makes explicit which assumption supports it; it does not estimate a universal repair threshold for LLMs.',
'Figure~\\ref{fig:repair_geometry} varies uncertainty and budget in a specified three-requirement model. Its unresolved region separates guaranteed repair from proved infeasibility; every infeasible point has a verified dual witness. The thresholds apply to that model.')]
for old,new in shorten:
    assert old in main,old[:70]
    main=main.replace(old,new)
write('R142_MAIN_BODY',main)

control=base('CONTROL_RESULTS')
control=control.replace('These are descriptive averages of complete batches drawn from repeated benchmark pools, not estimates from eight independent deployment populations.','')
control=control.replace('Figure~\\ref{fig:control} reports all eight selected outcomes per rule;',r'Figure~\ref{fig:control} reports all eight selected outcomes per rule, and Extended Data Fig.~\ref{fig:ed_candidates} retains all 128 candidate--strength outcomes;')
control=control.replace('The declaration results explain part of this distinction.',r'Extended Data Fig.~\ref{fig:ed_declarations} separates declaration errors from total loss. The declaration results explain part of this distinction.')
control=control.replace('These totals are descriptive because families can recur across batches. The rule selected an update in every batch; finite bounds did not imply wide declaration coverage. Its selection objective counts families with positive lower bounds, leaving the remaining families outside that objective. A candidate can therefore receive more declarations while harming more undeclared tasks. The sharp total-loss bound above identifies this limitation without contradicting the scoped false-declaration theorem. Neither three observed errors nor sixty-five observed correct declarations establishes population calibration under unverified sampling assumptions.',
'Families can recur across batches. Counting positive lower bounds leaves other families outside the selection objective, so more declarations can coexist with more total loss. The sharp bound explains this limitation. These finite counts do not establish population calibration under unverified sampling assumptions.')
control=control.replace('The methods also used different computations.',r'Extended Data Fig.~\ref{fig:ed_costs} provides the complementary call, token and time accounts. The methods also used different computations.')
control=control.replace('These analyses identify settings for a future registered comparison;',r'Extended Data Fig.~\ref{fig:ed_sensitivity} reports the sensitivity surfaces. These analyses identify settings for a future registered comparison;')
control=control.replace('Finally, saved margins allow sensitivity analyses without changing a primary choice. Increasing the pilot from 5\\% to 25\\% reduced the joint residual allowance in seven of eight batches; at 50\\% it did so in all eight. More accurate path approximation did not produce universal task retention or a prospective improvement of the frozen selector. Confidence and calibration-size sweeps include the infinite-bound cases, while new-benefit sweeps show which audited options remain available as requirements tighten. Extended Data Fig.~\\ref{fig:ed_sensitivity} reports the sensitivity surfaces. These analyses identify settings for a future registered comparison; they are not used to replace unfavourable primary results.',
r'Post hoc sensitivity analyses retain every frozen choice. A 25\% pilot reduced the residual allowance in seven of eight batches; a 50\% pilot did so in all eight. Extended Data Fig.~\ref{fig:ed_sensitivity} shows the calibration, confidence and benefit sweeps, including infinite bounds. These exploratory calculations identify future comparisons rather than replace primary outcomes.')
control += '\n'+r'\sref{sec:control_experiment}{Complete batch tables and frozen selection record}; \sref{sec:current_conformal}{declaration guarantee and its limit}.'+'\n'+r'\input{R142_COMPATIBILITY_MAIN.tex}'+'\n'
control=control.replace('Repeated benchmark pools preclude interpreting the markers as independent deployment populations.}',r'Repeated benchmark pools preclude interpreting the markers as independent deployment populations. \textbf{d--f}, the separate supplementary training comparison shows protected-family loss, new-task success and additional training seconds for the original update, MUSCLE-loss continuation and ordinary cross-entropy continuation. Lines and markers identify the eight batches. The full old pools are jointly reevaluated in panels d--f; their original-count reconstruction differences are documented in the Supplement. Teacher computation is included in training time.}')
control=control.replace(r'\begin{figure}[H]',r'\begin{figure}[!htbp]')
write('R142_CONTROL_RESULTS',control)

methods=base('METHODS')
methods=methods.replace('Full schemas and prompt hashes are retained so the grouping can be re-audited.',r'Full schemas and prompt hashes are retained so the grouping can be re-audited. Explicit schema and data validation are established components of production machine-learning pipelines.\cite{polyzotis2019}')
methods=methods.replace(r'\section{Methods}',r'\section{Methods}'+'\n'+anchor('main:methods'))
methods=methods.replace('The study contains three distinct forms of evidence: analytic results conditional on a specified model, numerical checks of those results under known synthetic laws, and language-model experiments with task-specific evaluators. The mathematical claims are established by proofs, not by the number of simulations. Historical experiments are retained with their original identifiers and available provenance; they are not relabelled as fresh R142 replications. A claim table records each result\'s task, model state, evaluation unit, permitted interpretation and primary-data status. The original packages and figures are preserved separately so that corrected interpretations remain traceable.',
'Analytic results state their model assumptions; numerical checks use known synthetic laws; language-model experiments use explicit task evaluators. The claim and support tables link each result to its model state, evaluation unit and primary-data status. Historical identifiers and original packages remain available for provenance.')
methods=methods.replace('Full-strength capability-validation accuracy ranks the eight candidates; closeness to the median update norm and candidate name break the stated shortlist ties. The top four form the fixed candidate set. They are evaluated at strengths $0.25,0.50,0.75,1.00$. An admissible candidate--strength pair must achieve at least 25\\% capability-validation accuracy and improve by at least fifteen percentage points over the incumbent on that validation pool. Both conditions are repeated on the independent capability audit for the same final pair. Passing validation does not guarantee passing audit. These floors define a useful improvement for this controlled experiment; they have no status as general industry safety requirements.',
'Full-strength capability-validation accuracy ranks eight candidates; distance from the median update norm and candidate name break shortlist ties. The top four are evaluated at strengths $0.25,0.50,0.75,1.00$. Admissibility requires at least 25\\% validation accuracy and a fifteen-percentage-point gain over the incumbent; the same pair must pass both floors on audit. These experimental floors are not industry standards.')
methods=methods.replace('``Parallel'' means several required calls, which may invoke the same function more than once. It is not synonymous with choosing among several available tools. BFCL\'s multiple-function categories concern available choices and were used as proxy labels in the earlier routing study; their labels are not literal required-call counts. The corrected task description therefore separates routing classification, generated function calls, and stateful tool execution. A correct banking-intent class in BANKING77 is another distinct endpoint.\\cite{casanueva2020} None of these measurements is treated as a real banking deployment outcome.',
'``Parallel'' requires several calls, possibly to the same function. The earlier routing study instead used available-function categories as proxy labels; these do not count required calls. BANKING77 intent classification and stateful tool execution have separate endpoints.\\cite{casanueva2020}')
parallel_start=methods.index('``Parallel')
parallel_end=methods.index('\n\n',parallel_start)
methods=methods[:parallel_start]+r"``Parallel'' requires several calls, possibly to the same function. The earlier routing study used available-function categories as proxy labels; these do not count required calls. BANKING77 intent classification and stateful tool execution have separate endpoints.\cite{casanueva2020}"+methods[parallel_end:]
methods=methods.replace('The old audit includes only incumbent successes, so it cannot estimate entries into old-task correctness or net overall old-task accuracy.', 'The original selection audit includes only incumbent successes. The added compatibility comparison evaluates the entire old audit pool and therefore also records entries into correctness and overall old-task accuracy.')
methods=methods.replace('These selectors are distinct operational controls, not faithful reproductions of all neighbouring learning algorithms. MUSCLE changes the update-training strategy, and forgetting predictors can use additional historical refinement tasks.\\cite{echterhoff2024,jin2025} Their training, access and calibration costs would have to be included in a claim of comparative superiority. The manuscript makes no such superiority claim from these simple selectors.',r'The six selectors are operational controls. The additional MUSCLE-loss experiment changes training and records that extra work explicitly; forgetting predictors trained on historical refinement tasks require a different information budget.\cite{echterhoff2024,jin2025}')
# Move an overly long operational incident account into the existing complete SI.
start=methods.index('A subsequent adapter-scaling lookup failed')
end=methods.index(' These details specify the experimental recipe;',start)
methods=methods[:start]+r'An adapter-interface failure was corrected by recovering the exact saved weights and cached results, verified by checksums; \sref{sec:control_experiment}{the complete record} documents this continuation and its missing shared-stage timings.'+methods[end:]
start=methods.index('The R138 ToolSandbox extension uses a stateful execution environment.')
end=methods.index(r'\subsection*{Theory, numerical sensitivity and statistics}',start)
methods=methods[:start]+r'''The R138 ToolSandbox extension distinguishes functional milestones from the final user-facing answer.\cite{toolsandbox2025} Its selected pair departed from the written exact-success criterion and was not reselected retrospectively. A required-answer audit of the ten checksum-verified recovered archives shows that both full-strength candidates omitted confirmations on all nine protected old variants; those variants represented two base tasks. On 152 new variants from nineteen base tasks, the earlier two successes per candidate become zero and one; the remaining success has no benchmark final-answer requirement. Two parser errors count as failures. Historical base overlap, eight protected calibration variants and the absence of independent semantic annotation prevent a strong population claim. Complete trajectories, denominators and corrections are in \sref{sec:r138_answers}{the required-answer audit}.

'''+methods[end:]
methods=methods.replace(r'\subsection*{Data and code availability}',r'''\subsection*{Compatibility training on previously used audit pools}
For every batch, we recover the incumbent and the previously selected full-strength update from checksummed saved adapters. The selected candidate and a new training seed are fixed before new compatibility outputs are produced. The MUSCLE-loss adapter starts from that update and uses the old teacher whenever the current student's ground-truth-token prediction is wrong; otherwise it uses the frozen new teacher.\cite{echterhoff2024} Loss is forward KL from the selected teacher to the student, averaged over response tokens including stopping, at temperature one. We use the original update examples plus the same forty replay examples, one epoch, rank-eight query/value adapters and the original optimizer settings. These resource choices differ from the published paper's rank-128 all-linear-layer, ten-epoch task setup. This is a reproduction of its specified loss in our setting, not its reported benchmark numbers.

An ordinary cross-entropy continuation starts from the same adapter with the same row order, seed and number of optimizer steps. It controls additional training duration; teacher forward passes make compatibility training more expensive, and their costs are recorded. The final checkpoint is evaluated without outcome-dependent tuning. The incumbent, original update and both continuations undergo joint evaluation on every old audit row and on new validation and audit rows. Paired transition tables therefore include both lost and newly correct old tasks. These audit pools were already examined in R141; the comparison is supplementary development evidence. It does not create a new untouched task distribution. Algorithm~\ref{alg:muscle} and \sref{sec:compatibility}{the complete comparison} specify all settings, numerical outcomes and reconstruction checks.

\subsection*{Finite records and downstream work}
The new deployment analysis assumes a fixed bounded evaluation record and an explicit threshold. Stopping uses the attainable range of its complete sum; an ambiguous range requires another observation. We exhaust all binary tapes of lengths one to twelve at the distinct extreme and majority thresholds. We also compare every returned minimum-cut solution against exhaustive enumeration in 600 small graphs, including zero prices and forced reopening. A separate synthetic workflow study uses 120 tapes of eighty decisions and thirty-one observations, repeated under separate, shared and unequal-cost dependencies. Outcome equality is checked both at each decision and at each downstream artifact. The observation laws, dependency sets and prices are simulation inputs, not fitted enterprise quantities. \sref{sec:deployment}{The proofs and complete inputs} state the validity boundary. Algorithms~\ref{alg:select}--\ref{alg:tape} make the selection, repair and deployment procedures executable.

\input{R142_ALGORITHMS.tex}

\subsection*{Data and code availability}''')
methods+=r'''
\paragraph{Supporting analytical scope.}
The \sref{sec:support_index}{linked supporting map} connects the retained extensions to the five results: multi-action and continuous decisions extend the direction result; shared-parameter, partial and robust recovery extend repair; Bellman decisions, paid service tiers and finite repeated-run rules specify alternative decision thresholds; information allocation, provider information and repeated adoption extend evidence and deployment work. These are conditional model extensions. Historical numerical records without recoverable primary outputs remain labelled provenance and do not establish new quantitative claims.
'''
# Keep the supporting-scope explanation with the analytical methods instead of
# allowing it to form a short isolated page after the unbroken algorithms.
scope_start=methods.index(r'\paragraph{Supporting analytical scope.}')
scope=methods[scope_start:].strip()
methods=methods[:scope_start].rstrip()+'\n'
availability_start=methods.index(r'\subsection*{Data and code availability}')
availability=methods[availability_start:].strip()
methods=methods[:availability_start].rstrip()+'\n'
methods=methods.replace(r'\input{R142_ALGORITHMS.tex}',scope+'\n\n'+availability+'\n\n'+r'\input{R142_ALGORITHMS.tex}')
write('R142_METHODS',methods)

# Current and retained SI: explicit section anchors and meaningful return routes.
def topic(title):
    t=title.lower()
    if any(x in t for x in ['repair','recovery','intervention','effect','optimizer','tractab','sealed-mixture','nonlinear neural']):return ('main:repair','Result 3: feasible repair')
    if any(x in t for x in ['information','evidence','validation','certification','reliability','budget','tier','sequential','bellman','release history','repeated','multi-release','migration','adoption','cost','deployment','carried','reuse','no-return','provider','data order','training versus']):return ('main:information','Result 4: evidence and deployment work')
    if any(x in t for x in ['direction','boundary','margin','preservation','continuous','affine','aggregate','representation','decision state','crossing','turnover']):return ('main:direction','Result 2: direction and preservation')
    if any(x in t for x in ['control','candidate','calibration','finite-data','compatibility','prospective','r134','bfcl','r138','toolsandbox','answer']):return ('main:control','Result 5: complete update evaluation')
    if any(x in t for x in ['same-case','frontier','checkpoint','human','empirical']):return ('main:joint','Result 1: paired task outcomes')
    return ('main:methods','Methods: scope, assumptions and provenance')
sections=[];objects=[]
def route(text,file):
    # Starred section headings have no numeric counter; name those links
    # explicitly instead of displaying an inherited part number.
    section_names={'sec:assumptions':'sources and assumptions table',
                   'sec:current_information':'Gaussian information analysis',
                   'sec:r138_answer_audit':'required-answer audit'}
    for label,name in section_names.items():
        old=r'Section~\ref{'+label+'}'
        link=r'\sref{'+label+'}{'+name+'}'
        text=text.replace('. '+old,'. The '+link)
        text=text.replace(old,'the '+link)
    text=re.sub(r'\\url\{([0-9a-f]{32,64})\}',lambda z:r'\nolinkurl{'+z[1]+'}',text)
    text=re.sub(r'\\cite\{([^}]+)\}',lambda z:r'\cite{'+','.join(z[1].split(',')[:3])+'}',text)
    matches=list(re.finditer(r'\\section\*?\{([^{}]+)\}',text));parts=[];last=0
    for i,m in enumerate(matches):
        parts.append(text[last:m.end()]);title=m[1];label=f'support:{file}:{i+1}';target,desc=topic(title)
        if file=='R142_DEPLOYMENT_THEORY':target,desc='main:information','Result 4: evidence and deployment work'
        if file=='R142_COMPATIBILITY_SUPPLEMENT':target,desc='main:control','Result 5: complete update evaluation'
        if file=='R142_PAIRED_RECORDS':target,desc='main:joint','Result 1: paired outcomes and execution control'
        parts.append('\n'+anchor(label)+r'{\small\mref{'+target+'}{'+desc+r'}.}\par'+'\n')
        sections.append({'section':title,'label':label,'main_label':target,'main_support':desc,'file':file+'.tex'})
        last=m.end()
    parts.append(text[last:]);text=''.join(parts)
    # Every supplementary formal result and display has an explicit return link.
    headings=list(re.finditer(r'\\section\*?\{([^{}]+)\}',text))
    def annotate_object(m):
        kind=m[1];block=m[0]
        preceding=[h[1] for h in headings if h.start()<m.start()]
        target,desc=topic(preceding[-1] if preceding else file)
        if file=='R142_DEPLOYMENT_THEORY':target,desc='main:information','Result 4: evidence and deployment work'
        if file=='R142_COMPATIBILITY_SUPPLEMENT':target,desc='main:control','Result 5: complete update evaluation'
        if file=='R142_PAIRED_RECORDS':target,desc='main:joint','Result 1: paired outcomes and execution control'
        labels=re.findall(r'\\label\{([^}]+)\}',block)
        if not labels and kind in ['theorem','proposition','lemma']:
            label=f'object:{file}:{len(objects)+1}'
            block=re.sub(r'(\\begin\{'+kind+r'\}(?:\[[^\]]*\])?)',lambda z:z[0]+'\\label{'+label+'}',block,count=1);labels=[label]
        if not labels and kind=='longtable':
            label=f'object:{file}:{len(objects)+1}';start=block.index(r'\begin{longtable}')+len(r'\begin{longtable}')
            while block[start].isspace():start+=1
            if block[start]=='[':start=block.index(']',start)+1
            while block[start].isspace():start+=1
            assert block[start]=='{';depth=1;end=start+1
            while depth:
                if block[end]=='{':depth+=1
                elif block[end]=='}':depth-=1
                end+=1
            title=preceding[-1] if preceding else 'Supporting analysis'
            block=block[:end]+'\n'+r'\caption{'+title+r'.}\label{'+label+r'}\\'+'\n'+block[end:];labels=[label]
        if labels:objects.append({'kind':kind,'label':labels[0],'main_label':target,'main_support':desc,'file':file+'.tex'})
        return block+'\n'+r'{\footnotesize\mref{'+target+'}{'+desc+r'}.}\par'+'\n'
    text=re.sub(r'\\begin\{(theorem|proposition|lemma|figure|table|longtable)\}(?:\[[^\]]*\])?.*?\\end\{\1\}',annotate_object,text,flags=re.S)
    if file=='R142_SUPPORTING_RECORD':
        text=text.replace(r'\section{Supporting migration-work reconstruction: validate, repair and reopen}',r'\section{Supporting migration-work reconstruction: validate, repair and reopen}'+'\n'+anchor('hist:migration'))
        text=text.replace(r'\section{Prospective effect transfer, official-BFCL confirmation and stateful execution (R132--R138)}',r'\section{Prospective effect transfer, official-BFCL confirmation and stateful execution (R132--R138)}'+'\n'+anchor('hist:prospective'))
    return text
for name in ['CURRENT_THEORY','ASSUMPTIONS','CONTROL_SUPPLEMENT','R138_ENDPOINT_AUDIT','SUPPORTING_RECORD']:
    text=base(name)
    if name=='ASSUMPTIONS':
        text=text.replace(r'\bottomrule',r'''Fixed finite-record stopping & Complete-score bounds in Theorem~\ref{thm:tape}; threshold specified by the application. & Bounded observations yield an exact completion interval. The illustrative majority threshold is a simulation choice, not a population-risk or industry requirement.\\
Fixed certificate prices and shared work & Shared fixed-cost network flows and graph closure.\cite{rhys1970,picard1976} & Supports the minimum-cut subproblem with known nonnegative prices and available sound certificates. Unknown certificate acquisition, omitted state and non-idempotent tool effects require additional modelling.\\
Published compatibility loss & MUSCLE's student-error mask and teacher KL.\cite{echterhoff2024} & The loss is implemented under this study's adapter and training budget. The paired extra-epoch control does not equalize teacher compute, and reused audit tasks do not become independent confirmation.\\
\bottomrule''',1)
    if name=='R138_ENDPOINT_AUDIT':text=text.replace(r'\section{Retrospective audit of required ToolSandbox answers}',r'\section{Retrospective audit of required ToolSandbox answers}'+'\n'+anchor('sec:r138_answers'))
    write('R142_'+name,route(text,'R142_'+name))
# Do not repeatedly add routes to hand-authored files.
for filename in ['R142_DEPLOYMENT_THEORY','R142_COMPATIBILITY_SUPPLEMENT','R142_PAIRED_RECORDS']:
    p=MAN/(filename+'.tex')
    if p.exists():
        text=p.read_text();text=re.sub(r'\\phantomsection\\label\{support:[^}]+\}\n\{\\small\\mref\{[^}]+\}\{[^}]+\}\.\}\\par\n','',text)
        text=re.sub(r'\n\{\\footnotesize\\mref\{[^}]+\}\{[^}]+\}\.\}\\par\n','\n',text)
        write(filename,route(text,filename))

# Full section map: the main manuscript has a concise result-to-proof map.
index=[r'\section{Complete supporting map}',anchor('sec:support_index'),r'Each entry links to its supporting section and back to the corresponding main result. Conditional extensions and historical provenance do not enlarge the evidentiary scope of the current experiments. Main Table~\ref{tab:main_support} provides the shorter reading route.',r'\begingroup\small\singlespacing',r'\begin{longtable}{p{.60\textwidth}p{.33\textwidth}}\toprule Supporting section & Main-text use\\\midrule\endhead']
for row in sections:index.append(r'\sref{'+row['label']+'}{'+row['section']+r'} & \mref{'+row['main_label']+'}{'+row['main_support']+r'}\\')
index.extend([r'\bottomrule\end{longtable}\endgroup'])
index.extend([r'\subsection*{Formal-result and display index}',r'Each object below has its own return link. Equations supporting that object remain in its linked section.',r'\begingroup\small\singlespacing\begin{longtable}{p{.38\textwidth}p{.54\textwidth}}\toprule Object & Main-text use\\\midrule\endhead'])
for row in objects:
    kind={'longtable':'Table','table':'Table','figure':'Figure','theorem':'Theorem','proposition':'Proposition','lemma':'Lemma'}[row['kind']]
    index.append(r'\sref{'+row['label']+'}{'+kind+r'~\ref{'+row['label']+r'}} & \mref{'+row['main_label']+'}{'+row['main_support']+r'}\\')
index.append(r'\bottomrule\end{longtable}\endgroup')
write('R142_SUPPORT_INDEX','\n'.join(index))
(ROOT/'audit/R142_SECTION_SUPPORT_MAP.json').write_text(json.dumps(sections,indent=2))
(ROOT/'audit/R142_OBJECT_SUPPORT_MAP.json').write_text(json.dumps(objects,indent=2))

write('R142_MAIN_SUPPORT',r'''\begin{table}[H]\centering\small
\caption{\textbf{Reading map from each result to its evidence and implementation.} Links lead to the complete proofs, denominators and supporting records.}\label{tab:main_support}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{.25\textwidth}>{\raggedright\arraybackslash}X}\toprule
Main result & Supporting evidence and executable procedure\\\midrule
1. Old loss and new benefit & Fig.~\ref{fig:joint}; Extended Data Fig.~\ref{fig:ed_turnover}; \sref{sec:joint_records}{paired reevaluation and batch-composition control}; \sref{sec:control_experiment}{current selection record}.\\
2. Direction and uncertainty & Fig.~\ref{fig:calibration}; Extended Data Figs.~\ref{fig:ed_boundaries}, \ref{fig:ed_declarations}, \ref{fig:ed_sensitivity}; Algorithm~\ref{alg:select}; \sref{sec:current_direction}{direction proof} and \sref{sec:current_conformal}{simultaneous calibration}.\\
3. Repair and incompatibility & Fig.~\ref{fig:repair_geometry}; Extended Data Figs.~\ref{fig:ed_repair}, \ref{fig:ed_prospective}; Algorithm~\ref{alg:repair}; \sref{sec:current_robust}{robust alternative and dual witnesses}.\\
4. Evidence and deployment & Fig.~\ref{fig:information}; Extended Data Fig.~\ref{fig:ed_deployment}; Algorithm~\ref{alg:tape}; \sref{sec:current_information}{information bounds}, \sref{sec:current_reuse}{reuse}, \sref{sec:deployment}{finite records and minimum cut}, \sref{hist:migration}{supporting lifecycle model}.\\
5. Complete update comparison & Fig.~\ref{fig:control}; Extended Data Figs.~\ref{fig:ed_candidates}, \ref{fig:ed_costs}, \ref{fig:ed_compatibility}; Algorithm~\ref{alg:muscle}; \sref{sec:control_experiment}{all selection results}, \sref{sec:compatibility}{compatibility training}, \sref{sec:r138_answers}{stateful endpoint audit}.\\\bottomrule
\end{tabularx}
\end{table}''')
main=(MAN/'R142_MAIN_BODY.tex').read_text().replace(r'\section{Results}',r'\input{R142_MAIN_SUPPORT.tex}'+'\n'+r'\section{Results}');write('R142_MAIN_BODY',main)

# ED: keep all sixteen old panels and add six data-backed displays.
ed=base('EXTENDED_DATA')
ed=ed.replace(r'\setcounter{figure}{0}\renewcommand{\thefigure}{Extended Data \arabic{figure}}',r'\setcounter{figure}{0}\renewcommand{\thefigure}{\arabic{figure}}\renewcommand{\figurename}{Extended Data Fig.}')
edlinks=[('fig:ed_turnover','main:joint','hist:prospective'),('fig:ed_boundaries','main:direction','sec:current_direction'),('fig:ed_repair','main:repair','sec:current_robust'),('fig:ed_prospective','main:repair','hist:prospective')]
for label,t,s in edlinks:ed=ed.replace('\\label{'+label+r'}\end{figure}',r'\label{'+label+r'}\end{figure}'+'\n'+r'\mref{'+t+r'}{Return to the corresponding main result}; \sref{'+s+r'}{supporting analysis and protocol}.'+'\n')
newed=[
('R142_ED5_candidate_grid','fig:ed_candidates','All candidate--strength outcomes in all eight batches.',r'Each panel shows four frozen candidates at four strengths, joined in increasing strength order. Candidate colour, marker and line style agree within a panel. Crosses mark audit benefit failures, including either the 25\% absolute floor or the fifteen-percentage-point gain floor. These audit outcomes were not used to replace the frozen primary choices.','main:control','sec:control_experiment'),
('R142_ED6_declarations','fig:ed_declarations','Declarations and total task loss are different outcomes.',r'\textbf{a}, declared and all protected families. \textbf{b}, false declarations and all protected-family losses at each calibrated choice. \textbf{c}, declaration frequency versus total old loss; each symbol identifies a batch. \textbf{d}, the sharp probability bound $\min(1,1-\pi+\alpha)$, where $\pi$ is a population declaration probability. Panel d is a theorem illustration and does not substitute the observed batch frequencies for an independently established population probability.','main:direction','sec:current_conformal'),
('R142_ED7_sensitivity','fig:ed_sensitivity','Sensitivity to the evidence and benefit requirements.',r'\textbf{a,b}, the largest saved calibration size per batch at 90\% nominal coverage; larger pilots change the joint residual allowance and the number of families declared by at least one pair. These are post hoc availability calculations, not reselected primary outcomes. \textbf{c}, exact finite-order-statistic thresholds as the error level changes. \textbf{d}, mean audit-eligible candidate--strength pairs under alternative benefit floors, out of sixteen. The complete 480 calibration settings and 160 benefit settings accompany the manuscript.','main:control','sec:control_experiment'),
('R142_ED8_costs','fig:ed_costs','Observation counts do not imply equal computation.',r'Totals across eight batches for method-specific old-task selection: \textbf{a}, record observations; \textbf{b}, measured elapsed seconds; \textbf{c}, teacher-forced sequence-token positions; \textbf{d}, generated tokens. Colours and hatches identify the same six selectors. Shared incumbent preparation, candidate training, capability screening and final scientific audit are excluded from these panels and retained separately in the ledger. Zero is zero additional old-task selection work, not free model deployment.','main:control','sec:control_experiment'),
('R142_ED9_compatibility','fig:ed_compatibility','A published compatibility loss with a matched additional-training control.',r'Each panel reports the complete old-task transition counts relative to the jointly reevaluated incumbent for one batch: old-correct/new-wrong losses and old-wrong/new-correct entries. The original full update, MUSCLE-loss adapter and cross-entropy continuation are evaluated at their actual final strength. Hatching distinguishes entries from losses; colours distinguish methods. All original audit rows are included. These previously inspected audit pools provide supplementary evidence, not a new independent confirmation.','main:control','sec:compatibility'),
('R142_ED10_deployment','fig:ed_deployment','Exact finite-record evaluation and shared-work planning.',r'\textbf{a}, complete versus exact-stopping observations for 120 synthetic tapes. \textbf{b}, specified operation work under three dependency structures sharing each tape. All 360 cases reproduce complete decisions and artifacts. \textbf{c}, the minimum-cut objective equals exhaustive enumeration in every one of 600 small problems. \textbf{d}, an explicit equal-count construction with one changed root in each case and a varying downstream price. These are algorithm checks and mathematical illustrations; none is an estimate of real organizational savings.','main:information','sec:deployment')]
for image,label,title,caption,target,support in newed:
    ed+='\n'+r'\clearpage\begin{figure}[H]\centering'+'\n'+r'\includegraphics[width=.98\textwidth]{figures/'+image+'.pdf}\n'+r'\caption{\textbf{'+title+'} '+caption+'}'+r'\label{'+label+r'}\end{figure}'+'\n'+r'\mref{'+target+r'}{Return to the corresponding main result}; \sref{'+support+r'}{supporting proof, protocol and complete results}.'+'\n'
write('R142_EXTENDED_DATA',ed)

# Shared preamble and three linked deliverables.
pre=(MAN/'NATURE_GENAI_UPDATE_R141_MAIN.tex').read_text().split(r'\begin{document}')[0]
pre=pre.replace(r'\usepackage[colorlinks=true',r'\usepackage{xr-hyper}'+'\n'+r'\usepackage[colorlinks=true')
pre+=r'''
\newcommand{\sref}[2]{\hyperref[#1]{#2}}
\newcommand{\mref}[2]{\hyperref[#1]{#2}}
\newcounter{algorithm}
\newenvironment{algorithmblock}[2]{\par\addvspace{.6em}\noindent\begin{minipage}{\textwidth}\small\singlespacing\refstepcounter{algorithm}\label{#1}\hrule\vspace{5pt}\textbf{Algorithm \thealgorithm. #2}\par\vspace{4pt}}{\par\vspace{4pt}\hrule\end{minipage}\par\addvspace{.7em}}
'''
mainstart=r'\begin{document}\linenumbers\onehalfspacing'+'\n'+r'\begin{center}{\LARGE\bfseries '+TITLE+r'\par}\vspace{1.2em}{\large Anonymous authors\par}\end{center}\vspace{.6em}\begin{center}\textbf{Abstract}\end{center}'+'\n'+r'\input{R142_ABSTRACT.tex}\clearpage\input{R142_MAIN_BODY.tex}\input{R142_METHODS.tex}'+'\n'
mainend=r'\clearpage\begingroup\singlespacing\input{R142_REFERENCES.tex}\endgroup\clearpage\input{R142_EXTENDED_DATA.tex}'+'\n'
sistart=r'''\clearpage
\setcounter{equation}{0}\setcounter{table}{0}\setcounter{figure}{0}
\renewcommand{\theequation}{S\arabic{equation}}\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}\renewcommand{\figurename}{Figure}
\begin{center}{\LARGE\bfseries Supplementary Information\par}\vspace{.7em}{\large '''+TITLE+r'''\par}\end{center}
\section*{Scope and reading map}
The current proofs, algorithms and complete experimental records come first. Each section links back to the main result it supports. The later technical record preserves the earlier derivations and their explicit corrections; historical numerical summaries without primary outputs remain provenance, not newly reproduced evidence. The \sref{sec:support_index}{complete supporting map} provides links to every section. Main Table~\ref{tab:main_support} gives the shortest route through the five results.
\input{R142_CURRENT_THEORY.tex}\input{R142_ASSUMPTIONS.tex}\input{R142_DEPLOYMENT_THEORY.tex}
\clearpage\part{Complete experimental results and endpoint audit}
\input{R142_PAIRED_RECORDS.tex}\input{R142_CONTROL_SUPPLEMENT.tex}\input{R142_COMPATIBILITY_SUPPLEMENT.tex}\input{R142_R138_ENDPOINT_AUDIT.tex}
\clearpage\part{Supporting derivations and corrected historical record}
\input{R142_SUPPORTING_RECORD.tex}\clearpage\input{R142_SUPPORT_INDEX.tex}
'''
write('R142_MAIN',pre+r'\externaldocument[][nocite]{R142_SUPPLEMENT}[R142_SUPPLEMENT.pdf]'+'\n'+mainstart+mainend+r'\end{document}')
write('R142_SUPPLEMENT',pre+r'\externaldocument[][nocite]{R142_MAIN}[R142_MAIN.pdf]'+'\n'+r'\begin{document}\linenumbers\onehalfspacing'+sistart+r'\clearpage\begingroup\singlespacing\input{R142_REFERENCES.tex}\endgroup\end{document}')
write('R142_COMBINED',pre+mainstart+mainend+sistart+r'\end{document}')

# A single numerical reference sequence keeps standalone and combined links aligned.
master=(MAN/'R141_BIBLIOGRAPHY_MASTER.tex').read_text()
master+=r'\bibitem{picard1976}Picard, J.-C. Maximal Closure of a Graph and Applications to Combinatorial Problems. \textit{Management Science} \textbf{22}, 1268--1272 (1976). \url{https://doi.org/10.1287/mnsc.22.11.1268}.'+'\n'
master+=r'\bibitem{rhys1970}Rhys, J. M. W. A Selection Problem of Shared Fixed Costs and Network Flows. \textit{Management Science} \textbf{17}, 200--207 (1970). \url{https://doi.org/10.1287/mnsc.17.3.200}.'+'\n'
master+=r'\bibitem{renggli2019}Renggli, C. et al. Continuous Integration of Machine Learning Models with ease.ml/ci: Towards a Rigorous Yet Practical Treatment. \textit{Proceedings of Machine Learning and Systems} \textbf{1} (2019). \url{https://proceedings.mlsys.org/paper_files/paper/2019/hash/4284d31e68c0a4a39dcdad167ac4bd72-Abstract.html}.'+'\n'
master+=r'\bibitem{polyzotis2019}Polyzotis, N., Zinkevich, M., Roy, S., Breck, E. \& Whang, S. Data Validation for Machine Learning. \textit{Proceedings of Machine Learning and Systems} \textbf{1} (2019). \url{https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html}.'+'\n'
items={m[1]:m[2].strip() for m in re.finditer(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)',master,re.S)}
order=[]
def expand(text,seen=()):
    def sub(m):
        p=MAN/m[1]
        if not p.suffix:p=p.with_suffix('.tex')
        if not p.exists() or p.name in seen:return ''
        return expand(p.read_text(),seen+(p.name,))
    return re.sub(r'\\input\{([^}]+)\}',sub,text)
content=expand(mainstart+mainend.replace(r'\input{R142_REFERENCES.tex}','')+sistart)
for m in re.finditer(r'\\cite\w*\{([^}]+)\}',content):
    for key in m[1].split(','):
        if key not in order:order.append(key)
missing=set(order)-set(items);assert not missing,missing
write('R142_REFERENCES',r'\begin{thebibliography}{99}'+'\n'+'\n'.join('\\bibitem{'+k+'}'+items[k] for k in order)+'\n'+r'\end{thebibliography}')
print(json.dumps({'linked_SI_sections':len(sections),'references':len(order),'main_headline_results':5,'extended_data_displays':10},indent=2))
