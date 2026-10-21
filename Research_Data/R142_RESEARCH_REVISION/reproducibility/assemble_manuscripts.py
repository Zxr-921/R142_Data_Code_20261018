"""Assemble editable R141 documents; retain original releases separately."""
from pathlib import Path
import re,json,shutil
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
MAN=ROOT/'manuscript'
original=(ROOT/'reproducibility/source_inputs/R140_ORIGINAL_SI.tex').read_text()
legacy=original[original.index(r'\section{Historical source map'):original.index(r'\section{R138 control rule')]
changes=[]
def replace(old,new):
    global legacy
    n=legacy.count(old)
    if not n:raise ValueError('Missing source: '+old[:100])
    legacy=legacy.replace(old,new);changes.append({'source':old[:110],'occurrences':n})

replace('The general model does not admit a universal radical expression because the first-order condition can contain polynomial subclasses of degree five or higher. The exact convex or root characterization is therefore the correct closure.',
        'For the stated convex class, the optimization problem and, when applicable, its monotone first-order root are the solution characterization. No algebraic impossibility theorem is needed or claimed here.')
replace('The historical theory gives information lower bounds for the pilot and final stages separately; regular likelihood-ratio or confidence-sequence procedures achieve the same first-order order under separated alternatives.',
        r'The complete observation transcript must be used for information accounting: reusable pilot observations are counted once. Stage-specific lower bounds cannot generally be added. Section~\ref{sec:current_information} proves a Gaussian upper bound and states the conditions for restricted additive accounting. General first-order attainability is not asserted.')
start=legacy.index('If $\\Gamma(x)=0$, finite expected validation cost')
end=legacy.index(r'\begin{equation}',legacy.index('Thus, within the regular subclass',start))
legacy=legacy[:start]+r'''If $\Gamma(x)=0$, the information inequality rules out finite expected validation cost for a uniformly correct rule in the stated alternative class. The previously asserted generic iterated-logarithm upper remainder is withdrawn. For the independent known-variance Gaussian class only, the explicit rule and finite-error upper bound in Section~\ref{sec:current_information} give
\begin{equation}
 \mathbb E_gT\le n_\eta+
 \frac{\exp[-n_\eta\eta^2g^2/(2\sigma^2)]}{1-\exp[-\eta^2g^2/(2\sigma^2)]}.
 \label{eq:s_gamma_upper}
\end{equation}
All quantities and the proof are given in that section. Outside a proved attainable class, the following expression is an information-based leading design surrogate, not an achieved general optimum:
'''+legacy[end:]
replace('A universal radicals formula is impossible for the unrestricted convex class because it can contain first-order conditions with a generic nonsolvable quintic.',
        'This characterizes the optimization problem without asserting an unproved general algebraic impossibility result.')
start=legacy.index('Pilot experiments may be needed before choosing the production intervention.')
end=legacy.index(r'\section{',start)
legacy=legacy[:start]+r'''Pilot experiments may be needed before choosing the production intervention. The general result is the stopped-transcript information inequality in Section~\ref{sec:current_information}. A sum of separate information costs is justified only for a restricted architecture: the intervention $x$ is fixed, pilot identification is uniformly $\alpha_I$-correct, final testing uses fresh independent observations without reusing pilot evidence, and the final test is uniformly $\alpha_V$-correct conditional on every admissible earlier history. With the stage information rates defined for those same experiments,
\begin{equation}
 \mathbb E C_{\rm total}\ge
 \frac{\operatorname{kl}(1-\alpha_I,\alpha_I)}{\Gamma_{\rm rec}(\theta)}
 +C_T(x)+\frac{\operatorname{kl}(1-\alpha_V,\alpha_V)}{\Gamma(x)},
 \qquad \alpha_I+\alpha_V\le\alpha.
 \label{eq:s_end_to_end_info}
\end{equation}
Apply the change-of-measure inequality to the pilot experiment and, conditional on the history, to the fresh final experiment; then take expectations and add the disjoint costs. This proves the bound for that restricted architecture. If $x$ is random, the latter two terms belong inside an expectation, and a history-conditional information rate is required. If observations are reused, these stage bounds do not supply a valid unrestricted sum. No generic matching upper bound is asserted.

Under an explicitly attainable leading logarithmic approximation with fixed positive coefficients $A=1/\Gamma_{\rm rec}$ and $B=1/\Gamma(x)$, minimizing $A\log(1/\alpha_I)+B\log(1/\alpha_V)$ subject to the error budget gives
\begin{equation}
 \alpha_I^*=\alpha\frac{A}{A+B},\qquad
 \alpha_V^*=\alpha\frac{B}{A+B}.
 \label{eq:s_error_split}
\end{equation}
This follows by differentiating the strictly convex scalar objective. The more expensive stage receives a larger allowed error. It is an optimum of the approximation, not a proved finite-sample optimum for arbitrary adaptive pipelines.

'''+legacy[end:]
replace('Finite separated menu with regular identifiable channels admits first-order tracking',
        'General lower bound; Gaussian upper rule proved in Part I')
replace('Convex confidence sets preserve convexity; boxes give an LP',
        'Any fixed uncertainty set gives a convex intersection; boxes give an LP')
replace('Regular finite-dimensional families admit first-order GLR tracking',
        'Known-variance independent Gaussian subclass has proved upper and lower bounds')
replace('General convex costs can yield nonsolvable polynomial first-order equations',
        'The information-based objective need not have an elementary minimizer')
replace('arbitrary smooth roots have no universal radicals form',
        'arbitrary smooth paths need not have a unique or elementary root')
replace('The unique optimum of Equation~\\ref{eq:s_train_certify_cost} is',
        'For $g,\\beta,c_T,c_V,\\sigma>0$ and $0<\\alpha<1$, the unique optimum of the leading continuous approximation in Equation~\\ref{eq:s_train_certify_cost} is')
replace('The result is not used as a main-paper recommendation.',
        'This is an exact optimizer of a stated leading approximation, not of every finite-sample sequential test. An allowed upper intervention limit clips the optimizer to that interval.')
replace('where $D$ is positive diagonal, $H\\ge0$, $D^{-1}H$ is irreducible, $d\\ge0$ is nonzero, and $c>0$ componentwise.',
        'where $D$ is positive diagonal, $H\\ge0$, $D^{-1}H$ is irreducible with strictly positive spectral radius, $d\\ge0$ is nonzero, and $c>0$ componentwise. The positive-radius assumption excludes the zero-interference scalar case, which has no finite singular threshold.')

# The corrected functional historical endpoint is retained with its true name.
replace(r'\subsection{R138: corrected executable-agent stress test and evidence boundary}',
        r'''\subsection{R138: historical functional endpoint and its current correction}
\textbf{Current reading rule.} The functional counts below deliberately retain the earlier endpoint for comparison. They exclude required final answers. Section~\ref{sec:r138_answer_audit} supplies the current answer audit: both full-strength candidates lose all nine protected complete tasks; new-task required-milestone counts are zero and one. Functional retention must not be described as complete task retention.
''')
replace('Pure agent-to-user wording is retained in ToolSandbox\'s continuous similarity score but is not part of the binary functional endpoint.',
        'Agent-to-user communication is excluded from this historical functional endpoint. Such communication may contain a required factual answer or confirmation, not merely optional wording. The current answer audit therefore checks it separately.')
replace('Thus the endpoint correction repairs the task semantics but does not convert R138 into a broad positive replication.',
        'Thus that correction repairs query/tool scoring only; it does not establish complete task semantics. The required-answer audit changes the broader conclusion.')
replace('corrected exact old-task damage is 0/9 for both U2 and U4.',
        'functional-milestone loss is 0/9 for both U2 and U4, whereas loss under the required-answer audit is 9/9 for both.')
replace('Exact old damage on protected variants & -- & 0/9 & 0/9',
        'Functional old loss (answer excluded) & -- & 0/9 & 0/9')
replace('R138 corrected executable-agent audit. The endpoint correction repairs query/read scoring but leaves the main scientific conclusion unchanged: the capability gains are sparse, and the historical 90\\% certificate must abstain because only eight protected transfer-fit requirements are available.',
        'Historical R138 functional-milestone counts, excluding required answers. Complete required-task counts are corrected in Table~\\ref{tab:r138_answer_audit}; the calibration abstention still follows from only eight protected fit variants.')
replace('The sparse gains show that the updates can create some held-out executable successes absent in the incumbent, but they do not establish broad cross-family capability improvement.',
        'These sparse functional gains do not establish complete task gains: three of the four candidate successes omit a required final answer. The one remaining case lacks a benchmark final-answer milestone. No broad cross-family improvement follows.')
replace('The next claim-bearing experiment is therefore prospective control rather than another descriptive replication.',
        'The R141 experiment completes a new prospective selection and audit procedure; its recorded results are reported in the main text and current experimental supplement. The following historical design describes the intended scope, including comparisons not yet supported.')
replace('This future algorithm comparison is kept separate from R138',
        'The R141 operational-selector comparison is kept separate from R138; faithful comparisons with the named learning algorithms remain necessary for an algorithmic superiority claim. It is kept separate')

# Plain terms in editable prose; the frozen originals remain byte-identical.
legacy=legacy.replace(r'\textsc{Qualify}', 'assess an update').replace(r'\textsc{Predict}', 'predict changes').replace(r'\textsc{Control}', 'choose or repair an update').replace(r'\textsc{Lifecycle}', 'evaluate repeated releases')
legacy=legacy.replace('Exact solution and tractability map for the retained theory.', 'Solutions and assumptions for the supporting theory.')
legacy=legacy.replace(r'\section{A solution-closure theorem for the retained model}', r'\section{Summary of tractable mathematical cases}')
legacy=legacy.replace(r'\begin{theorem}[Retained solution closure]',r'\begin{proposition}[Summary under the assumptions of each cited case]')
ix=legacy.index(r'\begin{proposition}[Summary under')
j=legacy.index(r'\end{theorem}',ix)
legacy=legacy[:j]+legacy[j:].replace(r'\end{theorem}',r'\end{proposition}',1)
legacy=legacy.replace('This theorem deliberately closes', 'This proposition summarizes')

# Literature is attached to mathematical families, not used to assert that their
# assumptions hold empirically. Every retained formal result has an explanation.
interpretations={
'Marginal transfer guarantee at a fixed radius':('A fixed-radius guarantee answers whether a measured error bound transfers to one new unit. It does not cover a radius chosen after inspecting those same calibration errors; Part I provides the simultaneous alternative.','vovk2005,angelopoulos2023'),
'Perfect exit recovery can force collateral failure':('Recovering every previously failed task may still sacrifice a survivor. The construction below is an existence example for signed effects, not a universal empirical frequency.','lopezpaz2017,boyd2004'),
'Exact decision preservation':('Each inequality compares the incumbent action with one rival. Keeping every comparison nonnegative preserves the action under the specified tie rule.','boyd2004'),
'Directional first crossing':('Current reserve divided by the adverse directional rate gives the affine first crossing. The direction must be specified; old reserve alone does not determine the result.','nocedal2006'),
'Second-order crossing':('Curvature can accelerate or reverse an initially predicted movement. The root is exact for the quadratic margin model; a Taylor approximation needs a controlled remainder.','nocedal2006'),
'Change-of-measure lower bound':('To distinguish two laws that demand different decisions, a procedure must collect enough information. Channel costs and shared observations determine how expensive that distinction is.','garivier2016,kaufmann2021'),
'Current-state Gaussian crossing--information identity':('The information distance is evaluated at the current model state. As its remaining margin shrinks, a nearby opposite-sign Gaussian law becomes harder to distinguish.','wald1945,garivier2016'),
'Farkas recovery alternative':('Either allowed interventions cover all deficits, or a nonnegative weighting exposes an incompatible combination. This is the standard linear alternative applied to the stated intervention library.','boyd2004'),
'Value-preserving repair alternative':('Survivors and newly acquired requirements impose reserves that repair cannot freely consume. The same signed system must include them as well as the losses to recover.','lopezpaz2017,boyd2004'),
'Exact obstruction test for a new direction':('A new direction helps only if it removes every remaining obstruction. Positive projection on one witness does not prove feasibility.','boyd2004'),
'Exact single-direction feasibility':('One intervention must be large enough to cover positive deficits and small enough not to exhaust negatively affected reserves. Feasibility requires the lower and upper limits to overlap.','boyd2004'),
'Conditioned M-matrix repair threshold':('The spectral result applies to the stated nonnegative interference matrix and nonnegative deficits. It describes how mutual adverse effects can overwhelm direct gains; arbitrary signed systems need the general linear program.','berman1994'),
'No-support identifiability limit':('If the data never reveal a relevant counterfactual and no structural relation supplies it, observationally identical models can demand different decisions. Additional support or intervention changes this information problem.','sutton2018'),
'Exact boundary for when changed-decision count is sufficient':('A count predicts downstream work only under the specified equal-work structure. Shared deterministic operations must be charged once; correlation alone does not create free computation.','sculley2015,amershi2019'),
'Minimal prospective decision state':('If every independent margin direction is allowed, probing one coordinate at a time reveals each margin. The injectivity result does not rule out useful compression for restricted directions.','nocedal2006'),
'Cube-root train--then--certify rule':('The cube root balances intervention expense against the leading reduction in future evidence cost. It solves the displayed approximation under its positive-parameter assumptions.','wald1945,boyd2004'),
'Final-horizon affine recovery':('Known affine release movements can be added over a fixed horizon. Requiring preservation at every intermediate release adds constraints beyond restoring the final endpoint.','boyd2004,sutton2018'),
'No-return certificate':('A no-return conclusion quantifies only over the explicitly allowed future interventions and horizon. It cannot exclude a future method outside that set.','boyd2004,sutton2018'),
'Necessary and sufficient robust preservation certificate':('To preserve an action without learning the exact target state, its worst allowed comparison must remain favourable. The uncertainty set must contain the true state for the conclusion to apply.','bental2009'),
'Summary under the assumptions of each cited case':('The list below gathers solutions already derived under separate assumptions. It does not combine all assumptions into a universal theorem about neural-network deployment.','boyd2004,bental2009')}
for title,(explanation,refs) in interpretations.items():
    pat=re.compile(r'(\\begin\{(?:theorem|proposition)\}\['+re.escape(title)+r'\])')
    legacy,n=pat.subn(lambda m:r'\paragraph{Interpretation and scope.} '+explanation+r'\cite{'+refs+'}\n\n'+m.group(1),legacy)
    changes.append({'result_explanation':title,'occurrences':n})

# Number all display math in the editable record.
legacy=legacy.replace(r'\[',r'\begin{equation}').replace(r'\]',r'\end{equation}')
legacy=legacy.replace(r'\begin{align*}',r'\begin{align}').replace(r'\end{align*}',r'\end{align}')
legacy=legacy.replace(r'\begin{equation*}',r'\begin{equation}').replace(r'\end{equation*}',r'\end{equation}')
legacy=re.sub(r'\\texttt\{([0-9a-f]{24,})\}',r'\\nolinkurl{\1}',legacy)
legacy=legacy.replace(r'\texttt{exact\_state\_success}',r'\nolinkurl{exact_state_success}')
# Historical hard page breaks add blank space without preserving information.
# Immutable original PDFs retain their original pagination separately.
legacy=legacy.replace(r'\clearpage','').replace(r'\newpage','')
legacy=legacy.replace(r'\part{Editable mechanism reconstruction from R120}',r'\section{Supporting mechanism reconstruction from R120}')
(MAN/'R141_SUPPORTING_RECORD.tex').write_text(legacy)
(ROOT/'audit'/'editorial_transformation_log.json').write_text(json.dumps(changes,indent=2))

preamble=r'''\documentclass[11pt,letterpaper]{article}
\usepackage[letterpaper,margin=0.75in]{geometry}
\usepackage{mathptmx}
\usepackage{graphicx}
\let\Bbbk\relax
\usepackage{amsmath,amssymb,mathtools,amsthm}
\usepackage{booktabs,tabularx,array,longtable,multirow}
\usepackage{microtype,setspace,xurl,float,caption,xcolor,pdfpages,enumitem}
\usepackage[super,sort&compress]{natbib}
\usepackage[mathlines]{lineno}
\usepackage[colorlinks=true,hypertexnames=false,allcolors=blue]{hyperref}
\usepackage{bookmark}
\DeclareCaptionFont{singlespace}{\setstretch{1.0}}
\captionsetup{font={small,singlespace},labelfont=bf,justification=raggedright,singlelinecheck=false,skip=5pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.35em}\raggedbottom
\setlength{\emergencystretch}{1.5em}
\setcounter{secnumdepth}{0}
\modulolinenumbers[1]
\renewcommand\linenumberfont{\normalfont\scriptsize\sffamily}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newcommand{\R}{\mathbb R}
\newcommand{\1}{\mathbf 1}
'''
title='Evaluating language-model updates under competing task requirements'
main=preamble+r'''\begin{document}\linenumbers\onehalfspacing
\begin{center}{\LARGE\bfseries '''+title+r'''\par}
\vspace{1.2em}{\large Anonymous authors\par}\end{center}
\vspace{0.6em}\begin{center}\textbf{Abstract}\end{center}
\input{R141_ABSTRACT.tex}
\clearpage
\input{R141_MAIN_BODY.tex}
\input{R141_METHODS.tex}
\clearpage
\begingroup\singlespacing
\input{R141_MAIN_REFERENCES.tex}
\endgroup
\clearpage
\input{R141_EXTENDED_DATA.tex}
\end{document}
'''
(MAN/'NATURE_GENAI_UPDATE_R141_MAIN.tex').write_text(main)
si=preamble+r'''\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}\linenumbers\onehalfspacing
\begin{center}{\LARGE\bfseries Supplementary Information\par}
\vspace{.7em}{\large '''+title+r'''\par}
\vspace{.7em}Anonymous authors\end{center}
\section*{Scope and reading map}
This is the integrated R141 Supplement. Part I contains the current formal results, complete proofs, interpretation and assumption sources. It distinguishes fixed task records from population reliability, provides a simultaneous candidate--strength calibration rule, separates guaranteed repair from optimistic infeasibility, proves Gaussian evidence bounds, and corrects information reuse. Part II presents the new experiment and the recovered required-answer audit. Part III retains supporting derivations and historical evidence with the corrections applied in place. Original releases, source figures and the complete earlier PDF snapshots are preserved unchanged in the separate preservation archive.

Current R141 statements govern this manuscript. In particular, no generic iterated-logarithm validation upper bound, unrestricted sum of stagewise information costs, or complete ToolSandbox success inferred from functional state alone is claimed. Historical numerical summaries with unavailable primary assets are explicitly identified as reported supporting records, not newly reproduced experiments. All existing original figure panels remain available; current displays are rebuilt from the supplied source tables where those tables are available.

\begingroup\singlespacing\small\setlength{\parskip}{0pt}
\setcounter{tocdepth}{1}\tableofcontents\endgroup\clearpage
\input{R141_CURRENT_THEORY.tex}
\input{R141_ASSUMPTIONS.tex}
\clearpage\part{Current experimental results and endpoint audit}
\input{R141_CONTROL_SUPPLEMENT.tex}
\input{R141_R138_ENDPOINT_AUDIT.tex}
\clearpage\part{Supporting derivations and corrected historical record}
\input{R141_SUPPORTING_RECORD.tex}
\clearpage\input{R141_SI_REFERENCES.tex}
\end{document}
'''
(MAN/'NATURE_GENAI_UPDATE_R141_SI.tex').write_text(si)

# Each document numbers citations in its own order of first appearance.
reference_text=(MAN/'R141_REFERENCES.tex').read_text().replace(r'\input{R141_NEW_REFERENCES.tex}',(MAN/'R141_NEW_REFERENCES.tex').read_text())
entries={}
for chunk in reference_text.split(r'\bibitem{')[1:]:
    key,body=chunk.split('}',1)
    entries[key]=r'\bibitem{'+key+'}'+body.split(r'\end{thebibliography}')[0].strip()+'\n'
def expanded(name):
    text=(MAN/name).read_text()
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expanded(m[1]),text)
for label,names in [('MAIN',['R141_MAIN_BODY.tex','R141_METHODS.tex','R141_EXTENDED_DATA.tex']),('SI',['R141_CURRENT_THEORY.tex','R141_ASSUMPTIONS.tex','R141_CONTROL_SUPPLEMENT.tex','R141_R138_ENDPOINT_AUDIT.tex','R141_SUPPORTING_RECORD.tex'])]:
    ordered=[]
    for name in names:
        for match in re.finditer(r'\\cite(?:p|t)?\{([^}]+)\}',expanded(name)):
            for key in match[1].split(','):
                key=key.strip()
                if key not in ordered:ordered.append(key)
    assert all(key in entries for key in ordered)
    (MAN/f'R141_{label}_REFERENCES.tex').write_text('\\begin{thebibliography}{99}\n'+''.join(entries[k] for k in ordered)+'\\end{thebibliography}\n')
(MAN/'R141_BIBLIOGRAPHY_MASTER.tex').write_text(''.join(entries.values()))
print(json.dumps({'legacy_characters':len(legacy),'transformations':len(changes),'main_wrapper_written':True,'si_wrapper_written':True}))
