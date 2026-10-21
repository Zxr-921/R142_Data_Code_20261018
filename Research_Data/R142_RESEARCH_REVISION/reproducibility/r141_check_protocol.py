"""Check scientifically consequential edge cases in the executed selector.

The selector block and scoring functions are compiled from the frozen runner,
not duplicated as a second implementation. No GPU or new training is needed.
"""
from pathlib import Path
import ast,math,json,hashlib,textwrap,importlib.util
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'reproducibility/r141_control_experiment.py'
spec=importlib.util.spec_from_file_location('frozen_runner',source)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
text=source.read_text();tree=ast.parse(text)
funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in {'group_values','safe_fraction'}]
definitions=compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),'frozen_scoring','exec')
block=textwrap.dedent(text[text.index('    def tie(pair):'):text.index('    # Equal cap on candidate-record observations.')])
selector=compile(block,'frozen_selector','exec')
checks=[]
def record(name,ok):
 assert ok,name
 checks.append({'check':name,'passed':True})
for n,a,finite in [(0,.1,False),(8,.1,False),(9,.1,True),(18,.05,False),(19,.05,True),(98,.01,False),(99,.01,True)]:
 q,k=runner.quantile_bound(range(n),a)
 record(f'quantile_n{n}_alpha{a}',math.isfinite(q)==finite)
def choose(q,margin=1.,empty=False,admissible=True):
 sel=[] if empty else [{'family':'f'}]
 env={'math':math,'np':np,'defaultdict':defaultdict,'MIN_MARGIN':runner.MIN_MARGIN,
      'q':q,'sel':sel,'gsel':np.array([] if empty else [margin]),
      'dsel':{'V1':np.array([] if empty else [0.])},'shortlist':['V1'],
      'admissible':[('V1',.25),('V1',1.)] if admissible else [],
      'candidates':{'V1':{'norm':1.,'full_cap_validation':.5}},
      'cap_grid':{'V1@0.25':.5,'V1@1.00':.5}}
 exec(definitions,env);exec(selector,env)
 return env['choices']
record('infinite_bound_abstains',choose(math.inf)['calibrated_prediction'] is None)
record('finite_bound_without_positive_declaration_abstains',choose(2.)['calibrated_prediction'] is None)
record('exact_tie_buffer_abstains',choose(0.,margin=runner.MIN_MARGIN)['calibrated_prediction'] is None)
record('empty_selection_abstains',choose(0.,empty=True)['calibrated_prediction'] is None)
record('empty_admissible_menu_abstains',all(v is None for v in choose(0.,admissible=False).values()))
record('informative_bound_permits_choice',choose(.5)['calibrated_prediction']==('V1',.25))
record('full_update_and_scaling_are_distinct',choose(.5)['full_update']!=choose(.5)['simple_scaling'])
a=[{'f':{'x':[1]}},{'g':{'y':[2]}},{'f':{'x':[3]}}]
record('family_key_ignores_order',runner.canonical_family(a)==runner.canonical_family(a[::-1]))
record('family_key_retains_call_multiplicity',runner.canonical_family(a)!=runner.canonical_family(a[:2]))
record('quoted_bracket_parser',runner.extract_calls('text [{"f":{"x":"]"}}] tail')==[{'f':{'x':']'}}])
record('malformed_output_rejected',runner.extract_calls('not valid [ {') is None)
result={'runner_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checks':checks,
        'scope':'Exact frozen selector edge cases; complete-batch numerical consistency is checked by r141_analyze_control.py.'}
(ROOT/'audit/protocol_edge_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':len(checks),'failed':0}))
