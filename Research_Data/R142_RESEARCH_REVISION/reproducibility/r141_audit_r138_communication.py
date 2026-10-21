"""Retrospective endpoint audit of recovered ToolSandbox trajectories.

Separates functional milestones from required communication. For the nine
incumbent functional successes, final confirmations were inspected against
the fixed requests; no automatic ROUGE=1 requirement is imposed on paraphrases.
Missing or malformed user-facing answers are observable necessary-condition
failures. This is a retrospective audit, not a newly registered experiment.
"""
import json,re,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results'/'historical_raw'
OUT=ROOT/'results'/'R138_communication_audit';OUT.mkdir(exist_ok=True)

def walk(x):
    if isinstance(x,dict):
        yield x
        for v in x.values():yield from walk(v)
    elif isinstance(x,list):
        for v in x:yield from walk(v)

rows=[]
for path in sorted(RAW.glob('*/scientific_record.json')):
    data=json.loads(path.read_text());ledger=data.get('R138_SCENARIO_LEDGER.jsonl')
    if not ledger:continue
    conversations={k.rsplit('/',2)[1]:v for k,v in data.items() if k.endswith('/conversation.json')}
    for line in ledger.splitlines():
        if not line.strip():continue
        r=json.loads(line);ident=f"{r['phase']}__{r['update']}__s{r['strength']}__{r['scenario']}"
        conv=conversations.get(ident)
        template=conv
        if template is None:
            donors=[k for k in conversations if ident.startswith(k+'_')]
            assert donors or r.get('error'), ('No task-milestone definition',ident)
            template=conversations[max(donors,key=len)] if donors else []
        milestones={}
        for obj in walk(template):
            if 'milestone_index' in obj and 'milestone' in obj:
                milestones[int(obj['milestone_index'])]=obj['milestone']
        required_answer=[]
        for ix,m in milestones.items():
            for con in m.get('snapshot_constraints',[]):
                for target in con.get('target_dataframe') or []:
                    if target.get('sender')=='AGENT' and target.get('recipient')=='USER' and target.get('content'):
                        required_answer.append(ix)
        scores={int(k):float(v[1]) for k,v in r.get('milestone_mapping',{}).items()}
        functional_ix=[ix for ix in milestones if ix not in required_answer]
        functional=not r.get('error') and bool(functional_ix) and all(scores.get(ix,0)>=1-1e-8 for ix in functional_ix) and r.get('minefield_similarity',0)<=1e-8
        final=(next((x.get('content') or '' for x in reversed(conv) if x.get('role')=='assistant'), '')
               if conv is not None else (r.get('agent_steps') or [{}])[-1].get('generated',''))
        has_answer=bool(final.strip()) and '<tool_call>' not in final and not re.match(r'^\s*\[?\s*\{',final)
        passed=False;reason='functional_milestone_failure';assessment='not_needed'
        if functional:
            if not required_answer:
                passed=True;reason='all_required_noncommunication_milestones_passed';assessment='benchmark_has_no_final_answer_milestone'
            elif not has_answer:
                reason='missing_or_malformed_required_final_answer';assessment='mechanically_observable_failure'
            elif r['update']=='M0' and r['phase']=='fresh_m0' and (r['scenario'].startswith('remove_contact_with_id') or r['scenario']=='send_message_with_phone_number_and_content'):
                # These nine actual final responses were read in full. They
                # correctly acknowledge the requested action and identifier.
                passed=True;reason='functional_milestones_and_correct_final_confirmation';assessment='retrospective_assistant_semantic_review'
            else:
                passed=None;reason='requires_semantic_review';assessment='unresolved'
        rows.append({'source_job':path.parent.name.replace('R138_',''),'phase':r['phase'],'model':r['update'],'strength':r['strength'],'scenario':r['scenario'],'original_exact_state_success':r['exact_state_success'],'functional_success':functional,'required_answer':bool(required_answer),'answer_present':has_answer,'complete_required_task':passed,'reason':reason,'assessment':assessment,'native_similarity':r['similarity'],'milestone_scores':json.dumps(scores),'final_response':final})

with (OUT/'trajectory_endpoint_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for phase,model in sorted({(r['phase'],r['model']) for r in rows}):
    sub=[r for r in rows if (r['phase'],r['model'])==(phase,model)]
    summary.append({'phase':phase,'model':model,'n':len(sub),'original_success_n':sum(r['original_exact_state_success'] for r in sub),'functional_n':sum(r['functional_success'] for r in sub),'complete_required_task_n':sum(r['complete_required_task'] is True for r in sub),'unresolved_n':sum(r['complete_required_task'] is None for r in sub),'missing_required_answer_n':sum(r['functional_success'] and r['required_answer'] and not r['answer_present'] for r in sub)})
result={'status':'retrospective reanalysis of verified historical archives','endpoint':'All required functional milestones plus a valid final response when the benchmark requires one','semantic_review':'Nine incumbent confirmations reviewed by the revising assistant; not an independent human annotation study','scope':'A benchmark without a final-answer milestone is not called complete conversational-agent success','summary':summary,'csv_sha256':hashlib.sha256((OUT/'trajectory_endpoint_audit.csv').read_bytes()).hexdigest()}
(OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
