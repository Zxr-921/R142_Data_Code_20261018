"""Exhaustive algorithm checks and explicitly synthetic workflow experiments."""
from pathlib import Path
import json,csv,itertools,time,hashlib
import numpy as np
from r142_deployment_algorithms import finite_tape_decision,minimum_work_plan,plan_cost,downstream_closure,certificate_reusable
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/deployment';OUT.mkdir(parents=True,exist_ok=True)
SEED=1420921
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    rng=np.random.default_rng(SEED);checks=[];tapes=[];graphs=[]
    for n in range(1,13):
        for bits in itertools.product([0,1],repeat=n):
            for threshold in {1,(n+1)//2,n}:
                result=finite_tape_decision(bits,threshold)
                assert result['decision']==int(sum(bits)>=threshold)
                seen=result['transcript'][:-1];lo=sum(x[1] for x in seen);hi=lo+n-len(seen)
                # Until the last read, both decisions are possible.
                assert lo<threshold<=hi
                checks.append({'n':n,'threshold':threshold,'bits':''.join(map(str,bits)),'reads':result['reads'],'decision':result['decision']})
    write('exhaustive_tape_checks.csv',checks)
    optimization=[]
    for rep in range(600):
        n=int(rng.integers(2,11));m=int(rng.integers(1,13));k=rng.integers(0,21,n).tolist();r=rng.integers(0,11,n).tolist();w=rng.integers(0,31,m).tolist()
        parents=[rng.choice(n,size=int(rng.integers(0,n+1)),replace=False).tolist() for _ in range(m)]
        forced=set(rng.choice(n,size=int(rng.integers(0,n+1)),replace=False).tolist())
        start=time.perf_counter();plan=minimum_work_plan(k,r,parents,w,forced);elapsed=time.perf_counter()-start
        exact=min(plan_cost(u,k,r,parents,w) for mask in range(1<<n) if forced.issubset(u:={i for i in range(n) if mask>>i&1}))
        assert abs(plan['cost']-exact)<1e-8
        optimization.append({'replicate':rep,'roots':n,'operations':m,'forced':len(forced),'min_cut':plan['cost'],'exhaustive':exact,'seconds':elapsed})
        graphs.append({'replicate':rep,'certificate_costs':k,'reopen_costs':r,'operation_parents':parents,'operation_costs':w,'forced':sorted(forced),'plan':plan})
    write('exhaustive_cut_checks.csv',optimization);(OUT/'cut_instances.json').write_text(json.dumps(graphs))
    workflows=[];workflow_records=[]
    for rep in range(120):
        n=80;length=31;prob=rng.beta(5,2,n);tape=(rng.random((n,length))<prob[:,None]).astype(int)
        old=np.ones(n,dtype=int);full=(tape.sum(axis=1)>=16).astype(int)
        decisions=[finite_tape_decision(x,16) for x in tape];selective=np.array([x['decision'] for x in decisions]);assert np.array_equal(full,selective)
        changed=set(np.flatnonzero(full!=old).tolist())
        for topology in ['separate','shared','unequal']:
            parents=[[i] for i in range(n)] if topology=='separate' else [list(range(j,min(j+8,n))) for j in range(0,n,8)]
            costs=[3.]*len(parents) if topology!='unequal' else [float(2**(j%8)) for j in range(len(parents))]
            full_artifacts=[sum(int(full[i]) for i in p) for p in parents]
            cached=[sum(int(old[i]) for i in p) for p in parents]
            active=[v for v,p in enumerate(parents) if changed.intersection(p)]
            for v in active:cached[v]=sum(int(selective[i]) for i in parents[v])
            assert cached==full_artifacts
            workflows.append({'replicate':rep,'topology':topology,'root_decisions':n,'changed_roots':len(changed),'full_reads':n*length,'selective_reads':sum(x['reads'] for x in decisions),'all_decisions_equal':True,'all_artifacts_equal':True,'full_operations':len(parents),'selective_operations':len(active),'full_operation_work':sum(costs),'selective_operation_work':sum(costs[v] for v in active)})
        workflow_records.append({'replicate':rep,'tapes':tape.tolist(),'full_decisions':full.tolist(),'transcripts':decisions})
    write('workflow_replays.csv',workflows);(OUT/'workflow_tapes.json').write_text(json.dumps(workflow_records))
    # Equal changed counts, arbitrary cost separation; no fitted LLM quantities.
    counter=[]
    for heavy in [1,2,5,10,20,50,100,200,500,1000]:
        counter.append({'heavy_operation_cost':heavy,'changed_count_A':1,'changed_count_B':1,'work_A':heavy,'work_B':1,'ratio':heavy})
    write('equal_count_different_work.csv',counter)
    # Evidence keys and transitive invalidation: every field change rejects reuse.
    fp={x:'v1' for x in ['model','input','evaluator','environment','policy']};key_checks=[]
    assert certificate_reusable(fp,fp)
    for key in fp:
        new={**fp,key:'v2'};assert not certificate_reusable(fp,new);key_checks.append({'changed_field':key,'reuse_allowed':False})
    assert downstream_closure(['a'],[('a','b'),('b','c'),('d','c')])=={'a','b','c'}
    summary={'seed':SEED,'tape_cases':len(checks),'optimization_cases':len(optimization),'max_optimization_gap':max(abs(x['min_cut']-x['exhaustive']) for x in optimization),'workflow_cases':len(workflows),'workflow_equality_failures':sum(not x['all_decisions_equal'] or not x['all_artifacts_equal'] for x in workflows),'full_reads_per_workflow':2480,'selective_reads_mean':float(np.mean([x['selective_reads'] for x in workflows])),'selective_reads_min':min(x['selective_reads'] for x in workflows),'selective_reads_max':max(x['selective_reads'] for x in workflows),'fingerprint_checks':key_checks,'evidence_status':'synthetic bounded-tape and graph validation; no measured enterprise savings','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
