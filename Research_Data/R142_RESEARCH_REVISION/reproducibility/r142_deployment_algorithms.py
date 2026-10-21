"""Executable deployment primitives with explicit validity conditions.

No routine infers population safety from a finite benchmark certificate.
Minimum cut is the classical maximum-closure construction (Picard, 1976).
"""
from collections import deque
from dataclasses import dataclass
import math

def finite_tape_decision(tape, threshold, order=None):
    """Exact decision 1{sum(tape)>=threshold} with unseen entries in [0,1].
    Accesses only entries in the supplied outcome-independent order, stopping
    when every possible completion has the same decision. Returns a transcript.
    """
    n=len(tape); order=list(range(n)) if order is None else list(order)
    if sorted(order)!=list(range(n)):raise ValueError('order must be a permutation')
    if not 0<threshold<=n:raise ValueError('threshold outside (0,n]')
    total=0.;seen=[]
    for i in order:
        value=float(tape[i])
        if not 0<=value<=1:raise ValueError('observations must be bounded in [0,1]')
        seen.append((i,value));total+=value;remaining=n-len(seen)
        if total>=threshold:return {'decision':1,'reads':len(seen),'lower':total,'upper':total+remaining,'transcript':seen}
        if total+remaining<threshold:return {'decision':0,'reads':len(seen),'lower':total,'upper':total+remaining,'transcript':seen}
    raise AssertionError('complete tape must determine a decision')

def downstream_closure(roots, edges):
    """All descendants of roots, including roots; directed dependency edges."""
    adj={}
    for a,b in edges:adj.setdefault(a,[]).append(b)
    found=set(roots);todo=deque(roots)
    while todo:
        for b in adj.get(todo.popleft(),[]):
            if b not in found:found.add(b);todo.append(b)
    return found

def minimum_work_plan(certificate_costs,reopen_costs,operation_parents,operation_costs,forced=()):
    """Exact plan for fixed available sound certificates with additive prices.

    Prices must be known before optimizing; this is not an optimizer of an
    adaptive experiment whose success and price are still unknown. Values
    returned by a numerical solver must satisfy the explicit objective check.
    """
    n=len(certificate_costs);m=len(operation_costs)
    if len(reopen_costs)!=n or len(operation_parents)!=m:raise ValueError('shape mismatch')
    vals=list(certificate_costs)+list(reopen_costs)+list(operation_costs)
    if any(not math.isfinite(x) or x<0 for x in vals):raise ValueError('finite nonnegative prices required')
    forced=set(forced)
    if not forced.issubset(range(n)):raise ValueError('invalid forced root')
    if any(not set(p).issubset(range(n)) for p in operation_parents):raise ValueError('invalid parent')
    infinity=1.+sum(vals);s=n+m;t=s+1;adj=[[] for _ in range(t+1)];edge_count=0
    def edge(a,b,capacity):
        nonlocal edge_count
        adj[a].append([b,float(capacity),len(adj[b])]);adj[b].append([a,0.,len(adj[a])-1]);edge_count+=1
    for i,(a,b) in enumerate(zip(certificate_costs,reopen_costs)):
        edge(s,i,a+(infinity if i in forced else 0));edge(i,t,b)
    for v,(parents,w) in enumerate(zip(operation_parents,operation_costs)):
        node=n+v;edge(node,t,w)
        for i in set(parents):edge(i,node,infinity)
    # Dinic's blocking-flow algorithm. No optional graph package is required.
    value=0.
    while True:
        level=[-1]*len(adj);level[s]=0;queue=deque([s])
        while queue:
            a=queue.popleft()
            for b,cap,rev in adj[a]:
                if cap>0 and level[b]<0:level[b]=level[a]+1;queue.append(b)
        if level[t]<0:break
        cursor=[0]*len(adj)
        def augment(a,amount):
            if a==t:return amount
            while cursor[a]<len(adj[a]):
                item=adj[a][cursor[a]];b,cap,rev=item
                if cap>0 and level[b]==level[a]+1:
                    pushed=augment(b,min(amount,cap))
                    if pushed>0:item[1]-=pushed;adj[b][rev][1]+=pushed;return pushed
                cursor[a]+=1
            return 0.
        while (pushed:=augment(s,infinity))>0:value+=pushed
    reopened={i for i in range(n) if level[i]>=0}
    activated={v for v,p in enumerate(operation_parents) if reopened.intersection(p)}
    objective=sum(x for i,x in enumerate(certificate_costs) if i not in reopened)+sum(reopen_costs[i] for i in reopened)+sum(operation_costs[v] for v in activated)
    assert forced.issubset(reopened)
    assert math.isclose(value,objective,rel_tol=1e-9,abs_tol=1e-8)
    return {'reopened':sorted(reopened),'operations':sorted(activated),'cost':objective,'cut_value':value,'nodes':len(adj),'edges':edge_count}

def plan_cost(reopened,certificate_costs,reopen_costs,operation_parents,operation_costs):
    u=set(reopened)
    return sum(x for i,x in enumerate(certificate_costs) if i not in u)+sum(reopen_costs[i] for i in u)+sum(w for p,w in zip(operation_parents,operation_costs) if u.intersection(p))

def certificate_reusable(old_fingerprint,new_fingerprint):
    """All fields matter: model, inputs, task evaluator, environment and policy.
    Exact tuple equality is used; hashes additionally assume no collision.
    """
    required={'model','input','evaluator','environment','policy'}
    if set(old_fingerprint)!=required or set(new_fingerprint)!=required:raise ValueError('incomplete fingerprint')
    return old_fingerprint==new_fingerprint
