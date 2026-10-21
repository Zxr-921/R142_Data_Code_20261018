import os,re,base64,gzip,hashlib,math,json,gc,tarfile
from pathlib import Path
from huggingface_hub import HfApi

OLD_JOB=os.environ['R138_OLD_JOB']
CAND=os.environ['R138_CANDIDATE']
EXPECTED_ADAPTER_SHA=os.environ['R138_EXPECTED_ADAPTER_SHA']
CURRENT_PROTOCOL_SHA='d494350d71204132198ff834058660e989d2001887af75cca583106d335a38c8'
ORIGINAL_RUNNER_SHA='d57a2e180798741afaeec33a4bdde6ef32d4eae7f67386b237d91747b625c4a2'

api=HfApi(token=os.environ['HF_TOKEN'])
j=api.inspect_job(job_id=OLD_JOB)
s=j.command[2]
m=re.search(r"s='([^']+)'\nd=gzip\.decompress",s)
assert m, 'embedded runner not found'
raw=gzip.decompress(base64.b64decode(m.group(1)))
assert hashlib.sha256(raw).hexdigest()==ORIGINAL_RUNNER_SHA
src=raw.decode()
# instrumentation only: retain every prompt so later executable calls have exact prompt provenance
old="'prompt':prompt if len(self.capture)==0 else None"
new="'prompt':prompt"
assert old in src
src_instrumented=src.replace(old,new)
wrapper_source_sha=hashlib.sha256(src_instrumented.encode()).hexdigest()
# Execute definitions/setup only; do NOT execute old bottom-level candidate training.
prefix=src_instrumented.split("name=os.environ['R138_CANDIDATE']",1)[0]
ns={'__name__':'__main__'}
exec(compile(prefix,'/tmp/r138_original_instrumented.py','exec'),ns,ns)
provenance={
    'candidate':CAND,'old_job':OLD_JOB,'original_runner_sha256':ORIGINAL_RUNNER_SHA,
    'instrumented_source_sha256':wrapper_source_sha,'current_protocol_sha256':CURRENT_PROTOCOL_SHA,
    'embedded_protocol_sha256':ns.get('PROTOCOL_SHA')}
print('R138_STAGE2_PROVENANCE',json.dumps(provenance,sort_keys=True),flush=True)

# helpers from original runner
SPLITS=ns['SPLITS']; variants=ns['variants']; run_one=ns['run_one']; load_base=ns['load_base']
teacher_margin=ns['teacher_margin']; canonical_call_text=ns['canonical_call_text']; adapter_strength=ns['adapter_strength']
SEEDS=ns['SEEDS']; STRENGTHS=ns['STRENGTHS']; RADII=ns['RADII']; PILOT_S=ns['PILOT_S']
seed_all=ns['seed_all']; LoraConfig=ns['LoraConfig']; get_peft_model=ns['get_peft_model']; DataLoader=ns['DataLoader']
EncDataset=ns['EncDataset']; collate=ns['collate']; BATCH=ns['BATCH']; LR=ns['LR']; EPOCHS=ns['EPOCHS']; GRAD_ACCUM=ns['GRAD_ACCUM']
delta_norm_from_model=ns['delta_norm_from_model']; OUT=ns['OUT']; torch=ns['torch']; adapter_sha256=ns['adapter_sha256']

assert CAND in ('U2','U4')

def first_executable(step_rows):
    for st in step_rows:
        calls=st.get('calls') or []
        codes=st.get('executed_code') or []
        prompt=st.get('prompt')
        if calls and codes and prompt:
            return prompt, canonical_call_text(calls[0]), calls[0]
    return None,None,None

# Phase A: endpoint-blind M0 on transfer-fit and transfer-select only.
base=load_base(train=False)
protected={'old_transfer_fit':[],'old_transfer_select':[]}
for split in ('old_transfer_fit','old_transfer_select'):
    for sn in variants(SPLITS[split]):
        row=run_one(sn,base,'m0_'+split,'M0',0.0)
        if not row.get('exact_state_success',False):
            continue
        prompt,target,call=first_executable(row.get('agent_steps') or [])
        if prompt is None:
            print('R138_PROTECTED_EXCLUDED_NO_EXECUTABLE',json.dumps({'candidate':CAND,'split':split,'scenario':sn}),flush=True)
            continue
        g0=float(teacher_margin(base,prompt,target,0.0))
        rec={'split':split,'scenario':sn,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
             'target_sha256':hashlib.sha256(target.encode()).hexdigest(),'call':call,'g0':g0,
             'm0_similarity':row['similarity'],'m0_exact_state_success':True,'_prompt':prompt,'_target':target}
        protected[split].append(rec)
        print('R138_M0_PROTECTED',json.dumps({k:v for k,v in rec.items() if k!='call'},sort_keys=True),flush=True)
print('R138_M0_COUNTS',json.dumps({k:len(v) for k,v in protected.items()},sort_keys=True),flush=True)
del base; gc.collect(); torch.cuda.empty_cache()

# Phase B: deterministic candidate retraining, byte-hash check before any transfer use.
seed=SEEDS[CAND]; seed_all(seed); base_train=load_base(train=True)
cfg=LoraConfig(r=ns['LORA_R'],lora_alpha=ns['LORA_ALPHA'],lora_dropout=0.0,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM')
model=get_peft_model(base_train,cfg); model.train()
gen=torch.Generator(); gen.manual_seed(seed)
dl=DataLoader(EncDataset(),batch_size=BATCH,shuffle=True,generator=gen,collate_fn=collate,num_workers=0)
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR)
opt.zero_grad(set_to_none=True); accum=0; losses=[]; optsteps=0
for ep in range(EPOCHS):
    for bi,batch in enumerate(dl):
        batch={k:v.to('cuda') for k,v in batch.items()}
        out=model(**batch,use_cache=False); loss=out.loss/GRAD_ACCUM; loss.backward(); losses.append(float(out.loss.detach().cpu())); accum+=1
        if accum==GRAD_ACCUM or bi==len(dl)-1:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step(); opt.zero_grad(set_to_none=True); accum=0; optsteps+=1
    print('R138_STAGE2_TRAIN_EPOCH',CAND,ep,'mean_loss',sum(losses)/len(losses),'opt_steps',optsteps,flush=True)
path=OUT/(CAND+'_stage2'); model.save_pretrained(path,safe_serialization=True)
norm=float(delta_norm_from_model(model)); model.eval(); model.config.use_cache=True
sha=adapter_sha256(path)
print('R138_STAGE2_ADAPTER',json.dumps({'candidate':CAND,'adapter_sha256':sha,'expected':EXPECTED_ADAPTER_SHA,'delta_norm':norm,'seed':seed},sort_keys=True),flush=True)
assert sha==EXPECTED_ADAPTER_SHA, f'adapter hash mismatch {sha} != {EXPECTED_ADAPTER_SHA}'

# Phase C: pilot + finite-grid paths. No fresh-old/audit is opened here.
all_paths={k:[] for k in protected}
for split,recs in protected.items():
    for rec in recs:
        prompt,target=rec['_prompt'],rec['_target']
        g0=rec['g0']
        g05=float(teacher_margin(model,prompt,target,PILOT_S))
        d=(g05-g0)/PILOT_S
        gs={}
        for ss in STRENGTHS:
            gs[str(ss)]=float(teacher_margin(model,prompt,target,float(ss)))
        pathrec={'scenario':rec['scenario'],'g0':g0,'g05':g05,'d_hat':d,'g':gs}
        all_paths[split].append(pathrec)
        print('R138_MARGIN_PATH',json.dumps({'candidate':CAND,'split':split,**pathrec},sort_keys=True),flush=True)

def sgrid(R): return [float(s) for s in STRENGTHS if float(s)<=float(R)+1e-12]
def resid(path,R):
    return max(abs(float(path['g'][str(s)]) - path['g0'] - s*path['d_hat']) for s in sgrid(R))
def realized_safe(path,R):
    return all(float(path['g'][str(s)])>0.0 for s in sgrid(R))
def robust_decl(path,R,q):
    return all(path['g0'] + s*path['d_hat'] - q > 0.0 for s in sgrid(R))
def naive_decl(path,R):
    return all(path['g0'] + s*path['d_hat'] > 0.0 for s in sgrid(R))

fit=all_paths['old_transfer_fit']; sel=all_paths['old_transfer_select']
q_by_R={}; select_metrics={}
for R in RADII:
    errs=sorted(resid(p,R) for p in fit); n=len(errs); k=math.ceil((n+1)*0.90)
    q=float('inf') if k>n or n==0 else float(errs[k-1])
    q_by_R[str(R)]={'q_R':q,'n_fit':n,'k':k,'fit_residuals':errs}
    cov_n=sum(resid(p,R)<=q for p in sel); nsel=len(sel); coverage=(cov_n/nsel if nsel else float('nan'))
    robust=[p for p in sel if robust_decl(p,R,q)]
    rfalse=sum(not realized_safe(p,R) for p in robust)
    rrate=(rfalse/len(robust) if robust else 0.0)
    naive=[p for p in sel if naive_decl(p,R)]
    nfalse=sum(not realized_safe(p,R) for p in naive)
    nrate=(nfalse/len(naive) if naive else 0.0)
    min_safe=max(5,0.10*nsel)
    passed=bool(nsel and coverage>=0.90 and len(robust)>=min_safe and rrate<=0.10)
    met={'R':float(R),'q_R':q,'protected_select':nsel,'coverage_n':cov_n,'coverage':coverage,
         'robust_safe_n':len(robust),'robust_false_safe_n':rfalse,'robust_false_safe_rate':rrate,
         'naive_safe_n':len(naive),'naive_false_safe_n':nfalse,'naive_false_safe_rate':nrate,
         'min_safe_required':min_safe,'passed':passed}
    select_metrics[str(R)]=met
    print('R138_RADIUS_SELECT',json.dumps({'candidate':CAND,**met},sort_keys=True),flush=True)
passing=[float(R) for R in RADII if select_metrics[str(R)]['passed']]
Rstar=max(passing) if passing else 0.0
summary={'candidate':CAND,'adapter_sha256':sha,'delta_norm':norm,'protected_fit':len(fit),'protected_select':len(sel),
         'q_by_R':q_by_R,'select_metrics':select_metrics,'R_star':Rstar,
         'current_protocol_sha256':CURRENT_PROTOCOL_SHA,'original_runner_sha256':ORIGINAL_RUNNER_SHA}
print('R138_STAGE2_RESULT',json.dumps(summary,sort_keys=True),flush=True)

# Persistent-by-log archive. The HF OAuth available to Jobs has no write-repos scope,
# so every raw artifact is bundled and emitted in checksummed base64 chunks.
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'R138_STAGE2_PROVENANCE.json').write_text(json.dumps(provenance,sort_keys=True,indent=2))
(OUT/'R138_STAGE2_SUMMARY.json').write_text(json.dumps(summary,sort_keys=True,indent=2))
(OUT/'R138_PROTECTED_RAW.json').write_text(json.dumps(protected,sort_keys=True,indent=2))
(OUT/'R138_MARGIN_PATHS.json').write_text(json.dumps(all_paths,sort_keys=True,indent=2))
(OUT/'R138_QR_BY_RADIUS.json').write_text(json.dumps(q_by_R,sort_keys=True,indent=2))
(OUT/'R138_RADIUS_SELECT.json').write_text(json.dumps(select_metrics,sort_keys=True,indent=2))
(OUT/'R138_ORIGINAL_RUNNER.py').write_bytes(raw)
(OUT/'R138_INSTRUMENTED_RUNNER.py').write_text(src_instrumented)
try:
    (OUT/'R138_STAGE2_WRAPPER.py').write_text(Path(__file__).read_text())
except Exception:
    pass
# File-level hashes before archive.
entries=[]
for fp in sorted(OUT.rglob('*')):
    if fp.is_file() and fp.name!='R138_STAGE2_ARCHIVE.tar.gz':
        b=fp.read_bytes(); entries.append({'path':str(fp.relative_to(OUT)),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(OUT/'R138_FILE_MANIFEST.json').write_text(json.dumps(entries,sort_keys=True,indent=2))
sha_lines=[f"{e['sha256']}  {e['path']}" for e in entries]
(OUT/'SHA256SUMS').write_text('\n'.join(sha_lines)+'\n')
archive=Path('/tmp')/f'R138_STAGE2_{CAND}_ARCHIVE.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
    tf.add(OUT,arcname=f'R138_STAGE2_{CAND}')
ab=archive.read_bytes(); archive_sha=hashlib.sha256(ab).hexdigest()
meta={'candidate':CAND,'archive_sha256':archive_sha,'archive_bytes':len(ab),'file_count':len(entries),'wrapper_runtime_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
print('R138_ARCHIVE_META',json.dumps(meta,sort_keys=True),flush=True)
b64=base64.b64encode(ab).decode('ascii'); chunk_size=4000
chunks=(len(b64)+chunk_size-1)//chunk_size
for i in range(chunks):
    print(f'R138_ARCHIVE_CHUNK {CAND} {i:06d}/{chunks:06d} '+b64[i*chunk_size:(i+1)*chunk_size],flush=True)
print('R138_ARCHIVE_END',json.dumps({'candidate':CAND,'archive_sha256':archive_sha,'chunks':chunks},sort_keys=True),flush=True)
print('R138_STAGE2_COMPLETE',CAND,flush=True)
