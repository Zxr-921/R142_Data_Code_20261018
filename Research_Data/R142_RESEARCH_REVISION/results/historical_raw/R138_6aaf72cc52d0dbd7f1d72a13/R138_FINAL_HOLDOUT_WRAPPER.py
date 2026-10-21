import os,re,json,base64,hashlib,io,tarfile,gc,shutil
from pathlib import Path
import requests
from huggingface_hub import HfApi

CURRENT_PROTOCOL_SHA='d494350d71204132198ff834058660e989d2001887af75cca583106d335a38c8'
ORIGINAL_RUNNER_SHA='d57a2e180798741afaeec33a4bdde6ef32d4eae7f67386b237d91747b625c4a2'
INSTRUMENTED_RUNNER_SHA='33c318eff244e93c165a90a602ef7f75c5bec5db2dd59add3631793002768f27'
FINAL_FREEZE_SCOPED_SHA='6047380c0b4f83fa4d603d4905a728c46d420aa93fa830dc6db5ffe5b156d53d'
FINAL_FREEZE_FILE_SHA='da427d60ab5a52dd8a25c10c3c24f180f03bcbb872d595ff23fa7ae51952da44'
STAGE2_U2_SHA='1bb4927de421deba0962ab01d6cc5e1f9df3d9f685088c5e7dd91554c56f9484'
ADAPTER_SHA={'U2':'f3ee811bd8a7495810b8c98766acd21650ee55d9e68bade725e163ad636694c9','U4':'de0221fab606890641f3a6bf2f4a8b0ca34f53e0dbc7a28e73a4686940b2c98f'}
MODE=os.environ['R138_MODE']
CAND=os.environ.get('R138_CANDIDATE','M0')
SOURCE_JOB=os.environ.get('R138_STAGE2_JOB','6aaf38c952d0dbd7f1d71c72')
SOURCE_SHA=os.environ.get('R138_STAGE2_ARCHIVE_SHA',STAGE2_U2_SHA)
BASE_START=int(os.environ.get('R138_BASE_START','0'))
BASE_END=int(os.environ.get('R138_BASE_END','999'))
FRESH_M0_JOB=os.environ.get('R138_FRESH_M0_JOB','')
EXPECTED_FRESH_M0_SHA=os.environ.get('R138_FRESH_M0_ARCHIVE_SHA','')
HF_TOKEN=os.environ['HF_TOKEN']
freeze_bytes=base64.b64decode(os.environ['R138_FREEZE_B64'])
assert hashlib.sha256(freeze_bytes).hexdigest()==FINAL_FREEZE_FILE_SHA
FREEZE=json.loads(freeze_bytes)
assert FREEZE['freeze_sha256']==FINAL_FREEZE_SCOPED_SHA
assert FREEZE['protocol_sha256']==CURRENT_PROTOCOL_SHA
assert FREEZE['selected_pair']==['U2','U4']
assert FREEZE['stage2']['U2']['R_star']==0.0 and FREEZE['stage2']['U4']['R_star']==0.0

api=HfApi(token=HF_TOKEN)
namespace=api.whoami()['name']
headers=api._build_hf_headers(token=HF_TOKEN)

def events(job_id,tail):
    r=requests.get(f'https://huggingface.co/api/jobs/{namespace}/{job_id}/logs',headers=headers,params={'tail':tail},timeout=180)
    r.raise_for_status(); out=[]
    for raw in r.text.splitlines():
        if raw.startswith('data: {'):
            try: out.append(json.loads(raw[6:]).get('data',''))
            except Exception: pass
    return out

def recover(job_id,end_marker,chunk_marker,expected_sha=None,tail=50000):
    chunks={}; end=None
    for line in events(job_id,tail):
        if chunk_marker+' ' in line:
            toks=line.split(chunk_marker+' ',1)[1].split(' ')
            fi=next((i for i,t in enumerate(toks) if re.fullmatch(r'\d+/\d+',t)),None)
            if fi is not None:
                idx,total=map(int,toks[fi].split('/')); chunks[idx]=(total,''.join(toks[fi+1:]).strip())
        if end_marker+' ' in line:
            try: end=json.loads(line.split(end_marker+' ',1)[1])
            except Exception: pass
    assert end is not None,(job_id,end_marker)
    total=int(end['chunks'])
    assert len(chunks)==total and min(chunks)==0 and max(chunks)==total-1 and all(i in chunks for i in range(total))
    blob=base64.b64decode(''.join(chunks[i][1] for i in range(total)),validate=True)
    sha=hashlib.sha256(blob).hexdigest()
    assert sha==end['archive_sha256']
    if expected_sha: assert sha==expected_sha,(sha,expected_sha)
    return blob,end

# Recover verified Stage-2 source and adapter.
stage2_blob,_=recover(SOURCE_JOB,'R138_ARCHIVE_END','R138_ARCHIVE_CHUNK',SOURCE_SHA,3100)
stage2_dir=Path('/tmp/r138_stage2_source'); shutil.rmtree(stage2_dir,ignore_errors=True); stage2_dir.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(stage2_blob),mode='r:gz') as tf: tf.extractall(stage2_dir)
rf=list(stage2_dir.rglob('R138_INSTRUMENTED_RUNNER.py')); of=list(stage2_dir.rglob('R138_ORIGINAL_RUNNER.py'))
assert len(rf)==1 and len(of)==1
runner_src=rf[0].read_text(); original_bytes=of[0].read_bytes()
assert hashlib.sha256(runner_src.encode()).hexdigest()==INSTRUMENTED_RUNNER_SHA
assert hashlib.sha256(original_bytes).hexdigest()==ORIGINAL_RUNNER_SHA
prefix=runner_src.split("name=os.environ['R138_CANDIDATE']",1)[0]
ns={'__name__':'__main__'}; exec(compile(prefix,'/tmp/r138_final_runner.py','exec'),ns,ns)
assert ns['TOOLSANDBOX_COMMIT']=='c8571d7854316d2e1c5f288e59fe1e34e53f6dd1'
assert ns['MODEL']=='NousResearch/Hermes-3-Llama-3.2-3B'

OUT=Path(f'/tmp/r138_final_{MODE}_{CAND}_{BASE_START}_{BASE_END}')
shutil.rmtree(OUT,ignore_errors=True); OUT.mkdir(parents=True)
ns['OUT']=OUT; ns['LEDGER']=OUT/'R138_SCENARIO_LEDGER.jsonl'
(OUT/'R138_FREEZE.json').write_bytes(freeze_bytes)
(OUT/'R138_INSTRUMENTED_RUNNER.py').write_text(runner_src)
(OUT/'R138_ORIGINAL_RUNNER.py').write_bytes(original_bytes)
try: (OUT/'R138_FINAL_HOLDOUT_WRAPPER.py').write_text(Path(__file__).read_text())
except Exception: pass

# Preserve raw action/tool information while avoiding redundant copies of every full prompt.
def compact_row(row):
    z=dict(row); steps=[]
    for j,st in enumerate(row.get('agent_steps') or []):
        q=dict(st)
        if j and not ((q.get('calls') or []) and (q.get('executed_code') or [])): q['prompt']=None
        steps.append(q)
    z['agent_steps']=steps; return z

def append_compact(row):
    with ns['LEDGER'].open('a') as f: f.write(json.dumps(compact_row(row),sort_keys=True,default=str,separators=(',',':'))+'\n')
ns['append_ledger']=append_compact

SPLITS=ns['SPLITS']; variants=ns['variants']; run_one=ns['run_one']; load_base=ns['load_base']; teacher_margin=ns['teacher_margin']
PeftModel=ns['PeftModel']; torch=ns['torch']; adapter_sha256=ns['adapter_sha256']; STRENGTHS=ns['STRENGTHS']; PILOT_S=ns['PILOT_S']; RADII=ns['RADII']; canonical_call_text=ns['canonical_call_text']

def first_exec(steps):
    for st in steps or []:
        calls=st.get('calls') or []; codes=st.get('executed_code') or []; prompt=st.get('prompt')
        if calls and codes and prompt: return prompt,canonical_call_text(calls[0]),calls[0]
    return None,None,None

def emit(prefix,key,blob):
    sha=hashlib.sha256(blob).hexdigest(); s=base64.b64encode(blob).decode(); step=3000; n=(len(s)+step-1)//step
    meta={'key':key,'archive_sha256':sha,'chunks':n,'bytes':len(blob)}
    print(prefix+'_META',json.dumps(meta,sort_keys=True),flush=True)
    for i in range(n): print(prefix+'_CHUNK',key,f'{i:06d}/{n:06d}',s[i*step:(i+1)*step],flush=True)
    print(prefix+'_END',json.dumps(meta,sort_keys=True),flush=True)
    return sha,n

def tar_paths(paths):
    bio=io.BytesIO()
    with tarfile.open(fileobj=bio,mode='w:gz',compresslevel=6) as tf:
        for p in paths:
            p=Path(p)
            if p.exists(): tf.add(p,arcname=str(p.relative_to(OUT)))
    return bio.getvalue()

def checkpoint(key,record_file,phase_prefix,base):
    paths=[record_file]; traj=OUT/'tool_sandbox_runs'/'trajectories'
    if traj.exists():
        for p in sorted(traj.iterdir()):
            if p.is_dir() and p.name.startswith(phase_prefix) and ('__'+base in p.name or p.name.endswith('__'+base)): paths.append(p)
    blob=tar_paths(paths); sha,n=emit('R138_FINAL_CHECKPOINT',key,blob)
    return {'key':key,'sha256':sha,'chunks':n,'bytes':len(blob)}

prov={'mode':MODE,'candidate':CAND,'source_stage2_job':SOURCE_JOB,'source_stage2_archive_sha256':SOURCE_SHA,
      'freeze_scoped_sha256':FINAL_FREEZE_SCOPED_SHA,'freeze_file_sha256':FINAL_FREEZE_FILE_SHA,
      'protocol_sha256':CURRENT_PROTOCOL_SHA,'original_runner_sha256':ORIGINAL_RUNNER_SHA,
      'instrumented_runner_sha256':INSTRUMENTED_RUNNER_SHA,'base_start':BASE_START,'base_end':BASE_END,'opened_after_freeze':True}
(OUT/'R138_FINAL_PROVENANCE.json').write_text(json.dumps(prov,sort_keys=True,indent=2))
print('R138_FINAL_PROVENANCE',json.dumps(prov,sort_keys=True),flush=True)
checkpoints=[]

if MODE in ('fresh_m0','audit_m0'):
    model=load_base(train=False)
    if MODE=='fresh_m0':
        all_rows=[]; protected=[]
        for base in SPLITS['old_fresh']:
            rows=[]; recs=[]
            for sn in variants([base]):
                row=run_one(sn,model,'fresh_m0','M0',0.0); cr=compact_row(row); rows.append(cr); all_rows.append(cr)
                if row.get('exact_state_success',False):
                    prompt,target,call=first_exec(row.get('agent_steps') or [])
                    if prompt is not None:
                        rec={'base':base,'scenario':sn,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                             'target':target,'target_sha256':hashlib.sha256(target.encode()).hexdigest(),'call':call,
                             'g0':float(teacher_margin(model,prompt,target,0.0)),'m0_similarity':row['similarity'],'m0_exact_state_success':True}
                        recs.append(rec); protected.append(rec)
            f=OUT/'checkpoints'/f'fresh_m0__{base}.json'; f.parent.mkdir(exist_ok=True); f.write_text(json.dumps({'rows':rows,'protected':recs},sort_keys=True,indent=2))
            checkpoints.append(checkpoint(f'fresh_m0::{base}',f,'fresh_m0__M0__s0.0',base))
        (OUT/'R138_FRESH_M0_PROTECTED.json').write_text(json.dumps(protected,sort_keys=True,indent=2))
        summary={'mode':MODE,'variant_n':len(all_rows),'protected_n':len(protected),
                 'mean_similarity':sum(r['similarity'] for r in all_rows)/len(all_rows),
                 'exact_state_rate':sum(bool(r['exact_state_success']) for r in all_rows)/len(all_rows),
                 'strong_old_damage_claim_allowed':len(protected)>=10}
    else:
        bases=SPLITS['new_capability_audit'][BASE_START:min(BASE_END,len(SPLITS['new_capability_audit']))]; all_rows=[]
        for base in bases:
            rows=[]
            for sn in variants([base]):
                row=run_one(sn,model,'cap_audit_m0','M0',0.0); cr=compact_row(row); rows.append(cr); all_rows.append(cr)
            f=OUT/'checkpoints'/f'cap_audit_m0__{base}.json'; f.parent.mkdir(exist_ok=True); f.write_text(json.dumps({'rows':rows},sort_keys=True,indent=2))
            checkpoints.append(checkpoint(f'cap_audit_m0::{base}',f,'cap_audit_m0__M0__s0.0',base))
        summary={'mode':MODE,'base_n':len(bases),'variant_n':len(all_rows),
                 'mean_similarity':sum(r['similarity'] for r in all_rows)/len(all_rows),
                 'exact_state_rate':sum(bool(r['exact_state_success']) for r in all_rows)/len(all_rows),'bases':bases}
    del model; gc.collect(); torch.cuda.empty_cache()

elif MODE in ('audit_candidate','fresh_candidate'):
    assert CAND in ('U2','U4')
    ads=[p.parent for p in stage2_dir.rglob('adapter_model.safetensors') if p.parent.name==CAND+'_stage2']; assert len(ads)==1
    adapter_dir=ads[0]; assert adapter_sha256(adapter_dir)==ADAPTER_SHA[CAND]
    base_model=load_base(train=False); model=PeftModel.from_pretrained(base_model,str(adapter_dir),is_trainable=False); model.eval(); model.config.use_cache=True
    # Preserve exact adapter bytes in this final archive too.
    shutil.copytree(adapter_dir,OUT/(CAND+'_adapter'),dirs_exist_ok=True)
    if MODE=='audit_candidate':
        bases=SPLITS['new_capability_audit'][BASE_START:min(BASE_END,len(SPLITS['new_capability_audit']))]; all_rows=[]
        for base in bases:
            rows=[]
            for sn in variants([base]):
                row=run_one(sn,model,'cap_audit',CAND,1.0); cr=compact_row(row); rows.append(cr); all_rows.append(cr)
            f=OUT/'checkpoints'/f'cap_audit__{CAND}__{base}.json'; f.parent.mkdir(exist_ok=True); f.write_text(json.dumps({'rows':rows},sort_keys=True,indent=2))
            checkpoints.append(checkpoint(f'cap_audit::{CAND}::{base}',f,f'cap_audit__{CAND}__s1.0',base))
        summary={'mode':MODE,'candidate':CAND,'adapter_sha256':ADAPTER_SHA[CAND],'base_n':len(bases),'variant_n':len(all_rows),
                 'mean_similarity':sum(r['similarity'] for r in all_rows)/len(all_rows),
                 'exact_state_rate':sum(bool(r['exact_state_success']) for r in all_rows)/len(all_rows),'bases':bases}
    else:
        assert FRESH_M0_JOB and EXPECTED_FRESH_M0_SHA
        fresh_blob,_=recover(FRESH_M0_JOB,'R138_FINAL_ARCHIVE_END','R138_FINAL_ARCHIVE_CHUNK',EXPECTED_FRESH_M0_SHA,50000)
        fd=Path('/tmp/r138_fresh_m0_source'); shutil.rmtree(fd,ignore_errors=True); fd.mkdir()
        with tarfile.open(fileobj=io.BytesIO(fresh_blob),mode='r:gz') as tf: tf.extractall(fd)
        pp=list(fd.rglob('R138_FRESH_M0_PROTECTED.json')); assert len(pp)==1
        protected=json.loads(pp[0].read_text()); bybase={b:[] for b in SPLITS['old_fresh']}
        for rec in protected: bybase[rec['base']].append(rec)
        all_rows=[]; all_paths=[]
        for base in SPLITS['old_fresh']:
            rows=[]; paths=[]
            for rec in bybase[base]:
                row=run_one(rec['scenario'],model,'fresh_candidate',CAND,1.0); cr=compact_row(row); rows.append(cr); all_rows.append(cr)
                prompt,target,g0=rec['prompt'],rec['target'],float(rec['g0'])
                g05=float(teacher_margin(model,prompt,target,PILOT_S)); d=(g05-g0)/PILOT_S
                gs={str(s):float(teacher_margin(model,prompt,target,float(s))) for s in STRENGTHS}
                residuals={}
                for R in RADII:
                    grid=[float(s) for s in STRENGTHS if float(s)<=float(R)+1e-12]
                    residuals[str(R)]=max(abs(gs[str(s)]-g0-s*d) for s in grid)
                pr={'base':base,'scenario':rec['scenario'],'g0':g0,'g05':g05,'d_hat':d,'g':gs,'residuals':residuals,
                    'frozen_q_R':{str(R):'Infinity' for R in RADII},'frozen_R_star':0.0,'certificate_abstain':True}
                paths.append(pr); all_paths.append(pr)
            f=OUT/'checkpoints'/f'fresh_candidate__{CAND}__{base}.json'; f.parent.mkdir(exist_ok=True); f.write_text(json.dumps({'rows':rows,'margin_paths':paths},sort_keys=True,indent=2))
            checkpoints.append(checkpoint(f'fresh_candidate::{CAND}::{base}',f,f'fresh_candidate__{CAND}__s1.0',base))
        (OUT/'R138_FRESH_MARGIN_PATHS.json').write_text(json.dumps(all_paths,sort_keys=True,indent=2))
        pn=len(protected); damage=sum(not bool(r['exact_state_success']) for r in all_rows)
        m0mean=sum(float(r['m0_similarity']) for r in protected)/pn if pn else None
        cmean=sum(float(r['similarity']) for r in all_rows)/len(all_rows) if all_rows else None
        summary={'mode':MODE,'candidate':CAND,'adapter_sha256':ADAPTER_SHA[CAND],'protected_n':pn,'candidate_evaluated_n':len(all_rows),
                 'old_damage_n':damage,'old_damage_rate':damage/pn if pn else None,'m0_mean_similarity':m0mean,
                 'candidate_mean_similarity':cmean,'similarity_change':(cmean-m0mean) if pn else None,
                 'frozen_R_star':0.0,'certificate_abstain':True,'strong_old_damage_claim_allowed':pn>=10}
    del model,base_model; gc.collect(); torch.cuda.empty_cache()
else:
    raise ValueError(MODE)

(OUT/'R138_FINAL_SUMMARY.json').write_text(json.dumps(summary,sort_keys=True,indent=2))
(OUT/'R138_CHECKPOINT_INDEX.json').write_text(json.dumps(checkpoints,sort_keys=True,indent=2))
print('R138_FINAL_SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
entries=[]
for fp in sorted(OUT.rglob('*')):
    if fp.is_file() and not fp.name.endswith('.tar.gz'):
        b=fp.read_bytes(); entries.append({'path':str(fp.relative_to(OUT)),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(OUT/'R138_FILE_MANIFEST.json').write_text(json.dumps(entries,sort_keys=True,indent=2))
(OUT/'SHA256SUMS').write_text('\n'.join(f"{e['sha256']}  {e['path']}" for e in entries)+'\n')
bio=io.BytesIO()
with tarfile.open(fileobj=bio,mode='w:gz',compresslevel=6) as tf:
    for fp in sorted(OUT.rglob('*')):
        if fp.is_file() and not fp.name.endswith('.tar.gz'): tf.add(fp,arcname=str(fp.relative_to(OUT)))
archive=bio.getvalue(); asha,an=emit('R138_FINAL_ARCHIVE',f'{MODE}::{CAND}::{BASE_START}:{BASE_END}',archive)
print('R138_FINAL_COMPLETE',json.dumps({'mode':MODE,'candidate':CAND,'summary':summary,'archive_sha256':asha,'archive_chunks':an,'archive_bytes':len(archive)},sort_keys=True),flush=True)
