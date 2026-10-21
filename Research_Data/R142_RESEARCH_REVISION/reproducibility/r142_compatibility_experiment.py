"""R142: published MUSCLE loss versus matched extra cross-entropy training.

This is a new experiment on previously used R141 audit tasks, not a new
independent confirmatory test and not a reproduction of MUSCLE's benchmark.
The two teachers are frozen; the token mask uses the current student's error.
All settings are frozen in R142_COMPATIBILITY_PROTOCOL.json before execution.
"""
import os, sys, json, time, random, hashlib, gzip, base64, gc, re, copy
from pathlib import Path
from collections import defaultdict

def recover(job, run, target, selected):
    import urllib.request, socket
    wanted = {'configuration.json', 'benchmark_rows.json'}
    groups = {}
    req = urllib.request.Request(
        'https://huggingface.co/api/jobs/jianlau7550/'+job+'/logs?tail=100000',
        headers={'Authorization':'Bearer '+os.environ['HF_TOKEN'], 'Accept':'text/event-stream'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            for raw in response:
                if not raw.startswith(b'data: '): continue
                event = json.loads(raw[6:])['data']
                if event.startswith('===== Job finished'): break
                for line in event.splitlines():
                    m = re.search(r'R141_ARTIFACT_(BEGIN|END) (\{.*\})', line)
                    if m:
                        v=json.loads(m[2]); name=v['name']
                        if name in wanted or name.startswith(('adapters/source_adapter/', 'adapters/'+selected+'/')):
                            assert v['run']==run
                            groups.setdefault(name, {'chunks':{}})['meta']=v
                    m = re.search(r'R141_ARTIFACT_CHUNK (\S+) (\S+) (\d+)/(\d+) ([A-Za-z0-9+/=]+)',line)
                    if m and (m[2] in wanted or m[2].startswith(('adapters/source_adapter/', 'adapters/'+selected+'/'))):
                        groups.setdefault(m[2], {'chunks':{}})['chunks'][int(m[3])]=m[5]
    except (TimeoutError, socket.timeout): pass
    report=[]
    for name,v in groups.items():
        meta=v['meta']; assert set(v['chunks'])==set(range(meta['chunks'])),name
        raw=gzip.decompress(base64.b64decode(''.join(v['chunks'][i] for i in range(meta['chunks']))))
        assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256'],name
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);report.append(meta)
    for name in ['configuration.json','benchmark_rows.json','adapters/source_adapter/adapter_model.safetensors',f'adapters/{selected}/adapter_model.safetensors']:
        assert (target/name).exists(),name
    return report

def run():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from r141_control_experiment import extract_calls
    run_id=sys.argv[1]
    protocol=json.loads(Path('/tmp/R142_COMPATIBILITY_PROTOCOL.json').read_text())
    spec=protocol['batches'][run_id];root=Path('/tmp/r142')/run_id;root.mkdir(parents=True,exist_ok=True)
    source=root/'source';source.mkdir(exist_ok=True)
    provenance=recover(spec['source_job'],run_id,source,spec['candidate'])
    cfg=json.loads((source/'configuration.json').read_text())
    split=json.loads((source/'benchmark_rows.json').read_text())
    assert cfg['run']==run_id
    seed=spec['training_seed']; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    os.environ['TOKENIZERS_PARALLELISM']='false';torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    dtype=torch.bfloat16 if run_id.startswith('qwen') else torch.float16
    events=[];index=[]
    def emit(name,obj,binary=False):
        raw=bytes(obj) if binary else json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
        fp=root/name;fp.parent.mkdir(parents=True,exist_ok=True);fp.write_bytes(raw)
        packed=base64.b64encode(gzip.compress(raw,compresslevel=6)).decode();chunks=[packed[i:i+12000] for i in range(0,len(packed),12000)]
        meta={'run':run_id,'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'chunks':len(chunks)}
        print('R142_ARTIFACT_BEGIN '+json.dumps(meta),flush=True)
        for i,x in enumerate(chunks):
            print(f'R142_ARTIFACT_CHUNK {run_id} {name} {i:05d}/{len(chunks):05d} {x}',flush=True)
            if len(chunks)>10: time.sleep(.02)
        print('R142_ARTIFACT_END '+json.dumps(meta),flush=True);index.append(meta)
    emit('protocol.json',{'protocol':protocol,'spec':spec,'configuration':cfg,'provenance':provenance,
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'prior_audit_reused':True})
    emit('benchmark_rows.json',split)
    tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    eos=tok.eos_token_id
    alias='R142-Checker-Alias';MODEL_CONFIG_MAPPING[alias]=copy.copy(MODEL_CONFIG_MAPPING['gorilla-openfunctions-v2']);MODEL_CONFIG_MAPPING[alias].underscore_to_dot=False
    def base(path,pinned=False):
        return AutoModelForCausalLM.from_pretrained(path,revision=cfg['revision'] if pinned else None,
             torch_dtype=dtype,attn_implementation='eager').to('cuda')
    m0=PeftModel.from_pretrained(base(cfg['model'],True),str(source/'adapters/source_adapter')).merge_and_unload()
    m0.save_pretrained(root/'M0');m0.eval()
    def candidate(trainable=False):
        return PeftModel.from_pretrained(base(str(root/'M0')),str(source/'adapters'/spec['candidate']),is_trainable=trainable)
    @torch.no_grad()
    def evaluate(m,rows,tag):
        m.eval();details=[];torch.cuda.synchronize();start=time.monotonic()
        for s in range(0,len(rows),4):
            batch=rows[s:s+4];width=max(len(x['prompt_ids']) for x in batch)
            ids=[[tok.pad_token_id]*(width-len(x['prompt_ids']))+x['prompt_ids'] for x in batch]
            masks=[[0]*(width-len(x['prompt_ids']))+[1]*len(x['prompt_ids']) for x in batch]
            out=m.generate(input_ids=torch.tensor(ids,device='cuda'),attention_mask=torch.tensor(masks,device='cuda'),
                do_sample=False,max_new_tokens=512,min_new_tokens=0,repetition_penalty=1.,no_repeat_ngram_size=0,
                renormalize_logits=False,pad_token_id=tok.pad_token_id,eos_token_id=eos,use_cache=True)
            for i,row in enumerate(batch):
                tokens=out[i,width:].tolist()
                if eos in tokens:tokens=tokens[:tokens.index(eos)+1]
                text=tok.decode(tokens,skip_special_tokens=True);parsed=extract_calls(text)
                try:
                    checked=ast_checker(row['record']['function'],parsed,row['gt'],Language.PYTHON,row['category'],alias) if parsed is not None else {}
                    valid=bool(checked.get('valid',False));error=None if valid else checked.get('error_type','json_parse')
                except Exception as exc: valid=False;error='checker_exception:'+type(exc).__name__
                details.append({'id':row['id'],'family':row['family'],'valid':valid,'error':error,'text':text,'tokens':tokens,'prompt_tokens':len(row['prompt_ids'])})
        torch.cuda.synchronize();events.append({'kind':'endpoint','tag':tag,'seconds':time.monotonic()-start,'records':len(rows),
            'prompt_tokens':sum(x['prompt_tokens'] for x in details),'generated_tokens':sum(len(x['tokens']) for x in details)})
        emit('rows/'+tag+'.json',details);return details
    summaries={}
    def audit(m,name):
        summaries[name]={}
        for pool in ['fresh_old','cap_validation','cap_audit']:
            out=evaluate(m,split[pool],name+'_'+pool);summaries[name][pool]={'n':len(out),'success':sum(x['valid'] for x in out)}
        print('R142_STAGE '+run_id+' '+name+'_audited',flush=True)
    audit(m0,'M0')
    new=candidate(False);new.eval()
    for teacher in [m0,new]:
        for p in teacher.parameters():p.requires_grad_(False)
    audit(new,'full_update')
    training_rows=split['update_train']+split['source_train'][:40]
    for method in ['muscle_loss','extra_cross_entropy']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        student=candidate(True);student.train();student.config.use_cache=False
        params=[p for p in student.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=2e-4)
        order=list(range(len(training_rows)));random.Random(seed).shuffle(order)
        optimizer.zero_grad(set_to_none=True);losses=[];steps=0;teacher_tokens=0;old_mask=0;target_tokens=0
        torch.cuda.synchronize();start=time.monotonic()
        for j,i in enumerate(order):
            row=training_rows[i];start_token=len(row['prompt_ids'])-1
            ids=torch.tensor([row['prompt_ids']+row['target_ids']],device='cuda');target=torch.tensor(row['target_ids'],device='cuda')
            logits=student(input_ids=ids,use_cache=False).logits[0,start_token:-1].float()
            assert logits.shape[0]==target.shape[0]
            if method=='muscle_loss':
                with torch.no_grad():
                    old_logits=m0(input_ids=ids,use_cache=False).logits[0,start_token:-1].float()
                    new_logits=new(input_ids=ids,use_cache=False).logits[0,start_token:-1].float()
                    mask=logits.detach().argmax(-1).ne(target)
                    teacher=torch.where(mask[:,None],old_logits,new_logits)
                    teacher_log=F.log_softmax(teacher,dim=-1)
                # KL(teacher || student), temperature one; only response tokens.
                loss=F.kl_div(F.log_softmax(logits,dim=-1),teacher_log,log_target=True,reduction='sum')/target.numel()
                teacher_tokens+=2*ids.numel();old_mask+=int(mask.sum())
                del old_logits,new_logits,teacher,teacher_log,mask
            else:loss=F.cross_entropy(logits,target)
            assert bool(torch.isfinite(loss)),(method,j,'nonfinite loss')
            accum=min(8,len(order)-(j//8)*8);(loss/accum).backward()
            losses.append(float(loss.detach()));target_tokens+=target.numel()
            del logits,loss,ids,target
            if (j+1)%8==0 or j+1==len(order):
                torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        events.append({'kind':'training','method':method,'seconds':elapsed,'records':len(order),'optimizer_steps':steps,
            'epochs':1,'teacher_forward_token_positions':teacher_tokens,'target_tokens':target_tokens,'old_teacher_tokens':old_mask})
        dest=root/'adapters'/method;student.save_pretrained(dest)
        emit('training/'+method+'.json',{'losses':losses,'seed':seed,'order':order,'event':events[-1]})
        for fp in sorted(dest.glob('*')):
            if fp.is_file():emit('adapters/'+method+'/'+fp.name,fp.read_bytes(),True)
        del optimizer
        audit(student,method)
        del student,params;gc.collect();torch.cuda.empty_cache()
    emit('cost_events.json',events);emit('RESULT.json',{'run':run_id,'candidate':spec['candidate'],'summary':summaries,'prior_audit_reused':True})
    emit('artifact_manifest.json',index)
    print('R142_COMPLETED '+run_id,flush=True)

if __name__=='__main__':run()
