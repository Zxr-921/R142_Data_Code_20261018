import os, re, json, math, time, uuid, random, hashlib, gzip, base64, gc, traceback
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_file

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.cli_role import CliUser, InteractiveMessage
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.common.execution_context import RoleType, DatabaseNamespace, get_current_context
from tool_sandbox.common.message_conversion import Message, to_openai_messages, openai_tool_call_to_python_code
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.common.tool_discovery import ToolBackend

# ---------------- frozen constants ----------------
PROTOCOL_SHA='71a0553c27ccb88c50b53e65e1ec71d914ccf4ec6d5dab8c2f6a7e12c3132d91'
TOOLSANDBOX_COMMIT='c8571d7854316d2e1c5f288e59fe1e34e53f6dd1'
MODEL='NousResearch/Hermes-3-Llama-3.2-3B'
DATASET_REPO='Adorg/ToolBench'
DATASET_REV='0c25df5e6937aae854c2cc1bb203297600afce8b'
DATASET_FILE='toolllama_G123_dfs_train.json'
DATASET_SHA='ffa0a7c5fef95b5693ccffdf19c30d2d2baff963b9df35cfd31dd4f42a9dbc78'
SEEDS={'U1':55101,'U2':55102,'U3':55103,'U4':55104}
MAXLEN=1536
BATCH=1
GRAD_ACCUM=4
LR=2e-4
EPOCHS=1
LORA_R=8
LORA_ALPHA=16
MAX_NEW_TOKENS=192
STRENGTHS=[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0]
RADII=[0.25,0.5,0.75,1.0]
PILOT_S=0.05
ALPHA=0.10
VAR_SUFFIXES=['','_10_distraction_tools','_3_distraction_tools','_3_distraction_tools_arg_description_scrambled','_3_distraction_tools_arg_type_scrambled','_3_distraction_tools_tool_description_scrambled','_3_distraction_tools_tool_name_scrambled','_all_tools']

SPLITS={
'old_transfer_fit':['search_sender_phone_number_with_content','search_phone_number_with_name','cellular_off','search_relationship_with_phone_number'],
'old_transfer_select':['update_contact_with_id_and_phone_number','find_thanksgiving_timestamp','get_wifi'],
'old_fresh':['search_name_with_relationship','send_message_with_phone_number_and_content','remove_contact_with_id','get_cellular'],
'new_capability_validation':['search_reminder_with_creation_recency_yesterday','modify_contact_with_message_recency','search_message_with_recency_latest','add_reminder_content_and_date_and_time_alt','add_reminder_content_and_week_delta_and_time','search_message_with_recency_latest_alt','search_message_with_recency_oldest_alt','turn_on_cellular_low_battery_mode','turn_on_wifi_low_battery_mode_implicit','add_reminder_content_and_week_delta_and_time_alt','search_reminder_with_recency_yesterday','search_reminder_with_creation_recency_yesterday_implicit','remove_reminder_with_recency_latest','add_reminder_content_and_weekday_delta_and_time','turn_on_wifi_low_battery_mode','turn_on_cellular_low_battery_mode_implicit','find_days_till_holiday','update_contact_relationship_with_relationship_alt','remove_contact_by_phone_alt'],
'new_capability_audit':['add_reminder_content_and_weekday_delta_and_time_alt','search_reminder_with_recency_upcoming','modify_contact_with_message_recency_alt','search_message_with_recency_oldest','remove_contact_by_phone_ambiguous','find_days_till_holiday_wifi_off_alt','remove_contact_by_phone_ambiguous_alt','turn_on_location_low_battery_mode','search_reminder_with_recency_upcoming_implicit','send_message_with_contact_content_cellular_off_alt','update_contact_relationship_with_relationship','find_days_till_holiday_alt','modify_reminder_with_recency_latest_alt','add_reminder_content_and_date_and_time','turn_on_location_low_battery_mode_implicit','search_reminder_with_recency_yesterday_implicit','remove_reminder_with_recency_latest_alt','find_days_till_holiday_wifi_off','modify_reminder_with_recency_latest']
}

OUT=Path('/tmp/r138_claim'); OUT.mkdir(parents=True,exist_ok=True)
LEDGER=OUT/'R138_SCENARIO_LEDGER.jsonl'
SUMMARY=OUT/'R138_RESULT.json'

# Make runtime as reproducible as feasible.
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
try: torch.use_deterministic_algorithms(True, warn_only=False)
except Exception: pass

print('R138_PROTOCOL_SHA256',PROTOCOL_SHA,flush=True)
print('R138_RECIPE',json.dumps({'maxlen':MAXLEN,'batch':BATCH,'grad_accum':GRAD_ACCUM,'lr':LR,'epochs':EPOCHS,'lora_r':LORA_R,'lora_alpha':LORA_ALPHA,'max_new_tokens':MAX_NEW_TOKENS,'strengths':STRENGTHS,'radii':RADII,'pilot_s':PILOT_S},sort_keys=True),flush=True)

# ---------------- utilities ----------------
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sha_text(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

def append_ledger(obj):
    with open(LEDGER,'a',encoding='utf-8') as f: f.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
    # Claim-bearing raw line is also persisted to provider logs.
    print('R138_LEDGER',json.dumps(obj,ensure_ascii=False,separators=(',',':')),flush=True)

def chatml(role,content): return f'<|im_start|>{role}\n{content}<|im_end|>\n'

FC_SCHEMA='{"properties":{"arguments":{"title":"Arguments","type":"object"},"name":{"title":"Name","type":"string"}},"required":["arguments","name"],"title":"FunctionCall","type":"object"}'

# ---------------- data ----------------
api=HfApi()
try:
    model_revision=api.model_info(MODEL).sha
except Exception as e:
    model_revision='UNKNOWN:'+type(e).__name__
print('MODEL_REVISION',model_revision,flush=True)

data_path=hf_hub_download(repo_id=DATASET_REPO,filename=DATASET_FILE,repo_type='dataset',revision=DATASET_REV)
actual_sha=hashlib.sha256(open(data_path,'rb').read()).hexdigest()
assert actual_sha==DATASET_SHA,(actual_sha,DATASET_SHA)
rows=json.load(open(data_path,encoding='utf-8'))
assert len(rows)==26999
order=[]
for idx,row in enumerate(rows):
    rid=str(row.get('id',''))
    h=hashlib.sha256(f'R138V2|train|{rid}'.encode()).hexdigest()
    order.append((h,idx,rid))
order.sort(key=lambda x:(x[0],x[1]))
chosen=order[:600]
train_rows=[rows[i] for _,i,_ in chosen]
selected_ids=[rid for _,_,rid in chosen]
selected_hash=hashlib.sha256(('\n'.join(selected_ids)+'\n').encode()).hexdigest()
print('TRAIN_SELECTED_N',len(train_rows),'TRAIN_SELECTED_IDS_SHA256',selected_hash,flush=True)
(OUT/'R138_TRAIN_SELECTED_IDS.txt').write_text('\n'.join(selected_ids)+'\n',encoding='utf-8')

# ---------------- tokenizer and training examples ----------------
tok=AutoTokenizer.from_pretrained(MODEL,revision=model_revision if not str(model_revision).startswith('UNKNOWN') else None)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token

def convert_assistant(v):
    # Convert ToolBench Action/Action Input output into Hermes XML when parsable.
    ma=re.search(r'Action\s*:\s*([^\n]+)',v)
    mi=re.search(r'Action Input\s*:\s*(\{.*\})\s*$',v,re.S)
    if ma and mi:
        name=ma.group(1).strip()
        try: args=json.loads(mi.group(1))
        except Exception: args=None
        if isinstance(args,dict):
            return '<tool_call>\n'+json.dumps({'arguments':args,'name':name},ensure_ascii=False,separators=(',',':'))+'\n</tool_call>'
    return v

def encode_row(row):
    ids=[]; labs=[]
    for m in row['conversations']:
        role=m.get('from'); val=m.get('value','')
        if role=='assistant': content=convert_assistant(val); rr='assistant'; learn=True
        elif role=='function': content='<tool_response>\n'+val+'\n</tool_response>'; rr='tool'; learn=False
        elif role=='system': content=val; rr='system'; learn=False
        else: content=val; rr='user'; learn=False
        seg=tok(chatml(rr,content),add_special_tokens=False).input_ids
        ids.extend(seg); labs.extend(seg if learn else [-100]*len(seg))
    # Keep the tail: ToolBench puts tool schemas near the end of the long system message and assistant turns follow it.
    if len(ids)>MAXLEN:
        ids=ids[-MAXLEN:]; labs=labs[-MAXLEN:]
    if all(x==-100 for x in labs):
        # deterministic fallback: supervise the final 128 tokens rather than silently dropping a frozen row
        k=min(128,len(ids)); labs[-k:]=ids[-k:]
    return {'input_ids':ids,'labels':labs}

encoded=[encode_row(x) for x in train_rows]
print('TRAIN_TOKEN_STATS',json.dumps({'min':min(map(lambda x:len(x['input_ids']),encoded)),'max':max(map(lambda x:len(x['input_ids']),encoded)),'mean':sum(len(x['input_ids']) for x in encoded)/len(encoded)},sort_keys=True),flush=True)

class EncDataset(Dataset):
    def __len__(self): return len(encoded)
    def __getitem__(self,i): return encoded[i]

def collate(batch):
    L=max(len(x['input_ids']) for x in batch)
    inp=[]; lab=[]; att=[]
    for x in batch:
        n=len(x['input_ids']); pad=L-n
        inp.append(x['input_ids']+[tok.pad_token_id]*pad)
        lab.append(x['labels']+[-100]*pad)
        att.append([1]*n+[0]*pad)
    return {'input_ids':torch.tensor(inp,dtype=torch.long),'labels':torch.tensor(lab,dtype=torch.long),'attention_mask':torch.tensor(att,dtype=torch.long)}

# ---------------- ToolSandbox adapter ----------------
class AutoEndUser(CliUser):
    def __init__(self): pass
    def user_input(self): return InteractiveMessage(tool_call='end_conversation()')

class Hermes3ToolAgent(BaseRole):
    role_type=RoleType.AGENT
    def __init__(self,model_obj,scale=1.0,capture=None): self.model_obj=model_obj; self.scale=scale; self.capture=capture if capture is not None else []
    def build_prompt(self,messages,tools):
        tooltxt=' '.join(json.dumps(t,separators=(',',':')) for t in tools)
        sys=("You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. "
             "You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. "
             f"Here are the available tools: <tools> {tooltxt} </tools> "
             f"Use the following pydantic model json schema for each tool call you will make: {FC_SCHEMA} "
             "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
             '<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call>')
        p=chatml('system',sys)
        for m in messages:
            role=m['role']
            if role=='system': p+=chatml('system',m.get('content',''))
            elif role=='user': p+=chatml('user',m.get('content',''))
            elif role=='assistant':
                if m.get('tool_calls'):
                    s=''
                    for tc in m['tool_calls']:
                        ff=tc['function']; args=ff.get('arguments',{})
                        if isinstance(args,str):
                            try: args=json.loads(args)
                            except Exception: pass
                        s+='<tool_call>\n'+json.dumps({'arguments':args,'name':ff['name']},ensure_ascii=False,separators=(',',':'))+'\n</tool_call>\n'
                    p+=chatml('assistant',s.rstrip())
                else: p+=chatml('assistant',m.get('content','') or '')
            elif role=='tool': p+=chatml('tool','<tool_response>\n'+json.dumps({'name':m.get('name'),'content':m.get('content')},ensure_ascii=False)+'\n</tool_response>')
        return p+'<|im_start|>assistant\n'
    def parse_calls(self,text):
        blocks=re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>',text,flags=re.S)
        objs=[]
        for b in blocks:
            try: objs.append(json.loads(b))
            except Exception: pass
        if not objs:
            # bounded balanced-brace fallback
            starts=[m.start() for m in re.finditer(r'\{',text)]
            for st in starts:
                depth=0
                for j in range(st,len(text)):
                    if text[j]=='{': depth+=1
                    elif text[j]=='}':
                        depth-=1
                        if depth==0:
                            try: x=json.loads(text[st:j+1])
                            except Exception: break
                            if isinstance(x,dict) and isinstance(x.get('name'),str) and 'arguments' in x: objs=[x]
                            break
                if objs: break
        out=[]
        for x in objs:
            args=x.get('arguments',{})
            if isinstance(args,str):
                try: args=json.loads(args)
                except Exception: continue
            if isinstance(args,dict): out.append((x['name'],args))
        return out
    def respond(self,ending_index=None):
        msgs=self.get_messages(ending_index=ending_index); self.messages_validation(msgs); msgs=self.filter_messages(msgs)
        if msgs[-1].sender==RoleType.SYSTEM: return
        available=self.get_available_tools(); names=set(available.keys()); tools=convert_to_openai_tools(available)
        om,_=to_openai_messages(msgs); prompt=self.build_prompt(om,tools)
        input_ids=tok(prompt,return_tensors='pt',truncation=True,max_length=7168).to('cuda')
        with torch.no_grad():
            out=self.model_obj.generate(**input_ids,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0,input_ids['input_ids'].shape[1]:],skip_special_tokens=False)
        if '<|im_end|>' in text: text=text.split('<|im_end|>',1)[0]
        calls=self.parse_calls(text); rs=[]; codes=[]
        if calls:
            ctx=get_current_context()
            for j,(name,args) in enumerate(calls,1):
                cid=f'call_{uuid.uuid4().hex[:10]}_{j}'
                tc=ChatCompletionMessageToolCall(id=cid,type='function',function=Function(name=name,arguments=json.dumps(args)))
                try:
                    ename=ctx.get_execution_facing_tool_name(name)
                    code=openai_tool_call_to_python_code(tc,names,ename)
                    rs.append(Message(sender=RoleType.AGENT,recipient=RoleType.EXECUTION_ENVIRONMENT,content=code,openai_tool_call_id=cid,openai_function_name=name)); codes.append(code)
                except Exception:
                    pass
        if not rs:
            clean=re.sub(r'<\|[^>]+\|>','',text).strip()
            rs=[Message(sender=RoleType.AGENT,recipient=RoleType.USER,content=clean or 'Unable to complete.')]
        self.capture.append({'prompt_sha256':sha_text(prompt),'prompt':prompt if len(self.capture)==0 else None,'tools_sha256':sha_text(json.dumps(tools,sort_keys=True,default=str)),'generated':text,'calls':calls,'executed_code':codes})
        self.add_messages(rs)

scenarios=named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
for split,bases in SPLITS.items():
    for b in bases:
        for suf in VAR_SUFFIXES:
            assert b+suf in scenarios,(split,b,suf)
print('SCIENTIFIC_VARIANTS_ASSERTED',flush=True)

def milestone_primary_state_indices(scenario):
    inds=[]
    for i,m in enumerate(scenario.evaluation.milestone_matcher.milestones):
        primary=False
        for c in m.snapshot_constraints:
            fn=getattr(c.snapshot_constraint,'__name__','')
            if str(c.database_namespace)!='SANDBOX' and fn!='guardrail_similarity': primary=True
        if primary: inds.append(i)
    return inds

def mapping_sim(mapping,i):
    v=mapping.get(i,mapping.get(str(i))) if isinstance(mapping,dict) else None
    if v is None: return None
    if isinstance(v,(list,tuple)) and len(v)>=2: return float(v[1])
    if isinstance(v,dict):
        for k in ['similarity','score','milestone_similarity']:
            if k in v: return float(v[k])
    try: return float(v)
    except Exception: return None

def compact_final_state():
    ctx=get_current_context(); out={}
    for ns in [DatabaseNamespace.SETTING,DatabaseNamespace.CONTACT,DatabaseNamespace.MESSAGING,DatabaseNamespace.REMINDER]:
        try:
            df=ctx.get_database(ns,get_all_history_snapshots=False,drop_sandbox_message_index=False)
            # Final-state rows, capped only by benchmark's tiny local DBs.
            out[str(ns)]=df.to_dicts()
        except Exception as e: out[str(ns)]={'error':type(e).__name__}
    return out

def run_one(name,model_obj,phase,update,strength,save_prompt=False):
    capture=[]; agent=Hermes3ToolAgent(model_obj,strength,capture)
    roles={RoleType.USER:AutoEndUser(),RoleType.EXECUTION_ENVIRONMENT:ExecutionEnvironment(),RoleType.AGENT:agent}
    t0=time.time(); err=None
    try:
        r=scenarios[name].play_and_evaluate(roles,OUT/'tool_sandbox_runs',f'{phase}__{update}__s{strength}__{name}')
        e=r.evaluation_result
        pinds=milestone_primary_state_indices(scenarios[name])
        psims=[mapping_sim(e.milestone_mapping,i) for i in pinds]
        exact_state=bool(pinds) and all(x is not None and x>=1-1e-8 for x in psims)
        row={'kind':'scenario','phase':phase,'update':update,'strength':strength,'scenario':name,'similarity':float(e.similarity),'milestone_similarity':float(e.milestone_similarity),'minefield_similarity':float(e.minefield_similarity),'turn_count':int(e.turn_count),'primary_state_indices':pinds,'primary_state_sims':psims,'exact_state_success':exact_state,'milestone_mapping':e.milestone_mapping,'minefield_mapping':e.minefield_mapping,'agent_steps':capture,'final_state':compact_final_state(),'elapsed_sec':time.time()-t0}
    except Exception as ex:
        err=traceback.format_exc(limit=5)
        row={'kind':'scenario','phase':phase,'update':update,'strength':strength,'scenario':name,'similarity':0.0,'milestone_similarity':0.0,'minefield_similarity':0.0,'turn_count':None,'primary_state_indices':milestone_primary_state_indices(scenarios[name]),'primary_state_sims':[],'exact_state_success':False,'agent_steps':capture,'error':type(ex).__name__,'traceback':err,'elapsed_sec':time.time()-t0}
    finally:
        for role in roles.values():
            try: role.teardown()
            except Exception: pass
    append_ledger(row); return row

def variants(bases): return [b+s for b in bases for s in VAR_SUFFIXES]

# ---------------- LoRA scaling / margin ----------------
def lora_layers(model_obj):
    for m in model_obj.modules():
        if isinstance(m,LoraLayer) and hasattr(m,'scaling') and 'default' in m.scaling: yield m

@contextmanager
def adapter_strength(model_obj,s):
    saved=[]
    for m in lora_layers(model_obj):
        old=m.scaling['default']; saved.append((m,old)); m.scaling['default']=old*float(s)
    try: yield
    finally:
        for m,old in saved: m.scaling['default']=old

def canonical_call_text(call):
    name,args=call
    return '<tool_call>\n'+json.dumps({'arguments':args,'name':name},ensure_ascii=False,separators=(',',':'))+'\n</tool_call>'

def teacher_margin(model_obj,prompt,target,s):
    pids=tok(prompt,add_special_tokens=False).input_ids; tids=tok(target,add_special_tokens=False).input_ids
    maxctx=7168
    if len(pids)+len(tids)>maxctx: pids=pids[-(maxctx-len(tids)):]
    ids=torch.tensor([pids+tids],device='cuda')
    with adapter_strength(model_obj,s), torch.no_grad(): logits=model_obj(ids[:,:-1],use_cache=False).logits[0].float()
    start=len(pids)-1; margins=[]
    for j,tid in enumerate(tids):
        pos=start+j
        z=logits[pos]; corr=float(z[tid]); z2=z.clone(); z2[tid]=-float('inf'); alt=float(z2.max()); margins.append(corr-alt)
    return min(margins) if margins else float('nan')

def delta_norm_from_model(model_obj):
    ss=0.0
    for m in lora_layers(model_obj):
        A=m.lora_A['default'].weight.detach().float().cpu(); B=m.lora_B['default'].weight.detach().float().cpu(); scale=float(m.scaling['default'])
        d=(B@A)*scale; ss+=float((d*d).sum())
    return math.sqrt(ss)

# ---------------- model train/eval ----------------
def load_base(train=False):
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_revision if not str(model_revision).startswith('UNKNOWN') else None,torch_dtype=torch.float16,device_map={'':0},attn_implementation='eager')
    model.config.use_cache=not train
    if train:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    return model

def train_candidate(name,seed):
    seed_all(seed); base=load_base(train=True)
    cfg=LoraConfig(r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=0.0,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(base,cfg); model.train()
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
        print('TRAIN_EPOCH',name,ep,'mean_loss',sum(losses)/len(losses),'opt_steps',optsteps,flush=True)
    path=OUT/name; model.save_pretrained(path,safe_serialization=True)
    norm=delta_norm_from_model(model)
    model.eval(); model.config.use_cache=True
    vals=[]
    for sn in variants(SPLITS['new_capability_validation']):
        with adapter_strength(model,1.0): row=run_one(sn,model,'cap_val',name,1.0)
        vals.append(row['similarity'])
    score=float(np.mean(vals))
    print('CAND',name,json.dumps({'cap_val':score,'delta_norm':norm,'seed':seed}),flush=True)
    del model,base,opt,dl; gc.collect(); torch.cuda.empty_cache()
    return {'cap_val':score,'delta_norm':norm,'seed':seed,'adapter':str(path)}


def adapter_sha256(path):
    files=sorted(Path(path).glob('*'))
    h=hashlib.sha256()
    for fp in files:
        if fp.is_file():
            h.update(fp.name.encode('utf-8')); h.update(b'\\0'); h.update(fp.read_bytes()); h.update(b'\\0')
    return h.hexdigest()

name=os.environ['R138_CANDIDATE']
assert name in SEEDS, name
result=train_candidate(name,SEEDS[name])
result['adapter_sha256']=adapter_sha256(result['adapter'])
result['protocol_sha256']=PROTOCOL_SHA
result['train_selected_ids_sha256']=selected_hash
result['model_revision']=model_revision
result['dataset_sha256']=actual_sha
result['ledger_sha256']=hashlib.sha256(LEDGER.read_bytes()).hexdigest() if LEDGER.exists() else hashlib.sha256(b'').hexdigest()
print('PARALLEL_CANDIDATE_RESULT',name,json.dumps(result,sort_keys=True),flush=True)
