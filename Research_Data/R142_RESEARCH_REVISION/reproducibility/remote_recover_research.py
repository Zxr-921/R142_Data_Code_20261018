"""Read owned research job archives through the authorized Hugging Face SDK.

Only declared scientific artifacts are emitted. Authentication material and
job environment metadata are never printed or copied into artifacts.
"""
import os,sys,json,time,re,gzip,base64,hashlib,io,tarfile
from huggingface_hub import HfApi
api=HfApi(token=os.environ['HF_TOKEN'])

def emit(run,name,obj):
    raw=json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()
    packed=base64.b64encode(gzip.compress(raw,6)).decode()
    parts=[packed[i:i+12000] for i in range(0,len(packed),12000)]
    meta={'run':run,'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'chunks':len(parts)}
    print('R141_ARTIFACT_BEGIN '+json.dumps(meta),flush=True)
    for i,s in enumerate(parts):print(f'R141_ARTIFACT_CHUNK {run} {name} {i:05d}/{len(parts):05d} {s}',flush=True)
    print('R141_ARTIFACT_END '+json.dumps(meta),flush=True)

def current(job):
    # Complete runs are fetched once with an explicit server-side line limit.
    # Waiting happens in the remote recovery job, not the interactive session.
    for _ in range(240):
        status=api.inspect_job(job_id=job,namespace='jianlau7550').status.stage
        if status in {'COMPLETED','ERROR','CANCELED','DELETED'}:break
        time.sleep(30)
    groups={}
    for event in api.fetch_job_logs(job_id=job,namespace='jianlau7550',tail=100000,follow=False):
        for line in event.splitlines():
            m=re.search(r'R141_ARTIFACT_(BEGIN|END) (\{.*\})',line)
            if m:
                meta=json.loads(m[2]);key=(meta['run'],meta['name'])
                if key[1].endswith('.safetensors'):continue
                groups.setdefault(key,{'chunks':{}})['meta']=meta
                continue
            m=re.search(r'R141_ARTIFACT_CHUNK (\S+) (\S+) (\d+)/(\d+) ([A-Za-z0-9+/=]+)',line)
            if m and not m[2].endswith('.safetensors'):
                key=(m[1],m[2]);groups.setdefault(key,{'chunks':{}})['chunks'][int(m[3])]=m[5]
    recovered={};report=[];run='unknown'
    for (run,name),entry in groups.items():
        meta=entry.get('meta');n=meta['chunks'] if meta else None
        if not meta or set(entry['chunks'])!=set(range(n)):
            report.append({'name':name,'status':'incomplete','present':len(entry['chunks']),'expected':n});continue
        raw=gzip.decompress(base64.b64decode(''.join(entry['chunks'][i] for i in range(n))))
        assert hashlib.sha256(raw).hexdigest()==meta['sha256'],name
        try:obj=json.loads(raw)
        except (ValueError,UnicodeError):obj=raw.decode()
        recovered[name]=obj;report.append({**meta,'status':'verified'})
    # A completed runner already emits a compact record. Avoid nesting and
    # duplicating it inside another copy of the same scientific data.
    payload=recovered.get('scientific_record.json',recovered)
    emit(run,'scientific_record.json',payload)
    emit(run,'sdk_recovery_report.json',{'source_job':job,'job_status':status,'report':report,'checkpoint_payloads':'Separate original job artifacts; excluded from this compact scientific record.'})
    print('R141_SDK_RECOVERY_COMPLETE '+job,flush=True)

def legacy(job):
    chunks={};meta=None
    for event in api.fetch_job_logs(job_id=job,namespace='jianlau7550',tail=100000,follow=False):
        for line in event.splitlines():
            m=re.search(r'R138(?:_FINAL)?_ARCHIVE_(?:META|END) (\{.*\})',line)
            if m:meta=json.loads(m[1])
            m=re.search(r'R138(?:_FINAL)?_ARCHIVE_CHUNK (?:\S+ )?(\d+)/(\d+) ([A-Za-z0-9+/=]+)',line)
            if m:chunks[int(m[1])]=m[3]
    if not meta:
        emit('R138_'+job,'recovery_status.json',{'status':'missing_archive_metadata','job':job});return
    n=meta['chunks'];assert set(chunks)==set(range(n)),(job,len(chunks),n)
    raw=base64.b64decode(''.join(chunks[i] for i in range(n)))
    expected=meta.get('archive_sha256',meta.get('sha256'));assert hashlib.sha256(raw).hexdigest()==expected
    files={};index=[]
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            value=tar.extractfile(member).read()
            rec={'name':member.name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}
            try:
                text=value.decode('utf-8')
                try:files[member.name]=json.loads(text)
                except ValueError:files[member.name]=text
                rec['included']=True
            except UnicodeError:rec['included']=False
            index.append(rec)
    emit('R138_'+job,'scientific_record.json',files)
    emit('R138_'+job,'recovery_status.json',{'status':'verified','source_job':job,'archive_meta':meta,'index':index,'binary_payloads':'Hashes retained; omitted from this compact research-text archive.'})

if __name__=='__main__':
    if sys.argv[1]=='legacy':
        for job in sys.argv[2:]:legacy(job)
    else:
        for job in sys.argv[1:]:current(job)
