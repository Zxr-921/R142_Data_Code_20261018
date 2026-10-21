"""Recover checksummed research artifacts from saved provider tool results."""
import json,base64,gzip,hashlib,re,sys
from pathlib import Path

def recover(log_json,destination):
    data=json.loads(Path(log_json).read_text())
    if 'structuredContent' in data:data=data['structuredContent']
    lines=data.get('outcome',data).get('lines',data if isinstance(data,list) else [])
    groups={}
    for event in lines:
        for line in event.splitlines():
            m=re.search(r'R142_ARTIFACT_(BEGIN|END) (\{.*\})',line)
            if m:
                meta=json.loads(m[2]);key=(meta['run'],meta['name'])
                groups.setdefault(key,{'chunks':{}})['meta']=meta
                continue
            m=re.search(r'R142_ARTIFACT_CHUNK (\S+) (\S+) (\d+)/(\d+) ([A-Za-z0-9+/=]+)',line)
            if m:
                key=(m[1],m[2]);groups.setdefault(key,{'chunks':{}})['chunks'][int(m[3])]=m[5]
    result=[]
    out=Path(destination)
    for (run,name),entry in groups.items():
        meta=entry.get('meta');n=meta['chunks'] if meta else None
        if meta is None or set(entry['chunks'])!=set(range(n)):
            result.append({'run':run,'name':name,'status':'incomplete','present':len(entry['chunks']),'expected':n});continue
        raw=gzip.decompress(base64.b64decode(''.join(entry['chunks'][i] for i in range(n))))
        assert hashlib.sha256(raw).hexdigest()==meta['sha256'], name
        assert len(raw)==meta['bytes']
        fp=(out/run/name).resolve();assert fp.is_relative_to(out.resolve())
        fp.parent.mkdir(parents=True,exist_ok=True);fp.write_bytes(raw)
        if name=='scientific_record.json':
            for rel,obj in json.loads(raw).items():
                p=(out/run/rel).resolve();assert p.is_relative_to((out/run).resolve())
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(obj if isinstance(obj,str) and not p.suffix=='.json' else json.dumps(obj,sort_keys=True,ensure_ascii=False))
        result.append({**meta,'status':'verified'})
    return result

if __name__=='__main__':
    results=recover(sys.argv[1],sys.argv[2]);report=Path(sys.argv[1]).with_suffix('.recovery.json');report.write_text(json.dumps(results,indent=2))
    print(json.dumps({'verified':sum(x['status']=='verified' for x in results),'incomplete':sum(x['status']!='verified' for x in results),'report':str(report)}))
