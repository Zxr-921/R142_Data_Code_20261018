"""Package current PDFs, all scientific evidence and reproducible sources."""
from pathlib import Path
import csv, hashlib, json, re, shutil, zipfile
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent/'R142_DELIVERABLES'
STEMS = ['R142_MAIN','R142_SUPPLEMENT','R142_COMBINED',
         'R142_RESPONSE_TO_COMMENTS','R142_COVER_LETTER_DRAFT']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def included(path):
    rel=path.relative_to(ROOT)
    if '__pycache__' in rel.parts or path.suffix in {'.pyc','.aux','.log','.out','.toc','.fls','.fdb_latexmk','.synctex'}: return False
    if path.name=='FILE_MANIFEST_SHA256.csv' or path.name.endswith('.tmp'): return False
    if rel.parts[0]=='qa':
        if len(rel.parts)<3 or rel.parts[1]!='final': return False
        return ('contact_' in path.name or path.name in {
            'R142_MAIN_001.png','R142_MAIN_013.png','R142_MAIN_020.png','R142_MAIN_021.png',
            'R142_SUPPLEMENT_001.png','R142_COMBINED_035.png','R142_COMBINED_047.png',
            'R142_COVER_LETTER_DRAFT_001.png'})
    return True
def main():
    OUT.mkdir(exist_ok=True)
    audit=json.loads((ROOT/'audit/R142_RELEASE_AUDIT.json').read_text())
    layout=json.loads((ROOT/'audit/R142_layout_verification.json').read_text())
    assert audit['recovered_artifacts']==210 and not layout['link_errors']
    for stem in STEMS:
        source=ROOT/'manuscript'/(stem+'.pdf')
        assert sha(source)==layout['documents'][stem]['sha256']
        shutil.copy2(source,OUT/source.name)
    shutil.copy2(ROOT/'R142_REVISION_REPORT_CN.md',OUT/'R142_REVISION_REPORT_CN.md')
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and included(p))
    manifest=ROOT/'FILE_MANIFEST_SHA256.csv'
    with manifest.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','bytes','sha256']);writer.writeheader()
        for p in files:writer.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    target=OUT/'R142_COMPLETE_REVISION.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for p in files+[manifest]:archive.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        table=list(csv.DictReader(archive.read(ROOT.name+'/FILE_MANIFEST_SHA256.csv').decode().splitlines()))
        assert len(table)==len(files) and len(archive.namelist())==len(files)+1
        for row in table:
            raw=archive.read(ROOT.name+'/'+row['path'])
            assert len(raw)==int(row['bytes']) and hashlib.sha256(raw).hexdigest()==row['sha256']
    outputs=[OUT/(stem+'.pdf') for stem in STEMS]+[OUT/'R142_REVISION_REPORT_CN.md',target]
    report={'zip_files':len(files)+1,'zip_crc_and_every_manifest_hash_valid':True,'deliverables':[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in outputs]}
    (OUT/'R142_DELIVERY_VERIFICATION.json').write_text(json.dumps(report,indent=2))
    (OUT/'R142_SAVE_REQUEST.json').write_text(json.dumps({'uploads':[{'local_path':str(p),'purpose':'create_library_file'} for p in outputs]},indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
