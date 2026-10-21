"""Compile from an isolated staging directory and atomically install valid PDFs."""
from pathlib import Path
import shutil,subprocess,fitz,json,hashlib
ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript';BUILD=Path('/tmp/r142_typeset')
def main():
    if BUILD.exists():shutil.rmtree(BUILD)
    BUILD.mkdir()
    for p in MAN.glob('R142*.tex'):shutil.copy2(p,BUILD/p.name)
    (BUILD/'figures').mkdir()
    for p in (MAN/'figures').glob('*.pdf'):shutil.copy2(p,BUILD/'figures'/p.name)
    sequence=['R142_COMBINED','R142_COMBINED','R142_MAIN','R142_SUPPLEMENT','R142_MAIN','R142_SUPPLEMENT']
    for optional in ['R142_RESPONSE_TO_COMMENTS','R142_COVER_LETTER_DRAFT']:
        if (BUILD/(optional+'.tex')).exists():sequence.extend([optional,optional])
    for i,name in enumerate(sequence):
        proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=BUILD,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (BUILD/(name+f'_pass{i}.stdout')).write_text(proc.stdout)
        if proc.returncode:raise RuntimeError(proc.stdout[-2400:])
        doc=fitz.open(BUILD/(name+'.pdf'));assert len(doc)>0 and not doc.is_repaired
        assert '\\gdef \\@abspage@last{' in (BUILD/(name+'.aux')).read_text()
    # xr-hyper emits local-file fragment links. Turn these into native PDF
    # GoToR actions with named destinations, supported by standard PDF readers.
    for name in ['R142_MAIN','R142_SUPPLEMENT']:
        path=BUILD/(name+'.pdf');doc=fitz.open(path)
        for page in doc:
            for link in page.get_links():
                file=link.get('file','')
                if '#nameddest=' not in file:continue
                filename,destination=file.split('#nameddest=',1)
                target=fitz.open(BUILD/filename);assert destination in target.resolve_names(),(name,filename,destination)
                page.update_link({'kind':fitz.LINK_GOTOR,'xref':link['xref'],'from':link['from'],'file':filename,'page':-1,'to':destination})
        temp=path.with_suffix('.fixed.pdf');doc.save(temp,garbage=4,deflate=True);doc.close();temp.replace(path)
    result=[]
    for name in dict.fromkeys(sequence):
        p=BUILD/(name+'.pdf');d=fitz.open(p)
        for ext in ['pdf','tex','aux','log']:
            source=BUILD/(name+'.'+ext);dest=MAN/source.name;tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_bytes(source.read_bytes());tmp.replace(dest)
        result.append({'file':p.name,'pages':len(d),'bytes':len(p.read_bytes()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (ROOT/'audit/R142_typeset.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
