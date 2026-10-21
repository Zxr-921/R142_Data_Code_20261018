"""Check scientific artifact recovery, references, layout and render every page."""
from pathlib import Path
from collections import Counter
import json,re,hashlib,math
import fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript';QA=ROOT/'qa/final';QA.mkdir(parents=True,exist_ok=True)
STEMS=['R142_MAIN','R142_SUPPLEMENT','R142_COMBINED','R142_RESPONSE_TO_COMMENTS','R142_COVER_LETTER_DRAFT']
def expand(s):return re.sub(r'\\input\{([^}]+)\}',lambda m:expand((MAN/m[1]).read_text()),s)
def prose_count(s):
    s=re.sub(r'\\begin\{(?:figure|table|algorithmblock)\}.*?\\end\{(?:figure|table|algorithmblock)\}','',s,flags=re.S)
    s=re.sub(r'\\begin\{equation\}.*?\\end\{equation\}','',s,flags=re.S)
    s=re.sub(r'\\(?:cite|ref|label|nolinkurl)\{[^}]*\}','',s);s=re.sub(r'\\[sm]ref\{[^}]*\}\{([^}]*)\}',r'\1',s)
    s=re.sub(r'\$[^$]*\$','',s);s=re.sub(r'\\\w+[*]?','',s)
    return len(re.findall(r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)*",s))
def main():
    docs={name:fitz.open(MAN/(name+'.pdf')) for name in STEMS};names={name:doc.resolve_names() for name,doc in docs.items()};report={'documents':{},'link_errors':[]};contacts=[]
    for stem,doc in docs.items():
        if stem!='R142_COMBINED':
            for stale in QA.glob(stem+'_*.png'):stale.unlink()
        assert not doc.is_repaired and len(doc)>0
        log=(MAN/(stem+'.log')).read_text();assert not any(x in log for x in ['undefined','Overfull','multiply defined','does not exist']),stem
        counts=Counter();outside=[];critical=[];thin=[];thumbs=[];fonts=Counter()
        for i,page in enumerate(doc):
            assert abs(page.rect.width-612)<1 and abs(page.rect.height-792)<1
            text=page.get_text();words=len(text.split())
            if words<70:thin.append({'page':i+1,'words':words,'text':text[:160]})
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        box=fitz.Rect(span['bbox']);fonts[span['font']]+=len(span['text'])
                        if box.x0<0 or box.y0<0 or box.x1>612.2 or box.y1>792.2:outside.append({'page':i+1,'text':span['text'],'bbox':list(box)})
            for link in page.get_links():
                counts[str(link['kind'])]+=1
                if link['kind'] in [fitz.LINK_GOTO,fitz.LINK_NAMED]:
                    dest=link.get('nameddest');okay=(dest in names[stem]) if dest else 0<=link.get('page',-1)<len(doc)
                elif link['kind']==fitz.LINK_GOTOR:
                    target=Path(link.get('file','')).stem;dest=link.get('to')
                    okay=target in names and (dest in names[target] if isinstance(dest,str) else 0<=link.get('page',-1)<len(docs[target]))
                elif link['kind']==fitz.LINK_URI:okay=link.get('uri','').startswith(('https://','http://','mailto:'))
                elif link['kind']==fitz.LINK_LAUNCH and '#nameddest=' in link.get('file',''):
                    filename,dest=link['file'].split('#nameddest=',1);target=Path(filename).stem
                    # PyMuPDF exposes a named GoToR as a file-fragment link;
                    # inspect the actual PDF action before accepting it.
                    action=doc.xref_get_key(link['xref'],'A/S')[1]
                    okay=action=='/GoToR' and target in names and dest in names[target]
                else:okay=False
                if not okay:report['link_errors'].append({'file':stem,'page':i+1,'kind':link['kind'],'file_target':link.get('file'),'destination':str(link.get('to',link.get('nameddest'))),'uri':link.get('uri')})
            # Combined repeats the entire contents; every unique standalone page
            # is rendered and inspected, while every combined link is audited.
            if stem!='R142_COMBINED':
                pix=page.get_pixmap(matrix=fitz.Matrix(.55,.55),alpha=False);im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);thumbs.append(im)
                if stem=='R142_MAIN' or i<2 or i==len(doc)-1 or any(x in text for x in ['Fixed certificate prices','Algorithm 1.','Algorithm 4.','Complete old transitions','Same-weight batch-composition']):
                    page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(QA/f'{stem}_{i+1:03d}.png');critical.append(i+1)
        assert not outside,(stem,outside[:3])
        if thumbs:
            for start in range(0,len(thumbs),16):
                canvas=Image.new('RGB',(4*360,4*470),'#d3d3d3');draw=ImageDraw.Draw(canvas)
                for j,im in enumerate(thumbs[start:start+16]):
                    x=(j%4)*360;y=(j//4)*470;canvas.paste(im,(x+12,y+24));draw.text((x+12,y+5),f'{stem} p{start+j+1}',fill='black')
                p=QA/f'{stem}_contact_{start//16+1:02d}.png';canvas.save(p);contacts.append(str(p.relative_to(ROOT)))
        report['documents'][stem]={'pages':len(doc),'sha256':hashlib.sha256((MAN/(stem+'.pdf')).read_bytes()).hexdigest(),'link_counts':dict(counts),'named_destinations':len(names[stem]),'undefined_refs':0,'overfull_boxes':0,'outside_page':0,'thin_pages_to_review':thin,'full_pages_rendered':critical,'fonts':dict(fonts)}
    source=expand((MAN/'R142_COMBINED.tex').read_text());assert not re.search(r'\\\[|\\begin\{(?:equation|align|gather|multline|alignat)\*\}|\$\$',source)
    counts={'main_prose_excluding_captions_tables_equations':prose_count(expand((MAN/'R142_MAIN_BODY.tex').read_text())),'methods_prose_excluding_algorithm_blocks':prose_count((MAN/'R142_METHODS.tex').read_text()),'abstract_words':len((MAN/'R142_ABSTRACT.tex').read_text().split()),'main_figures':expand((MAN/'R142_MAIN_BODY.tex').read_text()).count(r'\begin{figure}'),'extended_data':(MAN/'R142_EXTENDED_DATA.tex').read_text().count(r'\begin{figure}'),'algorithms':(MAN/'R142_ALGORITHMS.tex').read_text().count(r'\begin{algorithmblock}'),'references':(MAN/'R142_REFERENCES.tex').read_text().count(r'\bibitem'),'max_refs_per_citation':max(len(x.split(',')) for x in re.findall(r'\\cite\w*\{([^}]+)\}',source))}
    assert counts['main_figures']==5 and counts['extended_data']==10 and counts['algorithms']==4
    assert 150<=counts['abstract_words']<=200
    assert counts['max_refs_per_citation']<=3
    assert not report['link_errors'],report['link_errors'][:10]
    report['format']=counts;report['contacts']=contacts
    (ROOT/'audit/R142_layout_verification.json').write_text(json.dumps(report,indent=2));(ROOT/'audit/R142_link_audit.json').write_text(json.dumps({'documents':{k:{'pages':v['pages'],'links':v['link_counts'],'destinations':v['named_destinations']} for k,v in report['documents'].items()},'broken_links':report['link_errors'],'section_routes':len(json.loads((ROOT/'audit/R142_SECTION_SUPPORT_MAP.json').read_text())),'formal_object_routes':len(json.loads((ROOT/'audit/R142_OBJECT_SUPPORT_MAP.json').read_text()))},indent=2))
    print(json.dumps({'format':counts,'pages':{k:v['pages'] for k,v in report['documents'].items()},'contacts':len(contacts),'broken_links':len(report['link_errors'])},indent=2))
if __name__=='__main__':main()
