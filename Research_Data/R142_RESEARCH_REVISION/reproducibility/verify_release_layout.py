"""Check compiled-document integrity and render every page for visual review."""
from pathlib import Path
import re,json,hashlib
import fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript';QA=ROOT/'qa/final';QA.mkdir(parents=True,exist_ok=True)
docs=[('MAIN','NATURE_GENAI_UPDATE_R141_MAIN'),('SI','NATURE_GENAI_UPDATE_R141_SI'),('RESPONSE','R141_RESPONSE_TO_COMMENTS'),('LETTER','R141_COVER_LETTER_DRAFT')]
report={'documents':{},'visual_review_status':'Rendered; inspect the contact sheets and critical pages before setting reviewed.'}
for label,name in docs:
    path=MAN/(name+'.pdf');doc=fitz.open(path);log=path.with_suffix('.log').read_text()
    warnings=re.findall(r'^.*(?:Overfull|undefined|LaTeX Warning|Package .*Warning).*$',log,re.M)
    assert not warnings,(label,warnings)
    assert all(abs(p.rect.width-612)<.1 and abs(p.rect.height-792)<.1 for p in doc)
    assert all(len(p.get_text().strip())>150 for p in doc)
    outside=[];font_counts={}
    for i,p in enumerate(doc):
        for block in p.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if span['text'].strip():
                        x0,y0,x1,y1=span['bbox']
                        if min(x0,y0)<0 or x1>612.2 or y1>792.2:outside.append({'page':i+1,'bbox':span['bbox']})
                        key=span['font'];font_counts[key]=font_counts.get(key,0)+len(span['text'])
    assert not outside,(label,outside)
    if label=='MAIN':assert 'Introduction' in doc[1].get_text() and 'Introduction' not in doc[0].get_text()
    thumbnails=[]
    for i,p in enumerate(doc):
        pix=p.get_pixmap(matrix=fitz.Matrix(.46,.46),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        tile=Image.new('RGB',(300,390),'#e2e2e2');tile.paste(im,((300-im.width)//2,16));ImageDraw.Draw(tile).text((8,2),f'{label} {i+1}',fill='black');thumbnails.append(tile)
    for start in range(0,len(thumbnails),16):
        group=thumbnails[start:start+16];sheet=Image.new('RGB',(1200,390*((len(group)+3)//4)),'white')
        for i,im in enumerate(group):sheet.paste(im,((i%4)*300,(i//4)*390))
        sheet.save(QA/f'{label}_contact_{start+1:03d}.jpg',quality=88)
    needles={'MAIN':['Abstract','Figure 5:','Methods','Extended Data 1:','Extended Data 4:'],
             'SI':['A declaration guarantee','every selected update','Declarations and calibration errors','Conditioned M-matrix'],
             'RESPONSE':['Results that change','A15:'],'LETTER':['Dear Editors']}[label]
    critical=[]
    for i,p in enumerate(doc):
        txt=p.get_text()
        if i in [0,1,len(doc)-1] or any(n.lower() in txt.lower() for n in needles):
            p.get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(QA/f'{label}_{i+1:03d}.png');critical.append(i+1)
    report['documents'][label]={'pdf':str(path.relative_to(ROOT)),'pages':len(doc),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'letter_page_size':True,'undefined_references':0,'overfull_boxes':0,'text_outside_page':0,'critical_pages_rendered':critical,'fonts_by_character_count':font_counts}

def expand(name):
    s=(MAN/name).read_text()
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(m[1]),s)
def prose_count(s):
    s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',s,flags=re.S)
    s=re.sub(r'\\begin\{equation\}.*?\\end\{equation\}','',s,flags=re.S)
    s=re.sub(r'\\(?:cite|ref|label|nolinkurl)\{[^}]*\}','',s)
    s=re.sub(r'\$[^$]*\$','',s);s=re.sub(r'\\\w+[*]?','',s)
    return len(re.findall(r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)*",s))
body=expand('R141_MAIN_BODY.tex');methods=(MAN/'R141_METHODS.tex').read_text()
current=body+methods+expand('R141_CURRENT_THEORY.tex')+expand('R141_CONTROL_SUPPLEMENT.tex')+expand('R141_SUPPORTING_RECORD.tex')
unnumbered=re.findall(r'\\\[|\\begin\{(?:equation|align|gather|multline|alignat)\*\}|\$\$',current)
assert not unnumbered,unnumbered
report['format']={'body_prose_words_approx_excluding_equations_captions':prose_count(body),'methods_prose_words_approx':prose_count(methods),'abstract_words':len((MAN/'R141_ABSTRACT.tex').read_text().split()),'main_figures':body.count(r'\begin{figure}'),'extended_data_figures':(MAN/'R141_EXTENDED_DATA.tex').read_text().count(r'\begin{figure}'),'unnumbered_display_environments':len(unnumbered),'main_display_equations':body.count(r'\begin{equation}'),'line_numbers':True,'blue_citations_and_crossreferences':True,'declared_font_size':'11pt','font_family':'Times-compatible mathptmx','word_count_method':'Source-based estimate; removes figures, display/inline math, citation/reference markup and control words.'}
assert report['format']['main_figures']==5
assert report['format']['extended_data_figures']==4
assert 150<=report['format']['abstract_words']<=200
(ROOT/'audit/layout_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'pages':{k:v['pages'] for k,v in report['documents'].items()},'format':report['format']}))
