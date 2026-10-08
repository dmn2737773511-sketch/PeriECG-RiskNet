from pathlib import Path
# Reuse the standalone-panel renderer, without executing the seven main figures.
src=Path(__file__).with_name('build_panels.py').read_text().split('# FIGURE 1:')[0]
exec(compile(src,'panel_renderer','exec'))
from lxml import etree
NS='http://www.w3.org/2000/svg'
leads=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
for code in [8,9]:layouts[code]=[3,3,3,3]
for number,code,state,cols,bandlabels in [(1,8,'raw',['low_rms_uV','line_rms_uV'],['L','P']),(2,9,'filtered',['slow_rms_uV','fast_rms_uV'],['S','F'])]:
 for k,lead in enumerate(leads):
  letter=string.ascii_lowercase[k];fig,ax=panel(code,letter);vals=[];labels=[]
  for col,bl in zip(cols,bandlabels):
   for group in G:
    v=spec[(spec.lead==lead)&(spec.group==group)&(spec.preprocessing==state)].sort_values('phase')[col].values;vals.append(np.log10(v));labels.append(group+' '+bl)
  im=heat(fig,ax,np.array(vals),list(range(1,7)),labels,cbar=True)
  xy(ax,'Acquisition stage','Source / band');ax.text(.98,1.025,lead,transform=ax.transAxes,ha='right',fontsize=8,fontweight='bold');ax.tick_params(axis='y',labelsize=6)
  done(fig,code,letter,f'All-stage {state} component amplitudes for {lead}','spectral_profiles.csv; log10(RMS in µV)')
  manifest[-1]['figure']=f'S{number}';manifest[-1]['panel_file']=f'FigS{number}{letter}.svg'
  for ext in ['pdf','png','svg']:(P/f'Fig{code}{letter}.{ext}').rename(P/f'FigS{number}{letter}.{ext}')
 W=7.08*72;H=(4*HEIGHT+.08)*72;doc=fitz.open();page=doc.new_page(width=W,height=H);svg=etree.Element('{%s}svg'%NS,nsmap={None:NS});svg.set('width',f'{W}pt');svg.set('height',f'{H}pt');svg.set('viewBox',f'0 0 {W} {H}')
 for k in range(12):
  letter=string.ascii_lowercase[k];ww=(W-16)/3;xx=(k%3)*(ww+8);yy=(k//3)*HEIGHT*72;pd=fitz.open(P/f'FigS{number}{letter}.pdf');page.show_pdf_page(fitz.Rect(xx,yy,xx+ww,yy+HEIGHT*72),pd,0)
  root=etree.parse(str(P/f'FigS{number}{letter}.svg')).getroot();vb=[float(x) for x in root.attrib['viewBox'].split()];scale=min(ww/vb[2],HEIGHT*72/vb[3]);gp=etree.SubElement(svg,'{%s}g'%NS);gp.set('transform',f'translate({xx} {yy}) scale({scale})');prefix=f's{number}{letter}_'
  for el in root.iter():
   if 'id' in el.attrib:el.attrib['id']=prefix+el.attrib['id']
   for key,val in list(el.attrib.items()):
    if 'url(#' in val:el.attrib[key]=val.replace('url(#','url(#'+prefix)
    elif key.endswith('href') and val.startswith('#'):el.attrib[key]='#'+prefix+val[1:]
  for child in list(root):gp.append(child)
 doc.save(F/f'FigS{number}.pdf',deflate=True);page.get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(F/f'FigS{number}.png');page.get_pixmap(matrix=fitz.Matrix(110/72,110/72),alpha=False).save(F/f'FigS{number}_preview.png');etree.ElementTree(svg).write(str(F/f'FigS{number}.svg'),encoding='UTF-8',xml_declaration=True);doc.close()
json.dump(manifest,open(D/'supplement_panel_manifest.json','w'),indent=2)
print('Two supplementary plates completed, with 12 lead-specific panels each.')
