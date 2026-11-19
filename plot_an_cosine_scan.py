#!/usr/bin/env python3
"""AN reference cosine scan: conditional SR efficiency and reconstructed mass shapes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from plot_sj_tagging_efficiency import efficiency,digest,integer


def read(task):
    import ROOT
    path=Path(task['output']);receipt=json.loads((path.parent/'worker_result.json').read_text())
    if receipt.get('success') is not True or any(receipt.get(k)!=task[k] for k in ('index','sample','trial_token')):
        raise ValueError('Invalid worker receipt: '+str(path))
    f=ROOT.TFile.Open(str(path),'READ')
    if not f or f.IsZombie():raise ValueError('Cannot open '+str(path))
    m=f.Get('compactScan/Metadata');t=f.Get('compactScan/Events');m.GetEntry(0)
    cuts=list(map(float,m.referenceCosThrustCuts))
    if cuts!=task['reference_cos_thrust_cuts'] or str(m.sampleName)!=task['sample']:
        raise ValueError('Wrong sample or AN cosine grid')
    if str(m.referenceReconstruction)!='AN23-067-PATAK8-CA8-Thrust-source-port-v1':
        raise ValueError('Unsupported reconstruction')
    size=len(cuts);n=0;sw=0.;sw2=0.;negative=0
    passed=[0]*size;sk=[0.]*size;sk2=[0.]*size;rk2=[0.]*size;guard=[0]*size;invalid=[0]*size
    masses={c:dict(sj=[],suu=[]) for c in cuts};identities=hashlib.sha256()
    required=['run','lumi','event','genWeight','referenceWeight','passesBaseline','analysisNBTags',
              'referenceScanRecoStatus','referenceScanRegion','referenceScanSJ1Mass','referenceScanSJ2Mass',
              'referenceScanSuuMass','referenceScanSJ1NCA4E300','referenceScanSJ2NCA4E300']
    t.SetBranchStatus('*',0)
    for name in required:
        if not t.GetBranch(name):raise ValueError('Missing '+name)
        t.SetBranchStatus(name,1)
    for e in t:
        identities.update(f'{int(e.run)}:{int(e.lumi)}:{int(e.event)}\n'.encode())
        if not e.passesBaseline:continue
        n+=1;w=float(e.genWeight)*float(e.referenceWeight)
        if not math.isfinite(w):raise ValueError('Nonfinite weight')
        sw+=w;sw2+=w*w;negative+=w<0
        for i,c in enumerate(cuts):
            status=integer(e.referenceScanRecoStatus[i]);sr=integer(e.referenceScanRegion[i])==1
            explicit=(status==0 and e.analysisNBTags>=1 and e.referenceScanSJ1NCA4E300[i]>=2 and e.referenceScanSJ2NCA4E300[i]>=2)
            if sr!=explicit:raise ValueError('SR inconsistent with tag definition')
            invalid[i]+=status!=0;guard[i]+=status==6
            if sr:
                passed[i]+=1;sk[i]+=w;sk2[i]+=w*w
                masses[c]['sj'].extend([float(e.referenceScanSJ1Mass[i]),float(e.referenceScanSJ2Mass[i])])
                masses[c]['suu'].append(float(e.referenceScanSuuMass[i]))
            else:rk2[i]+=w*w
    if int(t.GetEntries())!=int(m.processedEvents) or int(t.GetEntries())!=task['max_events']:
        raise ValueError('Incomplete requested input coverage')
    rows=[]
    for i,c in enumerate(cuts):
        row=dict(sample=task['sample'],reconstruction='AN PAT-AK8 / event CA8 / SJ CA4',cos_thrust_cut=c,
                 processed_events=int(t.GetEntries()),preselected_events=n,sr_events=passed[i],
                 preselected_sumw=sw,preselected_sumw2=sw2,sr_sumw=sk[i],sr_sumw2=sk2[i],
                 negative_weight_preselected_events=negative,invalid_preselected_events=invalid[i],
                 guarded_preselected_events=guard[i],invalid_fraction=invalid[i]/n if n else None,
                 complexity_guard_fraction=guard[i]/n if n else None)
        row.update(efficiency(n,passed[i],sw,sk[i],sk2[i],rk2[i]));rows.append(row)
    audit=dict(sample=task['sample'],path=str(path),sha256=digest(path),processed_events=int(t.GetEntries()),
               preselected_events=n,event_identity_sha256=identities.hexdigest())
    f.Close();return rows,masses,audit


def render(rows,shapes,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    samples=sorted(shapes);colors=['#0072B2','#D55E00','#009E73']
    fig,ax=plt.subplots(figsize=(8,6))
    for sample,color in zip(samples,colors):
        curve=[r for r in rows if r['sample']==sample];x=[r['cos_thrust_cut'] for r in curve]
        y=[r['efficiency'] for r in curve];_,ms,mc=sample.split('_')
        ax.errorbar(x,y,yerr=[[max(0,r['efficiency']-r['efficiency_low']) for r in curve],
                              [max(0,r['efficiency_high']-r['efficiency']) for r in curve]],
                    fmt='-o',ms=3.5,capsize=2,color=color,label=f'WbWb {int(ms)/1000:g}/{int(mc)/1000:g} TeV')
    ax.set(xlabel=r'SJ assignment cosine threshold $c$',ylabel=r'$N_{\rm SR}(c)/N_{\rm preselection}$',xlim=(-.02,1.02),ylim=(0,1),
           title='Signal SR efficiency: AN reference reconstruction')
    ax.legend();ax.grid(alpha=.2)
    fig.text(.5,.015,'PAT AK8; event-frame CA8; SJ-rest-frame CA4; nominal UL2017 MC\nUnweighted counts; 68.27% Wilson intervals; fixed preselection denominator',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,1))
    for ext in ('png','pdf'):fig.savefig(output/f'sr_efficiency_vs_cosine.{ext}',dpi=180)
    plt.close(fig)
    histograms=[]
    def draw_mass(sample,cuts,kind):
        _,ms,mc=sample.split('_');nominal=int(mc if kind=='sj' else ms)/1000
        edges=np.linspace(0,6 if kind=='sj' else 12,81)
        fig,ax=plt.subplots(figsize=(8,5.6))
        for cut in cuts:
            values=np.asarray(shapes[sample][cut][kind],float)/1000
            counts,_=np.histogram(values,bins=edges);total=int(counts.sum())
            density=counts/(total*np.diff(edges)) if total else np.zeros(len(edges)-1)
            ax.stairs(density,edges,label=f'c = {cut:g}',lw=1.7)
            histograms.append(dict(sample=sample,cos_thrust_cut=cut,observable=kind,edges_tev=edges.tolist(),
                counts=counts.tolist(),entries=len(values),underflow=int((values<edges[0]).sum()),
                overflow=int((values>edges[-1]).sum())))
        ax.axvline(nominal,color='black',ls='--',lw=1.4,label=f'Nominal mass {nominal:g} TeV')
        ax.set(xlabel='Reconstructed SJ mass [TeV]' if kind=='sj' else 'Reconstructed Suu mass [TeV]',
               ylabel='Unit-area density [1/TeV]',xlim=(edges[0],edges[-1]),
               title=sample.replace('_',' '))
        ax.set_ylim(bottom=0);ax.legend(fontsize=10);fig.tight_layout();return fig
    for sample in samples:
        representative=[c for c in shapes[sample] if any(abs(c-v)<1e-9 for v in (0,.5,.85,1))]
        for kind in ('sj','suu'):
            fig=draw_mass(sample,representative,kind)
            for ext in ('png','pdf'):fig.savefig(output/f'{sample}_{kind}_mass_shapes.{ext}',dpi=180)
            plt.close(fig)
        with PdfPages(output/f'{sample}_mass_shapes_all_cosines.pdf') as pdf:
            for cut in shapes[sample]:
                for kind in ('sj','suu'):
                    fig=draw_mass(sample,[cut],kind);pdf.savefig(fig);plt.close(fig)
    (output/'mass_histograms.json').write_text(json.dumps(dict(selection='SR at each cosine threshold',
        normalization='unit area within displayed range; unweighted counts',curves=histograms),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,5.6))
    for sample,color in zip(samples,colors):
        curve=[r for r in rows if r['sample']==sample]
        ax.plot([r['cos_thrust_cut'] for r in curve],[r['invalid_fraction'] for r in curve],color=color,label=sample+' invalid')
        ax.plot([r['cos_thrust_cut'] for r in curve],[r['complexity_guard_fraction'] for r in curve],color=color,ls='--',label=sample+' guard')
    ax.set(xlabel='Cosine threshold c',ylabel='Fraction of preselected events',ylim=(0,1),xlim=(0,1))
    ax.legend(fontsize=9);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'reconstruction_losses_vs_cosine.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();manifest=json.loads(args.manifest.read_text());rows=[];shapes={};audits=[]
    for task in manifest['tasks']:
        values,masses,audit=read(task);rows.extend(values);shapes[task['sample']]=masses;audits.append(audit)
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    with (output/'efficiency.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'efficiency.json').write_text(json.dumps(dict(definition=manifest['definition'],rows=rows,input_audits=audits),indent=2,allow_nan=False)+'\n')
    render(rows,shapes,output)
    (output/'COMPLETE.json').write_text(json.dumps(dict(status='complete',inputs=audits,
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='COMPLETE.json'}),indent=2)+'\n')
    print(f'Wrote {len(rows)} AN efficiency points and reconstructed-only mass shapes to {output}')


if __name__=='__main__':main()
