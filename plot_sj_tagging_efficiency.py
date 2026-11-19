#!/usr/bin/env python3
"""Event-level SR efficiency conditional on fixed AN preselection.

Primary curves: unweighted counts, 68.27% Wilson intervals. Nominal weighted
ratios and correlated ratio errors are exported alongside counts. Reconstruction
failures remain in the denominator. Requires CMSSW Python/ROOT and matplotlib.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def efficiency(n, k, sw, sk, selected_w2, rejected_w2):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError('Expected a nonempty denominator containing the numerator')
    p = k/n
    center = (p + .5/n)/(1+1/n)
    half = math.sqrt(p*(1-p)/n + .25/n**2)/(1+1/n)
    weighted = sk/sw if sw else None
    error = math.sqrt((1-weighted)**2*selected_w2 + weighted**2*rejected_w2)/abs(sw) if sw else None
    return dict(efficiency=p, efficiency_low=center-half, efficiency_high=center+half,
                nominal_weighted_efficiency=weighted, nominal_weighted_error=error)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def integer(value):
    """PyROOT exposes vector<unsigned char> elements as one-character strings."""
    return ord(value) if isinstance(value, str) else int(value)


def read_sample(task):
    import ROOT
    path = Path(task['output'])
    receipt = json.loads((path.parent/'worker_result.json').read_text())
    if receipt.get('success') is not True or any(receipt.get(k)!=task[k] for k in ('sample','index','trial_token')):
        raise ValueError(f'Missing or mismatched success receipt for {path}')
    root = ROOT.TFile.Open(str(path), 'READ')
    if not root or root.IsZombie():
        raise ValueError(f'Cannot read {path}')
    try:
        meta, tree = root.Get('compactScan/Metadata'), root.Get('compactScan/Events')
        if not meta or not tree or meta.GetEntries()!=1:
            raise ValueError(f'Missing metadata/events in {path}')
        meta.GetEntry(0)
        if int(meta.schemaVersion)!=3 or str(meta.sampleName)!=task['sample']:
            raise ValueError(f'Wrong schema/sample in {path}')
        cuts = task['scan_grid']['cos_thrust_cuts']
        actual = list(map(float,meta.configCosThrust))
        if len(actual)!=len(cuts) or any(abs(a-b)>1e-6 for a,b in zip(actual,cuts)):
            raise ValueError(f'Unexpected cosine grid in {path}')
        for name,wanted in [('configCollectionPtCut',task['parameters']['t_keep']),('configCaRadius',task['parameters']['r_ca'])]:
            if any(abs(float(v)-wanted)>1e-6 for v in getattr(meta,name)):
                raise ValueError(f'Unexpected {name} in {path}')
        if abs(float(meta.akRadius)-task['parameters']['r_ak'])>1e-6:
            raise ValueError('Unexpected AK radius')
        required = ['run','lumi','event','genWeight','analysisWeight','passesBaseline','analysisNBTags',
                    'recoStatus','passesRecoJetVeto','passesSignalRegion','sj1NCA4E300','sj2NCA4E300']
        tree.SetBranchStatus('*',0)
        for name in required:
            if not tree.GetBranch(name):
                raise ValueError(f'Missing {name} in {path}')
            tree.SetBranchStatus(name,1)
        size = len(cuts)
        passed, selected_w, selected_w2 = [0]*size, [0.]*size, [0.]*size
        rejected_w2, guarded, invalid = [0.]*size, [0]*size, [0]*size
        n, sumw, sumw2, negative = 0, 0., 0., 0
        events, baseline = hashlib.sha256(), hashlib.sha256()
        for event in tree:
            identity = f'{int(event.run)}:{int(event.lumi)}:{int(event.event)}:{float(event.genWeight):.17g}\n'.encode()
            events.update(identity)
            if not event.passesBaseline:
                continue
            baseline.update(identity)
            w = float(event.genWeight)*float(event.analysisWeight)
            if not math.isfinite(w):
                raise ValueError('Nonfinite nominal event weight')
            n+=1; sumw+=w; sumw2+=w*w; negative+=w<0
            for j in range(size):
                status = integer(event.recoStatus[j])
                sr = (status==0 and bool(integer(event.passesRecoJetVeto[j])) and int(event.analysisNBTags)>=1
                      and int(event.sj1NCA4E300[j])>=2 and int(event.sj2NCA4E300[j])>=2)
                if sr!=bool(integer(event.passesSignalRegion[j])):
                    raise ValueError(f'SR flag disagrees with explicit selection in {path}')
                guarded[j]+=status==6; invalid[j]+=status!=0
                if sr:
                    passed[j]+=1; selected_w[j]+=w; selected_w2[j]+=w*w
                else:
                    rejected_w2[j]+=w*w
        if tree.GetEntries()!=int(meta.processedEvents) or tree.GetEntries()!=task['max_events']:
            raise ValueError(f'Incomplete event coverage in {path}')
        rows=[]
        for j,c in enumerate(cuts):
            row=dict(sample=task['sample'],ak_radius=task['parameters']['r_ak'],ca_radius=task['parameters']['r_ca'],
                     collection_pt_cut=task['parameters']['t_keep'],cos_thrust_cut=c,processed_events=int(tree.GetEntries()),
                     preselected_events=n,sr_events=passed[j],preselected_sumw=sumw,preselected_sumw2=sumw2,
                     sr_sumw=selected_w[j],sr_sumw2=selected_w2[j],negative_weight_preselected_events=negative,
                     invalid_preselected_events=invalid[j],guarded_preselected_events=guarded[j],
                     invalid_fraction=invalid[j]/n if n else None,complexity_guard_fraction=guarded[j]/n if n else None)
            row.update(efficiency(n,passed[j],sumw,selected_w[j],selected_w2[j],rejected_w2[j]));rows.append(row)
        return rows,dict(sample=task['sample'],ak_radius=task['parameters']['r_ak'],path=str(path),sha256=digest(path),
                         event_identity_sha256=events.hexdigest(),preselection_identity_sha256=baseline.hexdigest(),
                         processed_events=int(tree.GetEntries()),preselected_events=n,preselected_sumw=sumw)
    finally:
        root.Close()


def render(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    radii=sorted({r['ak_radius'] for r in rows});samples=sorted({r['sample'] for r in rows})
    colors=dict(zip(samples,['#0072B2','#D55E00','#009E73','#CC79A7']))
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    for diagnostic,stem in [(False,'sr_efficiency_vs_cosine'),(True,'reconstruction_losses_vs_cosine')]:
        fig,axes=plt.subplots(1,len(radii),figsize=(7*len(radii),5.6),squeeze=False,sharey=True)
        for ax,radius in zip(axes[0],radii):
            for sample in samples:
                curve=sorted([r for r in rows if r['sample']==sample and r['ak_radius']==radius],key=lambda r:r['cos_thrust_cut'])
                x=[r['cos_thrust_cut'] for r in curve];_,ms,mc=sample.split('_')
                label=f'WbWb {int(ms)/1000:g}/{int(mc)/1000:g} TeV'
                if diagnostic:
                    ax.plot(x,[r['invalid_fraction'] for r in curve],'-o',ms=3,color=colors[sample],label=label+' invalid')
                    ax.plot(x,[r['complexity_guard_fraction'] for r in curve],'--',color=colors[sample],label=label+' guard')
                else:
                    y=[r['efficiency'] for r in curve]
                    errors=[[max(0,r['efficiency']-r['efficiency_low']) for r in curve],
                            [max(0,r['efficiency_high']-r['efficiency']) for r in curve]]
                    ax.errorbar(x,y,yerr=errors,color=colors[sample],label=label,fmt='-o',ms=3.5,capsize=2,lw=1.6)
                ax.set_title(f'AK R = {radius:g}, intermediate CA R = {curve[0]["ca_radius"]:g}')
            ax.set_xlabel(r'SJ assignment cosine threshold $c$');ax.set_xlim(-.02,1.02);ax.set_ylim(0,1)
            ax.grid(alpha=.2);ax.legend(fontsize=9)
        axes[0][0].set_ylabel('Fraction of preselected signal events' if diagnostic else r'$N_{\rm SR}(c)\,/\,N_{\rm preselection}$')
        fig.suptitle('Reconstruction failures within preselection' if diagnostic else 'Signal SR efficiency after preselection',fontsize=16)
        fig.text(.5,.015,'UL2017 MC; collection pT > 120 GeV; no extra jet gate; unweighted counts'
                 +('' if diagnostic else '; 68.27% Wilson intervals'),ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.045,1,.94))
        for extension in ('png','pdf'):
            fig.savefig(output/(stem+'.'+extension),dpi=180)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());rows=[];audits=[]
    for task in manifest['tasks']:
        values,audit=read_sample(task);rows.extend(values);audits.append(audit)
    for sample in {a['sample'] for a in audits}:
        matching=[a for a in audits if a['sample']==sample]
        for key in ('event_identity_sha256','preselection_identity_sha256','processed_events','preselected_events'):
            if len({a[key] for a in matching})!=1:
                raise ValueError(f'Denominator/event population differs between radii for {sample}: {key}')
        weights=[a['preselected_sumw'] for a in matching]
        if max(weights)-min(weights)>1e-10*max(1,abs(weights[0])):
            raise ValueError(f'Preselection weights differ between radii for {sample}')
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    with (output/'efficiency.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'efficiency.json').write_text(json.dumps(dict(definition=manifest['definition'],rows=rows,input_audits=audits),indent=2,allow_nan=False)+'\n')
    render(rows,output)
    (output/'COMPLETE.json').write_text(json.dumps(dict(status='complete',inputs=audits,
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='COMPLETE.json'}),indent=2)+'\n')
    print(f'Wrote {len(rows)} efficiency points to {output}')


if __name__=='__main__':
    main()
