#!/usr/bin/env python3
"""Validate the optional AN reference scan, including the fixed 0.85 anchor."""
import math
import sys
import ROOT


def integer(value):
    return ord(value) if isinstance(value,str) else int(value)


def validate(path):
    f=ROOT.TFile.Open(path,'READ')
    if not f or f.IsZombie():raise ValueError('Cannot open '+path)
    m=f.Get('compactScan/Metadata');t=f.Get('compactScan/Events')
    if not m or not t or m.GetEntries()!=1:
        raise ValueError('Missing metadata/events in '+path)
    if not m.GetBranch('referenceCosThrustCuts'):
        raise ValueError('Missing reference cosine grid')
    m.GetEntry(0)
    cuts=list(map(float,m.referenceCosThrustCuts))
    if not cuts or cuts!=sorted(set(cuts)) or any(not math.isfinite(c) or not 0<=c<=1 for c in cuts):
        raise ValueError('Invalid reference cosine grid')
    if str(m.referenceReconstruction)!='AN23-067-PATAK8-CA8-Thrust-source-port-v1':
        raise ValueError('Unexpected AN reference implementation')
    pairs=[('referenceScanRecoStatus','referenceRecoStatus'),('referenceScanRegion','referenceRegion'),
           ('referenceScanSJ1Mass','referenceSJ1Mass'),('referenceScanSJ2Mass','referenceSJ2Mass'),
           ('referenceScanSuuMass','referenceSuuMass'),('referenceScanSJ1NCA4E300','referenceSJ1NCA4E300'),
           ('referenceScanSJ2NCA4E300','referenceSJ2NCA4E300')]
    vectors=[a for a,b in pairs]+['referenceScanNAmbiguous']
    anchor=next((i for i,c in enumerate(cuts) if abs(c-.85)<1e-12),None)
    t.SetBranchStatus('*',0)
    for name in vectors+[b for a,b in pairs]+['passesBaseline','analysisNBTags']:
        if not t.GetBranch(name):raise ValueError('Missing '+name)
        t.SetBranchStatus(name,1)
    for event in t:
        for name in vectors:
            if len(getattr(event,name))!=len(cuts):raise ValueError('Wrong width: '+name)
        for i,c in enumerate(cuts):
            status=integer(event.referenceScanRecoStatus[i]);region=integer(event.referenceScanRegion[i])
            if status not in range(9) or region not in range(5):raise ValueError('Unknown status/region')
            tagged=(event.referenceScanSJ1NCA4E300[i]>=2 and event.referenceScanSJ2NCA4E300[i]>=2)
            sr=bool(event.passesBaseline) and status==0 and event.analysisNBTags>=1 and tagged
            if (region==1)!=sr:raise ValueError('SR flag inconsistent with explicit cuts')
            for name in ('referenceScanSJ1Mass','referenceScanSJ2Mass','referenceScanSuuMass'):
                mass=float(getattr(event,name)[i])
                if status==0 and (not math.isfinite(mass) or mass<0):raise ValueError('Invalid mass for valid reco')
                if status!=0 and not math.isnan(mass):raise ValueError('Failed reco must have NaN masses')
        if anchor is not None:
            for vector,scalar in pairs:
                a=getattr(event,vector)[anchor];b=getattr(event,scalar)
                if 'Status' in vector or 'Region' in vector:a,b=integer(a),integer(b)
                else:a,b=float(a),float(b)
                if a!=b and not (isinstance(a,float) and math.isnan(a) and math.isnan(b)):
                    raise ValueError('0.85 anchor mismatch: '+vector)
    if t.GetEntries()!=int(m.processedEvents):raise ValueError('Processed-event count mismatch')
    print(f'PASS: {t.GetEntries()} events x {len(cuts)} AN reference cosine points; SR, masses, and 0.85 anchor')
    f.Close()


if __name__=='__main__':validate(sys.argv[1])
