import numpy as np, json
DT=0.02
def load(f):
    z=np.load(f)
    return {k:z[k] for k in ('t','cL','cR','lfz','rfz','lfy','rfy','z','wz','phase','jq')}

def analyse(name,f):
    z=load(f)
    T,N=z['cL'].shape
    out={}
    rows=[]
    for e in range(N):
        cL=np.nan_to_num(z['cL'][:,e])>0.5      # (T,) for env e  <-- correct axis
        cR=np.nan_to_num(z['cR'][:,e])>0.5
        ok=~np.isnan(z['t'][:,e])
        if ok.sum()<100: continue
        n=ok.sum()
        cL=cL[:n]; cR=cR[:n]
        tdL=np.count_nonzero(cL[1:]&~cL[:-1]); tdR=np.count_nonzero(cR[1:]&~cR[:-1])
        rows.append(dict(
            e=e, dur=n*DT, tdL=tdL, tdR=tdR,
            duty_L=cL.mean(), duty_R=cR.mean(),
            dbl=(cL&cR).mean(), fly=(~cL&~cR).mean(),
            clr_L=(np.nanpercentile(z['lfz'][:n,e],95)-np.nanpercentile(z['lfz'][:n,e],5))*1000,
            clr_R=(np.nanpercentile(z['rfz'][:n,e],95)-np.nanpercentile(z['rfz'][:n,e],5))*1000,
        ))
    a=lambda k: np.array([r[k] for r in rows],float)
    res=dict(name=name, n=len(rows),
        step_hz_per_foot=float((a('tdL')/a('dur')).mean()),
        stride_hz=float(((a('tdL')+a('tdR'))/a('dur')/2).mean()),
        duty_L=float(a('duty_L').mean()), duty_R=float(a('duty_R').mean()),
        double_support=float(a('dbl').mean()), flight=float(a('fly').mean()),
        clr_L=float(a('clr_L').mean()), clr_R=float(a('clr_R').mean()),
        tdL_med=float(np.median(a('tdL'))), tdL_min=float(a('tdL').min()), tdL_max=float(a('tdL').max()),
        n_zero_step=int((a('tdL')==0).sum()))
    # walkers-only (>=10 total touchdowns) resplit
    tot=a('tdL')+a('tdR'); m=tot>=10
    if m.sum() and (~m).sum():
        res['walkers_n']=int(m.sum())
        res['walkers_duty']=float(a('duty_L')[m].mean())
        res['walkers_step_hz']=float((a('tdL')[m]/a('dur')[m]).mean())
        res['walkers_dbl']=float(a('dbl')[m].mean())
        res['standers_n']=int((~m).sum())
        res['standers_duty']=float(a('duty_L')[~m].mean())
        res['standers_step_hz']=float((a('tdL')[~m]/a('dur')[~m]).mean())
        res['standers_dbl']=float(a('dbl')[~m].mean())
    print(json.dumps(res,indent=1))
    return res

for n,f in [('official','/tmp/agentD/q_official.npz'),('ck7000','/tmp/agentD/q_ck7000.npz'),('align_fwd','/tmp/agentD/q_align_fwd.npz')]:
    analyse(n,f)
