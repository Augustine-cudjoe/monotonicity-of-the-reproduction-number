"""Vertex optimisation over boxes of edge rates: spurious coordinatewise optima and coordinate-descent success. Seed fixed."""
import numpy as np, itertools
from edge_common import *
rng=np.random.default_rng(20261111)
def Gfrom(n,edges,vals):
    G=np.zeros((n,n))
    for (i,j),g in zip(edges,vals): G[i,j]=g
    return G
def descent(f,E,bits):
    bits=list(bits)
    for _ in range(100):
        ch=False
        for k in range(E):
            b1=bits.copy(); b1[k]^=1
            if f(b1)<f(bits)*(1-1e-12)-1e-14: bits=b1; ch=True
        if not ch: break
    return tuple(bits)
for n,trials in [(3,300),(4,40)]:
    edges=[(i,j) for i in range(n) for j in range(n) if i!=j]; E=len(edges)
    allv=list(itertools.product([0,1],repeat=E)); idx={b:k for k,b in enumerate(allv)}
    spur=s1=s5=0; maxgap=0.0
    for t in range(trials):
        B,p,h,xi,om,_=wide(rng,n,['full','diag','zero'][t%3]); gmax=np.exp(rng.uniform(np.log(.05),np.log(30),E))
        vals=np.array([rho(Kmat(B,p,h,xi,om,Gfrom(n,edges,np.array(b)*gmax),1.0)) for b in allv]); gm=vals.min()
        f=lambda b: vals[idx[tuple(b)]]
        has=False
        for k,b in enumerate(allv):
            if vals[k]>gm*(1+1e-7) and all(vals[idx[tuple(list(b[:m])+[1-b[m]]+list(b[m+1:]))]]>=vals[k]*(1-1e-9)-1e-13 for m in range(E)): has=True;break
        spur+=has
        res=[f(descent(f,E,rng.integers(0,2,E))) for _ in range(5)]
        s1+= res[0]<=gm*(1+1e-7); s5+= min(res)<=gm*(1+1e-7); maxgap=max(maxgap,(res[0]-gm)/gm)
    print(f"n={n}: {trials} instances; with a non-global coordinatewise local minimum: {spur}; descent finds global min: 1 start {s1}, 5 starts {s5}; worst 1-start gap {maxgap:.3f}")
