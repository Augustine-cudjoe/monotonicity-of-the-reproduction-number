"""R0 along a single edge rate: convex, concave, or neither? Seed fixed."""
import numpy as np
from edge_common import *
rng=np.random.default_rng(20261110)
cnt=dict(convex=0,concave=0,constant=0,neither=0); tot=0
for t in range(900):
    n=int(rng.integers(3,6)); B,p,h,xi,om,G=wide(rng,n,['full','diag','zero'][t%3]); al=1.0
    i,j=rng.choice(n,2,replace=False); G=with_edge(G,i,j,0.0)
    u0,v0=perron(Kmat(B,p,h,xi,om,G,al))
    if min(u0.min(),v0.min())<1e-9*max(u0.max(),v0.max()): continue
    gs=np.linspace(0,float(np.exp(rng.uniform(0,4))),400)
    v=np.array([rho(Kmat(B,p,h,xi,om,with_edge(G,i,j,g),al)) for g in gs])
    d2=np.diff(v,2); tol=1e-7*v.max(); tot+=1
    cx=(d2>tol).any(); cv=(d2<-tol).any()
    if cx and cv: cnt['neither']+=1
    elif cx: cnt['convex']+=1
    elif cv: cnt['concave']+=1
    else: cnt['constant']+=1
print("instances",tot,cnt)
