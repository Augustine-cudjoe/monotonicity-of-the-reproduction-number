import numpy as np
from edge_common import *
rng=np.random.default_rng(11)
edges=[(i,j) for i in range(3) for j in range(3) if i!=j]
gs=np.concatenate([[0],np.logspace(-2,3,80)]); best=None
for t in range(6000):
    P=rng.integers(1,4,3).astype(float);H=rng.integers(1,4,3).astype(float);Dw=rng.integers(1,4,3).astype(float);Dx=rng.integers(1,4,3).astype(float)
    B=np.zeros((3,3)); sel=rng.choice(6,3,replace=False); G=np.zeros((3,3))
    for s in sel[:2]: G[edges[s]]=1.0
    i,j=edges[sel[2]]
    v=np.array([rho(Kmat(B,P,H,Dx,Dw,with_edge(G,i,j,g),1.0)) for g in gs])
    if v[-1]>v[0]*(1+1e-6) and (np.diff(v)>=-1e-9).all():
        score=(v[-1]-v[0])/v[0]
        if best is None or score>best[0]: best=(score,P,H,Dw,Dx,G,(i,j),v)
s,P,H,Dw,Dx,G,e,v=best
print("rise",round(s,4),"P",P,"H",H,"Dw",Dw,"Dx",Dx,"G",G.tolist(),"varying edge (to,from)",e)
for g in [0,.5,1,2,10,100,1000]: print(g,round(rho(Kmat(np.zeros((3,3)),P,H,Dx,Dw,with_edge(G,*e,g),1.0)),4))
print("local ratios r=c*d:",(H/Dw)*(Dx*np.diag(P)))
