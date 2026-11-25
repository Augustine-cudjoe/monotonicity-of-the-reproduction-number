import numpy as np, itertools
from edge_common import *
rng=np.random.default_rng(7)
als=np.concatenate([[0],np.logspace(-2,3,120)])
best=None
edges=[(i,j) for i in range(3) for j in range(3) if i!=j]
for t in range(60000):
    P=rng.integers(1,4,3).astype(float);H=rng.integers(1,4,3).astype(float);Dw=rng.integers(1,4,3).astype(float);Dx=rng.integers(1,4,3).astype(float)
    B=rng.integers(0,3,(3,3)).astype(float)*(rng.random((3,3))<.5)
    sel=rng.choice(6,2,replace=False); G=np.zeros((3,3))
    for s in sel: G[edges[s]]=1.0
    v=np.array([rho(Kmat(B,P,H,Dx,Dw,G,a)) for a in als])
    d=np.diff(v); 
    if (d>1e-9).any() and (d<-1e-9).any():
        depth=(v[0]-v.min())/v[0]; rise=(v[-1]-v.min())/v[0]
        score=min(depth,rise)
        if best is None or score>best[0]: best=(score,P,H,Dw,Dx,B,G,v)
s,P,H,Dw,Dx,B,G,v=best
print("score",s);print("P",P,"H",H,"Dw",Dw,"Dx",Dx);print("B",B.tolist());print("G",G.tolist())
for a in [0,.1,.25,.5,1,2,5,20,100,1e4]: print(a,round(rho(Kmat(B,P,H,Dx,Dw,G,a)),4))
print("min",v.min(),"at",als[v.argmin()])
