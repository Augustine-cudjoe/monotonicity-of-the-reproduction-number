"""Box-vertex optimality (n=3,4), two-edge three-house systems, strongly connected Example check. Seed fixed."""
import numpy as np, itertools
from edge_common import *
rng=np.random.default_rng(20261109)
def Gfrom(n,edges,vals):
    G=np.zeros((n,n))
    for (i,j),g in zip(edges,vals): G[i,j]=g
    return G
for n,trials,npts in [(3,150,3000),(4,10,1500)]:
    edges=[(i,j) for i in range(n) for j in range(n) if i!=j]; E=len(edges)
    viol=0
    for t in range(trials):
        B,p,h,xi,om,_=wide(rng,n,['full','diag','zero'][t%3]); gmax=np.exp(rng.uniform(np.log(.05),np.log(30),E))
        f=lambda v: rho(Kmat(B,p,h,xi,om,Gfrom(n,edges,v),1.0))
        vv=[f(np.array(b)*gmax) for b in itertools.product([0,1],repeat=E)]; lo,hi=min(vv),max(vv)
        bad=False
        for _ in range(npts):
            v=rng.random(E)*gmax*(rng.random(E)<.7); r=f(v)
            if r<lo-1e-9*hi or r>hi+1e-9*hi: bad=True
        viol+=bad
    print(f"box-vertex check n={n}: {trials} instances x {npts} interior points, violations {viol}")
# two-edge / one-edge three-house systems: monotone in alpha?
als=np.concatenate([[0],np.logspace(-3,4,60)])
for nedges,label in [(1,"single edge"),(2,"two edges")]:
    cnt=0;N=20000
    for t in range(N):
        n=3; P=LU(rng,-1,1,n);H=LU(rng,-1,1,n);Dw=LU(rng,-1,1,n);Dx=LU(rng,-1,1,n)
        B=LU(rng,-3,1,(n,n))*(rng.random((n,n))<.8)
        all_e=[(i,j) for i in range(n) for j in range(n) if i!=j]
        sel=rng.choice(len(all_e),nedges,replace=False); G=np.zeros((n,n))
        for s in sel: G[all_e[s]]=LU(rng,-2,2)
        v=np.array([rho(Kmat(B,P,H,Dx,Dw,G,a)) for a in als]); d=np.diff(v);tol=1e-9*v.max()
        cnt+= (d>tol).any() and (d<-tol).any()
    print(f"three houses, {label}: nonmonotone in alpha in {cnt} of {N}")
# Example 5 made strongly connected
D=np.diag
B=np.array([[0,0,0],[0,0,0],[0,1,2.]]);P=D([2,1,2.]);H=D([1,3,1.]);Dw=np.array([1,2,3.]);Dx=D([3,2,1.])
for eps in [1e-3,1e-2]:
    G=np.array([[0,1,0],[1,0,1],[0,0,0.]]); G[0,2]=eps;G[2,0]=eps;G[2,1]=eps
    ag=np.concatenate([[0],np.logspace(-3,3,400)])
    v=np.array([rho((B+H@np.linalg.inv(np.diag(Dw)+a*(np.diag(G.sum(axis=0))-G))@Dx)@P) for a in ag])
    print(f"Auxiliary three-house system (not in the manuscript) + extra edges of rate {eps}: R0(0)={v[0]:.3f}, min {v.min():.3f} at alpha={ag[v.argmin()]:.3f}, limit {v[-1]:.3f}")
