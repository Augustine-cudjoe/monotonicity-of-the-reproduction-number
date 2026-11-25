"""Theorem 5 checks: single-edge monotonicity, sign-at-zero rule, closed-form limit. Seed fixed."""
import numpy as np, sys
from edge_common import *
rng=np.random.default_rng(20261108)
grid=np.concatenate([[0],np.logspace(-4,4,200)])
tot=nonmono=signmis=lim_tested=lim_bad=0; worst=0.0; skipped=0; inc=dec=const=0
for t in range(3000):
    n=int(rng.integers(3,8)); B,p,h,xi,om,G=wide(rng,n,['full','diag','zero'][t%3])
    al=float(np.exp(rng.uniform(-2,2))); i,j=rng.choice(n,2,replace=False); G=with_edge(G,i,j,0.0)
    K0=Kmat(B,p,h,xi,om,G,al); u0,v0=perron(K0)
    if min(u0.min(),v0.min())<1e-9*max(u0.max(),v0.max()): skipped+=1; continue   # K0 numerically reducible
    vals=np.array([rho(Kmat(B,p,h,xi,om,with_edge(G,i,j,g),al)) for g in grid])
    d=np.diff(vals); tol=1e-8*vals.max(); tot+=1
    up=(d>tol).any(); dn=(d<-tol).any()
    if up and dn: nonmono+=1
    elif up: inc+=1
    elif dn: dec+=1
    else: const+=1
    Q=np.diag(G.sum(axis=0))-G; M0=np.diag(om)+al*Q; Mi=np.linalg.inv(M0); e=np.eye(n)
    a=np.diag(h)@Mi@(e[j]-e[i]); b=e[j]@Mi@np.diag(xi)@np.diag(p)
    typ=-np.sign((u0@a)*(b@v0)); net=vals[-1]-vals[0]
    if abs(net)>1e-7*vals.max() and np.sign(net)!=typ: signmis+=1
    tau=e[j]@Mi@(e[j]-e[i])
    if tau>1e-9:
        lim_tested+=1; big=rho(Kmat(B,p,h,xi,om,with_edge(G,i,j,1e9),al)); form=rho(K0-np.outer(a,b)/tau)
        err=abs(big-form)/big; worst=max(worst,err); lim_bad+= err>1e-4
print(f"instances with irreducible K0: {tot} (skipped {skipped})")
print(f"increasing {inc}, decreasing {dec}, constant {const}, NONMONOTONE {nonmono}")
print(f"sign-at-zero rule mismatches: {signmis}")
print(f"closed-form limit tested {lim_tested}, > 1e-4: {lim_bad}, worst relative error {worst:.2e}")
