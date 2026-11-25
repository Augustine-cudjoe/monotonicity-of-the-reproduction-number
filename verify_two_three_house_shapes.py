import numpy as np
rng=np.random.default_rng(11)
alphas=np.concatenate([[0],np.logspace(-4,4,300)])
def batch(N,k):
    L=lambda lo,hi,s: 10**rng.uniform(lo,hi,s)
    B=L(-3,1,(N,k,k))*(rng.random((N,k,k))<.8); P=L(-1,1,(N,k)); h=L(-1,1,(N,k)); w=L(-1,1,(N,k)); xi=L(-1,1,(N,k))
    G=L(-2,2,(N,k,k))*(rng.random((N,k,k))<.8); idx=np.arange(k); G[:,idx,idx]=0
    return B,P,h,w,xi,G
def curves(B,P,h,w,xi,G):
    N,k=P.shape; Q=-G.copy(); idx=np.arange(k); Q[:,idx,idx]=G.sum(1)
    out=np.empty((N,len(alphas)))
    for j,a in enumerate(alphas):
        M=np.zeros((N,k,k)); M[:,idx,idx]=w; M=M+a*Q
        Minv=np.linalg.inv(M)
        K=(B+h[:,:,None]*Minv*xi[:,None,:])*P[:,None,:]
        out[:,j]=np.abs(np.linalg.eigvals(K)).max(1)
    return out
for k,N in ((2,150000),(3,20000)):
    nonmono=0; tot=0; worst=0
    for _ in range(N//10000 if k==2 else 2):
        n=10000 if k==2 else N//2
        c=curves(*batch(n,k)); d=np.diff(c,axis=1); scale=np.abs(c).max(1,keepdims=True)
        up=(d>1e-9*scale).any(1); dn=(d<-1e-9*scale).any(1)
        nm=up&dn; nonmono+=nm.sum(); tot+=n
    print(f"k={k}: {tot} instances, non-monotone (both rises and falls on grid): {nonmono}")
