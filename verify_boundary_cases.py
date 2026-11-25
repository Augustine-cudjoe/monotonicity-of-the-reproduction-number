"""Two houses, movement in ONE direction only (g12=0 or g21=0) and no movement: is R0(alpha) monotone?
Sampling: B_ij = 10^U(-3,1) with prob .8 else 0; p,h,omega,xi = 10^U(-1,1); the nonzero g = 10^U(-2,2);
alpha grid {0} U logspace(-3,4,60); a curve is 'nonmonotone' if it rises and falls by more than 1e-9*max."""
import numpy as np
rng=np.random.default_rng(77)
alphas=np.concatenate([[0],np.logspace(-3,4,60)])
def rho2(K):
    tr=K[:,0,0]+K[:,1,1]; det=K[:,0,0]*K[:,1,1]-K[:,0,1]*K[:,1,0]
    return tr/2+np.sqrt(np.maximum((tr/2)**2-det,0))
def run(mode,N=40000):
    L=lambda lo,hi,s: 10**rng.uniform(lo,hi,s)
    B=L(-3,1,(N,2,2))*(rng.random((N,2,2))<.8); P,h,w,xi=(L(-1,1,(N,2)) for _ in range(4))
    g12=L(-2,2,N); g21=L(-2,2,N)
    if mode=="g12=0": g12[:]=0
    if mode=="g21=0": g21[:]=0
    if mode=="none": g12[:]=0; g21[:]=0
    R=np.empty((N,len(alphas)))
    for j,a in enumerate(alphas):
        Q=np.zeros((N,2,2)); Q[:,0,0]=g21; Q[:,0,1]=-g12; Q[:,1,0]=-g21; Q[:,1,1]=g12
        M=np.zeros((N,2,2)); M[:,0,0]=w[:,0]; M[:,1,1]=w[:,1]; M=M+a*Q
        Minv=np.linalg.inv(M)
        K=(B+h[:,:,None]*Minv*xi[:,None,:])*P[:,None,:]; R[:,j]=rho2(K)
    d=np.diff(R,axis=1); tol=1e-9*R.max(1,keepdims=True)
    up=(d>tol).any(1); dn=(d<-tol).any(1); return N,(up&dn).sum(),up.sum(),dn.sum()
for m in ("g12=0","g21=0","none"):
    N,nm,u,dn=run(m); print(f"{m:6s}: {N} instances | nonmonotone {nm} | increasing {u} | decreasing {dn}")
