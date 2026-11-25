import numpy as np
def rho(A): return max(np.linalg.eigvals(A).real)
def Kmat(B,p,h,xi,om,G,al):
    Q=np.diag(G.sum(axis=0))-G; M=np.diag(om)+al*Q
    return (B+np.diag(h)@np.linalg.inv(M)@np.diag(xi))@np.diag(p)
def LU(rng,lo,hi,size=None): return 10**rng.uniform(lo,hi,size)
def wide(rng,n,mode):
    f=lambda lo,hi,sz=None: np.exp(rng.uniform(np.log(lo),np.log(hi),sz))
    p=f(.05,20,n);h=f(.05,20,n);xi=f(.05,20,n);om=f(.05,20,n)
    if mode=='full': B=f(.01,10,(n,n))*(rng.random((n,n))<.7)
    elif mode=='diag': B=np.diag(f(.01,10,n))
    else: B=np.zeros((n,n))
    G=f(.01,20,(n,n))*(rng.random((n,n))<.6); np.fill_diagonal(G,0)
    return B,p,h,xi,om,G
def perron(Kk):
    w,V=np.linalg.eig(Kk); v=np.abs(V[:,np.argmax(w.real)].real)
    w2,U=np.linalg.eig(Kk.T); u=np.abs(U[:,np.argmax(w2.real)].real); return u,v
def with_edge(G,i,j,g):
    G2=G.copy(); G2[i,j]=g; return G2
