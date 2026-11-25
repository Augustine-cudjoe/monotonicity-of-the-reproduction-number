"""Checks Proposition (first-order form) and Corollary (sign persistence). Run: python3 verify_sign_persistence.py"""
import numpy as np
rng=np.random.default_rng(7)
def build(k):
    LU=lambda a,b,s:10**rng.uniform(a,b,s)
    P=LU(-1,1,k);H=LU(-1,1,k);w=LU(-1,1,k);x=LU(-1,1,k)
    B=LU(-2,0.5,(k,k))*(rng.random((k,k))<.7)
    G=LU(-1.5,1,(k,k))*(rng.random((k,k))<.6);np.fill_diagonal(G,0)
    return P,H,w,x,B,G
def pot(P,H,w,x,B,G,al):
    Q=np.diag(G.sum(0))-G;Mi=np.linalg.inv(np.diag(w)+al*Q)
    K=(B+np.diag(H)@Mi@np.diag(x))@np.diag(P)
    ev,V=np.linalg.eig(K);i=np.argmax(ev.real);v=np.abs(V[:,i].real)
    ev2,U=np.linalg.eig(K.T);j=np.argmax(ev2.real);u=np.abs(U[:,j].real)
    return ev[i].real,Mi.T@(H*u),Mi@(x*P*v),u,v
bad=tot=mism=0
for t in range(1500):
    k=rng.integers(3,7);P,H,w,x,B,G=build(k);al=10**rng.uniform(-1,1)
    i,j=rng.choice(k,2,replace=False);G[i,j]=0
    R0,Phi,Psi,u,v=pot(P,H,w,x,B,G,al)
    if min(u)<1e-8*max(u) or min(v)<1e-8*max(v):continue
    tot+=1;signs=set()
    for g in [0,1e-3,1e-2,.1,1,10,100,1e3]:
        G2=G.copy();G2[i,j]=g;_,Ph,_,_,_=pot(P,H,w,x,B,G2,al)
        d=Ph[i]-Ph[j];signs.add(0 if abs(d)<1e-9*abs(Ph).max() else int(np.sign(d)))
    bad+=len(signs-{0})>1
    g=1e-6;G2=G.copy();G2[i,j]=g;R,_,_,_,_=pot(P,H,w,x,B,G2,al)
    mism+=abs((R-R0)/g-al*Psi[j]*(Phi[i]-Phi[j])/(u@v))>1e-3*max(1,abs(al*Psi[j]*(Phi[i]-Phi[j])/(u@v)))
print("instances",tot,"| sign changes of Phi_i-Phi_j along edge:",bad,"| first-order formula mismatches vs finite difference:",mism)
