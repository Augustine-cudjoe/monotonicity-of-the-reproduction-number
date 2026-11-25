import numpy as np
rng=np.random.default_rng(21)
def Q_of(G): return np.diag(G.sum(0))-G
def Kmat(G,B,H,X,P,w): return (B+np.diag(H)@np.linalg.inv(np.diag(w)+Q_of(G))@np.diag(X))@np.diag(P)
def rho(A): return max(abs(np.linalg.eigvals(A)))
def lam_vertex(i,B,H,X,P,w):                    # R_inf(e_i) = rho( BP + (h_i/w_i) e_i d^T ),  d_j = xi_j p_j
    k=len(w); A=B@np.diag(P); A=A.copy(); A[i,:]+= (H[i]/w[i])*(X*P); return rho(A)
def Rinf(pit,B,H,X,P,w):                         # pit = removal distribution
    return rho(B@np.diag(P)+np.outer(H*pit/w,X*P))
def inst(k,sparse):
    w,H,X,P=(rng.random((4,k))+.05)*rng.choice([.5,1,4],4)[:,None]
    G=rng.random((k,k))*(rng.random((k,k))>sparse)*rng.choice([.1,1,10,100]); np.fill_diagonal(G,0)
    B=rng.random((k,k))*(rng.random((k,k))>sparse)*rng.choice([.2,1,3]); B+=1e-3
    return B,H,X,P,w,G
# 1. exact representation R0(G) = R_inf(pi~) with pi~ = T y / 1^T y,  T = D_w M^{-1}, y = D v
e1=e2=0; viol=0; N=20000
for t in range(N):
    k=rng.integers(2,8); B,H,X,P,w,G=inst(k,rng.choice([0,.5]))
    Km=Kmat(G,B,H,X,P,w); ev,V=np.linalg.eig(Km); j=np.argmax(ev.real); lam=ev[j].real; v=V[:,j].real; v=v if v.sum()>0 else -v
    if v.min()<=0: continue
    T=np.diag(w)@np.linalg.inv(np.diag(w)+Q_of(G)); y=np.diag(X*P)@v; pit=T@y/(np.ones(k)@y)
    Kinf=B@np.diag(P)+np.outer(H*pit/w,X*P)
    e1=max(e1,np.abs(Kinf@v-lam*v).max()/lam); e2=max(e2,abs(rho(Kinf)-lam)/lam)
    lams=[lam_vertex(i,B,H,X,P,w) for i in range(k)]
    if lam>max(lams)*(1+1e-9) or lam<min(lams)*(1-1e-9): viol+=1
print(f"representation: max rel. error of K_inf(pi~) v = lam v : {e1:.1e};  |rho(K_inf(pi~)) - R0| / R0 : {e2:.1e}")
print(f"sharp envelope violations: {viol} of {N}")
# 2. hub networks approach the endpoints
errs=[]
for t in range(300):
    k=rng.integers(2,7); B,H,X,P,w,G=inst(k,0)
    lams=np.array([lam_vertex(i,B,H,X,P,w) for i in range(k)])
    for i in (int(np.argmax(lams)),int(np.argmin(lams))):
        n=1e7; Gh=np.zeros((k,k)); Gh[i,:]=n; Gh[i,i]=0
        errs.append(abs(rho(Kmat(Gh,B,H,X,P,w))-lams[i])/lams[i])
print(f"hub network (all flow into house i, rate 1e7): max relative gap to R_inf(e_i) = {max(errs):.1e}")
# 3. the paper's examples
def show(name,B,H,X,P,w,G):
    k=len(w); lams=np.array([lam_vertex(i,B,H,X,P,w) for i in range(k)])
    print(f"{name}: R0(no movement)={rho(Kmat(np.zeros((k,k)),B,H,X,P,w)):.4f}  lambda_i={np.round(lams,4)}  ->  attainable range [{lams.min():.4f}, {lams.max():.4f}]")
show("two-house",np.array([[0,2.],[0,4]]),np.array([1.,3]),np.array([3.,1]),np.array([2.,1]),np.array([2.,1]),None)
r5=np.random.default_rng(20260913); k=5
mu=np.linspace(.08,.16,k); sig=np.linspace(.65,.95,k); r=np.linspace(.38,.72,k); beta=np.linspace(.25,.55,k); eta=np.linspace(.7,1.4,k); q=np.linspace(.8,1.3,k); xi=np.linspace(.7,1.2,k); om=np.linspace(.45,.85,k)
C=np.full((k,k),.025); np.fill_diagonal(C,0); B5=np.diag(beta)+C; G5=np.zeros((k,k))
for j in range(k):
    for i in range(k):
        if i!=j: G5[i,j]=.02+.08*r5.random()
P5=(sig/r)/(sig+mu)
show("five-house",B5,eta*q,xi,P5,om,G5)
print("   five-house R0(alpha=1), R0(20):",rho(Kmat(G5,B5,eta*q,xi,P5,om)).round(4),rho(Kmat(20*G5,B5,eta*q,xi,P5,om)).round(4))
G4=np.array([[0.,.18676055,1.13409212,.81236097],[.98523368,0.,.81327833,.50700571],[.59144856,1.02896409,0.,.40809778],[.12089721,1.40339853,1.43800821,0.]])
show("four-house",np.array([[1.07918447,.13585585,.20192385,.44449085],[.37123652,1.18721717,.05377471,.13350669],[.05072876,.35474165,.67764017,.29408311],[.15211338,.29210047,.08140161,1.05927646]]),
     np.array([.53669501,.40102555,1.36073196,2.91421398]),np.array([1.12802281,.71707664,1.89141466,.34346463]),np.array([1.33294022,1.05874074,1.8659112,1.294221]),np.array([.62478676,1.44991748,1.99386879,1.20422586]),G4)
