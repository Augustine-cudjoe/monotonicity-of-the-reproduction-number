import numpy as np
rng=np.random.default_rng(5)
def Q_of(G): return np.diag(G.sum(0))-G
def K(a,B,H,X,P,w,G): return (B+np.diag(H)@np.linalg.inv(np.diag(w)+a*Q_of(G))@np.diag(X))@np.diag(P)
def R0(a,*p): return max(abs(np.linalg.eigvals(K(a,*p))))
def rand_inst(k,sym_G=False,recip_B=False,sparse=.0):
    w,H,X,P=(rng.random((4,k))+.1)*rng.choice([.5,1,3],4)[:,None]
    G=rng.random((k,k))*(rng.random((k,k))>sparse)*rng.choice([.3,1,4]); np.fill_diagonal(G,0)
    if sym_G: G=(G+G.T)/2
    if recip_B:
        S=rng.random((k,k))*rng.choice([.3,1,3]); S=(S+S.T)/2; B=S@np.diag(X/H)       # B diag(H/X) = S symmetric
    else: B=rng.random((k,k))*rng.choice([.3,1,3])*(rng.random((k,k))>sparse)
    return B,H,X,P,w,G
alphas=[0,.05,.3,1,4,20,100,1e4]
# ---- (A) universal envelope ----
viol=0; ratios=[]
for t in range(6000):
    k=rng.integers(2,8); B,H,X,P,w,G=rand_inst(k,sparse=rng.choice([0,.5]))
    c=w/H; th=P*(X+c@B)*H/w           # theta_j = p_j (h_j/w_j)(xi_j + sum_i (w_i/h_i) B_ij)
    vals=np.array([R0(a,B,H,X,P,w,G) for a in alphas])
    if vals.max()>th.max()*(1+1e-9) or vals.min()<th.min()*(1-1e-9): viol+=1
    ratios.append((vals.max()-vals.min())/(th.max()-th.min()+1e-300))
print("(A) envelope violations:",viol,"of 6000;  median fraction of the envelope actually used by R0(alpha):",np.median(ratios).round(3))
# paper's examples
two=(np.array([[0,2.],[0,4]]),np.array([1.,3]),np.array([3.,1]),np.array([2.,1]),np.array([2.,1]),np.array([[0,1.],[3,0]]))
B,H,X,P,w,G=two; c=w/H; th=P*(X+c@B)*H/w
print("two-house theta =",th," R0 range over alpha:",min(R0(a,*two) for a in alphas),max(R0(a,*two) for a in alphas))
# ---- (B) reversibility theorem ----
nonmono=nonconv=badsim=0
for t in range(3000):
    k=rng.integers(2,7); B,H,X,P,w,G=rand_inst(k,sym_G=True,recip_B=True,sparse=rng.choice([0,.4]))
    V=np.diag(np.sqrt(H/(X*P))); Sd=np.linalg.inv(V)@B@np.diag(P)@V; Lam=np.diag(np.sqrt(H*X*P))
    asym=np.abs(Sd-Sd.T).max()
    grid=np.concatenate([[0],np.logspace(-2,3,40)]); vals=[]
    for a in grid:
        Ms=np.diag(w)+a*Q_of(G); Sym=Sd+Lam@np.linalg.inv(Ms)@Lam
        e=max(np.linalg.eigvalsh((Sym+Sym.T)/2)); vals.append(e)
        if abs(e-R0(a,B,H,X,P,w,G))>1e-8*max(1,e) or asym>1e-9: badsim+=1
    vals=np.array(vals)
    if np.any(np.diff(vals)>1e-9): nonmono+=1
    # convexity in alpha on a uniform grid
    ug=np.linspace(0,5,60); uv=[]
    for a in ug:
        Ms=np.diag(w)+a*Q_of(G); uv.append(max(np.linalg.eigvalsh(Sd+Lam@np.linalg.inv(Ms)@Lam)))
    if np.any(np.diff(uv,2)<-1e-9): nonconv+=1
print("(B) symmetric G + reciprocal B: similarity failures",badsim,"; non-monotone instances",nonmono,"; non-convex instances",nonconv,"of 3000")
# ---- necessity: can either asymmetry alone produce amplification? ----
def count_up(sym_G,recip_B,n=4000):
    ups=0; ex=None
    for t in range(n):
        k=rng.integers(2,6); inst=rand_inst(k,sym_G=sym_G,recip_B=recip_B,sparse=rng.choice([0,.4]))
        vals=np.array([R0(a,*inst) for a in alphas]); 
        if vals.max()>vals[0]*(1+1e-7) and np.any(np.diff(vals)>1e-9):
            ups+=1; ex=ex or (k,vals[0],vals.max())
    return ups,ex
for name,sg,rb in (("symmetric G, non-reciprocal B",True,False),("directed G, reciprocal B",False,True),("directed G, non-reciprocal B",False,False),("symmetric G, reciprocal B",True,True)):
    u,ex=count_up(sg,rb); print(f"  {name}: instances where R0 rises above R0(0) or increases somewhere: {u}/4000   example (k,R0(0),max R0): {ex}")
