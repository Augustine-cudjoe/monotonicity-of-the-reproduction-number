"""Five-house application (parameters as listed in Appendix A of the manuscript). Reproduces R0(0), R0(20),
classifies every directed edge at alpha=1 by the sign of Phi_i-Phi_j (Theorem 5 / Corollary sign-persistence),
and tests two interventions. Run: python3 five_house_application.py"""
import numpy as np, itertools
k=5
mu=np.linspace(.08,.16,k); sig=np.linspace(.65,.95,k); r=np.linspace(.38,.72,k); beta=np.linspace(.25,.55,k)
eta=np.linspace(.7,1.4,k); q=np.linspace(.8,1.3,k); xi=np.linspace(.7,1.2,k); om=np.linspace(.45,.85,k)
H=np.diag(eta*q); P=np.diag(sig/(r*(sig+mu))); Dx=np.diag(xi); Dw=np.diag(om)
C=np.full((k,k),.025); np.fill_diagonal(C,0); B=np.diag(beta)+C
rng=np.random.default_rng(20260913)
U=rng.random(k*(k-1)); G=np.zeros((k,k)); n=0
for j in range(k):
    for i in range(k):
        if i!=j: G[i,j]=.02+.08*U[n]; n+=1
def parts(G,al):
    Q=np.diag(G.sum(0))-G; Mi=np.linalg.inv(Dw+al*Q); K=(B+H@Mi@Dx)@P
    ev,V=np.linalg.eig(K); a=np.argmax(ev.real); v=np.abs(V[:,a].real)
    ev2,W=np.linalg.eig(K.T); b=np.argmax(ev2.real); u=np.abs(W[:,b].real)
    return ev[a].real,u,v,Mi
R=lambda G,al:parts(G,al)[0]
print("G (rounded):\n",np.round(G,4))
print("R0(0)=%.4f  R0(1)=%.4f  R0(20)=%.4f"%(R(G,0),R(G,1),R(G,20)))
al=1.0; R0,u,v,Mi=parts(G,al); Phi=Mi.T@(np.diag(H)*u); 
print("Phi (reproductive value of pathogen by house):",np.round(Phi/Phi.max(),4))
helpful=[];harm=[]
for j in range(k):
    for i in range(k):
        if i!=j:
            (helpful if Phi[i]<Phi[j] else harm).append((j+1,i+1))
print("alpha=1: helpful edges j->i (raising rate lowers R0):",len(helpful),helpful)
print("alpha=1: harmful edges:",len(harm),harm)
# verify types by direct finite change of each edge
bad=0
for (j,i) in helpful+harm:
    for g in [0,.01,.1,1,10]:
        G2=G.copy(); G2[i-1,j-1]=g; 
    seq=[]
    for g in [0,.05,.2,1,5,50]:
        G2=G.copy(); G2[i-1,j-1]=g; seq.append(R(G2,al))
    d=np.diff(seq); ok=(np.all(d<0) if (j,i) in helpful else np.all(d>0))
    bad+= (not ok)
print("edges whose observed direction disagrees with the sign rule:",bad,"of",k*(k-1))
# interventions at alpha=1
Gh=G.copy()                       # remove all harmful edges
for (j,i) in harm: Gh[i-1,j-1]=0
Gb=G.copy()                       # double helpful edges, remove harmful
for (j,i) in helpful: Gb[i-1,j-1]*=5
for (j,i) in harm: Gb[i-1,j-1]=0
Gu=G.copy()                       # remove all movement
print("R0 baseline            %.4f"%R(G,1))
print("R0 no movement         %.4f"%R(Gu*0,1))
print("R0 remove harmful      %.4f"%R(Gh,1))
print("R0 remove harmful, x5 helpful %.4f"%R(Gb,1))
# exact best over vertices {0, 5*g} for each edge is 2^20 ~ 1e6: do coordinate descent with exact endpoint comparison
Gc=G.copy(); lo,hi=0.0,None
for sweep in range(6):
    ch=False
    for j in range(k):
        for i in range(k):
            if i==j: continue
            best=None
            for val in [0.0,5*G[i,j]]:
                G2=Gc.copy(); G2[i,j]=val; rv=R(G2,al)
                if best is None or rv<best[0]-1e-12: best=(rv,val)
            if abs(Gc[i,j]-best[1])>1e-15: Gc[i,j]=best[1]; ch=True
    if not ch: break
print("R0 after coordinate descent on box [0,5g]: %.4f"%R(Gc,1))
print("edges at upper bound:",[(j+1,i+1) for j in range(k) for i in range(k) if i!=j and Gc[i,j]>0])
# range
A=B@P; c=np.diag(H)/np.diag(Dw); d=np.diag(Dx)*np.diag(P)
lam=[max(np.linalg.eigvals(A+c[i]*np.outer(np.eye(k)[i],d)).real) for i in range(k)]
print("hub values",np.round(lam,4))
