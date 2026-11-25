import numpy as np
rho=lambda K: max(np.linalg.eigvals(K).real)
def Kmat(a,B,P,h,w,xi,G):
    Q=np.diag(G.sum(0))-G; return (B+np.diag(h)@np.linalg.inv(np.diag(w)+a*Q)@np.diag(xi))@np.diag(P)
def R0(a,*p): return rho(Kmat(a,*p))
def XY(B,P,h,w,xi,G):
    c=h/w; d=xi*P; K0=B@np.diag(P)+np.diag(c*d); k11,k22,k12,k21=K0[0,0],K0[1,1],K0[0,1],K0[1,0]
    w1=w[1]*G[1,0]*d[0]; w2=w[0]*G[0,1]*d[1]
    return c[0]*c[1]*(k11-k22)+c[0]**2*k21-c[1]**2*k12, w1*w2*(k11-k22)+w1**2*k12-w2**2*k21, K0, (w1,w2)
# --- best-sink example
B=np.array([[0,1.],[1,3]]);P=np.array([2.,3]);h=np.array([2.,1]);w=np.array([2.,2]);xi=np.array([3.,2])
A=B@np.diag(P); d=xi*P; c=h/w
lam=[rho(A+c[i]*np.outer(np.eye(2)[i],d)) for i in range(2)]
print("sink example: r_i =",xi*P*h/w," lambda_i =",lam," (6+sqrt51 =",6+51**.5,")  R0(0)=",rho(A+np.diag(c*d)))
# hub networks approach lambda_i
for i in range(2):
    G=np.zeros((2,2)); G[i,1-i]=1e7; G[1-i,i]=0; G[i,1-i]=1e7  # all flow into house i
    # hub into i: g_{i,j}=n for j!=i, others 0
    print(f"  hub into house {i+1}: R0(alpha=1,n=1e7) =",R0(1.0,B,P,h,w,xi,G))
# --- manuscript two-house laboratory
Dw=np.array([2.,1]);G2=np.array([[0,1.],[3,0]]);H=np.array([1.,3]);X=np.array([3.,1]);P2=np.array([2.,1]);B2=np.array([[0,2.],[0,4]])
x,y,K0,ws=XY(B2,P2,H,Dw,X,G2)
print("lab: K(0)=",K0.tolist()," X=",x," Y=",y," -XY sign:",np.sign(-x*y)," R0(0,1,5,20,1e4)=",[round(R0(a,B2,P2,H,Dw,X,G2),4) for a in (0,1,5,20,1e4)])
# --- B=0 check
B0=np.zeros((2,2)); x,y,_,_=XY(B0,P2,H,Dw,X,G2); print("B=0: X,Y =",x,y," X*Y>=0 ->",x*y>=0)
# --- three-house example
B3=np.array([[0,0,0],[0,0,0],[0,1,2.]]);P3=np.array([2.,1,2]);h3=np.array([1.,3,1]);w3=np.array([1.,2,3]);xi3=np.array([3.,2,1])
G3=np.array([[0,1,0],[1,0,1],[0,0,0.]]); fine=np.concatenate([[0],np.logspace(-3,6,3000)]); v=np.array([R0(a,B3,P3,h3,w3,xi3,G3) for a in fine]); m=v.argmin()
print("auxiliary 3-house system (not in the manuscript): R0(0)=",v[0]," min=",round(v[m],4)," at alpha=",round(fine[m],4)," R0(1e3)=",round(R0(1e3,B3,P3,h3,w3,xi3,G3),4)," R0(1e6)=",round(v[-1],4))
for a in (0.1,0.25,0.475,1,2,5,20,100): print("   alpha",a,round(R0(a,B3,P3,h3,w3,xi3,G3),4))
print("   nonmonotone: min strictly interior and both ends higher:",v[0]>v[m]+1e-6 and v[-1]>v[m]+1e-6)
