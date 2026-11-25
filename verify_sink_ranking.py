import numpy as np
rho=lambda A: max(abs(np.linalg.eigvals(A)))
def lam(B,P,h,w,xi):
    A=B@np.diag(P); d=xi*P; k=len(h)
    return np.array([rho(A+(h[i]/w[i])*np.outer(np.eye(k)[i],d)) for i in range(k)])
# five-house data from reproduce script
k=5
mu=np.linspace(.08,.16,k);sig=np.linspace(.65,.95,k);r=np.linspace(.38,.72,k)
beta=np.linspace(.25,.55,k);eta=np.linspace(.7,1.4,k);q=np.linspace(.8,1.3,k)
xi=np.linspace(.7,1.2,k);om=np.linspace(.45,.85,k)
C=np.full((k,k),.025);np.fill_diagonal(C,0);B=np.diag(beta)+C
P=(sig/r)/(sig+mu);h=eta*q
L=lam(B,P,h,om,xi); ri=xi*P*h/om
print("five-house  lambda_i:",np.round(L,4)); print("            r_i     :",np.round(ri,4))
print("            argmin lambda",L.argmin()+1,"argmin r",ri.argmin()+1,"| argmax lambda",L.argmax()+1,"argmax r",ri.argmax()+1)
# random: how often does ranking by local ratio r_i pick the wrong best sink / worst hub?
rng=np.random.default_rng(1); N=20000; wrong_best=wrong_worst=0; gap=[]
for _ in range(N):
    k=rng.integers(3,7)
    B=rng.random((k,k))*rng.choice([.05,.3,1.0]); h=rng.random(k)+.1; w=rng.random(k)+.1; xi=rng.random(k)+.1; P=rng.random(k)+.1
    L=lam(B,P,h,w,xi); ri=xi*P*h/w
    wrong_best+= L.argmin()!=ri.argmin(); wrong_worst+= L.argmax()!=ri.argmax()
    gap.append((L.max()-L.min())/L.min())
print("random instances:",N,"| best sink != argmin r_i:",wrong_best/N,"| worst hub != argmax r_i:",wrong_worst/N)
# explicit small example where r-ranking fails
k=2
for _ in range(100000):
    B=rng.integers(0,4,(2,2)).astype(float); h=rng.integers(1,4,2).astype(float); w=rng.integers(1,4,2).astype(float); xi=rng.integers(1,4,2).astype(float); P=rng.integers(1,4,2).astype(float)
    L=lam(B,P,h,w,xi); ri=xi*P*h/w
    if L.argmin()!=ri.argmin() and abs(L[0]-L[1])>.2 and abs(ri[0]-ri[1])>.2 and rho(B@np.diag(P))>0:
        print("2-house example: B=",B.tolist(),"h=",h,"w=",w,"xi=",xi,"P=",P,"lambda=",np.round(L,4),"r=",np.round(ri,4));break
