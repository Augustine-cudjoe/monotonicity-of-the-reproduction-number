"""Exact values for the two new examples (two-edge nonmonotone example; B=0 cycle-closing example)."""
import numpy as np
from edge_common import *
D=np.diag
print("=== Example (three houses, two edges, one direct entry) ===")
B=np.array([[0,2,0],[0,0,0],[0,0,0.]]);P=D([2,1,1.]);H=D([1,1,1.]);Dw=np.array([3,2,1.]);Dx=D([1,2,2.])
Gv=np.zeros((3,3));Gv[0,2]=1.0;Gv[1,0]=1.0       # edges 3->1 and 1->2 (g_ij: from j to i)
R=lambda G,a: rho(Kmat(B,np.diag(P),np.diag(H),np.diag(Dx),Dw,G,a))
print("R0 at alpha=0,0.1,0.25,0.479,1,2,5,20,100,1e4:",[round(R(Gv,a),4) for a in [0,.1,.25,.479,1,2,5,20,100,1e4]])
ag=np.concatenate([[0],np.logspace(-3,3,2000)]);v=np.array([R(Gv,a) for a in ag]);print("min",round(v.min(),4),"at alpha",round(ag[v.argmin()],3))
A=B@P; c=np.diag(H)/Dw; d=np.diag(Dx)*np.diag(P); hub=[rho(A+c[i]*np.outer(np.eye(3)[i],d)) for i in range(3)]
print("hub values lambda_i:",[round(x,4) for x in hub],"(sink = house 2 of the chain gives limit",round(hub[1],4),")")
for eps in [1e-3,1e-2]:
    G=Gv.copy()
    for i in range(3):
        for j in range(3):
            if i!=j and G[i,j]==0: G[i,j]=eps
    v=np.array([R(G,a) for a in ag]);dd=np.diff(v)
    print(f"  all other edges at rate {eps} (strongly connected): R0(0)={v[0]:.4f}, min {v.min():.4f}, limit {v[-1]:.4f}, nonmonotone {(dd>1e-9).any() and (dd<-1e-9).any()}")
print("=== Example (B=0, closing a cycle raises R0) ===")
P=D([2,3,1.]);H=D([3,2,3.]);Dw=np.array([1,1,1.]);Dx=D([2,2,2.]);B=np.zeros((3,3))
G0=np.zeros((3,3));G0[1,0]=1.0;G0[2,1]=1.0        # edges 1->2, 2->3 (house indices 1,2,3)
R=lambda G,a: rho(Kmat(B,np.diag(P),np.diag(H),np.diag(Dx),Dw,G,a))
print("local ratios r_i = c_i d_i:",(np.diag(H)/Dw)*(np.diag(Dx)*np.diag(P)))
for g in [0,.5,1,2,10,100]:
    G=G0.copy();G[0,2]=g;print(f"  edge 3->1 at rate {g}: R0 = {R(G,1.0):.4f}")
