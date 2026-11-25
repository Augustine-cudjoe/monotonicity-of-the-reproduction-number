"""Checks the two-house criterion:  dR0/dalpha has the sign of  -X*Y,  where (k_ij = entries of K(0)),
   c_i = h_i/omega_i, d_i = xi_i p_i, w1 = omega_2 g_21 d_1, w2 = omega_1 g_12 d_2,
   X = c1 c2 (k11-k22) + c1^2 k21 - c2^2 k12,   Y = w1 w2 (k11-k22) + w1^2 k12 - w2^2 k21."""
import numpy as np
rng = np.random.default_rng(2026)
alphas = np.concatenate([[0], np.logspace(-3, 4, 60)])
def rho2(K):                      # largest eigenvalue of batched 2x2 nonnegative matrices
    tr = K[:, 0, 0] + K[:, 1, 1]; det = K[:, 0, 0]*K[:, 1, 1] - K[:, 0, 1]*K[:, 1, 0]
    return tr/2 + np.sqrt(np.maximum((tr/2)**2 - det, 0))
tot = viol = skipped = nonmono = inc = const_skip = 0
for _ in range(40):
    N = 5000; L = lambda lo, hi, s: 10**rng.uniform(lo, hi, s)
    B = L(-3, 1, (N, 2, 2)) * (rng.random((N, 2, 2)) < .8)          # arbitrary B >= 0, 20% zeros
    P, h, w, xi = (L(-1, 1, (N, 2)) for _ in range(4)); g12 = L(-2, 2, N); g21 = L(-2, 2, N)
    c = h/w; d = xi*P
    K0 = B*P[:, None, :] + np.einsum('ni,ij->nij', c*d, np.eye(2))
    k11, k22, k12, k21 = K0[:, 0, 0], K0[:, 1, 1], K0[:, 0, 1], K0[:, 1, 0]
    w1 = w[:, 1]*g21*d[:, 0]; w2 = w[:, 0]*g12*d[:, 1]
    X = c[:, 0]*c[:, 1]*(k11-k22) + c[:, 0]**2*k21 - c[:, 1]**2*k12
    Y = w1*w2*(k11-k22) + w1**2*k12 - w2**2*k21
    R = np.empty((N, len(alphas)))
    for j, a in enumerate(alphas):
        Delta = w[:, 0]*w[:, 1] + a*(w[:, 0]*g12 + w[:, 1]*g21)
        Minv = np.zeros((N, 2, 2))
        Minv[:, 0, 0] = (w[:, 1] + a*g12)/Delta; Minv[:, 0, 1] = a*g12/Delta
        Minv[:, 1, 0] = a*g21/Delta;             Minv[:, 1, 1] = (w[:, 0] + a*g21)/Delta
        KE = h[:, :, None]*Minv*(xi*P)[:, None, :]
        R[:, j] = rho2(B*P[:, None, :] + KE)
    dif = np.diff(R, axis=1); tol = 1e-9*R.max(1, keepdims=True)
    up = (dif > tol).any(1); dn = (dif < -tol).any(1)
    nonmono += (up & dn).sum()
    XY = X*Y; ok = np.abs(XY) > 1e-8*(np.abs(X)*np.abs(Y)+1e-300) 
    pred_inc = XY < 0
    mask = ok & (up ^ dn)
    viol += ((up & pred_inc == False) & mask & ~pred_inc & up).sum() + ((dn & pred_inc) & mask).sum()
    skipped += (~mask).sum(); tot += N; inc += up.sum()
print(f"{tot} two-house instances: non-monotone {nonmono}; criterion violations {viol}; increasing {inc}; undecidable/flat (skipped) {skipped}")
