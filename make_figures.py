"""Figures 1 and 2 of the manuscript. Run: python3 make_figures.py  (writes fig_three_house.pdf, fig_five_house.pdf)"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .6, "lines.linewidth": 1.6, "legend.frameon": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": .5, "pdf.fonttype": 42})
C1, C2, C3, INK = "#2a78d6", "#d9622b", "#1f9d7a", "#52514e"
rho = lambda A: max(np.linalg.eigvals(A).real)
def K(B, p, h, xi, om, G, al):
    Q = np.diag(G.sum(0)) - G
    return (B + np.diag(h) @ np.linalg.inv(np.diag(om) + al * Q) @ np.diag(xi)) @ np.diag(p)

# ---- Figure 1: Example (three houses, two edges)
B = np.zeros((3, 3)); B[0, 1] = 2; p = [2, 1, 1]; h = [1, 1, 1]; om = [3, 2, 1]; xi = [1, 2, 2]
G = np.zeros((3, 3)); G[0, 2] = 1; G[1, 0] = 1
al = np.logspace(-2, 2, 600)
b1 = 2 / (1 + al); b2 = (al + 5 + np.sqrt(9 * al**2 + 26 * al + 1)) / (2 * (al + 3))
R = np.array([rho(K(B, p, h, xi, om, G, a)) for a in al])
assert np.allclose(R, np.maximum(b1, b2))
Ge = G + 1e-2 * (1 - np.eye(3)) * (G == 0)
Re = np.array([rho(K(B, p, h, xi, om, Ge, a)) for a in al])
fig, ax = plt.subplots(figsize=(4.8, 3.5))
ax.plot(al, b1, ":", color=INK, lw=1, label=r"branch $2/(1+\alpha)$")
ax.plot(al, b2, "--", color=INK, lw=1, label="second branch")
ax.plot(al, R, "-", color=C1, label=r"$R_0(\alpha)$, two edges")
ax.plot(al, Re, "-.", color=C2, label=r"all missing edges at rate $10^{-2}$")
a_star = np.roots([1, 2, 3, -2]); a_star = a_star[np.isreal(a_star)].real[0]
ax.plot([a_star], [2 / (1 + a_star)], "o", color=C1, ms=4)
ax.annotate(r"minimum $\approx1.353$ at $\alpha^*\approx0.478$", (a_star, 2 / (1 + a_star)), (0.9, 1.12), fontsize=8,
            arrowprops=dict(arrowstyle="-", color=INK, lw=.5))
ax.axhline(2, color=INK, lw=.5)
ax.set_xscale("log"); ax.set_ylim(.9, 2.15); ax.set_xlabel(r"movement intensity $\alpha$"); ax.set_ylabel(r"$R_0$")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2); fig.tight_layout(); fig.savefig("fig_three_house.pdf"); plt.close(fig)
print("Fig 1: alpha*=%.4f min=%.4f; perturbed minimum %.4f" % (a_star, 2 / (1 + a_star), Re.min()))

# ---- Figure 2: five-house application
k = 5
mu = np.linspace(.08, .16, k); sig = np.linspace(.65, .95, k); r = np.linspace(.38, .72, k); beta = np.linspace(.25, .55, k)
eta = np.linspace(.7, 1.4, k); q = np.linspace(.8, 1.3, k); xi = np.linspace(.7, 1.2, k); om = np.linspace(.45, .85, k)
h = eta * q; p = sig / (r * (sig + mu)); Cm = np.full((k, k), .025); np.fill_diagonal(Cm, 0); B = np.diag(beta) + Cm
rng = np.random.default_rng(20260913); U = rng.random(k * (k - 1)); G = np.zeros((k, k)); n = 0
for j in range(k):
    for i in range(k):
        if i != j: G[i, j] = .02 + .08 * U[n]; n += 1
Rf = lambda G, a: rho(K(B, p, h, xi, om, G, a))
hub = []
for i in range(k):
    e = np.zeros(k); e[i] = 1
    hub.append(rho((B + np.outer(h, np.ones(k)) * 0 + np.diag(h) @ np.outer(e, np.ones(k)) / om[i] @ np.diag(xi)) @ np.diag(p)))
al = np.logspace(-2, 2, 300); Ra = [Rf(G, a) for a in al]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.4))
a1.axhspan(min(hub), max(hub), color="#e6e5e0", lw=0, label="attainable range (all networks)")
a1.plot(al, Ra, color=C1, label=r"$R_0(\alpha)$, baseline network"); a1.axhline(Rf(G, 0), color=INK, lw=.6, ls=":", label=r"$R_0(0)$")
a1.set_xscale("log"); a1.set_xlabel(r"movement intensity $\alpha$"); a1.set_ylabel(r"$R_0$"); a1.set_title("(a)", loc="left", fontsize=9)
a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=1)
g = np.logspace(-3, np.log10(50), 200)
def edge(j, i):
    out = []
    for x in g:
        G2 = G.copy(); G2[i - 1, j - 1] = x; out.append(Rf(G2, 1.0))
    return out
a2.plot(g, edge(5, 1), "-", color=C3, label=r"edge $5\to1$ (helpful)")
a2.plot(g, edge(1, 5), "--", color=C2, label=r"edge $1\to5$ (harmful)")
a2.axhline(Rf(G, 1.0), color=INK, lw=.6, ls=":", label="baseline")
a2.set_xscale("log"); a2.set_xlabel(r"rate $g$ of the single edge ($\alpha=1$)"); a2.set_title("(b)", loc="left", fontsize=9)
a2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=1); fig.tight_layout(); fig.savefig("fig_five_house.pdf"); plt.close(fig)
print("Fig 2: hub values", np.round(hub, 4), "R0(0)=%.4f R0(1)=%.4f" % (Rf(G, 0), Rf(G, 1)), "edge 5->1 at g=50: %.4f; edge 1->5 at g=50: %.4f" % (edge(5, 1)[-1], edge(1, 5)[-1]))
