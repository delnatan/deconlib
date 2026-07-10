"""Sweep the Hessian-log eps (curvature threshold) and isotropy, and learn a
ground-truth-free quality metric.

Uses a synthetic 3D object imaged through the *real* vectorial widefield PSF
(so the axial missing cone -- the null space that lets flux collapse onto one
plane -- is present), giving us a ground truth to score against. For each
(eps, r) we record the true relative error plus several cheap, ground-truth-free
observables, then ask which observable (or combination) best predicts the true
error -- i.e. which metric we should actually watch when tuning/converging on
real data where no ground truth exists.

Since the regularizer here is the curvature-only Hessian-log (no intensity term,
no quadratic floor -- see erdecon_mlx docstring), there is no collapse-driving
term and eps is the single knob: an absolute curvature threshold separating
noise from edges. The sweep spans eps from far below to far above the feature
curvature scale to locate its (broad) optimum.

Findings are printed and written to scripts/output/erdecon_eps_sweep.txt.
"""

from pathlib import Path

import numpy as np
import mlx.core as mx
from scipy.ndimage import gaussian_filter

from deconlib import compute_widefield_psf, fft_coords
from deconlib.deconvolution import make_forward_model, Hessian3D, erdecon_with_operator

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "erdecon_eps_sweep.txt"
_lines: list[str] = []


def log(msg=""):
    print(msg, flush=True)
    _lines.append(str(msg))


# =============================================================================
# GROUND-TRUTH SYNTHETIC with the real widefield PSF (true missing cone)
# =============================================================================
np.random.seed(0)
shape = (24, 48, 48)
sp = (0.15, 0.065, 0.065)  # (dz, dy, dx) um
truth = np.zeros(shape, np.float32)
# Extended axial structure: a filament climbing through z + a purely-axial
# segment (the worst case for the missing cone) + a few point sources.
for t in range(8, 40):
    z = int(5 + (t - 8) / 32 * 13)
    truth[z, t, 24] = 1.0
truth[6:18, 24, 12] = 0.8
rng0 = np.random.default_rng(5)
for _ in range(8):
    z, y, x = rng0.integers(4, 20), rng0.integers(8, 40), rng0.integers(8, 40)
    truth[z, y, x] += rng0.uniform(0.5, 1.0)
truth = gaussian_filter(truth, 0.6) + 0.01
truth /= truth.max()

pz = fft_coords(shape[0], spacing=sp[0])
psf = compute_widefield_psf(
    z=pz, shape=(31, 31), spacing=sp[1:], wavelength=0.6,
    na=1.4, ni=1.515, ns=1.45, normalize=True,
)
fm = make_forward_model(psf, shape, zoom=1.0)
tp = np.zeros(fm.padded_shape, np.float32)
tp[fm.valid_slices] = truth
clean = np.asarray(fm.op.forward(mx.array(tp)))
rng = np.random.default_rng(1)
obs = np.maximum(clean + 0.01 * rng.standard_normal(clean.shape), 0.0).astype(np.float32)

reg_r = sp[1] / sp[0]  # anisotropic voxel ratio (lateral/axial)
tnorm = np.linalg.norm(truth)


# =============================================================================
# Ground-truth-free observables + the gold-standard true error
# =============================================================================
def observables(result):
    g = np.asarray(result.restored[fm.valid_slices])
    plane = g.mean((1, 2))
    axial_excess = float(plane.max() / (np.median(plane) + 1e-12))
    peak_conc = float(g.max() / (g.sum() + 1e-12))
    rough = float(
        np.abs(np.diff(g, axis=0)).sum()
        + np.abs(np.diff(g, axis=1)).sum()
        + np.abs(np.diff(g, axis=2)).sum()
    ) / g.sum()
    idiv = result.data_misfit_history[-1]
    true_err = float(np.linalg.norm(g - truth) / tnorm)
    return dict(idiv=idiv, axial_excess=axial_excess, peak_conc=peak_conc,
               rough=rough, true_err=true_err)


# =============================================================================
# SWEEP
# =============================================================================
epsilons = [1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
rs = [("aniso", reg_r), ("iso", 1.0)]
lam = 0.02

log(f"Ground-truth synthetic: shape={shape}, real widefield PSF, "
    f"data max={obs.max():.3f}, raw err={np.linalg.norm(obs - truth) / tnorm:.3f}")
log(f"lambda={lam}, sweeping eps x r  "
    f"({len(epsilons)}x{len(rs)} = {len(epsilons) * len(rs)} configs)\n")

rows = []
header = f"{'eps':>8} {'r':>6} | {'true_err':>9} {'idiv':>7} {'ax_exc':>8} {'peakconc':>9} {'rough':>7}"
log(header)
log("-" * len(header))
for rname, rval in rs:
    for eps in epsilons:
        res = erdecon_with_operator(
            obs, fm.op, hessian=Hessian3D(r=rval),
            reg_weight=lam, eps_reg=eps,
            num_iter=60, eval_interval=60, tol=1e-7,
        )
        o = observables(res)
        o.update(eps=eps, r=rval, rname=rname)
        rows.append(o)
        log(f"{eps:>8.2g} {rname:>6} | {o['true_err']:>9.4f} "
            f"{o['idiv']:>7.3f} {o['axial_excess']:>8.2f} {o['peak_conc']:>9.2e} {o['rough']:>7.3f}")

best = min(rows, key=lambda d: d["true_err"])
worst = max(rows, key=lambda d: d["true_err"])
log(f"\nBEST  (lowest true err): eps={best['eps']:.0e} r={best['rname']}  "
    f"true_err={best['true_err']:.4f} ax_exc={best['axial_excess']:.2f} idiv={best['idiv']:.3f}")
log(f"WORST (highest true err): eps={worst['eps']:.0e} r={worst['rname']}  "
    f"true_err={worst['true_err']:.4f} ax_exc={worst['axial_excess']:.2f} idiv={worst['idiv']:.3f}")


# =============================================================================
# LEARN a metric: which observable(s) predict true error?
# =============================================================================
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


te = np.array([r["true_err"] for r in rows])
feat_names = ["idiv", "axial_excess", "peak_conc", "rough"]
feats = {n: np.array([r[n] for r in rows]) for n in feat_names}
# log-transform the heavy-tailed positive observables before correlating/fitting
featsT = {n: (np.log10(feats[n] + 1e-12) if n in ("axial_excess", "peak_conc") else feats[n])
          for n in feat_names}

log("\nRank correlation (Spearman) of each observable with TRUE error:")
for n in feat_names:
    log(f"  {n:>13}: rho = {spearman(feats[n], te):+.3f}")

# Standardized linear predictor of true error, with leave-one-out R^2.
X = np.stack([ (featsT[n] - featsT[n].mean()) / (featsT[n].std() + 1e-12) for n in feat_names ], 1)
X = np.hstack([np.ones((len(te), 1)), X])
y = (te - te.mean()) / (te.std() + 1e-12)


def loo_r2(cols):
    Xc = X[:, [0] + [1 + c for c in cols]]
    preds = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        beta, *_ = np.linalg.lstsq(Xc[m], y[m], rcond=None)
        preds[i] = Xc[i] @ beta
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


log("\nLeave-one-out R^2 predicting TRUE error from ground-truth-free observables:")
single = [(loo_r2([i]), feat_names[i]) for i in range(len(feat_names))]
for r2, n in sorted(single, reverse=True):
    log(f"  {n:>13} alone            : R^2 = {r2:+.3f}")
combos = {
    "idiv + axial_excess": [0, 1],
    "idiv + peak_conc": [0, 2],
    "idiv + axial_excess + rough": [0, 1, 3],
    "all four": [0, 1, 2, 3],
}
for name, cols in combos.items():
    log(f"  {name:>28}: R^2 = {loo_r2(cols):+.3f}")

with open(LOG, "w") as f:
    f.write("\n".join(_lines) + "\n")
log(f"\nWrote {LOG}")
