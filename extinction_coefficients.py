"""Extinction coefficients A_r'/E(B-V) and E(g'-r')/E(B-V) for the isochrone fit.

Computed from first principles, with no dust-map recalibration:
  * extinction law A(lambda)/A_V for diffuse Milky Way dust, R_V = 3.1:
      - Cardelli, Clayton & Mathis 1989, ApJ 345, 245 (optical, eq. 3a/3b)
      - O'Donnell 1994, ApJ 422, 158 (revised optical coefficients)
  * SDSS g and r system response curves (filter_curves/sdss_*.dat), from the
    SVO Filter Profile Service (Rodrigo et al. 2012; Rodrigo & Solano 2020)
  * blackbody stars at the temperatures of the fit anchors:
    4800 K (red clump), 5800 K (subgiants), 6500 K (turnoff)

For each band, A_b = -2.5 log10( int F T 10^(-0.4 A(lambda)) lambda dlam / int F T lambda dlam ),
evaluated for a small A_V (0.3 mag, similar to NGC 6819) and divided by E(B-V) = A_V / R_V.
The adopted value is the mean of the 6 law x temperature cases; half the range is the systematic.

Run:  .venv/bin/python extinction_coefficients.py
"""
import os
import numpy as np

R_V = 3.1
A_V = 0.30
TEMPS = (4800, 5800, 6500)
HERE = os.path.dirname(os.path.abspath(__file__))


def a_over_av(lam_A, rv=R_V, law="CCM89"):
    """A(lambda)/A_V in the optical range 1.1 <= x <= 3.3 um^-1 (both SDSS g and r lie inside it)."""
    x = 1e4 / lam_A
    assert np.all((x >= 1.1) & (x <= 3.3)), "wavelength outside the optical range of the law"
    y = x - 1.82
    if law == "CCM89":
        a = np.polyval([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.0], y)
        b = np.polyval([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.0], y)
    elif law == "ODonnell94":
        a = np.polyval([-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0], y)
        b = np.polyval([3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0], y)
    else:
        raise ValueError(law)
    return a + b / rv


def blackbody(lam_A, T):
    h, c, k = 6.626e-27, 2.998e10, 1.381e-16
    lam = lam_A * 1e-8
    return 1.0 / lam ** 5 / np.expm1(h * c / (lam * k * T))


def band_extinction(lam, trans, T, law):
    w = blackbody(lam, T) * trans * lam          # photon-counting weight
    return -2.5 * np.log10(np.trapezoid(w * 10 ** (-0.4 * A_V * a_over_av(lam, law=law)), lam)
                           / np.trapezoid(w, lam)) / (A_V / R_V)


def coefficients(verbose=True):
    curves = {b: np.loadtxt(os.path.join(HERE, "filter_curves", f"sdss_{b}.dat")) for b in "gr"}
    if verbose:
        for law in ("CCM89", "ODonnell94"):     # sanity check: the law gives back R_V from B (4400 A) and V (5500 A)
            print(f"check {law}: R_V from A_B/A_V = {1 / (a_over_av(np.array([4400.]), law=law)[0] - 1):.3f}")
    rows = []
    for law in ("CCM89", "ODonnell94"):
        for T in TEMPS:
            Ag, Ar = (band_extinction(curves[b][:, 0], curves[b][:, 1], T, law) for b in "gr")
            rows.append((law, T, Ag, Ar, Ag - Ar))
            if verbose:
                print(f"  {law:<10s} {T} K:  A_g/E(B-V) {Ag:.3f}  A_r/E(B-V) {Ar:.3f}  E(g-r)/E(B-V) {Ag - Ar:.3f}")
    ar = np.array([r[3] for r in rows]); egr = np.array([r[4] for r in rows])
    out = dict(A_r=ar.mean(), A_r_sys=np.ptp(ar) / 2, E_gr=egr.mean(), E_gr_sys=np.ptp(egr) / 2)
    if verbose:
        print(f"adopted: A_r/E(B-V) = {out['A_r']:.3f} +- {out['A_r_sys']:.3f},  "
              f"E(g-r)/E(B-V) = {out['E_gr']:.3f} +- {out['E_gr_sys']:.3f}  (mean +- half range, R_V = {R_V})")
    return out


if __name__ == "__main__":
    coefficients()
