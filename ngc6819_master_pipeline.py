"""
====================================================================================================
NGC 6819 (THE FOXHEAD) - END-TO-END GIT PHOTOMETRY & ISOCHRONE PIPELINE
====================================================================================================
Observer : Veeresh S K (4th year B.Tech, Electrical Engineering)
Telescope: 0.7 m GROWTH-India Telescope (GIT), Hanle
Frames   : 20261012170-258-RA.wcs.proc.fits  (SDSS r', 180 s)
           20261012170-727-RA.wcs.proc.fits  (SDSS g', 180 s)

Stages
  1. FITS pixels -> instrumental magnitudes
       Background2D sky model, DAOStarFinder detection, empirical FWHM, EPSFBuilder ePSF,
       PSFPhotometry + 1.5xFWHM aperture photometry (aperture fallback for bright/flagged stars),
       r'/g' cross-match, APT cross-check sample.
  2. Instrumental -> calibrated magnitudes and membership
       Pan-STARRS DR1 -> SDSS (Tonry et al. 2012) reference magnitudes, Siegel robust
       zero-point + colour-term fit with bootstrap errors, Gaia DR3 cross-match, proper-motion /
       parallax membership, manual SAOImageDS9 circle star counts.
  3. Calibrated CMD -> physical parameters
       Binary / blue-straggler masking, reddening-uniformity test, unforced multi-feature PARSEC
       v1.2S fit over the full (logAge, mu0, E(B-V)) grid, bootstrap + calibration-systematic
       uncertainties, turnoff mass, star-by-star Teff vs Gaia DR3 GSP-Phot.
  4. Outputs
       Five 300-dpi figures, ds9_cluster_members.reg, apt_crosscheck_sample.csv,
       pipeline_results.json and NGC6819_Final_Report.pdf.  Every number in the PDF is read from
       pipeline_results.json; the written discussion is read from report_text.md (your own words).

Usage
  python ngc6819_master_pipeline.py                  # default: --use-cache (reuse Stage 1 photometry)
  python ngc6819_master_pipeline.py --from-scratch   # redo Stage 1 from the FITS pixels (~10 min)
  python ngc6819_master_pipeline.py --report-only    # rebuild only the PDF after editing report_text.md
  python ngc6819_master_pipeline.py --n-boot 200     # more bootstrap resamples for the fit
  python ngc6819_master_pipeline.py --spatial-filter circle   # fit only Gaia members inside your DS9 circle
====================================================================================================
"""

import argparse
import gzip
import html
import inspect
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.table import Table
from astropy.nddata import NDData, Cutout2D
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

from scipy.spatial import cKDTree
from scipy.optimize import minimize, curve_fit
from scipy.stats import siegelslopes, chi2 as chi2_dist

warnings.filterwarnings("ignore", category=FITSFixedWarning)

# ==================================================================================================
# CONFIGURATION (every tunable choice lives here so it can be quoted in the report)
# ==================================================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

FRAMES = {"r": "20261012170-258-RA.wcs.proc.fits",
          "g": "20261012170-727-RA.wcs.proc.fits"}

CIRCLE_FILE = "manual_cluster_circle.reg"
BKG_ANNULUS_FACTORS = (1.5, 2.5)       # background ring = 1.5R .. 2.5R (on-chip area only)

REPORT_TEXT_FILE = "report_text.md"
SCREENSHOT_DIR = "screenshots"
RESULTS_JSON = "pipeline_results.json"
STAGE1_JSON = "stage1_diagnostics.json"
STAGE1_NPZ = "stage1_diagnostics.npz"
PDF_FILE = "NGC6819_Final_Report.pdf"
GITHUB_URL = "https://github.com/Veeresh-S-K/NCG_6819_analysis"

# Fixed literature inputs (observation_report.pdf, Section 4)
FEH = +0.09
Z_INI = 0.0185
# Extinction coefficients from extinction_coefficients.py: CCM89 and O'Donnell (1994) laws (R_V = 3.1) integrated over
# the SDSS g, r response curves (SVO) for 4800-6500 K stars; mean of the 6 cases (half range: +-0.017, +-0.039).
# (The proposal's 2.27 = Schlafly & Finkbeiner 2011 SDSS r per SFD-map E(B-V); not used: our E(B-V) is not from SFD.)
A_R_PER_EBV = 2.71          # A_r' / E(B-V)
E_GR_PER_EBV = 1.03         # E(g'-r') / E(B-V)
R_V = 3.1                   # (m-M)_V = mu0 + R_V E(B-V); ASSUMPTION: Milky Way diffuse-dust average

# Stage 1 photometry
BKG_BOX = 128
DETECT_NSIGMA = 4.0
BORDER_PIX = 25
APER_FACTORS = (1.0, 1.5, 2.0)     # aperture radii in units of FWHM (1.5 is the science aperture)
ANNULUS_FACTORS = (3.0, 5.0)       # local sky annulus in units of FWHM
PSF_FIT_SHAPE = (9, 9)
BRIGHT_PEAK_FALLBACK = 30000.0     # ADU above sky: use aperture photometry above this peak
MATCH_GR_ARCSEC = 1.0              # r'-g' cross-match radius
FLAT_FIELD_ERR = 0.0075            # per-pixel flat-field error: DAOPHOT II default "percent error" 0.75% (DAOPHOT II
                                   # User's Manual, parameter 18; Stetson 1987, 1992). Applied to the star signal only:
                                   # the measured sky rms already contains the flat-field structure of the sky.

# Stage 2 calibration (PS1 -> SDSS, Tonry et al. 2012, inverted to use PS1 (g-r) as argument)
PS1_TO_SDSS = {"g": (0.014, 0.162), "r": (-0.001, 0.011)}   # x_SDSS = x_PS1 + a + b (g-r)_PS1
PS1_REF_CUTS = dict(r_min=13.5, r_max=17.5, g_min=14.0, g_max=18.5, err_max=0.03,
                    psf_kron_max=0.05, inst_err_max=0.03, peak_max=45000.0)
CAL_SYSTEMATIC_FLOOR = 0.02        # ASSUMPTION: adopted systematic of the PS1->SDSS transformation [mag]
N_BOOT_ZP = 500
VIZIER_SERVERS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu")   # main site, then mirror
MAST_PS1_URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv"   # PS1 fallback
MAST_PAGESIZE = 1000000
MATCH_CAT_ARCSEC = 0.8             # GIT-PS1 and GIT-Gaia match radius

# Stage 2 membership
PMEMB_MIN = 0.70                   # Cantat-Gaudin et al. (2018) probability
RUWE_MAX = 1.4                     # single-star astrometry filter used for EDR3/DR3 by Fabricius et al. (2021)
PM_SIGMA = 0.25                    # mas/yr, adopted width of the cluster proper-motion clump
PM_NSIGMA = 3.0
PLX_RANGE = (0.25, 0.50)           # mas
GAIA_PLX_ZP = -0.017               # mas, Lindegren et al. (2021) median of faint QUASARS - shown for illustration only:
                                   # not valid for these bright stars (the per-star recipe was not applied)
GAIA_PLX_SYS = 0.010               # mas, floor on a cluster-mean parallax (Vasiliev & Baumgardt 2021)

# Stage 3 fit
MAG_SCALE = 1.0                    # plain CMD distance (no r' axis compression; 1-6 tested, same fit)
FEATURE_WINDOWS = {                # observed-CMD windows that define each anchor feature
    "MS":   dict(r=(15.3, 18.5), weight=1.0),     # equal weights: each feature's median distance counts once
    "MSTO": dict(r=(14.6, 15.3), weight=1.0),
    "SGB":  dict(r=(14.4, 15.0), gr=(0.60, 0.88), weight=1.0),
    "RC":   dict(r=(12.0, 13.5), weight=1.0),
}
# Each feature is compared only with its own evolutionary phase of the PARSEC isochrone ('label' column:
# 1 = main sequence, 2 = subgiant branch, 4 = core-He burning). Without this, red-clump stars can be matched to the
# nearby red-giant branch and the fit can slide to a wrong (age, mu0) solution.
FEATURE_PHASES = {"MS": (1,), "MSTO": (1, 2), "SGB": (2,), "RC": (4,)}
MU0_RANGE = (11.5, 12.8)
EBV_RANGE = (0.0, 0.25)
GRID_STEP_FINE = (0.02, 0.005)     # (mu0, E(B-V)) grid used for the main fit
GRID_STEP_BOOT = (0.04, 0.010)     # coarser grid inside the bootstrap (then Nelder-Mead refine)
BOOT_AGE_WINDOW = 0.10             # dex around the best logAge searched in each bootstrap
TEFF_MAG_LIMIT = 16.5

# Gaia-free membership: statistical field-star subtraction with the background ring
DECON_SCALE = (0.10, 0.50)         # CMD distance units: 0.10 mag in g'-r', 0.50 mag in r'
DECON_DMAX = 2.0                   # a field star removes a circle star at most 0.2 mag (colour) / 1 mag (r') away
DECON_K = 40                       # nearest circle stars considered per field star
N_DECON_REAL = 100                 # random realisations of the subtraction
P_STAT_MIN = 0.5                   # Gaia-free member = survives in at least half of the realisations

# Proposal Section 5 benchmarks (verbatim values from observation_report.pdf)
BENCH = {
    # values = the proposal's benchmarks (Section 5); source strings checked against the papers on ADS (2026-11-18)
    "mMV": (12.38, 0.04, "Brewer et al. (2016), 3 eclipsing binaries"),
    "mu0_astero": (11.85, 12.20, "Basu et al. (2011) 11.85+/-0.05; Balona et al. (2013) 12.20+/-0.06; red-giant asteroseismology"),
    "dist_kpc": (2.3, 2.8, "proposal range, method dependent"),
    "age_gyr": (2.2, 2.5, "proposal range; Brewer et al. (2016): 2.21+/-0.10+/-0.20 Gyr (CMD), 2.38+/-0.05+/-0.22 Gyr (binaries)"),
    "ebv": (0.10, 0.16, "Kalirai et al. (2001) adopted 0.10; Anthony-Twarog et al. (2014) measured 0.160+/-0.007 (uvby-Hbeta)"),
    "mto": (1.25, 0.06, "Brewer et al. (2016), preliminary mass of turnoff star WOCS 24009 A (1.251+/-0.057)"),
    "crit_mu": 0.15, "crit_age": 0.30,
}


# ==================================================================================================
# SMALL UTILITIES
# ==================================================================================================

def err_lo_hi(q):
    """(lower, upper) total error of a fit parameter; old results files only have the symmetric 'total'."""
    return q.get("total_lo", q["total"]), q.get("total_hi", q["total"])


def pm_txt(q, prec):
    lo, hi = err_lo_hi(q)
    return f"{q['value']:.{prec}f} +{hi:.{prec}f}/-{lo:.{prec}f}"


def pm_html(q, prec):
    lo, hi = err_lo_hi(q)
    return f"{q['value']:.{prec}f}<super>+{hi:.{prec}f}</super><sub>-{lo:.{prec}f}</sub>"


def banner(title):
    print("\n" + "=" * 92 + "\n" + title + "\n" + "=" * 92, flush=True)


def robust_std(x):
    """1.4826 x median absolute deviation (Gaussian-equivalent sigma, insensitive to outliers)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def find_catalog(name):
    for p in (name, os.path.join("reference_catalogs", name)):
        if os.path.exists(p):
            return p
    return None


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def ensure_fits(band):
    """Decompress <frame>.fits.gz once (the .gz archive is kept)."""
    path = FRAMES[band]
    if not os.path.exists(path):
        gz = path + ".gz"
        if not os.path.exists(gz):
            raise FileNotFoundError(f"Neither {path} nor {gz} found")
        print(f"[{band}] decompressing {gz} -> {path}")
        with gzip.open(gz, "rb") as fi, open(path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return path


def load_frame(band, with_data=True):
    path = ensure_fits(band)
    with fits.open(path) as hdul:
        hdr = hdul[0].header.copy()
        data = hdul[0].data.astype(np.float32) if with_data else None
    return data, hdr, WCS(hdr)


# ==================================================================================================
# TASK 2: POINTING CENTRE, FIELD OF VIEW AND CATALOGUE SEARCH RADIUS FROM THE FITS HEADER + WCS
# ==================================================================================================

def _parse_header_coord(ra_val, dec_val):
    """TARRA/TARDEC may be stored as decimal degrees or as sexagesimal strings."""
    if isinstance(ra_val, (int, float, np.floating)) and isinstance(dec_val, (int, float, np.floating)):
        return SkyCoord(float(ra_val) * u.deg, float(dec_val) * u.deg, frame="icrs")
    ra_s, dec_s = str(ra_val).strip(), str(dec_val).strip()
    if any(c in ra_s.lower() for c in ":hms "):
        return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
    return SkyCoord(float(ra_s) * u.deg, float(dec_s) * u.deg, frame="icrs")


def read_field_geometry(hdr, wcs):
    """
    Pointing centre from TARRA/TARDEC (fallback: WCS world position of the frame centre),
    detector FOV from the WCS pixel scales, and the radius needed to cover every detector pixel.
    """
    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    try:
        centre = _parse_header_coord(hdr["TARRA"], hdr["TARDEC"])
        source = "TARRA/TARDEC header keywords"
    except (KeyError, ValueError):
        centre = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        source = "WCS world position of the detector centre (TARRA/TARDEC missing)"
    scales = proj_plane_pixel_scales(wcs) * 3600.0          # arcsec / pixel along x, y
    fov_x, fov_y = nx * scales[0] / 60.0, ny * scales[1] / 60.0
    corners = wcs.pixel_to_world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
    frame_centre = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    cx, cy = wcs.world_to_pixel(centre)
    return dict(
        centre=centre, centre_source=source,
        centre_hms=centre.to_string("hmsdms", sep=":", precision=1),
        ra_deg=centre.ra.deg, dec_deg=centre.dec.deg,
        centre_pix=(float(cx), float(cy)), nx=nx, ny=ny,
        pix_scale_x=float(scales[0]), pix_scale_y=float(scales[1]),
        pix_scale=float(np.mean(scales)),
        fov_x_arcmin=float(fov_x), fov_y_arcmin=float(fov_y),
        r_inscribed_arcmin=float(0.5 * min(fov_x, fov_y)),
        r_circumscribed_arcmin=float(0.5 * np.hypot(fov_x, fov_y)),
        r_search_arcmin=float(centre.separation(corners).arcmin.max()),
        pointing_offset_arcsec=float(centre.separation(frame_centre).arcsec),
    )


def _vizier_query(label, fn):
    """Run fn(server) on each VizieR server in turn; return (table as DataFrame, server)."""
    from astroquery.vizier import conf
    conf.timeout = 600
    for server in VIZIER_SERVERS:
        try:
            res = fn(server)
            if len(res) and len(res[0]):
                return res[0].to_pandas(), f"VizieR ({server})"
            print(f"  {label}: VizieR server {server} returned no rows")
        except Exception as err:          # HTTP error, timeout
            print(f"  {label}: VizieR server {server} failed: {err}")
    return None, None


def _ps1_from_mast(centre, radius_deg):
    """
    PS1 DR1 mean photometry from the MAST catalogue API, renamed to the VizieR II/349 column names.
    nDetections >= 3 reproduces the VizieR II/349 selection (checked: all 50,000 rows of an older VizieR
    download are present with identical magnitudes). One request, no paging: MAST's paging is not stable
    (it returned duplicated and missing objects), so the query fails loudly if the page limit is reached.
    """
    import io, urllib.parse, urllib.request
    cols = ["objID", "raMean", "decMean", "nDetections", "gMeanPSFMag", "gMeanPSFMagErr", "rMeanPSFMag",
            "rMeanPSFMagErr", "gMeanKronMag", "rMeanKronMag"]
    q = urllib.parse.urlencode({"ra": centre.ra.deg, "dec": centre.dec.deg, "radius": radius_deg,
                                "nDetections.gte": 3, "columns": "[" + ",".join(cols) + "]",
                                "pagesize": MAST_PAGESIZE})
    with urllib.request.urlopen(f"{MAST_PS1_URL}?{q}", timeout=900) as resp:
        ps1 = pd.read_csv(io.StringIO(resp.read().decode()))
    if len(ps1) >= MAST_PAGESIZE or ps1["objID"].duplicated().any():
        raise RuntimeError(f"MAST returned {len(ps1)} rows (page limit {MAST_PAGESIZE}) or duplicate objIDs")
    ps1 = ps1.replace(-999.0, np.nan)
    return ps1.rename(columns={"raMean": "RAJ2000", "decMean": "DEJ2000", "gMeanPSFMag": "gmag",
                               "gMeanPSFMagErr": "e_gmag", "rMeanPSFMag": "rmag", "rMeanPSFMagErr": "e_rmag",
                               "gMeanKronMag": "gKmag", "rMeanKronMag": "rKmag"})


def query_reference_catalogs(geom):
    """
    Download PS1 DR1, Gaia DR3 and the Cantat-Gaudin (2018) NGC 6819 members out to the WCS-derived
    search radius, with no row limit. VizieR servers are tried in order; PS1 falls back to the MAST
    Pan-STARRS DR1 API (same survey release). Nothing is written until all three downloads have
    succeeded, so a failed query never leaves a mixed cache behind. Sources go to reference_catalogs/SOURCES.txt.
    """
    from astroquery.vizier import Vizier
    radius = geom["r_search_arcmin"] * u.arcmin
    centre = geom["centre"]
    print(f"Downloading reference catalogues around {geom['centre_hms']} with r = {radius:.2f} ...")
    ps1, src_ps1 = _vizier_query("PS1 DR1", lambda s: Vizier(
        columns=["RAJ2000", "DEJ2000", "objID", "gmag", "e_gmag", "rmag", "e_rmag", "rKmag", "gKmag"],
        row_limit=-1, vizier_server=s).query_region(centre, radius=radius, catalog="II/349/ps1"))
    if ps1 is None:
        try:
            ps1, src_ps1 = _ps1_from_mast(centre, radius.to(u.deg).value), "MAST Pan-STARRS DR1 mean-object API"
        except Exception as err:
            raise SystemExit(f"ERROR: PS1 download failed on VizieR and MAST ({err}); cached catalogues left untouched.")
    g, src_g = _vizier_query("Gaia DR3", lambda s: Vizier(
        columns=["Source", "RA_ICRS", "DE_ICRS", "Plx", "e_Plx", "pmRA", "e_pmRA", "pmDE", "e_pmDE",
                 "RUWE", "Gmag", "BPmag", "RPmag", "Teff"],
        row_limit=-1, vizier_server=s).query_region(centre, radius=radius, catalog="I/355/gaiadr3"))
    cg, src_cg = _vizier_query("Cantat-Gaudin 2018", lambda s: Vizier(
        row_limit=-1, column_filters={"Cluster": "NGC_6819"},
        vizier_server=s).query_constraints(catalog="J/A+A/618/A93/members"))
    if g is None or cg is None:
        raise SystemExit("ERROR: Gaia DR3 / Cantat-Gaudin download failed on every VizieR server; "
                         "cached catalogues left untouched.")
    g = g.rename(columns={"Source": "source_id", "RA_ICRS": "ra", "DE_ICRS": "dec", "Plx": "parallax",
                          "e_Plx": "parallax_error", "pmRA": "pmra", "e_pmRA": "pmra_error", "pmDE": "pmdec",
                          "e_pmDE": "pmdec_error", "RUWE": "ruwe", "Gmag": "phot_g_mean_mag",
                          "BPmag": "phot_bp_mean_mag", "RPmag": "phot_rp_mean_mag", "Teff": "teff_gspphot"})
    os.makedirs("reference_catalogs", exist_ok=True)
    ps1.to_csv("reference_catalogs/ps1.csv", index=False)
    g.to_csv("reference_catalogs/gaia_dr3.csv", index=False)
    cg.to_csv("reference_catalogs/cantat_gaudin.csv", index=False)
    with open("reference_catalogs/SOURCES.txt", "w") as f:
        f.write(f"Downloaded {datetime.now():%Y-%m-%d %H:%M} around {geom['centre_hms']}, r = {radius:.2f}, no row limit\n")
        f.write(f"ps1.csv            {len(ps1):>7,} rows  PS1 DR1 (II/349)        from {src_ps1}\n")
        f.write(f"gaia_dr3.csv       {len(g):>7,} rows  Gaia DR3 (I/355/gaiadr3) from {src_g}\n")
        f.write(f"cantat_gaudin.csv  {len(cg):>7,} rows  J/A+A/618/A93 (NGC 6819) from {src_cg}\n")
    print(open("reference_catalogs/SOURCES.txt").read())


# ==================================================================================================
# TASK 1: SAOImageDS9 CIRCLE REGION PARSER + SPATIAL STAR COUNTS
# ==================================================================================================

_RADIUS_UNITS = {'"': u.arcsec, "'": u.arcmin, "d": u.deg}
_CIRCLE_RE = re.compile(r"^circle\s*\(\s*([^,\s()]+)\s*,\s*([^,\s()]+)\s*,\s*([^,\s()]+)\s*\)$", re.IGNORECASE)


def _circle_error(path, msg):
    raise SystemExit(f"\nERROR in cluster circle file '{path}': {msg}\n"
                     "Expected format (SAOImageDS9: Region > Save, format 'ds9', coordinate system 'icrs'):\n"
                     "    icrs\n"
                     "    circle(19:41:18.0,+40:11:12,9.5')\n")


def parse_ds9_circle(path, wcs):
    """
    Read the cluster circle from a DS9 region file. Exactly one format is accepted:
        icrs                                   <- coordinate-system line, before the circle
        circle(RA, Dec, radius)                <- exactly one circle
    RA/Dec: sexagesimal (19:41:18.0, +40:11:12) or decimal degrees; radius must carry a unit:
    " (arcsec), ' (arcmin) or d (degrees). '#' comments and the 'global' line are ignored.
    Any other coordinate system (fk5, image, physical, galactic, ...), any other shape, an
    exclusion region, a second circle or a radius without a unit stops the pipeline with an error.
    """
    system, circles = None, []
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line or line.lower().startswith("global"):
                continue
            for part in (p.strip() for p in line.split(";")):
                if not part:
                    continue
                if part.lower() == "icrs":
                    system = "icrs"
                    continue
                m = _CIRCLE_RE.match(part)
                if not m:
                    _circle_error(path, f"line {n} '{part}' is not supported (only an 'icrs' line and one circle).")
                if system != "icrs":
                    _circle_error(path, f"line {n}: the circle comes before an 'icrs' coordinate-system line.")
                circles.append((n, part, m.groups()))
    if len(circles) != 1:
        _circle_error(path, f"found {len(circles)} circles; exactly one is required.")
    n, text, (a, b, rad) = circles[0]

    try:
        if ":" in a:
            centre = SkyCoord(a, b, unit=(u.hourangle, u.deg), frame="icrs")
        else:
            centre = SkyCoord(float(a) * u.deg, float(b) * u.deg, frame="icrs")
    except ValueError as err:
        _circle_error(path, f"line {n}: cannot read the centre '{a}, {b}' ({err}).")
    if rad[-1] not in _RADIUS_UNITS:
        _circle_error(path, f"line {n}: radius '{rad}' has no unit; write e.g. 9.5' or 570\".")
    radius_arcmin = (float(rad[:-1]) * _RADIUS_UNITS[rad[-1]]).to(u.arcmin).value

    scale_arcsec = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    x0, y0 = (float(v) for v in wcs.world_to_pixel(centre))     # 0-based (Python) pixels
    return dict(file=path, system="icrs", region_text=text, centre=centre,
                ra_deg=float(centre.ra.deg), dec_deg=float(centre.dec.deg),
                centre_hms=centre.to_string("hmsdms", sep=":", precision=1),
                radius_arcmin=float(radius_arcmin), x0=x0, y0=y0,
                r_pix=float(radius_arcmin * 60.0 / scale_arcsec))


def circle_edge_margin(circle, wcs, nx, ny, n_points=720):
    """Smallest distance (pixels) between the circle's edge and the detector edge; negative = off the chip."""
    pa = np.linspace(0.0, 360.0, n_points, endpoint=False) * u.deg
    edge = circle["centre"].directional_offset_by(pa, circle["radius_arcmin"] * u.arcmin)
    px, py = wcs.world_to_pixel(edge)
    # pixel centres run 0..nx-1, so the detector's outer boundary is at -0.5 and nx-0.5
    return float(np.min([px + 0.5, nx - 0.5 - px, py + 0.5, ny - 0.5 - py]))


def load_manual_circle(path, chips):
    """
    chips = {band: (wcs, nx, ny)}. The circle is parsed once (with the first band's WCS for the
    pixel position) and must lie entirely on every frame; otherwise the pipeline stops.
    """
    if not os.path.exists(path):
        _circle_error(path, "file not found. Draw the cluster circle in SAOImageDS9 and save it here.")
    first = next(iter(chips))
    circ = parse_ds9_circle(path, chips[first][0])
    scale = circ["radius_arcmin"] * 60.0 / circ["r_pix"]
    margins = {}
    for band, (wcs, nx, ny) in chips.items():
        m_pix = circle_edge_margin(circ, wcs, nx, ny)
        margins[band] = m_pix * scale / 60.0
        if m_pix < 0:
            _circle_error(path, f"part of the circle lies outside the {band}' frame (up to {-m_pix:.0f} px = "
                                f"{-margins[band]:.2f}' beyond the detector edge). The whole circle must be on the image; "
                                f"the largest radius that fits here is {circ['radius_arcmin'] + margins[band]:.2f}'.")
    circ["edge_margin_arcmin"] = margins
    print(f"  Manual circle (icrs): centre {circ['centre_hms']}, r = {circ['radius_arcmin']:.2f}' "
          f"(pixel x,y = {circ['x0'] + 1:.1f}, {circ['y0'] + 1:.1f}, r = {circ['r_pix']:.1f} px); fully on the chip, "
          + ", ".join(f"{b}' edge margin {m:.2f}'" for b, m in margins.items()))
    return circ


def onchip_area_arcmin2(centre, r1, r2, wcs, nx, ny, border=BORDER_PIX, step=0.12):
    """Area (arcmin^2) of the ring r1 <= r < r2 that falls on usable detector pixels (grid integration)."""
    g = np.arange(-r2, r2 + step, step)
    dx, dy = np.meshgrid(g, g)
    rr = np.hypot(dx, dy)
    sel = (rr >= r1) & (rr < r2)
    dec = centre.dec.deg + dy[sel] / 60.0
    ra = centre.ra.deg + dx[sel] / 60.0 / np.cos(np.radians(dec))
    px, py = wcs.world_to_pixel_values(ra, dec)
    ok = (px > border) & (px < nx - 1 - border) & (py > border) & (py < ny - 1 - border)
    return float(ok.sum() * step * step), float(sel.sum() * step * step)


def spatial_star_counts(circle, wcs, nx, ny, samples, members):
    """
    Cluster population from spatial over-density alone:
        N_cl = N_in - Sigma_bkg * A_in,  Sigma_bkg = N_ring / A_ring (on-chip areas),
        sigma(N_cl) = sqrt(N_in + (A_in / A_ring)^2 N_ring)       (Poisson).
    'samples' = {label: (ra, dec)}; members = (ra, dec) of the Gaia astrometric members.
    """
    centre, R = circle["centre"], circle["radius_arcmin"]
    r1, r2 = BKG_ANNULUS_FACTORS[0] * R, BKG_ANNULUS_FACTORS[1] * R
    a_in, a_in_full = onchip_area_arcmin2(centre, 0.0, R, wcs, nx, ny)
    a_ring, a_ring_full = onchip_area_arcmin2(centre, r1, r2, wcs, nx, ny)
    sep_mem = centre.separation(SkyCoord(members[0] * u.deg, members[1] * u.deg)).arcmin
    n_mem_in = int((sep_mem < R).sum())
    n_mem_ring = int(((sep_mem >= r1) & (sep_mem < r2)).sum())
    rows = []
    for label, (ra, dec) in samples.items():
        sep = centre.separation(SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)).arcmin
        n_in = int((sep < R).sum())
        n_out = int((sep >= R).sum())
        n_ring = int(((sep >= r1) & (sep < r2)).sum())
        dens = n_ring / a_ring
        excess = n_in - dens * a_in
        err = np.sqrt(n_in + (a_in / a_ring) ** 2 * n_ring)
        rows.append(dict(sample=label, n_total=len(sep), n_inside=n_in, n_outside=n_out, n_ring=n_ring,
                         bkg_density_per_arcmin2=dens, expected_field_inside=dens * a_in,
                         excess=excess, excess_err=err))
    # hybrid estimate for the Gaia-matched sample: remove astrometric members from the ring first
    gm = [r for r in rows if r["sample"].startswith("Gaia-matched")]
    hybrid = None
    if gm:
        r0 = gm[0]
        dens_f = (r0["n_ring"] - n_mem_ring) / a_ring
        hybrid = dict(field_density=dens_f, excess=r0["n_inside"] - dens_f * a_in,
                      excess_err=np.sqrt(r0["n_inside"] + (a_in / a_ring) ** 2 * (r0["n_ring"] - n_mem_ring)))
    return dict(radius_arcmin=R, ring_arcmin=(r1, r2), area_inside=a_in, area_inside_full=a_in_full,
                area_ring=a_ring, area_ring_full=a_ring_full, rows=rows,
                members_total=len(sep_mem), members_inside=n_mem_in, members_outside=len(sep_mem) - n_mem_in,
                members_in_ring=n_mem_ring, hybrid=hybrid)


# ==================================================================================================
# STAGE 1: FITS PIXELS -> INSTRUMENTAL MAGNITUDES
# ==================================================================================================

def detect_sources(bkg_sub, fwhm, threshold, nx, ny, broad=False, border=BORDER_PIX):
    from photutils.detection import DAOStarFinder
    kw = dict(sharpness_range=(0.0, 2.5), roundness_range=(-2.5, 2.5)) if broad else {}
    tbl = DAOStarFinder(threshold=threshold, fwhm=fwhm, **kw)(bkg_sub)
    x, y = np.asarray(tbl["x_centroid"]), np.asarray(tbl["y_centroid"])
    keep = (x > border) & (x < nx - 1 - border) & (y > border) & (y < ny - 1 - border)
    return tbl[keep]


def _isolated(x, y, all_x, all_y, radius):
    tree = cKDTree(np.c_[all_x, all_y])
    counts = np.array([len(v) for v in tree.query_ball_point(np.c_[x, y], r=radius)])
    return counts == 1


def _gauss2d_ell(xy, amp, x0, y0, sx, sy, th, c):
    x, y = xy
    ct, s = np.cos(th), np.sin(th)
    xr = (x - x0) * ct + (y - y0) * s
    yr = -(x - x0) * s + (y - y0) * ct
    return amp * np.exp(-0.5 * (xr ** 2 / sx ** 2 + yr ** 2 / sy ** 2)) + c


def psf_star_candidates(sources):
    """Bright unsaturated stars whose DAOFIND sharpness and roundness lie within 2 robust sigma of the typical
    bright star.  Cuts are relative to the frame's own PSF, so an elongated PSF is not rejected wholesale."""
    bright = (sources["peak"] > 2500) & (sources["peak"] < BRIGHT_PEAK_FALLBACK)
    sel = np.array(bright)
    for col in ("sharpness", "roundness1", "roundness2"):
        v = np.asarray(sources[col])
        sel &= np.abs(v - np.median(v[bright])) < 2.0 * robust_std(v[bright])
    return sources[sel]


def spread_over_chip(tbl, n, nx, ny, grid=8):
    """Pick up to n stars spread evenly over the detector: split it into grid x grid cells and take stars
    round-robin, brightest first in each cell. (Taking the first n rows of the detection table would pick
    stars from the bottom rows only, because DAOStarFinder lists sources in row order.)"""
    if len(tbl) <= n:
        return tbl
    x, y = np.asarray(tbl["x_centroid"]), np.asarray(tbl["y_centroid"])
    cell = (np.clip((y * grid / ny).astype(int), 0, grid - 1) * grid + np.clip((x * grid / nx).astype(int), 0, grid - 1))
    order = np.lexsort((-np.asarray(tbl["peak"]), cell))
    rank = np.empty(len(tbl), int)
    for c in np.unique(cell):
        idx = order[cell[order] == c]
        rank[idx] = np.arange(len(idx))
    pick = np.lexsort((cell, rank))[:n]
    return tbl[np.sort(pick)]


def measure_fwhm(bkg_sub, sources, max_stars=250, box=12):
    """Empirical seeing: elliptical 2-D Gaussian (free centre, axes, angle, pedestal) fitted to isolated
    bright stars.  FWHM = 2.355 sqrt(sigma_major sigma_minor); ellipticity = 1 - minor/major."""
    cand = psf_star_candidates(sources)
    iso = _isolated(cand["x_centroid"], cand["y_centroid"], sources["x_centroid"], sources["y_centroid"], 18.0)
    cand = spread_over_chip(cand[iso], max_stars, bkg_sub.shape[1], bkg_sub.shape[0])
    yy, xx = np.mgrid[-box:box + 1, -box:box + 1]
    fwhms, ell, ang = [], [], []
    for row in cand:
        xi, yi = int(round(row["x_centroid"])), int(round(row["y_centroid"]))
        stamp = bkg_sub[yi - box:yi + box + 1, xi - box:xi + box + 1]
        if stamp.shape != xx.shape:
            continue
        p0 = [float(row["peak"]), row["x_centroid"] - xi, row["y_centroid"] - yi, 2.0, 1.8, 0.1, 0.0]
        try:
            popt, _ = curve_fit(_gauss2d_ell, (xx.ravel(), yy.ravel()), stamp.ravel(), p0=p0, maxfev=4000)
        except RuntimeError:
            continue
        sa, sb = sorted([abs(popt[3]), abs(popt[4])])[::-1]
        if 0.8 < sb <= sa < 5.0:
            fwhms.append(2.35482 * np.sqrt(sa * sb))
            ell.append(1 - sb / sa)
            th = popt[5] if abs(popt[3]) >= abs(popt[4]) else popt[5] + np.pi / 2
            ang.append(np.degrees(th) % 180)
    fwhms = np.array(fwhms)
    return dict(fwhm=float(np.median(fwhms)), scatter=robust_std(fwhms), n=int(len(fwhms)),
                ellipticity=float(np.median(ell)), pa_deg=float(np.median(ang)))


def build_epsf(bkg_sub, sources, n_stars=80):
    """2x-oversampled empirical PSF (Anderson & King 2000) from bright, typical-shaped, isolated stars."""
    from photutils.psf import EPSFBuilder, extract_stars
    cand = psf_star_candidates(sources)
    iso = _isolated(cand["x_centroid"], cand["y_centroid"], sources["x_centroid"], sources["y_centroid"], 20.0)
    cand = spread_over_chip(cand[iso], n_stars, bkg_sub.shape[1], bkg_sub.shape[0])
    print(f"  ePSF stars: {len(cand)} spread over x {np.min(cand['x_centroid']):.0f}-{np.max(cand['x_centroid']):.0f}, "
          f"y {np.min(cand['y_centroid']):.0f}-{np.max(cand['y_centroid']):.0f} px")
    cat = Table()
    cat["x"], cat["y"] = cand["x_centroid"], cand["y_centroid"]
    stars = extract_stars(NDData(data=bkg_sub), cat, size=25)
    epsf, _ = EPSFBuilder(oversampling=2, maxiters=5, progress_bar=False)(stars)
    return epsf, len(cand)


def image_models(band, data, hdr):
    """Sky model, empirical FWHM and ePSF for one frame (the 'image -> model' half of Stage 1)."""
    from photutils.background import Background2D, MedianBackground
    ny, nx = data.shape
    bkg = Background2D(data, (BKG_BOX, BKG_BOX), filter_size=(3, 3), bkg_estimator=MedianBackground())
    bkg_sub = data - bkg.background
    sky_rms = float(bkg.background_rms_median)
    first = detect_sources(bkg_sub, 4.5, 5.0 * sky_rms, nx, ny, border=40)
    shape = measure_fwhm(bkg_sub, first)
    fwhm, fwhm_scatter, n_fwhm = shape["fwhm"], shape["scatter"], shape["n"]
    epsf_sources = detect_sources(bkg_sub, fwhm, 5.0 * sky_rms, nx, ny, border=35)
    epsf, n_psf = build_epsf(bkg_sub, epsf_sources)
    w = WCS(hdr)
    scale = float(np.mean(proj_plane_pixel_scales(w)) * 3600.0)
    info = dict(band=band, gain=float(hdr.get("GAIN", 2.10)), exptime=float(hdr.get("EXPTIME", 180.0)),
                airmass=float(hdr.get("AIRMASS", np.nan)), saturate=float(hdr.get("SATURATE", 58000.0)),
                date_obs=str(hdr.get("DATE-OBS", "")), filter=str(hdr.get("FILTER", band)),
                nx=nx, ny=ny, pix_scale=scale, sky_median=float(bkg.background_median), sky_rms=sky_rms,
                fwhm_pix=fwhm, fwhm_arcsec=fwhm * scale, fwhm_scatter_pix=fwhm_scatter, n_fwhm_stars=n_fwhm,
                psf_ellipticity=shape["ellipticity"], psf_pa_deg=shape["pa_deg"], n_psf_stars=n_psf)
    print(f"[{band}] sky = {info['sky_median']:.1f} ADU, rms = {sky_rms:.2f} ADU | FWHM = {fwhm:.3f} px "
          f"({info['fwhm_arcsec']:.2f}\", {n_fwhm} stars, scatter {fwhm_scatter:.2f} px), ellipticity "
          f"{shape['ellipticity']:.2f} at {shape['pa_deg']:.0f} deg | ePSF from {n_psf} stars")
    return bkg, bkg_sub, epsf, info


def aperture_and_psf_photometry(band, data, bkg, bkg_sub, epsf, info, wcs):
    """
    Measure every detected star in two ways and merge them:
      * aperture: sum inside r = 1.0/1.5/2.0 FWHM minus (local sky median from 3-5 FWHM annulus x area);
        sigma_F^2 = F/gain + A*sigma_sky^2 + A^2*sigma_sky^2/N_sky + f_ff^2 * sum(signal^2)
      * ePSF fit (PSFPhotometry) weighted by the per-pixel noise map sqrt(sky_rms^2 + signal/gain + (f_ff*signal)^2),
        f_ff = FLAT_FIELD_ERR (DAOPHOT II error model, applied to the star only); the fitted flux error is scaled by sqrt(reduced chi^2) when > 1.
      * aperture-fallback stars also carry the scatter of the aperture->PSF offset.
    m_inst = -2.5 log10(F / t_exp). Stars with peak >= 30,000 ADU, a non-zero PSF flag, or a poor PSF
    fit use the 1.5 FWHM aperture magnitude shifted onto the PSF scale by the median (m_PSF - m_ap1.5)
    of clean isolated stars.
    """
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
    from photutils.psf import PSFPhotometry
    fwhm, gain, texp = info["fwhm_pix"], info["gain"], info["exptime"]
    ny, nx = data.shape
    src = detect_sources(bkg_sub, fwhm, DETECT_NSIGMA * info["sky_rms"], nx, ny, broad=True)
    xy = np.c_[src["x_centroid"], src["y_centroid"]]
    print(f"[{band}] {len(src)} sources detected (>{DETECT_NSIGMA:.0f} sigma, broad shape cuts)")

    ann = CircularAnnulus(xy, r_in=ANNULUS_FACTORS[0] * fwhm, r_out=ANNULUS_FACTORS[1] * fwhm)
    st = ApertureStats(data, ann)
    sky = np.where(np.isfinite(st.median), st.median, info["sky_median"])
    sky_sig = np.where(np.isfinite(st.std), st.std, info["sky_rms"])
    n_sky = np.maximum(ann.area, 1.0)
    out = {}
    sig2 = np.clip(bkg_sub, 0, None) ** 2          # for the flat-field term: sum of signal^2 inside each aperture
    for k in APER_FACTORS:
        ap = CircularAperture(xy, r=k * fwhm)
        flux = np.asarray(aperture_photometry(data, ap)["aperture_sum"]) - ap.area * sky
        s2 = np.asarray(aperture_photometry(sig2, ap)["aperture_sum"])
        ferr = np.sqrt(np.maximum(flux, 0) / gain + ap.area * sky_sig ** 2 + ap.area ** 2 * sky_sig ** 2 / n_sky
                       + FLAT_FIELD_ERR ** 2 * s2)
        out[k] = (flux, ferr)

    # per-pixel noise: sky (incl. read noise) + photon noise of the star + flat-field error (DAOPHOT error model)
    error_map = np.sqrt(bkg.background_rms ** 2 + np.clip(bkg_sub, 0, None) / gain
                        + (FLAT_FIELD_ERR * np.clip(bkg_sub, 0, None)) ** 2)
    init = Table()
    init["x_0"], init["y_0"] = src["x_centroid"], src["y_centroid"]
    init["flux_0"] = np.maximum(out[1.5][0], 10.0)
    t0 = time.time()
    psf = PSFPhotometry(epsf, PSF_FIT_SHAPE, aperture_radius=1.5 * fwhm, n_threads=os.cpu_count() or 1)
    res = psf(bkg_sub, error=error_map, init_params=init)
    print(f"[{band}] ePSF fits done in {time.time() - t0:.0f} s")
    pf, pfe_raw, flags = (np.asarray(res["flux_fit"]), np.asarray(res["flux_err"]), np.asarray(res["flags"]))
    # The fit error assumes the ePSF matches the star exactly. When it does not (PSF mismatch, a neighbour in the
    # fit box, sky error), the residuals are larger than the noise model and reduced chi^2 > 1: scale the error by
    # sqrt(reduced chi^2), the standard correction for an imperfect model.
    rchi2 = np.asarray(res["reduced_chi2"], dtype=float)
    pfe = pfe_raw * np.sqrt(np.where(np.isfinite(rchi2), np.maximum(rchi2, 1.0), 1.0))

    def mag(f):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(f > 0, -2.5 * np.log10(f / texp), np.nan)

    def merr(f, e):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((f > 0) & (e > 0), 1.0857 * e / f, np.nan)

    m_psf, e_psf = mag(pf), merr(pf, pfe)
    m_ap = {k: mag(out[k][0]) for k in APER_FACTORS}
    e_ap15 = merr(*out[1.5])
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    peak = np.asarray(src["peak"])
    # raw (sky-included) maximum in a 5x5 box: compared with SATURATE
    xi, yi = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
    raw_peak = np.max([data[yi + dy, xi + dx] for dy in range(-2, 3) for dx in range(-2, 3)], axis=0)

    clean = ((nn > 18.0) & (e_psf < 0.04) & (e_ap15 < 0.04) & (flags == 0) & (peak > 2000) &
             (peak < BRIGHT_PEAK_FALLBACK))
    dm = (m_psf - m_ap[1.5])[clean]
    offset = float(np.median(dm))
    fallback = (peak >= BRIGHT_PEAK_FALLBACK) | (flags != 0) | ((e_psf > 0.04) & (peak > 5000)) | ~np.isfinite(m_psf)
    m_rob = np.where(fallback, m_ap[1.5] + offset, m_psf)
    # aperture stars are put on the PSF scale with the offset, whose star-to-star scatter adds to their error
    e_rob = np.where(fallback, np.hypot(e_ap15, robust_std(dm)), e_psf)
    with np.errstate(invalid="ignore"):
        fine = np.isfinite(rchi2) & (flags == 0)
        print(f"[{band}] PSF fit quality: median sqrt(reduced chi2) = {np.median(np.sqrt(rchi2[fine])):.2f} "
              f"(peak < 2000 ADU: {np.median(np.sqrt(rchi2[fine & (peak < 2000)])):.2f}; "
              f"peak 2000-30000 ADU: {np.median(np.sqrt(rchi2[fine & (peak >= 2000) & (peak < 30000)])):.2f})")
    print(f"[{band}] aperture->PSF offset = {offset:+.4f} mag (N = {clean.sum()}, scatter {robust_std(dm):.4f}); "
          f"aperture fallback for {fallback.sum()} stars; {np.sum(raw_peak >= info['saturate'])} saturated")

    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    df = pd.DataFrame({
        "id": np.arange(1, len(src) + 1), "x": xy[:, 0], "y": xy[:, 1], "ra": ra, "dec": dec,
        "peak": peak, "raw_peak": raw_peak, "sharpness": np.asarray(src["sharpness"]),
        "roundness1": np.asarray(src["roundness1"]), "roundness2": np.asarray(src["roundness2"]),
        "nn_dist_pix": nn, "sky_med": sky, "sky_rms": sky_sig,
        "flux_aper_1.0": out[1.0][0], "flux_aper_1.5": out[1.5][0], "flux_aper_2.0": out[2.0][0],
        "flux_err_aper_1.5": out[1.5][1],
        "mag_aper_1.0": m_ap[1.0], "mag_aper_1.5": m_ap[1.5], "mag_aper_2.0": m_ap[2.0], "mag_err_aper_1.5": e_ap15,
        "flux_psf": pf, "flux_err_psf": pfe, "flux_err_psf_raw": pfe_raw, "psf_reduced_chi2": rchi2,
        "mag_psf": m_psf, "mag_err_psf": e_psf, "psf_flags": flags,
        "mag_robust": m_rob, "mag_err_robust": e_rob, "use_aperture_fallback": fallback,
        "sat_flag": raw_peak >= info["saturate"],
    })
    stats = dict(n_detected=len(src), aperture_offset=offset, aperture_offset_scatter=robust_std(dm),
                 n_offset_stars=int(clean.sum()), n_fallback=int(fallback.sum()),
                 n_saturated=int(np.sum(raw_peak >= info["saturate"])),
                 r_aper_pix=1.5 * fwhm, r_sky_in_pix=ANNULUS_FACTORS[0] * fwhm, r_sky_out_pix=ANNULUS_FACTORS[1] * fwhm)
    return df, stats


def crossmatch_bands(df_r, df_g):
    """Nearest-neighbour r'-g' match within 1", keeping only the closest r' star for each g' star."""
    cr = SkyCoord(df_r["ra"].values * u.deg, df_r["dec"].values * u.deg)
    cg = SkyCoord(df_g["ra"].values * u.deg, df_g["dec"].values * u.deg)
    idx, d2d, _ = cr.match_to_catalog_sky(cg)
    sep = d2d.arcsec
    ok = sep < MATCH_GR_ARCSEC
    order = np.argsort(sep)
    seen, uniq = set(), np.zeros(len(sep), bool)
    for i in order:
        if ok[i] and idx[i] not in seen:
            uniq[i] = True
            seen.add(idx[i])
    mr = df_r[uniq].reset_index(drop=True)
    mg = df_g.iloc[idx[uniq]].reset_index(drop=True)
    cat = pd.concat([mr.add_suffix("_r"), mg.add_suffix("_g")], axis=1)
    cat["sep_arcsec"] = sep[uniq]
    return finalise_matched(cat)


def finalise_matched(cat):
    cat["mag_inst_r"], cat["mag_err_inst_r"] = cat["mag_robust_r"], cat["mag_err_robust_r"]
    cat["mag_inst_g"], cat["mag_err_inst_g"] = cat["mag_robust_g"], cat["mag_err_robust_g"]
    cat["color_inst_g_r"] = cat["mag_inst_g"] - cat["mag_inst_r"]
    cat["color_err_inst"] = np.hypot(cat["mag_err_inst_g"], cat["mag_err_inst_r"])
    cat["is_saturated"] = cat["sat_flag_r"].astype(bool) | cat["sat_flag_g"].astype(bool)
    cat["snr10_flag"] = ((cat["mag_err_inst_r"] < 0.1086) & (cat["mag_err_inst_g"] < 0.1086))
    return cat


def run_stage1(from_scratch):
    banner("STAGE 1: FITS PIXELS -> INSTRUMENTAL MAGNITUDES")
    have_cache = os.path.exists("photometry_matched.csv") and os.path.exists(STAGE1_JSON) and os.path.exists(STAGE1_NPZ)
    if not from_scratch and os.path.exists("photometry_matched.csv"):
        matched = pd.read_csv("photometry_matched.csv")
        if "is_saturated" not in matched:
            matched = finalise_matched(matched)
        print(f"--use-cache: loaded photometry_matched.csv ({len(matched):,} r'+g' stars)")
        if have_cache:
            with open(STAGE1_JSON) as f:
                diag = json.load(f)
            npz = np.load(STAGE1_NPZ)
            diag["_arrays"] = {k: npz[k] for k in npz.files}
            diag["source"] = diag.get("source", "from-scratch run") + " (loaded from cache)"
            return matched, diag
        print("  stage1_diagnostics.json missing -> rebuilding sky/FWHM/ePSF models only (no PSF photometry).")
        diag = {"bands": {}, "_arrays": {}, "source": "cache: photometry from an earlier run; image models rebuilt"}
        for band in ("r", "g"):
            data, hdr, wcs = load_frame(band)
            bkg, bkg_sub, epsf, info = image_models(band, data, hdr)
            dm = (matched[f"mag_psf_{band}"] - matched[f"mag_aper_1.5_{band}"])[
                (matched[f"psf_flags_{band}"] == 0) & (matched[f"nn_dist_pix_{band}"] > 18) &
                (matched[f"mag_err_psf_{band}"] < 0.04) & (matched[f"peak_{band}"].between(2000, BRIGHT_PEAK_FALLBACK))]
            info.update(n_detected=None, aperture_offset=float(np.median(dm)), aperture_offset_scatter=robust_std(dm),
                        n_offset_stars=int(len(dm)), n_fallback=int(matched[f"use_aperture_fallback_{band}"].sum()),
                        n_saturated=int(matched[f"sat_flag_{band}"].sum()),
                        r_aper_pix=1.5 * info["fwhm_pix"], r_sky_in_pix=3 * info["fwhm_pix"], r_sky_out_pix=5 * info["fwhm_pix"])
            diag["bands"][band] = info
            diag["_arrays"][f"bkg_{band}"] = bkg.background[::8, ::8]
            diag["_arrays"][f"epsf_{band}"] = epsf.data
            del data, bkg, bkg_sub
        diag["n_matched"] = len(matched)
        diag["n_snr10"] = int(matched["snr10_flag"].sum())
        return matched, diag

    diag = {"bands": {}, "_arrays": {}, "source": "from-scratch run"}
    tables = {}
    for band in ("r", "g"):
        data, hdr, wcs = load_frame(band)
        bkg, bkg_sub, epsf, info = image_models(band, data, hdr)
        df, stats = aperture_and_psf_photometry(band, data, bkg, bkg_sub, epsf, info, wcs)
        info.update(stats)
        tables[band] = df
        diag["bands"][band] = info
        diag["_arrays"][f"bkg_{band}"] = bkg.background[::8, ::8]
        diag["_arrays"][f"epsf_{band}"] = epsf.data
        df.to_csv(f"photometry_{band}_band.csv", index=False)
        del data, bkg, bkg_sub
    matched = crossmatch_bands(tables["r"], tables["g"])
    matched.to_csv("photometry_matched.csv", index=False)
    diag["n_matched"] = len(matched)
    diag["n_snr10"] = int(matched["snr10_flag"].sum())
    diag["match_sep_median_arcsec"] = float(matched["sep_arcsec"].median())
    print(f"r'+g' matched: {len(matched):,} stars ({diag['n_snr10']:,} with S/N > 10 in both bands); "
          f"median separation {diag['match_sep_median_arcsec']:.3f}\"")
    with open(STAGE1_JSON, "w") as f:
        json.dump(to_jsonable({k: v for k, v in diag.items() if k != "_arrays"}), f, indent=2)
    np.savez_compressed(STAGE1_NPZ, **diag["_arrays"])
    return matched, diag


APT_USER_COLS = ["apt_source_intensity_r", "apt_source_intensity_g", "apt_notes"]


def export_apt_sample(matched, diag, n=50):
    """50 isolated, unsaturated stars spread in brightness, with 1-based pixel positions for APT."""
    c = matched[(matched["nn_dist_pix_r"] > 22) & (matched["nn_dist_pix_g"] > 22) & (~matched["is_saturated"]) &
                (matched["mag_err_inst_r"] < 0.03) & (matched["mag_err_inst_g"] < 0.03) &
                (matched["peak_r"].between(2000, 55000))].sort_values("mag_inst_r").reset_index(drop=True)
    s = c.iloc[::max(1, len(c) // n)].iloc[:n].reset_index(drop=True)
    br, bg = diag["bands"]["r"], diag["bands"]["g"]
    apt = pd.DataFrame({
        "star_id": np.arange(1, len(s) + 1), "ra_deg": s["ra_r"].round(6), "dec_deg": s["dec_r"].round(6),
        "x_r_1based_APT": (s["x_r"] + 1).round(2), "y_r_1based_APT": (s["y_r"] + 1).round(2),
        "x_g_1based_APT": (s["x_g"] + 1).round(2), "y_g_1based_APT": (s["y_g"] + 1).round(2),
        "apt_aperture_radius_r_px": round(br["r_aper_pix"], 2), "apt_sky_inner_r_px": round(br["r_sky_in_pix"], 2),
        "apt_sky_outer_r_px": round(br["r_sky_out_pix"], 2),
        "apt_aperture_radius_g_px": round(bg["r_aper_pix"], 2), "apt_sky_inner_g_px": round(bg["r_sky_in_pix"], 2),
        "apt_sky_outer_g_px": round(bg["r_sky_out_pix"], 2),
        "sky_median_r": s["sky_med_r"].round(2),
        "flux_aper_1.5_r": s["flux_aper_1.5_r"].round(1), "flux_aper_1.5_g": s["flux_aper_1.5_g"].round(1),
        "mag_aper_1.0_r": s["mag_aper_1.0_r"].round(4), "mag_aper_1.5_r": s["mag_aper_1.5_r"].round(4),
        "mag_aper_2.0_r": s["mag_aper_2.0_r"].round(4), "mag_err_aper_r": s["mag_err_aper_1.5_r"].round(4),
        "mag_aper_1.5_g": s["mag_aper_1.5_g"].round(4),
        "mag_psf_r": s["mag_psf_r"].round(4), "mag_err_psf_r": s["mag_err_psf_r"].round(4),
        "mag_robust_r": s["mag_inst_r"].round(4), "mag_robust_g": s["mag_inst_g"].round(4),
    })
    for col in APT_USER_COLS:
        apt[col] = np.nan if col != "apt_notes" else ""
    # keep any APT measurements already typed into the existing file (matched by position)
    if os.path.exists("apt_crosscheck_sample.csv"):
        old = pd.read_csv("apt_crosscheck_sample.csv")
        user = [c for c in APT_USER_COLS if c in old.columns]
        if user and old[user].notna().any().any():
            co = SkyCoord(old["ra_deg"].values * u.deg, old["dec_deg"].values * u.deg)
            cn = SkyCoord(apt["ra_deg"].values * u.deg, apt["dec_deg"].values * u.deg)
            idx, d2d, _ = cn.match_to_catalog_sky(co)
            for i, (j, d) in enumerate(zip(idx, d2d.arcsec)):
                if d < 1.0:
                    for col in user:
                        apt.loc[i, col] = old.loc[j, col]
            print(f"  kept APT values typed into the previous apt_crosscheck_sample.csv ({(d2d.arcsec < 1).sum()} stars)")
    apt.to_csv("apt_crosscheck_sample.csv", index=False)
    print(f"Exported apt_crosscheck_sample.csv ({len(apt)} stars)")
    return apt


# ==================================================================================================
# STAGE 2: CALIBRATION (PAN-STARRS DR1 -> SDSS) AND MEMBERSHIP (GAIA DR3)
# ==================================================================================================

def load_reference_catalogs(geom, refresh=False):
    names = {"ps1": "ps1.csv", "gaia": "gaia_dr3.csv", "cg": "cantat_gaudin.csv"}
    if refresh or any(find_catalog(n) is None for n in names.values()):
        query_reference_catalogs(geom)
    ps1 = pd.read_csv(find_catalog("ps1.csv"))
    gaia = pd.read_csv(find_catalog("gaia_dr3.csv"))
    cg = pd.read_csv(find_catalog("cantat_gaudin.csv"))
    centre = geom["centre"]
    cov = {}
    for key, df, ra, de in (("ps1", ps1, "RAJ2000", "DEJ2000"), ("gaia", gaia, "ra", "dec")):
        r = centre.separation(SkyCoord(df[ra].values * u.deg, df[de].values * u.deg)).arcmin
        cov[key] = dict(n=len(df), r_max_arcmin=float(r.max()))
        if r.max() < geom["r_search_arcmin"] - 0.5:
            print(f"  NOTE: cached {key} reaches r = {r.max():.1f}' but the detector corners need "
                  f"{geom['r_search_arcmin']:.1f}' (use --refresh-catalogs to re-query).")
    print(f"Reference catalogues: PS1 {len(ps1):,} | Gaia DR3 {len(gaia):,} | Cantat-Gaudin {len(cg):,} "
          f"(required search radius from WCS: {geom['r_search_arcmin']:.2f}')")
    return ps1, gaia, cg, cov


def calibrate_photometry(cat, ps1):
    """
    Zero points and colour terms on the SDSS system:
      1) match GIT stars to PS1 (0.8"), keep clean unsaturated point sources 13.5 < r_PS1 < 17.5;
      2) PS1 -> SDSS:  x_SDSS = x_PS1 + a + b (g-r)_PS1          (Tonry et al. 2012);
      3) fit  x_SDSS - x_inst = ZP + C (g_inst - r_inst)  with Siegel's repeated-median slope;
      4) ZP/C uncertainties = std over 500 bootstrap resamples of the reference stars.
    """
    c_git = SkyCoord(cat["ra_r"].values * u.deg, cat["dec_r"].values * u.deg)
    c_ps1 = SkyCoord(ps1["RAJ2000"].values * u.deg, ps1["DEJ2000"].values * u.deg)
    idx, d2d, _ = c_git.match_to_catalog_sky(c_ps1)
    m = d2d.arcsec < MATCH_CAT_ARCSEC
    git, p = cat[m].reset_index(drop=True), ps1.iloc[idx[m]].reset_index(drop=True)
    k = PS1_REF_CUTS
    ref = ((p["rmag"].between(k["r_min"], k["r_max"])) & (p["gmag"].between(k["g_min"], k["g_max"])) &
           (p["e_rmag"] < k["err_max"]) & (p["e_gmag"] < k["err_max"]) &
           (np.abs(p["rmag"] - p["rKmag"]) < k["psf_kron_max"]) & (~git["is_saturated"]) &
           (git["mag_err_inst_r"] < k["inst_err_max"]) & (git["mag_err_inst_g"] < k["inst_err_max"]) &
           (git["peak_r"] < k["peak_max"]))
    git, p = git[ref].reset_index(drop=True), p[ref].reset_index(drop=True)
    c_ps = (p["gmag"] - p["rmag"]).values
    sdss = {b: p[f"{b}mag"].values + PS1_TO_SDSS[b][0] + PS1_TO_SDSS[b][1] * c_ps for b in ("g", "r")}
    c_inst = (git["mag_inst_g"] - git["mag_inst_r"]).values
    rng = np.random.default_rng(42)
    calib, refs = {}, pd.DataFrame({"ra": git["ra_r"], "dec": git["dec_r"], "x_r": git["x_r"], "y_r": git["y_r"],
                                    "c_inst": c_inst, "r_ps1": p["rmag"], "g_ps1": p["gmag"]})
    for b in ("r", "g"):
        y = sdss[b] - git[f"mag_inst_{b}"].values
        slope, zp = siegelslopes(y, c_inst)
        boot = []
        for _ in range(N_BOOT_ZP):
            j = rng.integers(0, len(y), len(y))
            boot.append(siegelslopes(y[j], c_inst[j]))
        boot = np.array(boot)
        resid = y - (zp + slope * c_inst)
        calib[b] = dict(zp=float(zp), colour_term=float(slope), zp_err=float(np.std(boot[:, 1])),
                        colour_term_err=float(np.std(boot[:, 0])), rms=robust_std(resid),
                        zp_sys=float(np.hypot(np.std(boot[:, 1]), CAL_SYSTEMATIC_FLOOR)))
        refs[f"{b}_sdss_ref"], refs[f"d{b}"], refs[f"resid_{b}"] = sdss[b], y, resid
        print(f"  {b}': {b}_SDSS - {b}_inst = {zp:.4f} (+/- {calib[b]['zp_err']:.4f}) "
              f"{slope:+.4f} (+/- {calib[b]['colour_term_err']:.4f}) (g-r)_inst | robust RMS {calib[b]['rms']:.4f} mag")
    calib["n_ref"] = int(len(y))
    calib["n_ps1_matched"] = int(m.sum())
    c_all = cat["color_inst_g_r"].values
    for b in ("r", "g"):
        cat[f"{b}_sdss"] = cat[f"mag_inst_{b}"] + calib[b]["zp"] + calib[b]["colour_term"] * c_all
        cat[f"e_{b}_sdss"] = np.hypot(cat[f"mag_err_inst_{b}"], calib[b]["zp_err"])
    cat["g_r_sdss"] = cat["g_sdss"] - cat["r_sdss"]
    cat["e_g_r_sdss"] = np.hypot(cat["e_g_sdss"], cat["e_r_sdss"])
    print(f"  {calib['n_ref']} PS1 reference stars (of {calib['n_ps1_matched']:,} matched)")
    refs.to_csv("calibration_reference_stars.csv", index=False)
    return cat, calib, refs


def select_members(cat, gaia, cg):
    """
    Gaia DR3 cross-match (0.8") and membership:
      member = P_memb(Cantat-Gaudin 2018) >= 0.70
               OR (RUWE < 1.4 AND |mu - mu_cl| < 3 x 0.25 mas/yr AND 0.25 <= parallax <= 0.50 mas).
    The cluster proper-motion centre is the median of the P_memb >= 0.70 stars.
    """
    c_git = SkyCoord(cat["ra_r"].values * u.deg, cat["dec_r"].values * u.deg)
    c_g = SkyCoord(gaia["ra"].values * u.deg, gaia["dec"].values * u.deg)
    idx, d2d, _ = c_git.match_to_catalog_sky(c_g)
    m = d2d.arcsec < MATCH_CAT_ARCSEC
    out = cat[m].reset_index(drop=True)
    g = gaia.iloc[idx[m]].reset_index(drop=True)
    pmemb = dict(zip(cg["Source"], cg["PMemb"]))
    g["PMemb"] = g["source_id"].map(pmemb).fillna(0.0)
    for col in ("source_id", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "ruwe",
                "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "teff_gspphot", "PMemb"):
        out[col] = g[col].values
    out["gaia_ra"], out["gaia_dec"] = g["ra"].values, g["dec"].values
    core = out["PMemb"] >= PMEMB_MIN
    pm0 = (float(out.loc[core, "pmra"].median()), float(out.loc[core, "pmdec"].median()))
    out["pm_dist_sigma"] = np.hypot(out["pmra"] - pm0[0], out["pmdec"] - pm0[1]) / PM_SIGMA
    kin = ((out["ruwe"] < RUWE_MAX) & (out["pm_dist_sigma"] < PM_NSIGMA) &
           out["parallax"].between(*PLX_RANGE))
    out["is_member"] = core | kin
    mem = out[out["is_member"]]
    good_plx = mem[(mem["parallax_error"] < 0.1) & (mem["ruwe"] < RUWE_MAX)]
    w = 1.0 / good_plx["parallax_error"] ** 2
    plx_w = float(np.sum(w * good_plx["parallax"]) / np.sum(w))
    stats = dict(n_gaia_matched=int(m.sum()), n_members=int(out["is_member"].sum()),
                 n_field=int((~out["is_member"]).sum()), n_by_pmemb=int(core.sum()), n_by_kinematics=int(kin.sum()),
                 n_cg_total=int(len(cg)), n_cg_found_by_id=int(g["source_id"].isin(cg["Source"]).sum()),
                 pmra0=pm0[0], pmdec0=pm0[1],
                 pmra_med=float(mem["pmra"].median()), pmdec_med=float(mem["pmdec"].median()),
                 pmra_sd=robust_std(mem["pmra"]), pmdec_sd=robust_std(mem["pmdec"]),
                 plx_med=float(mem["parallax"].median()), plx_sd=robust_std(mem["parallax"]),
                 plx_wmean=plx_w, plx_wmean_err=float(1.0 / np.sqrt(np.sum(w))), n_plx=int(len(good_plx)),
                 d_plx_raw_kpc=1.0 / plx_w, d_plx_zpcorr_kpc=1.0 / (plx_w - GAIA_PLX_ZP))
    print(f"  Gaia-matched {stats['n_gaia_matched']:,} | members {stats['n_members']:,} "
          f"(P_memb {stats['n_by_pmemb']:,}, kinematic {stats['n_by_kinematics']:,}) | field {stats['n_field']:,}")
    print(f"  PM centre ({pm0[0]:+.3f}, {pm0[1]:+.3f}) mas/yr | weighted parallax {plx_w:.4f} +/- "
          f"{stats['plx_wmean_err']:.4f} mas -> {stats['d_plx_raw_kpc']:.3f} kpc raw, "
          f"{stats['d_plx_zpcorr_kpc']:.3f} kpc after {GAIA_PLX_ZP} mas zero point")
    return out, stats


def run_stage2(matched, geom, circle, hdr_r, wcs_r, refresh_catalogs, spatial_filter):
    banner("STAGE 2: PAN-STARRS DR1 CALIBRATION, GAIA DR3 MEMBERSHIP, MANUAL DS9 CIRCLE")
    ps1, gaia, cg, cov = load_reference_catalogs(geom, refresh_catalogs)
    matched, calib, refs = calibrate_photometry(matched, ps1)
    cal, mstats = select_members(matched, gaia, cg)
    sep = circle["centre"].separation(SkyCoord(cal["ra_r"].values * u.deg, cal["dec_r"].values * u.deg)).arcmin
    cal["r_circle_arcmin"] = sep
    cal["in_circle"] = sep < circle["radius_arcmin"]
    cal.to_csv("photometry_calibrated.csv", index=False)
    mem = cal[cal["is_member"]]
    counts = spatial_star_counts(
        circle, wcs_r, geom["nx"], geom["ny"],
        {"All r'+g' detections": (matched["ra_r"], matched["dec_r"]),
         "Gaia-matched stars": (cal["ra_r"], cal["dec_r"]),
         "Gaia-matched, r' < 18": (cal.loc[cal["r_sdss"] < 18, "ra_r"], cal.loc[cal["r_sdss"] < 18, "dec_r"])},
        (mem["ra_r"].values, mem["dec_r"].values))
    print(f"\nManual DS9 circle (r = {counts['radius_arcmin']:.2f}', ring {counts['ring_arcmin'][0]:.2f}-"
          f"{counts['ring_arcmin'][1]:.2f}', on-chip areas {counts['area_inside']:.1f} / {counts['area_ring']:.1f} arcmin^2):")
    for r in counts["rows"]:
        print(f"  {r['sample']:<24s} inside {r['n_inside']:>5d} | outside {r['n_outside']:>5d} | ring {r['n_ring']:>5d} | "
              f"field {r['bkg_density_per_arcmin2']:.2f}/arcmin^2 -> excess {r['excess']:.0f} +/- {r['excess_err']:.0f}")
    print(f"  Gaia astrometric members: {counts['members_inside']} inside, {counts['members_outside']} outside, "
          f"{counts['members_in_ring']} in the background ring")
    if counts["hybrid"]:
        print(f"  Excess after removing ring members from the background: {counts['hybrid']['excess']:.0f} "
              f"+/- {counts['hybrid']['excess_err']:.0f}")
    if spatial_filter == "circle":
        members = mem[mem["in_circle"]].copy().reset_index(drop=True)
        print(f"  --spatial-filter circle: {len(members)} members inside the circle used downstream")
    else:
        members = mem.copy().reset_index(drop=True)
    return cal, members, calib, refs, mstats, counts, cov


def statistical_decontamination(matched, cal, circle, counts, n_real=N_DECON_REAL, seed=11):
    """
    Gaia-FREE membership: statistical field-star subtraction with the background ring (the classical
    comparison-field method, done star by star with nearest neighbours instead of fixed CMD cells).
      * circle sample = every r'+g' star with S/N > 10 in both bands inside the DS9 circle;
        field sample  = the same cut inside the background ring;
      * one realisation: each ring star is drawn with probability A_circle / A_ring (on-chip areas), and every
        drawn field star removes the nearest still-present circle star in the (g'-r', r') plane, distance
        hypot(d_colour / 0.10, d_r / 0.50) <= 2 (at most 0.2 mag in colour or 1 mag in r'); a field star with
        no circle star that close removes nothing ("unmatched");
      * P_stat = fraction of the realisations in which a circle star survives; Gaia-free member = P_stat >= 0.5.
    Only the calibrated GIT photometry and star positions are used. Gaia is used afterwards ONLY to score the
    result (agreement columns), never to select stars.
    """
    centre, R = circle["centre"], circle["radius_arcmin"]
    r1, r2 = counts["ring_arcmin"]
    ok = matched["snr10_flag"].astype(bool) & np.isfinite(matched["g_r_sdss"]) & np.isfinite(matched["r_sdss"])
    s = matched[ok]
    sep = centre.separation(SkyCoord(s["ra_r"].values * u.deg, s["dec_r"].values * u.deg)).arcmin
    circ = s[sep < R].copy().reset_index(drop=True)
    ring = s[(sep >= r1) & (sep < r2)].copy().reset_index(drop=True)
    ratio = counts["area_inside"] / counts["area_ring"]
    X = np.c_[circ["g_r_sdss"] / DECON_SCALE[0], circ["r_sdss"] / DECON_SCALE[1]]
    Y = np.c_[ring["g_r_sdss"] / DECON_SCALE[0], ring["r_sdss"] / DECON_SCALE[1]]
    k = min(DECON_K, len(circ))
    nd, nj = cKDTree(X).query(Y, k=k, distance_upper_bound=DECON_DMAX)
    nd, nj = np.atleast_2d(nd).tolist(), np.atleast_2d(nj).tolist()
    rng = np.random.default_rng(seed)
    survive = np.zeros(len(circ))
    removed, unmatched = [], []
    for _ in range(n_real):
        alive = np.ones(len(circ), bool)
        drawn = rng.permutation(np.flatnonzero(rng.random(len(ring)) < ratio))
        miss = 0
        for i in drawn:
            for d, j in zip(nd[i], nj[i]):
                if d == np.inf:
                    miss += 1
                    break
                if alive[j]:
                    alive[j] = False
                    break
            else:
                miss += 1
        survive += alive
        removed.append(len(drawn) - miss)
        unmatched.append(miss)
    circ["p_stat"] = survive / n_real
    circ["is_stat_member"] = circ["p_stat"] >= P_STAT_MIN

    # scoring against Gaia (after the selection; nothing above used Gaia)
    g = cal[["id_r", "is_member"]].rename(columns={"is_member": "gaia_member"})
    circ = circ.merge(g, on="id_r", how="left")
    has_g = circ["gaia_member"].notna()
    gm = circ["gaia_member"].fillna(False).astype(bool)
    sm = circ["is_stat_member"]
    st = dict(n_circle=int(len(circ)), n_ring=int(len(ring)), area_ratio=float(ratio),
              expected_field=float(len(ring) * ratio), removed_mean=float(np.mean(removed)),
              unmatched_mean=float(np.mean(unmatched)), n_real=int(n_real),
              sum_p_stat=float(circ["p_stat"].sum()), n_stat_members=int(sm.sum()),
              n_gaia_members_circle=int(gm.sum()),
              frac_stat_that_are_gaia=float((sm & gm).sum() / max((sm & has_g).sum(), 1)),
              frac_gaia_recovered=float((sm & gm).sum() / max(gm.sum(), 1)),
              scale=list(DECON_SCALE), dmax=DECON_DMAX, p_min=P_STAT_MIN)
    print(f"\nGaia-free statistical subtraction (S/N > 10 in both bands, {n_real} realisations):")
    print(f"  circle {st['n_circle']:,} stars | ring {st['n_ring']:,} stars x area ratio {ratio:.3f} = "
          f"{st['expected_field']:.0f} field stars expected; removed {st['removed_mean']:.0f} on average "
          f"({st['unmatched_mean']:.0f} field stars had no CMD neighbour within the limit)")
    print(f"  sum of P_stat = {st['sum_p_stat']:.0f} cluster stars; {st['n_stat_members']:,} stars with P_stat >= {P_STAT_MIN}")
    print(f"  scored against Gaia afterwards: {st['frac_stat_that_are_gaia']:.0%} of the Gaia-free members are Gaia members; "
          f"they recover {st['frac_gaia_recovered']:.0%} of the {st['n_gaia_members_circle']} Gaia members (S/N > 10) in the circle")
    circ.to_csv("statistical_membership.csv", index=False)
    return circ, ring, st


# ==================================================================================================
# STAGE 3: POPULATIONS, REDDENING TEST, UNFORCED PARSEC FIT, UNCERTAINTIES, Teff
# ==================================================================================================

def load_parsec(path="parsec_grid_ngc6819.dat"):
    rows, cols = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("# Zini"):
                cols = line.strip("# \n").split()
            elif not line.startswith("#") and line.strip():
                rows.append(line.split())
    return pd.DataFrame(rows, columns=cols).astype(float)


def isochrone_arrays(df_iso, logage, n_sub=8):
    """One PARSEC age, labels 0-4 (PMS..core-He burning), densified by linear interpolation along the
    track so the nearest-point distance measures the distance to the curve, not to sparse vertices.
    The RGB-tip -> core-He-burning jump (label 3 -> 4) is not interpolated."""
    s = df_iso[np.isclose(df_iso["logAge"], logage, atol=1e-4)]
    s = s[(s["label"] <= 4) & (s["rmag"] > -3.5) & (s["rmag"] < 8.5)]
    cols = {"gr": (s["gmag"] - s["rmag"]).values, "Mr": s["rmag"].values, "Mini": s["Mini"].values,
            "logTe": s["logTe"].values, "label": s["label"].values}
    lab = cols["label"]
    jump = (lab[:-1] == 3) & (lab[1:] == 4) | (lab[1:] < lab[:-1])
    t = np.linspace(0, 1, n_sub, endpoint=False)
    out = {}
    for k, v in cols.items():
        seg = v[:-1, None] + (v[1:] - v[:-1])[:, None] * (t if k != "label" else 0 * t)
        seg[jump, 1:] = np.nan
        out[k] = np.r_[seg.ravel(), v[-1:]]
    ok = np.isfinite(out["gr"])
    return {k: v[ok] for k, v in out.items()}


def turnoff_mass(iso):
    """Initial mass at the bluest point of the upper main sequence (label 1) = main-sequence turnoff."""
    ms = (iso["label"] == 1) & (iso["Mini"] > 0.8)
    i = np.argmin(np.where(ms, iso["gr"], np.inf))
    return float(iso["Mini"][i]), float(iso["Mr"][i]), float(iso["gr"][i])


def flag_populations(mem, gr="g_r_sdss", r="r_sdss"):
    """MS ridge line (35th colour percentile per r' bin, quadratic) -> unresolved binaries
    (0.065-0.28 mag redder), blue stragglers (brighter + bluer than the turnoff)."""
    msc = mem[mem[r].between(15.5, 18.5)]
    bins = np.linspace(15.5, 18.5, 10)
    cen = 0.5 * (bins[:-1] + bins[1:])
    ridge = [np.percentile(msc.loc[msc[r].between(a, b), gr], 35) for a, b in zip(bins[:-1], bins[1:])]
    poly = np.poly1d(np.polyfit(cen, ridge, 2))
    off = mem[gr] - poly(mem[r])
    mem["ms_colour_offset"] = off
    mem["is_bss"] = (mem[r] < 14.9) & (mem[gr] < 0.37) & (~mem["is_saturated"])
    mem["is_binary"] = mem[r].between(15.5, 18.5) & off.between(0.065, 0.28) & (~mem["is_bss"])
    mem["is_single_member"] = ((~mem["is_bss"]) & (~mem["is_binary"]) & (~mem["is_saturated"]) & (mem[r] < 19.5)
                               & np.isfinite(mem[gr]))
    mem["is_rc_region"] = mem[r].between(*FEATURE_WINDOWS["RC"]["r"]) & (~mem["is_bss"])
    return poly


N_LOCAL = 25   # neighbours used for the local (differential) reddening estimate


def _zone_test(ra, dec, values, centre):
    """Median of 'values' in 4 position-angle quadrants + 3 radial zones; chi^2 against the global median."""
    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    pa, rad = centre.position_angle(c).deg, centre.separation(c).arcmin
    values = np.asarray(values)
    glob = np.median(values)
    zones = []
    for name, sel in ([(f"PA {a}-{a + 90} deg", (pa >= a) & (pa < a + 90)) for a in (0, 90, 180, 270)] +
                      [(f"r {lo}-{hi}'", (rad >= lo) & (rad < hi)) for lo, hi in ((0, 4), (4, 8), (8, 99))]):
        v = values[sel]
        if len(v) >= 10:
            zones.append(dict(zone=name, n=int(len(v)), median=float(np.median(v) - glob),
                              err=float(1.253 * robust_std(v) / np.sqrt(len(v)))))
    chi = float(sum((z["median"] / z["err"]) ** 2 for z in zones))
    dof = len(zones) - 1
    return dict(zones=zones, chi2=chi, dof=dof, p_value=float(chi2_dist.sf(chi, dof)),
                max_offset=float(max(abs(z["median"]) for z in zones)))


def reddening_uniformity_test(mem, centre, refs):
    """
    Is E(B-V) constant across the field?
      * zone test: median MS colour offset from the ridge line in 4 quadrants + 3 radial zones, chi^2 vs. constant;
      * control: the same zone test on the PS1 calibration colour residuals (g - r). If those also vary,
        the pattern could be photometric (flat field), not dust;
      * local estimate: for every member, the median MS colour offset of its 25 nearest MS neighbours
        (Platais et al. 2013 style), and whether removing it reduces the MS colour scatter.
    """
    ms = mem[mem["is_single_member"] & mem["r_sdss"].between(15.5, 18.5) & mem["ms_colour_offset"].between(-0.15, 0.15)]
    off = ms["ms_colour_offset"].values
    zt = _zone_test(ms["ra_r"], ms["dec_r"], off, centre)
    ct = _zone_test(refs["ra"], refs["dec"], refs["resid_g"] - refs["resid_r"], centre)
    tree = cKDTree(np.c_[ms["x_r"], ms["y_r"]])
    d, j = tree.query(np.c_[mem["x_r"], mem["y_r"]], k=N_LOCAL + 1)
    local = np.empty(len(mem))
    for i in range(len(mem)):
        jj = j[i][d[i] > 1e-6][:N_LOCAL]          # never use the star itself
        local[i] = np.median(off[jj])
    local -= np.median(off)                       # zero-mean: only *differential* reddening
    mem["d_gr_local"] = local
    is_ms = mem.index.isin(ms.index)
    res = dict(n=int(len(ms)), zones=zt["zones"], chi2=zt["chi2"], dof=zt["dof"], p_value=zt["p_value"],
               max_zone_offset_mag=zt["max_offset"], max_zone_offset_ebv=zt["max_offset"] / E_GR_PER_EBV,
               cal_zones=ct["zones"], cal_chi2=ct["chi2"], cal_dof=ct["dof"], cal_p_value=ct["p_value"],
               cal_max_offset=ct["max_offset"],
               sigma_raw=robust_std(off), sigma_corrected=robust_std(off - local[is_ms]),
               local_ebv_range=[float(np.percentile(local, 5) / E_GR_PER_EBV), float(np.percentile(local, 95) / E_GR_PER_EBV)],
               uniform=bool(zt["p_value"] > 0.01))
    print(f"Reddening test (N = {res['n']} MS stars): chi2 = {zt['chi2']:.1f}/{zt['dof']} (p = {zt['p_value']:.3g}), "
          f"largest zone offset {zt['max_offset']:.3f} mag in g'-r' (= {res['max_zone_offset_ebv']:.3f} in E(B-V))")
    print(f"  control, PS1 calibration colour residuals: chi2 = {ct['chi2']:.1f}/{ct['dof']} (p = {ct['p_value']:.3g}), "
          f"largest zone offset {ct['max_offset']:.3f} mag")
    print(f"  local {N_LOCAL}-neighbour correction: MS colour sigma {res['sigma_raw']:.4f} -> {res['sigma_corrected']:.4f} mag; "
          f"local dE(B-V) 5-95%: {res['local_ebv_range'][0]:+.3f} to {res['local_ebv_range'][1]:+.3f}")
    print("  -> " + ("consistent with spatially uniform reddening." if res["uniform"]
                     else "zones differ significantly: differential reddening is present (or a photometric gradient)."))
    return res


def apply_fit_photometry(mem, correct):
    """Columns used by the fit and plots: raw calibrated photometry, or with the local differential
    reddening removed:  (g-r) -= d,  r -= (A_r / E(g-r)) d."""
    d = mem["d_gr_local"] if correct else 0.0
    mem["gr_fit"] = mem["g_r_sdss"] - d
    mem["r_fit"] = mem["r_sdss"] - (A_R_PER_EBV / E_GR_PER_EBV) * d


def feature_points(mem):
    singles = mem[mem["is_single_member"]]
    feats = {}
    for name, w in FEATURE_WINDOWS.items():
        base = mem[~mem["is_bss"]] if name == "RC" else singles
        sel = base["r_fit"].between(*w["r"])
        if "gr" in w:
            sel &= base["gr_fit"].between(*w["gr"])
        if name == "MS":
            sel &= base["r_fit"] > w["r"][0]
        sel &= np.isfinite(base["gr_fit"]) & np.isfinite(base["r_fit"])
        feats[name] = np.c_[base.loc[sel, "gr_fit"].values, base.loc[sel, "r_fit"].values]
    return feats


class IsochroneFitter:
    """
    Loss(logAge, mu0, E) = sum_f w_f * median_i  d_i,f
      d = distance from each star of feature f to the nearest point of the matching evolutionary phase
          (FEATURE_PHASES) of the shifted isochrone in the plane (g'-r', r'/MAG_SCALE), shifted by
          E(g'-r') = E_GR_PER_EBV E,  A_r' = A_R_PER_EBV E,  plus mu0.
    Shifting the isochrone by (dx, dy) is the same as shifting the stars by (-dx, -dy), so each age's
    k-d tree is built once and the whole (mu0, E) grid is evaluated in one vectorised query.
    """

    def __init__(self, df_iso):
        self.ages = np.array(sorted(df_iso["logAge"].unique()))
        self.iso = {a: isochrone_arrays(df_iso, a) for a in self.ages}
        self.trees = {}
        for a, v in self.iso.items():
            self.trees[a] = {}
            for k in FEATURE_WINDOWS:
                sel = np.isin(v["label"], FEATURE_PHASES[k])
                if sel.sum() < 2:                 # phase missing at this age: fall back to the whole isochrone
                    sel = np.ones_like(sel)
                self.trees[a][k] = cKDTree(np.c_[v["gr"][sel], v["Mr"][sel] / MAG_SCALE])
        self.weights = np.array([FEATURE_WINDOWS[k]["weight"] for k in FEATURE_WINDOWS])

    @staticmethod
    def pack(feats):
        pts = np.vstack([feats[k] for k in FEATURE_WINDOWS])
        pts = np.c_[pts[:, 0], pts[:, 1] / MAG_SCALE]
        edges = np.cumsum([0] + [len(feats[k]) for k in FEATURE_WINDOWS])
        return pts, edges

    def _loss_many(self, age, pts, edges, mu, eb, workers=1):
        shift = np.c_[E_GR_PER_EBV * eb, (mu + A_R_PER_EBV * eb) / MAG_SCALE]       # (n_grid, 2)
        meds = []
        for i, k in enumerate(FEATURE_WINDOWS):
            fp = pts[edges[i]:edges[i + 1]]
            q = (fp[None, :, :] - shift[:, None, :]).reshape(-1, 2)
            d = self.trees[age][k].query(q, workers=workers)[0].reshape(len(mu), len(fp))
            meds.append(np.median(d, axis=1))
        meds = np.stack(meds, axis=1)
        return meds @ self.weights

    def loss(self, age, pts, edges, mu0, ebv):
        if not (MU0_RANGE[0] <= mu0 <= MU0_RANGE[1] and EBV_RANGE[0] <= ebv <= EBV_RANGE[1]):
            return 1e6
        return float(self._loss_many(age, pts, edges, np.array([mu0]), np.array([ebv]))[0])

    def grid(self, age, pts, edges, step, workers=1):
        mus = np.arange(MU0_RANGE[0], MU0_RANGE[1] + 1e-9, step[0])
        ebs = np.arange(EBV_RANGE[0], EBV_RANGE[1] + 1e-9, step[1])
        MU, EB = np.meshgrid(mus, ebs)
        L = self._loss_many(age, pts, edges, MU.ravel(), EB.ravel(), workers).reshape(MU.shape)
        return mus, ebs, L

    def fit_age(self, age, pts, edges, step, workers=1):
        mus, ebs, L = self.grid(age, pts, edges, step, workers)
        j, i = np.unravel_index(np.argmin(L), L.shape)
        r = minimize(lambda p: self.loss(age, pts, edges, p[0], p[1]), [mus[i], ebs[j]], method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-6, maxiter=400, initial_simplex=[
                         [mus[i], ebs[j]], [mus[i] + step[0], ebs[j]], [mus[i], ebs[j] + step[1]]]))
        if r.fun <= L[j, i]:
            return float(r.fun), float(r.x[0]), float(r.x[1]), (mus, ebs, L)
        return float(L[j, i]), float(mus[i]), float(ebs[j]), (mus, ebs, L)

    def fit(self, feats, ages=None, step=GRID_STEP_FINE, workers=1, keep_grids=False):
        pts, edges = self.pack(feats)
        rows, grids = [], {}
        for a in (self.ages if ages is None else ages):
            loss, mu, eb, g = self.fit_age(a, pts, edges, step, workers)
            rows.append((a, loss, mu, eb))
            if keep_grids:
                grids[a] = g
        tab = pd.DataFrame(rows, columns=["logAge", "loss", "mu0", "ebv"])
        best = tab.loc[tab["loss"].idxmin()]
        return dict(logAge=float(best["logAge"]), mu0=float(best["mu0"]), ebv=float(best["ebv"]),
                    loss=float(best["loss"]), table=tab, grids=grids)


_FITTER, _FEATS, _AGES = None, None, None


def _boot_one(seed):
    rng = np.random.default_rng(seed)
    f = {k: v[rng.integers(0, len(v), len(v))] for k, v in _FEATS.items()}
    r = _FITTER.fit(f, ages=_AGES, step=GRID_STEP_BOOT)
    return r["logAge"], r["mu0"], r["ebv"], r["loss"]


def derived(logage, mu0, ebv, fitter):
    mto = turnoff_mass(fitter.iso[fitter.ages[np.argmin(np.abs(fitter.ages - logage))]])[0]
    return dict(age_gyr=10 ** logage / 1e9, mMV=mu0 + R_V * ebv, d_kpc=10 ** (mu0 / 5 + 1) / 1e3, mto=mto,
                mMr=mu0 + A_R_PER_EBV * ebv)


def run_stage3(members, calib, refs, centre, n_boot, jobs, diff_reddening="auto", label="main", fitter=None,
               do_teff=True, write_files=True):
    banner("STAGE 3: POPULATIONS, REDDENING TEST, UNFORCED PARSEC v1.2S FIT"
           + ("" if label == "main" else f"  [comparison sample: {COMPARE_LABELS[label]}]"))
    poly = flag_populations(members)
    redd = reddening_uniformity_test(members, centre, refs)
    correct = diff_reddening == "on" or (diff_reddening == "auto" and not redd["uniform"])
    redd["correction_applied"] = bool(correct)
    redd["correction_mode"] = diff_reddening
    apply_fit_photometry(members, correct)
    if correct:
        poly = flag_populations(members, "gr_fit", "r_fit")
        print(f"  Differential-reddening correction APPLIED to the fit photometry (--diff-reddening {diff_reddening}).")
    else:
        print(f"  No differential-reddening correction applied (--diff-reddening {diff_reddening}).")
    pops = dict(n_members=len(members), n_single=int(members["is_single_member"].sum()),
                n_binary=int(members["is_binary"].sum()), n_bss=int(members["is_bss"].sum()),
                n_rc_region=int(members["is_rc_region"].sum()), n_saturated=int(members["is_saturated"].sum()))
    print(f"Members {pops['n_members']:,}: singles {pops['n_single']:,} | binaries {pops['n_binary']} | "
          f"BSS {pops['n_bss']} | RC/giant window {pops['n_rc_region']} | saturated {pops['n_saturated']}")

    fitter = fitter or IsochroneFitter(load_parsec())
    feats = feature_points(members)
    print("Anchor features: " + ", ".join(f"{k} {len(v)} stars (w={FEATURE_WINDOWS[k]['weight']})" for k, v in feats.items()))
    t0 = time.time()
    best = fitter.fit(feats, workers=-1, keep_grids=True)
    print(f"Grid fit over {len(fitter.ages)} ages x (mu0, E) in {time.time() - t0:.0f} s -> logAge {best['logAge']:.2f}, "
          f"mu0 {best['mu0']:.3f}, E(B-V) {best['ebv']:.3f}, loss {best['loss']:.4f}")
    at_edge = [n for n, v, rg in (("mu0", best["mu0"], MU0_RANGE), ("E(B-V)", best["ebv"], EBV_RANGE))
               if min(abs(v - rg[0]), abs(v - rg[1])) < 1e-3] + (
        ["logAge"] if best["logAge"] in (fitter.ages[0], fitter.ages[-1]) else [])
    if at_edge:
        print(f"  WARNING: best fit sits on the search boundary for {at_edge}")

    # ---- bootstrap (statistical) ----
    global _FITTER, _FEATS, _AGES
    _FITTER, _FEATS = fitter, feats
    _AGES = fitter.ages[np.abs(fitter.ages - best["logAge"]) <= BOOT_AGE_WINDOW + 1e-6]
    t0 = time.time()
    if n_boot > 0:
        with mp.get_context("fork").Pool(jobs) as pool:
            boot = np.array(pool.map(_boot_one, range(1000, 1000 + n_boot)))
    else:
        boot = np.empty((0, 4))
    print(f"Bootstrap: {n_boot} resamples in {time.time() - t0:.0f} s")
    bdf = pd.DataFrame(boot, columns=["logAge", "mu0", "ebv", "loss"])
    if len(bdf):
        d = pd.DataFrame([derived(a, m, e, fitter) for a, m, e in bdf[["logAge", "mu0", "ebv"]].values])
        bdf = pd.concat([bdf, d], axis=1)
        edge_frac = float(np.mean(np.isin(bdf["logAge"].round(3), np.round([_AGES[0], _AGES[-1]], 3))))
    else:
        edge_frac = 0.0
    bdf.to_csv("isochrone_bootstrap_samples.csv" if label == "main" else f"isochrone_bootstrap_samples_{label}.csv", index=False)

    # ---- calibration systematics: refit with zero points shifted by +/- their uncertainty ----
    sys_r = calib["r"]["zp_sys"]
    sys_c = float(np.hypot(calib["g"]["zp_sys"], calib["r"]["zp_sys"]))
    shifts = {"r+": (0, sys_r), "r-": (0, -sys_r), "c+": (sys_c, 0), "c-": (-sys_c, 0)}
    sysfits = {}
    for k, (dc, dr) in shifts.items():
        f = {n: v + np.array([dc, dr]) for n, v in feats.items()}
        r = fitter.fit(f, ages=_AGES, step=GRID_STEP_FINE, workers=-1)
        sysfits[k] = dict(logAge=r["logAge"], mu0=r["mu0"], ebv=r["ebv"], **derived(r["logAge"], r["mu0"], r["ebv"], fitter))

    bd = derived(best["logAge"], best["mu0"], best["ebv"], fitter)
    params = {}
    for key, val in (("logAge", best["logAge"]), ("mu0", best["mu0"]), ("ebv", best["ebv"]),
                     ("age_gyr", bd["age_gyr"]), ("mMV", bd["mMV"]), ("d_kpc", bd["d_kpc"]), ("mto", bd["mto"]),
                     ("mMr", bd["mMr"])):
        if len(bdf):
            lo, hi = np.percentile(bdf[key], [16, 84])
            stat = float(0.5 * (hi - lo))
            stat_lo, stat_hi = float(max(val - lo, 0.0)), float(max(hi - val, 0.0))   # asymmetric, about the best fit
        else:
            lo = hi = stat = stat_lo = stat_hi = float("nan")
        # the age grid is discrete (0.01 dex): add its quantisation floor 0.01/sqrt(12) dex
        q_floor = {"logAge": 0.01 / np.sqrt(12), "age_gyr": np.log(10) * bd["age_gyr"] * 0.01 / np.sqrt(12)}.get(key, 0.0)
        stat, stat_lo, stat_hi = (float(np.hypot(x, q_floor)) for x in (stat, stat_lo, stat_hi))
        # calibration systematic: largest move from the best fit in each shift direction (r', colour), in quadrature
        sys = float(np.hypot(max(abs(sysfits["r+"][key] - val), abs(sysfits["r-"][key] - val)),
                             max(abs(sysfits["c+"][key] - val), abs(sysfits["c-"][key] - val))))
        params[key] = dict(value=float(val), stat=stat, sys=sys, total=float(np.hypot(stat, sys)),
                           stat_lo=stat_lo, stat_hi=stat_hi, total_lo=float(np.hypot(stat_lo, sys)),
                           total_hi=float(np.hypot(stat_hi, sys)), p16=float(lo), p84=float(hi))
    if len(bdf):
        params["mu0_ebv_corr"] = float(np.corrcoef(bdf["mu0"], bdf["ebv"])[0, 1])
    print("\nUNFORCED FIT RESULT (value, asymmetric bootstrap stat, calibration sys):")
    for key, lab, fmt in (("age_gyr", "Age [Gyr]", ".2f"), ("logAge", "logAge", ".3f"), ("mu0", "mu0 [mag]", ".3f"),
                          ("ebv", "E(B-V) [mag]", ".3f"), ("mMV", "(m-M)_V [mag]", ".3f"), ("d_kpc", "d [kpc]", ".3f"),
                          ("mto", "M_TO [Msun]", ".3f")):
        p = params[key]
        print(f"  {lab:<15s} {p['value']:{fmt}} +{p['stat_hi']:{fmt}}/-{p['stat_lo']:{fmt}} (stat) +/- {p['sys']:{fmt}} (sys)  "
              f"(total +{p['total_hi']:{fmt}}/-{p['total_lo']:{fmt}})")
    if edge_frac > 0.1:
        print(f"  NOTE: {edge_frac:.0%} of bootstrap ages hit the +/-{BOOT_AGE_WINDOW} dex search window edge.")

    # ---- star-by-star Teff ----
    iso = fitter.iso[fitter.ages[np.argmin(np.abs(fitter.ages - best["logAge"]))]]
    teff = teff_comparison(members, iso, best["mu0"], best["ebv"]) if do_teff else None
    if write_files:
        members.to_csv("cluster_members_cleaned.csv", index=False)
    mto_info = turnoff_mass(iso)
    fit = dict(best={k: best[k] for k in ("logAge", "mu0", "ebv", "loss")}, params=params, n_boot=int(len(bdf)),
               boot_age_window=BOOT_AGE_WINDOW, boot_edge_fraction=edge_frac, at_edge=at_edge,
               sys_shifts=dict(r=sys_r, colour=sys_c), sysfits=sysfits,
               feature_counts={k: int(len(v)) for k, v in feats.items()},
               age_table=best["table"].to_dict("list"), turnoff_Mr=mto_info[1], turnoff_gr0=mto_info[2])
    return dict(pops=pops, reddening=redd, fit=fit, teff=teff,
                _grids=best["grids"], _fitter=fitter, _boot=bdf, _feats=feats, _poly=poly, _members=members)


def teff_comparison(mem, iso, mu0, ebv):
    """Teff for each member = logTe of the nearest point on the best isochrone in the dereddened CMD
    (colour AND absolute magnitude, so dwarfs and giants of the same colour are told apart)."""
    tree = cKDTree(np.c_[iso["gr"], iso["Mr"] / MAG_SCALE])
    gr0 = mem["gr_fit"] - E_GR_PER_EBV * ebv
    Mr = mem["r_fit"] - mu0 - A_R_PER_EBV * ebv
    ok = np.isfinite(gr0) & np.isfinite(Mr)
    _, j = tree.query(np.c_[gr0[ok], Mr[ok] / MAG_SCALE])
    mem["gr0"], mem["Mr0"] = gr0, Mr
    mem["teff_git"] = np.nan
    mem.loc[ok, "teff_git"] = 10 ** iso["logTe"][j]
    s = mem[(mem["r_sdss"] < TEFF_MAG_LIMIT) & mem["teff_gspphot"].notna() & mem["teff_git"].notna() & mem["is_single_member"]]
    dT = (s["teff_git"] - s["teff_gspphot"]).values
    giants = (s["r_sdss"] < 14.4).values
    sig = robust_std(dT)
    res = dict(n=int(len(s)), median=float(np.median(dT)), median_err=float(1.253 * sig / np.sqrt(len(dT))),
               robust_sigma=sig, median_dwarfs=float(np.median(dT[~giants])), n_dwarfs=int((~giants).sum()),
               median_giants=float(np.median(dT[giants])) if giants.any() else float("nan"), n_giants=int(giants.sum()),
               mag_limit=TEFF_MAG_LIMIT)
    print(f"Teff (r' < {TEFF_MAG_LIMIT}, N = {res['n']}): median GIT - Gaia = {res['median']:+.0f} +/- "
          f"{res['median_err']:.0f} K, robust sigma {sig:.0f} K (dwarfs {res['median_dwarfs']:+.0f} K, "
          f"giants {res['median_giants']:+.0f} K)")
    return res


def benchmark_table(fit, mstats):
    """Compare with the proposal benchmarks. Errors are asymmetric: the side facing the benchmark is used for sigmas."""
    p = fit["params"]
    B = BENCH
    rows = []

    def row(quantity, key, **kw):
        lo, hi = err_lo_hi(p[key])
        rows.append(dict(quantity=quantity, value=p[key]["value"], err=p[key]["total"], err_lo=lo, err_hi=hi, **kw))

    def toward(key, delta):          # error on the side of the benchmark (value above it -> lower error)
        lo, hi = err_lo_hi(p[key])
        return lo if delta > 0 else hi

    d_mu = p["mMV"]["value"] - B["mMV"][0]
    row("(m-M)_V", "mMV", bench=f"{B['mMV'][0]:.2f} +/- {B['mMV'][1]:.2f}", source=B["mMV"][2], delta=d_mu,
        verdict=("criterion MET" if abs(d_mu) <= B["crit_mu"] else "criterion NOT met") +
        f" (|delta| {abs(d_mu):.3f} vs <= {B['crit_mu']} mag)")
    lo, hi = B["mu0_astero"][:2]
    v = p["mu0"]["value"]
    row("mu0", "mu0", bench=f"{lo:.2f} to {hi:.2f}", source=B["mu0_astero"][2],
        delta=0.0 if lo <= v <= hi else (v - hi if v > hi else v - lo),
        verdict="inside range" if lo <= v <= hi else f"outside range by {(v - hi if v > hi else lo - v):.3f} mag")
    for key, q, unit, prec in (("dist_kpc", "d", "kpc", 3), ("age_gyr", "Age", "Gyr", 2), ("ebv", "E(B-V)", "mag", 3)):
        pk = {"dist_kpc": "d_kpc", "age_gyr": "age_gyr", "ebv": "ebv"}[key]
        lo, hi = B[key][:2]
        v = p[pk]["value"]
        dlt = 0.0 if lo <= v <= hi else (v - hi if v > hi else v - lo)
        e = toward(pk, dlt)
        ns = abs(dlt) / e if e and e > 0 else np.nan
        if key == "age_gyr":
            verdict = ("criterion MET" if abs(dlt) <= B["crit_age"] else "criterion NOT met") + \
                      f" ({abs(dlt):.2f} Gyr from the range vs <= {B['crit_age']} Gyr)"
        else:
            verdict = "inside range" if dlt == 0 else f"outside range by {abs(dlt):.{prec}f} {unit} ({ns:.1f} sigma)"
        row(q, pk, bench=f"{lo} to {hi} {unit}", source=B[key][2], delta=dlt, verdict=verdict)
    if mstats:                       # Gaia geometric distance of the members (raw 1/parallax, no zero-point correction)
        d_plx = mstats["d_plx_raw_kpc"]
        e_plx = d_plx * np.hypot(mstats["plx_wmean_err"], GAIA_PLX_SYS) / mstats["plx_wmean"]
        dlt = p["d_kpc"]["value"] - d_plx
        ns = dlt / np.hypot(toward("d_kpc", dlt), e_plx)
        row("d vs Gaia", "d_kpc", bench=f"{d_plx:.3f} +/- {e_plx:.3f} kpc (1/parallax, uncorrected)",
            source="Gaia DR3 parallaxes of the members; +/-0.010 mas floor (Vasiliev & Baumgardt 2021); "
                   "zero-point correction NOT applied", delta=dlt, verdict=f"{ns:+.1f} sigma (combined errors)")
    v = p["mto"]["value"]
    t = (v - B["mto"][0]) / np.hypot(toward("mto", v - B["mto"][0]), B["mto"][1])
    row("M_TO", "mto", bench=f"{B['mto'][0]} +/- {B['mto'][1]} Msun", source=B["mto"][2],
        delta=v - B["mto"][0], verdict=f"{t:+.1f} sigma (combined errors)")
    return rows


COMPARE_LABELS = {"main": "Gaia members, whole frame",
                  "gaia_circle": "Gaia members, inside circle",
                  "gaia_free": "Gaia-free, inside circle"}


def run_comparison(cal, stat_circ, calib, refs, centre, n_boot, jobs, main_s3):
    """Refit the same PARSEC model to (a) Gaia members inside the circle and (b) the Gaia-free statistical members
    inside the circle, with exactly the main fit's settings (same differential-reddening decision, bootstrap size,
    zero-point systematics)."""
    mode = "on" if main_s3["reddening"]["correction_applied"] else "off"
    samples = {"gaia_circle": cal[cal["is_member"] & cal["in_circle"]].copy().reset_index(drop=True),
               "gaia_free": stat_circ[stat_circ["is_stat_member"]].copy().reset_index(drop=True)}
    out = {}
    for label, df in samples.items():
        out[label] = run_stage3(df, calib, refs, centre, n_boot, jobs, mode, label=label,
                                fitter=main_s3["_fitter"], do_teff=False, write_files=False)
    return out


def comparison_summary(main_s3, comp, decon):
    res = {"labels": COMPARE_LABELS, "decontamination": decon, "fits": {}}
    for label, s3 in [("main", main_s3)] + list(comp.items()):
        f = s3["fit"]
        res["fits"][label] = dict(
            n_stars=int(s3["pops"]["n_members"]), n_single=int(s3["pops"]["n_single"]),
            feature_counts=f["feature_counts"], correction_applied=bool(s3["reddening"]["correction_applied"]),
            reddening_chi2=float(s3["reddening"]["chi2"]), reddening_dof=int(s3["reddening"]["dof"]),
            params={k: f["params"][k] for k in ("age_gyr", "logAge", "mu0", "ebv", "mMV", "d_kpc", "mto")},
            at_edge=f["at_edge"], boot_edge_fraction=f["boot_edge_fraction"],
            benchmarks=benchmark_table(f, None))
    return res


# ==================================================================================================
# STAGE 4a: FIGURES (300 dpi)
# ==================================================================================================

DPI = 300


def fig_phase1(matched, diag):
    A, br, bg = diag["_arrays"], diag["bands"]["r"], diag["bands"]["g"]
    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(A["bkg_r"], origin="lower", cmap="plasma", extent=[0, br["nx"], 0, br["ny"]])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Sky [ADU]")
    ax.set_title(f"(a) Background2D sky model, r' ({BKG_BOX}x{BKG_BOX} px mesh)", fontweight="bold")
    ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    ax.text(0.03, 0.95, f"median {br['sky_median']:.1f} ADU\nrms {br['sky_rms']:.2f} ADU", transform=ax.transAxes,
            va="top", bbox=dict(fc="white", alpha=0.85))

    ax = fig.add_subplot(gs[0, 1])
    e = A["epsf_r"]
    im = ax.imshow(e, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("normalised ePSF")
    ax.set_title(f"(b) ePSF r' (2x oversampled, {br['n_psf_stars']} stars), FWHM {br['fwhm_pix']:.2f} px = "
                 f"{br['fwhm_arcsec']:.2f}\"", fontweight="bold", fontsize=11)
    ins = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    cy, cx = np.array(e.shape) // 2
    yy, xx = np.indices(e.shape)
    rr = np.hypot(xx - cx, yy - cy).ravel() / 2.0
    sel = rr < 8
    ins.scatter(rr[sel], e.ravel()[sel] / e.max(), s=2, c="crimson")
    sig = br["fwhm_pix"] / 2.35482
    rg = np.linspace(0, 8, 100)
    ins.plot(rg, np.exp(-0.5 * (rg / sig) ** 2), "k--", lw=1, label="Gaussian, measured FWHM")
    ins.set_xlabel("r [px]", fontsize=7); ins.tick_params(labelsize=6); ins.legend(fontsize=5)

    ax = fig.add_subplot(gs[1, 0])
    for b, col in (("r", "royalblue"), ("g", "seagreen")):
        v = matched[~matched["is_saturated"]]
        ax.scatter(v[f"mag_inst_{b}"], v[f"mag_err_inst_{b}"], s=2, alpha=0.3, color=col, label=f"{b}' measured")
        info = diag["bands"][b]
        mg = np.linspace(-15, -5.5, 200)
        F = info["exptime"] * 10 ** (-0.4 * mg)
        area = np.pi * info["r_aper_pix"] ** 2
        ax.plot(mg, 1.0857 * np.sqrt(F / info["gain"] + area * info["sky_rms"] ** 2) / F, color=col, ls="--", lw=1.5,
                label=f"{b}' expected: star + sky noise in r = {info['r_aper_pix']:.1f} px")
    ax.axhline(0.1086, color="red", ls=":", label="S/N = 10")
    ax.set_yscale("log"); ax.set_ylim(5e-4, 1); ax.set_xlim(-15, -5.5)
    ax.set_xlabel("instrumental magnitude  -2.5 log10(F/t)"); ax.set_ylabel("sigma_m [mag]")
    ax.set_title("(c) Photometric uncertainty vs. instrumental magnitude", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 1])
    for b, col in (("r", "royalblue"), ("g", "seagreen")):
        c = matched[(matched[f"psf_flags_{b}"] == 0) & (matched[f"nn_dist_pix_{b}"] > 18) &
                    (matched[f"mag_err_psf_{b}"] < 0.04) & matched[f"peak_{b}"].between(2000, BRIGHT_PEAK_FALLBACK)]
        off = diag["bands"][b]["aperture_offset"]
        ax.scatter(c[f"mag_psf_{b}"], c[f"mag_psf_{b}"] - c[f"mag_aper_1.5_{b}"], s=4, alpha=0.4, color=col,
                   label=f"{b}' clean isolated stars (N={len(c)})")
        ax.axhline(off, color=col, lw=2, label=f"{b}' median offset {off:+.3f} mag")
    ax.set_ylim(-0.25, 0.15)
    ax.set_xlabel("m_PSF (instrumental)"); ax.set_ylabel("m_PSF - m_ap(1.5 FWHM) [mag]")
    ax.set_title("(d) PSF vs. 1.5 FWHM aperture: offset used for the bright-star fallback", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig("phase1_diagnostics.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved phase1_diagnostics.png")


def fig_phase2(cal, members, calib, refs, mstats):
    fig = plt.figure(figsize=(21, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.28)
    for k, b in enumerate(("r", "g")):
        ax = fig.add_subplot(gs[0, k])
        c = calib[b]
        sc = ax.scatter(refs["c_inst"], refs[f"d{b}"], c=refs["r_ps1"], cmap="viridis_r", s=8, alpha=0.7)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("r_PS1")
        xx = np.linspace(refs["c_inst"].min(), refs["c_inst"].max(), 50)
        ax.plot(xx, c["zp"] + c["colour_term"] * xx, "r-", lw=2,
                label=f"Siegel fit: ZP = {c['zp']:.4f} +/- {c['zp_err']:.4f}\nC = {c['colour_term']:+.4f} +/- "
                      f"{c['colour_term_err']:.4f}, robust RMS = {c['rms']:.3f}")
        mid = np.median(refs[f"d{b}"])
        ax.set_ylim(mid - 0.3, mid + 0.3)
        ax.set_xlabel("(g - r)_inst"); ax.set_ylabel(f"{b}_SDSS(PS1) - {b}_inst [mag]")
        ax.set_title(f"({'ab'[k]}) {b}' zero point from {calib['n_ref']} PS1 stars", fontweight="bold")
        ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 2])
    lim = 3 * calib["r"]["rms"]
    sc = ax.scatter(refs["x_r"], refs["y_r"], c=refs["resid_r"], cmap="coolwarm", vmin=-lim, vmax=lim, s=10)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("r' residual [mag]")
    ax.set_aspect("equal"); ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    ax.set_title("(c) r' calibration residuals across the detector", fontweight="bold")

    field = cal[~cal["is_member"]]
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(field["pmra"], field["pmdec"], s=2, c="gray", alpha=0.25, label=f"field (N={len(field):,})")
    ax.scatter(members["pmra"], members["pmdec"], s=6, c="forestgreen", label=f"members (N={len(members):,})")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mstats["pmra0"] + PM_NSIGMA * PM_SIGMA * np.cos(t), mstats["pmdec0"] + PM_NSIGMA * PM_SIGMA * np.sin(t),
            "r--", label=f"{PM_NSIGMA:.0f} x {PM_SIGMA} mas/yr around ({mstats['pmra0']:+.2f}, {mstats['pmdec0']:+.2f})")
    ax.set_xlim(-12, 6); ax.set_ylim(-14, 4)
    ax.set_xlabel("pmRA* [mas/yr]"); ax.set_ylabel("pmDec [mas/yr]")
    ax.set_title("(d) Gaia DR3 proper-motion diagram", fontweight="bold"); ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-0.5, 1.5, 81)
    ax.hist(field["parallax"].dropna(), bins=bins, density=True, color="lightgray", label="field")
    ax.hist(members["parallax"].dropna(), bins=bins, density=True, color="forestgreen", alpha=0.7, label="members")
    ax.axvspan(*PLX_RANGE, color="orange", alpha=0.15, label=f"kinematic window {PLX_RANGE} mas")
    ax.axvline(mstats["plx_wmean"], color="darkred", lw=2,
               label=f"weighted mean {mstats['plx_wmean']:.4f} +/- {mstats['plx_wmean_err']:.4f} mas")
    ax.set_xlim(-0.3, 1.2); ax.set_xlabel("parallax [mas]"); ax.set_ylabel("density")
    ax.set_title("(e) Parallax distribution", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(field["g_r_sdss"], field["r_sdss"], s=2, c="silver", alpha=0.3, label="Gaia field stars")
    ax.scatter(members["g_r_sdss"], members["r_sdss"], s=5, c="black", label="Gaia members")
    ax.set_xlim(0, 2.2); ax.set_ylim(21.5, 11)
    ax.set_xlabel("g' - r' (SDSS)"); ax.set_ylabel("r' (SDSS)")
    ax.set_title("(f) All Gaia-matched stars vs. members", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig("phase2_calibration_and_membership.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved phase2_calibration_and_membership.png")


def _iso_curve(fitter, logage, mu0, ebv):
    iso = fitter.iso[fitter.ages[np.argmin(np.abs(fitter.ages - logage))]]
    x, y = iso["gr"] + E_GR_PER_EBV * ebv, iso["Mr"] + mu0 + A_R_PER_EBV * ebv
    lab = iso["label"]
    return [(x[lab <= 3], y[lab <= 3]), (x[lab == 4], y[lab == 4])]


def _plot_cmd(ax, cal, members, fitter, fit, legend_values=True, field=True):
    p = fit["params"]
    if field:
        f = cal[~cal["is_member"]]
        ax.scatter(f["g_r_sdss"], f["r_sdss"], s=2, c="lightgray", alpha=0.25, label=f"field (N={len(f):,})")
    b = members[members["is_binary"]]
    ax.scatter(b["gr_fit"], b["r_fit"], s=14, facecolors="none", edgecolors="darkorange", label=f"MS binaries (N={len(b)})")
    s = members[members["is_bss"]]
    ax.scatter(s["gr_fit"], s["r_fit"], s=30, marker="D", c="deepskyblue", edgecolors="blue", label=f"blue stragglers (N={len(s)})")
    rc = members[members["is_rc_region"]]
    ax.scatter(rc["gr_fit"], rc["r_fit"], s=30, marker="^", c="crimson", label=f"red clump/giant window (N={len(rc)})")
    sg = members[members["is_single_member"] & ~members["is_rc_region"]]
    ax.scatter(sg["gr_fit"], sg["r_fit"], s=6, c="black", label=f"single members (N={len(sg):,})")
    lab = (f"PARSEC v1.2S, [Fe/H] = {FEH:+.2f}\n"
           f"age = {pm_txt(p['age_gyr'], 2)} Gyr\n"
           f"mu0 = {pm_txt(p['mu0'], 3)},  E(B-V) = {pm_txt(p['ebv'], 3)}\n"
           f"(m-M)_V = {pm_txt(p['mMV'], 3)},  d = {pm_txt(p['d_kpc'], 3)} kpc\n"
           f"M_TO = {pm_txt(p['mto'], 2)} Msun")
    for i, (x, y) in enumerate(_iso_curve(fitter, fit["best"]["logAge"], fit["best"]["mu0"], fit["best"]["ebv"])):
        ax.plot(x, y, "r-", lw=2, zorder=6, label=lab if (i == 0 and legend_values) else None)
    ax.set_xlim(0.1, 2.2); ax.set_ylim(21, 11)
    corr = "corrected" if "gr_fit" in members and not np.allclose(members["gr_fit"], members["g_r_sdss"], equal_nan=True) else ""
    ax.set_xlabel("g' - r' (SDSS, calibrated" + (", members dereddened differentially)" if corr else ")"))
    ax.set_ylabel("r' (SDSS, calibrated)")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.92); ax.grid(alpha=0.3)


# Literature solutions drawn on the CMD with the same PARSEC grid and the same extinction coefficients:
# (label, age [Gyr], (m-M)_V, E(B-V), style); checked on ADS: Brewer et al. 2016 (CMD and binary ages, E(B-V) adopted from
# Anthony-Twarog et al. 2014), Kalirai et al. 2001 (2.5 Gyr, (m-M)_V = 12.30 for an adopted E(B-V) = 0.10).
LIT_SOLUTIONS = [
    ("Brewer+16, CMD age: 2.21 Gyr, (m-M)$_V$ = 12.38, E(B-V) = 0.16", 2.21, 12.38, 0.16, dict(c="tab:blue", ls="--")),
    ("Brewer+16, binary age: 2.38 Gyr, (m-M)$_V$ = 12.38, E(B-V) = 0.16", 2.38, 12.38, 0.16, dict(c="tab:blue", ls=":")),
    ("Kalirai+01: 2.5 Gyr, (m-M)$_V$ = 12.30, E(B-V) = 0.10", 2.50, 12.30, 0.10, dict(c="tab:green", ls="-.")),
]
LIT_FIG = "NGC6819_fit_vs_literature.png"


def fig_fit_vs_literature(members, fit, fitter, bdf):
    """(a) member CMD with this work's isochrone and the literature solutions; (b) mu0-E(B-V) plane with the loss surface,
    the bootstrap fits and the literature values."""
    best, p = fit["best"], fit["params"]
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.25))
    ax = axs[0]
    ot = members[~members["is_single_member"]]
    sg = members[members["is_single_member"]]
    ax.scatter(ot["gr_fit"], ot["r_fit"], s=9, facecolors="none", edgecolors="darkorange", lw=0.6,
               label=f"binaries / blue stragglers / saturated (N={len(ot)})")
    ax.scatter(sg["gr_fit"], sg["r_fit"], s=4, c="black", label=f"single members (N={len(sg):,})")
    for i, (x, y) in enumerate(_iso_curve(fitter, best["logAge"], best["mu0"], best["ebv"])):
        ax.plot(x, y, "r-", lw=2.6, zorder=6, label=(f"this work: {p['age_gyr']['value']:.2f} Gyr, (m-M)$_V$ = "
                                                       f"{p['mMV']['value']:.2f}, E(B-V) = {p['ebv']['value']:.3f}") if i == 0 else None)
    for lab, age, mMV, ebv, sty in LIT_SOLUTIONS:
        for i, (x, y) in enumerate(_iso_curve(fitter, np.log10(age * 1e9), mMV - R_V * ebv, ebv)):
            ax.plot(x, y, lw=1.8, zorder=5, label=lab if i == 0 else None, **sty)
    ax.set_xlim(0.1, 1.6); ax.set_ylim(19.0, 11.5)
    ax.set_xlabel("g' - r' (calibrated, differential reddening removed)"); ax.set_ylabel("r'")
    ax.set_title("(a) Member CMD: this work's fit vs. literature solutions", fontweight="bold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1); ax.grid(alpha=0.3)

    ax = axs[1]
    pts, edges = fitter.pack(feature_points(members))
    mus, ebs, L = fitter.grid(best["logAge"], pts, edges, GRID_STEP_FINE, workers=-1)
    cf = ax.contourf(ebs, mus, (L - L.min()).T, levels=np.linspace(0, 0.3, 31), cmap="magma_r", extend="max")
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label("loss - minimum (at the best age)")
    if len(bdf):
        ax.scatter(bdf["ebv"], bdf["mu0"], s=7, c="cyan", edgecolors="k", lw=0.2, label="bootstrap fits (this work)")
    elo, ehi = err_lo_hi(p["ebv"]); mlo, mhi = err_lo_hi(p["mu0"])
    ax.errorbar(best["ebv"], best["mu0"], xerr=[[elo], [ehi]], yerr=[[mlo], [mhi]], fmt="*", ms=17, c="gold", mec="k",
                ecolor="k", capsize=4, label="this work (best fit, total errors)", zorder=6)
    ax.errorbar(0.16, 12.38 - R_V * 0.16, yerr=0.04, fmt="s", c="tab:blue", mec="k", capsize=4,
                label="Brewer+16: (m-M)$_V$ = 12.38, E(B-V) = 0.16")
    ax.errorbar(0.10, 12.30 - R_V * 0.10, yerr=0.12, fmt="D", c="tab:green", mec="k", capsize=4,
                label="Kalirai+01: (m-M)$_V$ = 12.30, E(B-V) = 0.10 (adopted)")
    ax.axhline(11.85, c="dodgerblue", ls="--", label=r"Basu+11: $\mu_0$ = 11.85 (asteroseismic)")
    ax.axhline(12.20, c="dodgerblue", ls=":", label=r"Balona+13: $\mu_0$ = 12.20 (asteroseismic)")
    ax.axvspan(0.153, 0.167, color="tab:purple", alpha=0.25, label="Anthony-Twarog+14: E(B-V) = 0.160 $\\pm$ 0.007")
    ax.set_xlim(0, 0.25); ax.set_ylim(11.6, 12.5)
    ax.set_xlabel("E(B-V)"); ax.set_ylabel(r"$\mu_0$")
    ax.set_title(r"(b) Distance-reddening plane: this work vs. literature", fontweight="bold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2); ax.grid(alpha=0.3)
    fig.savefig(LIT_FIG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {LIT_FIG}")



def fig_phase3(cal, members, s3):
    fit, fitter, bdf, teff = s3["fit"], s3["_fitter"], s3["_boot"], s3["teff"]
    best = fit["best"]
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    tab = pd.DataFrame(fit["age_table"])
    ax.plot(10 ** tab["logAge"] / 1e9, tab["loss"], "ko-", ms=4, label="best loss at each age (mu0, E free)")
    if len(bdf):
        ax2 = ax.twinx()
        la = fitter.ages
        edges = 10 ** np.r_[la[0] - 0.005, 0.5 * (la[1:] + la[:-1]), la[-1] + 0.005] / 1e9
        ax2.hist(bdf["age_gyr"], bins=edges, color="tab:blue", alpha=0.3)
        ax2.set_ylabel(f"bootstrap count (N={len(bdf)})", color="tab:blue")
    ax.axvspan(*BENCH["age_gyr"][:2], color="green", alpha=0.12, label="proposal benchmark 2.2-2.5 Gyr")
    ax.set_xlabel("age [Gyr]"); ax.set_ylabel("loss")
    ax.set_title("(a) Age constraint: loss profile and bootstrap", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    mus, ebs, L = s3["_grids"][best["logAge"]]
    cf = ax.contourf(ebs, mus, (L - L.min()).T, levels=np.linspace(0, 0.3, 31), cmap="magma_r", extend="max")
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label("loss - min (at best age)")
    if len(bdf):
        ax.scatter(bdf["ebv"], bdf["mu0"], s=6, c="cyan", edgecolors="k", lw=0.2, label="bootstrap best fits")
    ax.plot(best["ebv"], best["mu0"], "*", ms=16, c="gold", mec="k", label="best fit")
    ax.errorbar(0.16, BENCH["mMV"][0] - R_V * 0.16, yerr=BENCH["mMV"][1], fmt="s", c="lime", mec="k", capsize=4,
                label="Brewer+16: (m-M)_V = 12.38, E(B-V) = 0.16")
    for v, ls in zip(BENCH["mu0_astero"][:2], ("--", ":")):
        ax.axhline(v, c="dodgerblue", ls=ls, label=f"asteroseismic mu0 = {v}")
    ax.axvspan(*BENCH["ebv"][:2], color="green", alpha=0.1, label="E(B-V) benchmark 0.10-0.16")
    ax.set_xlim(0, 0.25); ax.set_ylim(11.6, 12.6)
    ax.set_xlabel("E(B-V)"); ax.set_ylabel("mu0")
    ax.set_title(f"(b) mu0-E(B-V) degeneracy at logAge {best['logAge']:.2f}", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    _plot_cmd(ax, cal, members, fitter, fit)
    ax.set_title("(c) Decontaminated CMD + unforced PARSEC fit", fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    r = s3["reddening"]
    for zs, off, col, lab in ((r["zones"], -0.12, "tab:red", f"MS colour offset (dust?): chi2 = {r['chi2']:.0f}/{r['dof']}"),
                              (r["cal_zones"], 0.12, "tab:gray", f"PS1 calibration (g-r) residual: chi2 = {r['cal_chi2']:.0f}/{r['cal_dof']}")):
        ax.errorbar(np.arange(len(zs)) + off, [q["median"] for q in zs], yerr=[q["err"] for q in zs], fmt="o",
                    capsize=4, color=col, label=lab)
    ax.axhline(0, color="k", ls="--")
    ax.set_xticks(range(len(r["zones"]))); ax.set_xticklabels([q["zone"] for q in r["zones"]], rotation=30, fontsize=8)
    ax.set_ylabel("zone median - global median [mag in g'-r']")
    ax.set_title("(d) Reddening uniformity test" + (" (local correction applied)" if r["correction_applied"] else ""),
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    s = members[(members["r_sdss"] < TEFF_MAG_LIMIT) & members["teff_gspphot"].notna() & members["teff_git"].notna()
                & members["is_single_member"]]
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(s["teff_gspphot"], s["teff_git"], c=s["r_sdss"], cmap="plasma_r", s=12)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("r'")
    tt = np.linspace(4000, 7000, 10)
    ax.plot(tt, tt, "k--", label="1:1")
    ax.set_xlim(4000, 7000); ax.set_ylim(4000, 7000)
    ax.set_xlabel("Teff Gaia DR3 GSP-Phot [K]"); ax.set_ylabel("Teff GIT (PARSEC, nearest isochrone point) [K]")
    ax.set_title(f"(e) Star-by-star Teff (r' < {TEFF_MAG_LIMIT}, N = {teff['n']})", fontweight="bold")
    ax.text(0.03, 0.95, f"median dT = {teff['median']:+.0f} +/- {teff['median_err']:.0f} K\nrobust sigma = {teff['robust_sigma']:.0f} K",
            transform=ax.transAxes, va="top", bbox=dict(fc="white", alpha=0.85))
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(s["r_sdss"], s["teff_git"] - s["teff_gspphot"], s=10, c="royalblue", alpha=0.6)
    ax.axhline(0, c="k", ls="--")
    ax.axhline(teff["median"], c="r", label=f"median {teff['median']:+.0f} K")
    ax.axhspan(teff["median"] - teff["robust_sigma"], teff["median"] + teff["robust_sigma"], color="r", alpha=0.1,
               label=f"+/- robust sigma ({teff['robust_sigma']:.0f} K)")
    ax.set_ylim(-800, 800); ax.set_xlabel("r'"); ax.set_ylabel("Teff GIT - Teff Gaia [K]")
    ax.set_title("(f) Teff residuals vs. brightness", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig("phase3_isochrone_fit.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved phase3_isochrone_fit.png")


def fig_final_cmd(cal, members, s3, circle, counts, geom):
    data, hdr, wcs = load_frame("r")
    R = circle["radius_arcmin"]
    side = max(18.0, 2.0 * R * 1.12)
    cut = Cutout2D(data, circle["centre"], (side * u.arcmin, side * u.arcmin), wcs=wcs, mode="partial", fill_value=np.nan)
    fig = plt.figure(figsize=(17, 8.2))
    ax = fig.add_subplot(1, 2, 1, projection=cut.wcs)
    norm = ImageNormalize(cut.data[np.isfinite(cut.data)], interval=ZScaleInterval(contrast=0.25), stretch=AsinhStretch(0.1))
    ax.imshow(cut.data, origin="lower", cmap="inferno", norm=norm)
    cx, cy = cut.wcs.world_to_pixel(circle["centre"])
    ax.add_patch(Circle((cx, cy), circle["r_pix"], fill=False, ec="cyan", lw=2,
                        label=f"manual DS9 circle, r = {R:.2f}' ({counts['members_inside']} Gaia members inside)"))
    r1, r2 = counts["ring_arcmin"]
    for rr in (r1, r2):
        ax.add_patch(Circle((cx, cy), rr * 60 / geom["pix_scale"], fill=False, ec="white", lw=1, ls="--"))
    px, py = cut.wcs.world_to_pixel(geom["centre"])
    ax.plot(px, py, "+", c="lime", ms=16, mew=2, label="pointing centre (TARRA/TARDEC)")
    ax.set_xlim(0, cut.data.shape[1]); ax.set_ylim(0, cut.data.shape[0])
    # label the coordinates themselves: this frame is rotated ~90 deg (North points right), so RA runs
    # along the vertical axis and Dec along the horizontal one
    ax.coords[0].set_axislabel("RA (J2000)")
    ax.coords[1].set_axislabel("Dec (J2000)")
    n_dir = cut.wcs.wcs_world2pix([[circle["ra_deg"], circle["dec_deg"] + 0.05]], 0)[0] - np.array([cx, cy])
    tip = np.array([0.08, 0.08]) + 0.07 * n_dir / np.hypot(*n_dir)
    ax.annotate("", xy=tuple(tip), xytext=(0.08, 0.08), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    lab = tip + 0.03 * n_dir / np.hypot(*n_dir)
    ax.text(lab[0], lab[1], "N", transform=ax.transAxes, color="white", fontsize=11, fontweight="bold",
            ha="center", va="center")
    ax.set_title(f"GIT 0.7 m, SDSS r' 180 s, central {side:.0f}' x {side:.0f}'\n(dashed: inner edge of background ring {r1:.1f}')",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white", framealpha=0.7)
    ax2 = fig.add_subplot(1, 2, 2)
    _plot_cmd(ax2, cal, members, s3["_fitter"], s3["fit"])
    ax2.set_title("Decontaminated SDSS CMD of NGC 6819 with the unforced PARSEC v1.2S fit", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig("NGC6819_Final_CMD_Comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    del data
    print("Saved NGC6819_Final_CMD_Comparison.png")


def _plot_fit_sample(ax, members, s3, colour_by=None, title=""):
    p = s3["fit"]["params"]
    b = s3["fit"]["best"]
    if colour_by is not None:
        sc = ax.scatter(members["gr_fit"], members["r_fit"], s=7, c=members[colour_by], cmap="viridis",
                        vmin=P_STAT_MIN, vmax=1.0, zorder=3)
        plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.045).set_label("P_stat (fraction of realisations survived)")
    else:
        sg = members[members["is_single_member"]]
        ot = members[~members["is_single_member"]]
        ax.scatter(ot["gr_fit"], ot["r_fit"], s=10, facecolors="none", edgecolors="darkorange", zorder=3,
                   label=f"binaries / BSS / saturated (N={len(ot)})")
        ax.scatter(sg["gr_fit"], sg["r_fit"], s=5, c="black", zorder=3, label=f"single stars (N={len(sg):,})")
    lab = (f"age {pm_txt(p['age_gyr'], 2)} Gyr\n"
           f"(m-M)_V {pm_txt(p['mMV'], 3)}\n"
           f"E(B-V) {pm_txt(p['ebv'], 3)}\n"
           f"M_TO {pm_txt(p['mto'], 2)} Msun")
    for i, (x, y) in enumerate(_iso_curve(s3["_fitter"], b["logAge"], b["mu0"], b["ebv"])):
        ax.plot(x, y, "r-", lw=2, zorder=6, label=lab if i == 0 else None)
    ax.set_xlim(0.1, 2.2); ax.set_ylim(20.5, 11)
    ax.set_xlabel("g' - r' (calibrated" + (", dereddened differentially)" if s3["reddening"]["correction_applied"] else ")"))
    ax.set_ylabel("r'")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.92); ax.grid(alpha=0.3)


def fig_membership_comparison(stat_circ, ring, decon, main_s3, comp, members_main):
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(stat_circ["g_r_sdss"], stat_circ["r_sdss"], s=3, c="black")
    ax.set_title(f"(a) Every star inside your circle, S/N > 10 (N = {len(stat_circ):,})\nno membership information",
                 fontweight="bold", fontsize=11)
    ax = [ax, fig.add_subplot(gs[0, 1])]
    rng = np.random.default_rng(3)
    one = ring[rng.random(len(ring)) < decon["area_ratio"]]
    ax[1].scatter(one["g_r_sdss"], one["r_sdss"], s=3, c="grey")
    ax[1].set_title(f"(b) Background ring, scaled to the circle's area (one random draw, N = {len(one):,})\n"
                    f"this is what gets subtracted from (a)", fontweight="bold", fontsize=11)
    for a in ax:
        a.set_xlim(0.1, 2.2); a.set_ylim(20.5, 11); a.grid(alpha=0.3)
        a.set_xlabel("g' - r' (calibrated)"); a.set_ylabel("r'")
    _plot_fit_sample(fig.add_subplot(gs[0, 2]), comp["gaia_free"]["_members"], comp["gaia_free"], colour_by="p_stat",
                     title=f"(c) GAIA-FREE: survivors with P_stat >= {P_STAT_MIN} (N = {decon['n_stat_members']:,})\n+ its own unforced PARSEC fit")
    _plot_fit_sample(fig.add_subplot(gs[1, 0]), comp["gaia_circle"]["_members"], comp["gaia_circle"],
                     title=f"(d) GAIA: proper-motion/parallax members inside the circle\n(N = {comp['gaia_circle']['pops']['n_members']:,}) + fit")
    _plot_fit_sample(fig.add_subplot(gs[1, 1]), members_main, main_s3,
                     title=f"(e) GAIA: members on the whole frame (main result, N = {main_s3['pops']['n_members']:,}) + fit")
    sub = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1, 2], hspace=0.65)
    runs = [("main", main_s3, "black"), ("gaia_circle", comp["gaia_circle"], "tab:blue"), ("gaia_free", comp["gaia_free"], "tab:orange")]
    for i, (key, lab, bench) in enumerate((("age_gyr", "Age [Gyr]", BENCH["age_gyr"][:2]),
                                           ("mMV", "(m-M)_V [mag]", (BENCH["mMV"][0] - BENCH["mMV"][1], BENCH["mMV"][0] + BENCH["mMV"][1])),
                                           ("ebv", "E(B-V) [mag]", BENCH["ebv"][:2]),
                                           ("mto", "M_TO [Msun]", (BENCH["mto"][0] - BENCH["mto"][1], BENCH["mto"][0] + BENCH["mto"][1])))):
        a = fig.add_subplot(sub[i])
        a.axvspan(*bench, color="green", alpha=0.18, label="literature benchmark" if i == 0 else None)
        for j, (name, s3, col) in enumerate(runs):
            q = s3["fit"]["params"][key]
            lo, hi = err_lo_hi(q)
            a.errorbar(q["value"], -j, xerr=[[lo], [hi]], fmt="o", color=col, capsize=4, ms=6)
        a.set_yticks([0, -1, -2]); a.set_yticklabels([COMPARE_LABELS[r[0]] for r in runs], fontsize=8)
        a.set_ylim(-2.6, 0.6); a.set_title(lab, fontsize=10, loc="left"); a.grid(alpha=0.3, axis="x")
        if i == 0:
            a.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("NGC 6819 membership: Gaia-free statistical field subtraction vs. Gaia astrometry "
                 "(same calibration, same isochrone fit)", fontsize=14, fontweight="bold")
    fig.savefig("NGC6819_Gaia_vs_GaiaFree.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved NGC6819_Gaia_vs_GaiaFree.png")


def apt_summary(apt, exptime):
    s = apt.copy()
    out = dict(n_entered=0)
    if "apt_source_intensity_r" in s and s["apt_source_intensity_r"].notna().sum() >= 3:
        v = s[s["apt_source_intensity_r"] > 0]
        d = -2.5 * np.log10(v["apt_source_intensity_r"] / exptime) - v["mag_aper_1.5_r"]
        out = dict(n_entered=int(len(v)), median_diff=float(np.median(d)), robust_sigma=robust_std(d),
                   n_within_005=int(np.sum(np.abs(d - np.median(d)) < 0.05)))
    return out


def fig_apt(apt, diag):
    br = diag["bands"]["r"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    ax = axs[0, 0]
    ax.scatter(apt["mag_psf_r"], apt["mag_aper_1.5_r"], s=30, c="royalblue", edgecolors="k")
    xx = np.linspace(apt["mag_psf_r"].min(), apt["mag_psf_r"].max(), 10)
    ax.plot(xx, xx - br["aperture_offset"], "r--", label=f"m_ap = m_PSF - ({br['aperture_offset']:+.3f})")
    ax.set_xlabel("m_PSF r' (instrumental)"); ax.set_ylabel("m_ap(1.5 FWHM) r'")
    ax.set_title("(a) PSF vs. aperture, 50-star APT sample", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
    ax = axs[0, 1]
    r = np.array(APER_FACTORS) * br["fwhm_pix"]
    for _, row in apt.sort_values("mag_robust_r").iloc[5:15].iterrows():
        f = 10 ** (-0.4 * np.array([row["mag_aper_1.0_r"], row["mag_aper_1.5_r"], row["mag_aper_2.0_r"]]))
        ax.plot(r, f / f[-1], "o-", alpha=0.7)
    ax.set_xlabel("aperture radius [px] (1.0, 1.5, 2.0 FWHM)"); ax.set_ylabel("flux / flux(2.0 FWHM)")
    ax.set_title("(b) Curve of growth (10 bright sample stars)", fontweight="bold"); ax.grid(alpha=0.3)
    ax = axs[1, 0]
    ax.scatter(apt["mag_robust_r"], apt["mag_err_aper_r"], s=25, c="darkorange", label="aperture sigma")
    ax.scatter(apt["mag_robust_r"], apt["mag_err_psf_r"], s=25, marker="^", c="purple", label="ePSF sigma")
    ax.set_yscale("log"); ax.set_xlabel("m_inst r'"); ax.set_ylabel("sigma_m [mag]")
    ax.set_title("(c) Error estimates: aperture vs. ePSF", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3, which="both")
    ax = axs[1, 1]
    s = apt_summary(apt, br["exptime"])
    if s["n_entered"] >= 3:
        v = apt[apt["apt_source_intensity_r"] > 0]
        d = -2.5 * np.log10(v["apt_source_intensity_r"] / br["exptime"]) - v["mag_aper_1.5_r"]
        ax.scatter(v["mag_aper_1.5_r"], d, s=30, c="teal", edgecolors="k")
        ax.axhline(s["median_diff"], c="r", label=f"median {s['median_diff']:+.3f}, robust sigma {s['robust_sigma']:.3f}")
        ax.axhspan(s["median_diff"] - 0.05, s["median_diff"] + 0.05, color="r", alpha=0.1, label="+/- 0.05 mag")
        ax.set_xlabel("Python m_ap(1.5 FWHM) r'"); ax.set_ylabel("APT - Python [mag]")
        ax.set_title(f"(d) APT vs. Python, same aperture (N = {s['n_entered']})", fontweight="bold"); ax.legend()
    else:
        ax.text(0.5, 0.5, "No APT measurements entered yet.\n\nMeasure the stars in apt_crosscheck_sample.csv in APT\n"
                f"(aperture {br['r_aper_pix']:.2f} px, sky {br['r_sky_in_pix']:.1f}-{br['r_sky_out_pix']:.1f} px)\n"
                "and type APT's 'Source Intensity' (sky-subtracted, ADU)\ninto the column apt_source_intensity_r,"
                "\nthen rerun with --use-cache.", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(d) APT vs. Python (pending your APT measurements)", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("apt_vs_python_photometry.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved apt_vs_python_photometry.png")
    return s


def export_ds9_regions(cal, members, circle, counts):
    with open("ds9_cluster_members.reg", "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"# green = {len(members)} Gaia DR3 members, red = random field sample, cyan = manual circle, "
                "white dashed = background ring\n")
        f.write('global dashlist=8 3 width=1 font="helvetica 9 normal roman" select=1 highlite=1 dash=0 fixed=0 '
                'edit=1 move=1 delete=1 include=1 source=1\nfk5\n')
        c = circle
        f.write(f'circle({c["ra_deg"]:.6f},{c["dec_deg"]:.6f},{c["radius_arcmin"]:.3f}\') # color=cyan width=2 '
                f'text={{manual circle}}\n')
        for rr in counts["ring_arcmin"]:
            f.write(f'circle({c["ra_deg"]:.6f},{c["dec_deg"]:.6f},{rr:.3f}\') # color=white dash=1 background\n')
        for _, r in members.iterrows():
            f.write(f'circle({r["ra_r"]:.6f},{r["dec_r"]:.6f},3.5") # color=green text={{M{int(r["id_r"])}}}\n')
        field = cal[~cal["is_member"]]
        for _, r in field.sample(min(4000, len(field)), random_state=42).iterrows():
            f.write(f'circle({r["ra_r"]:.6f},{r["dec_r"]:.6f},2.5") # color=red\n')
    print("Exported ds9_cluster_members.reg")


# ==================================================================================================
# STAGE 4b: REPORT (numbers from pipeline_results.json, prose from report_text.md)
# ==================================================================================================

REPORT_TEMPLATE = r"""<!--
report_text.md  -  YOUR words for NGC6819_Final_Report.pdf.
The pipeline fills in every number, table, figure and code listing automatically; this file holds the
explanations, which the course rules require to be your own.

Rules
  * Sections start with a line "## key" (keep the keys).  Any section that still starts with TODO is shown
    in the PDF as a grey "to be written" box, so you can see what is missing.
  * Write numbers as placeholders so they always match the latest run, e.g. {{mu0}} or {{age_gyr_total}}.
    All available placeholders and their current values are listed in report_placeholders.txt.
  * Formatting: blank line = new paragraph, **bold**, *italic*, `code`, lines starting "- " are bullets,
    "### Heading" is a sub-heading, and a line  ![caption](screenshots/file.png)  inserts a screenshot.
  * HTML comments like this one are ignored.
Rebuild the PDF in a few seconds with:   python ngc6819_master_pipeline.py --report-only
-->

## original_goals
<!-- Copied verbatim from observation_report.pdf (Sections 1 and 5). Edit only if you want a different excerpt. -->
Every brightness measurement here comes from GIT images and nothing else. The external catalogues serve two narrow purposes: Pan-STARRS supplies stars already present in my frame whose magnitudes are known, which is how detector counts become real magnitudes, and Gaia supplies proper motions and parallaxes, which say which stars in my frame belong to the cluster rather than lying in front of or behind it. Neither contributes a point to the diagram itself. A cluster diagram measured in SDSS filters from a single telescope is what does not currently exist for NGC 6819.

Reddening, distance modulus, and age will be extracted from a single simultaneous isochrone fit, since the three are correlated. Unresolved double stars, which sit up to 0.75 mag above the main sequence, will be excluded from this fit. To verify the accuracy of my GIT photometry and fitting pipeline, every extracted parameter will be checked against independent literature benchmarks derived from entirely different physical techniques (e.g., asteroseismology, eclipsing binaries, and geometry).

**Success criterion:** My distance modulus should fall within ~0.15 mag of the eclipsing binary value, and my age within ~0.3 Gyr. Agreement at that level will thoroughly validate the full chain of photometry, calibration, membership selection, and isochrone fitting.

## data_analysis_overview
TODO: In your own words: how did you turn the two FITS images into magnitudes and positions? What did you do by hand (DS9: drawing the cluster circle, inspecting stars; APT: spot checks) and what was done in code? Why PSF photometry in this crowded field, and why the aperture fallback for bright stars? Refer to Figure {{fig_phase1}} (phase1) and your screenshots.

## zero_points_and_uncertainties
TODO: Explain how the zero points were measured (PS1 stars, the Tonry transformation, why a colour term, why Siegel's robust fit) and how you got the uncertainties: per-star photometric errors (Poisson + sky), the zero-point bootstrap ({{zp_r}} +/- {{zp_r_err}} in r'), the {{cal_sys_floor}} mag systematic floor you adopted, and the {{cal_rms_r}} mag star-to-star scatter. Say what you think limits the scatter.

## ds9_apt_checks
TODO: Describe your manual SAOImageDS9 work: how you chose the cluster circle (centre {{circle_centre}}, r = {{circle_radius}}'), what you saw, and the APT cross-check. Insert screenshots, e.g.
![DS9 with my cluster circle and the member regions](screenshots/ds9_circle.png)

## ai_declaration
TODO: Declare every AI tool you used and what it did for you, and show you understand it. For example (edit to match what actually happened): "I used Claude Code (Anthropic) to restructure ngc6819_master_pipeline.py: it wrote the DS9 region parser, read the pointing centre and field of view from the FITS header and WCS, added the bootstrap error estimates and the report generator. I checked ... and I understand that ...". Also list any other assistants used for earlier versions of the code.

## fit_method
TODO: In your own words: how do the numbers become age, distance, reddening and turnoff mass? Explain the loss function (four CMD features with equal weights, each compared only with its own isochrone phase: main sequence, subgiant branch, core-He-burning red clump), why mu0 and E(B-V) are degenerate (Figure {{fig_fit}}, panel b), how the bootstrap gives the asymmetric statistical errors and how the shifted zero points give the systematic errors.

## membership_comparison
TODO: Compare the Gaia-free CMD with the Gaia CMDs (Figure {{fig_compare}}, Table {{tab_compare}}) in your own words. Which CMD features survive without Gaia? How different are the age ({{age_gyr_free}} vs {{age_gyr_gcircle}} Gyr inside the same circle) and (m-M)_V ({{mMV_free}} vs {{mMV_gcircle}})? What does that say about how much your result depends on the membership method, and what are the weaknesses of each method?

## what_worked_failed
TODO: What worked, what failed or surprised you (for example the spatial over-density count {{excess_gaia}} vs {{members_inside}} Gaia members inside the circle, the reddening test, the Teff offset of {{teff_median}} K).

## literature_comparison
TODO: Compare with the literature using the benchmark table. Do the numbers agree? Why or why not? Discuss in particular the age ({{age_gyr}} Gyr vs 2.2-2.5 Gyr) and E(B-V) ({{ebv}} vs 0.10-0.16), and the mu0-E(B-V) degeneracy.

## improvements
TODO: What could you have done differently? What other data would have helped (i' band / short exposures from the proposal, a metallicity grid to propagate the +/-0.03 dex [Fe/H] uncertainty, spectroscopic binaries, ...)?

## conclusion
TODO: A short conclusion in your own words.
"""

REPORT_ORDER = [("original_goals", "1. Original goals"),
                ("data_analysis_overview", None), ("zero_points_and_uncertainties", None),
                ("ds9_apt_checks", None), ("ai_declaration", None),
                ("fit_method", None), ("what_worked_failed", None), ("literature_comparison", None),
                ("improvements", None), ("conclusion", None)]

REFERENCES = [   # every entry checked on NASA ADS (bibcode given); Efron and Siegel are not in ADS -> DOI (Crossref)
    "Anderson J., King I. R., 2000, PASP, 112, 1360 (empirical PSF) [2000PASP..112.1360A]",
    "Andrae R. et al., 2023, A&A, 674, A27 (Gaia DR3 GSP-Phot) [2023A&A...674A..27A]",
    "Anthony-Twarog B. J., Deliyannis C. P., Twarog B. A., 2014, AJ, 148, 51 (E(B-V) = 0.160) [2014AJ....148...51A]",
    "Astropy Collaboration, 2022, ApJ, 935, 167 [2022ApJ...935..167A]",
    "Balona L. A. et al., 2013, MNRAS, 430, 3472 [2013MNRAS.430.3472B]",
    "Basu S. et al., 2011, ApJ, 729, L10 [2011ApJ...729L..10B]",
    "Bonatto C., Bica E., 2007, MNRAS, 377, 1301 (field-star decontamination) [2007MNRAS.377.1301B]",
    "Bradley L. et al., photutils, Zenodo, doi:10.5281/zenodo.596036",
    "Bragaglia A. et al., 2001, AJ, 121, 327 [2001AJ....121..327B]",
    "Bressan A. et al., 2012, MNRAS, 427, 127 (PARSEC) [2012MNRAS.427..127B]",
    "Brewer L. N. et al., 2016, AJ, 151, 66 [2016AJ....151...66B]",
    "Cantat-Gaudin T. et al., 2018, A&A, 618, A93 [2018A&A...618A..93C]",
    "Cardelli J. A., Clayton G. C., Mathis J. S., 1989, ApJ, 345, 245 (extinction law) [1989ApJ...345..245C]",
    "Chambers K. C. et al., 2016, arXiv:1612.05560 (Pan-STARRS1) [2016arXiv161205560C]",
    "Efron B., 1979, Ann. Stat., 7, 1 (bootstrap), doi:10.1214/aos/1176344552",
    "Fabricius C. et al., 2021, A&A, 649, A5 (Gaia EDR3 catalogue validation; RUWE < 1.4) [2021A&A...649A...5F]",
    "Gaia Collaboration, Vallenari A. et al., 2023, A&A, 674, A1 (Gaia DR3) [2023A&A...674A...1G]",
    "Kalirai J. S. et al., 2001, AJ, 122, 266 [2001AJ....122..266K]",
    "Kumar H. et al., 2022, AJ, 164, 90 (GROWTH-India Telescope) [2022AJ....164...90K]",
    "Lindegren L. et al., 2021, A&A, 649, A4 (parallax zero point) [2021A&A...649A...4L]",
    "Magnier E. A. et al., 2013, ApJS, 205, 20 (PS1 photometric reference ladder) [2013ApJS..205...20M]",
    "Milone A. P. et al., 2012, A&A, 540, A16 (differential-reddening correction) [2012A&A...540A..16M]",
    "O'Donnell J. E., 1994, ApJ, 422, 158 (extinction law) [1994ApJ...422..158O]",
    "Platais I. et al., 2013, AJ, 146, 43 [2013AJ....146...43P]",
    "Rodrigo C., Solano E., Bayo A., 2012, IVOA Working Draft (SVO Filter Profile Service) [2012ivoa.rept.1015R]",
    "Rodrigo C., Solano E., 2020, XIV.0 Scientific Meeting of the SEA, 182 (SVO Filter Profile Service) [2020sea..confE.182R]",
    "Schlafly E. F., Finkbeiner D. P., 2011, ApJ, 737, 103 [2011ApJ...737..103S]",
    "Siegel A. F., 1982, Biometrika, 69, 242 (repeated-median regression), doi:10.1093/biomet/69.1.242",
    "Stetson P. B., 1987, PASP, 99, 191 (DAOPHOT / DAOFIND) [1987PASP...99..191S]",
    "Stetson P. B., 1992, ASP Conf. Ser., 25, 297 (DAOPHOT II) [1992ASPC...25..297S]",
    "Stetson P. B., User's Manual for DAOPHOT II (default 0.75% flat-field 'percent error'; not indexed in ADS)",
    "Tonry J. L. et al., 2012, ApJ, 750, 99 (PS1 photometric system) [2012ApJ...750...99T]",
    "Vasiliev E., Baumgardt H., 2021, MNRAS, 505, 5978 (Gaia EDR3 parallax systematics floor) [2021MNRAS.505.5978V]",
]


def placeholder_values(R):
    p = R["stage3"]["fit"]["params"]
    c = R["stage2"]["calibration"]
    cnt = R["stage2"]["spatial_counts"]
    gm = [r for r in cnt["rows"] if r["sample"] == "Gaia-matched stars"][0]
    v = {
        "age_gyr": f"{p['age_gyr']['value']:.2f}", "age_gyr_stat": f"{p['age_gyr']['stat']:.2f}",
        "age_gyr_sys": f"{p['age_gyr']['sys']:.2f}", "age_gyr_total": f"{p['age_gyr']['total']:.2f}",
        "logage": f"{p['logAge']['value']:.2f}",
        "excess_gaia": f"{gm['excess']:.0f} +/- {gm['excess_err']:.0f}",
        "members_inside": str(cnt["members_inside"]), "n_members": str(R["stage2"]["membership"]["n_members"]),
        "circle_centre": R["circle"]["centre_hms"], "circle_radius": f"{R['circle']['radius_arcmin']:.2f}",
        "zp_r": f"{c['r']['zp']:.4f}", "zp_r_err": f"{c['r']['zp_err']:.4f}", "zp_g": f"{c['g']['zp']:.4f}",
        "zp_g_err": f"{c['g']['zp_err']:.4f}", "ct_r": f"{c['r']['colour_term']:+.4f}", "ct_g": f"{c['g']['colour_term']:+.4f}",
        "cal_rms_r": f"{c['r']['rms']:.3f}", "cal_rms_g": f"{c['g']['rms']:.3f}", "n_ref": str(c["n_ref"]),
        "cal_sys_floor": f"{CAL_SYSTEMATIC_FLOOR:.2f}",
        "teff_median": f"{R['stage3']['teff']['median']:+.0f}", "teff_sigma": f"{R['stage3']['teff']['robust_sigma']:.0f}",
        "fwhm_r": f"{R['stage1']['bands']['r']['fwhm_arcsec']:.2f}", "fwhm_g": f"{R['stage1']['bands']['g']['fwhm_arcsec']:.2f}",
    }
    for k in ("age_gyr", "mu0", "ebv", "mMV", "d_kpc", "mto"):
        prec = 2 if k in ("mto", "age_gyr") else 3
        for part in ("value", "stat", "sys", "total"):
            v[k if part == "value" else f"{k}_{part}"] = f"{p[k][part]:.{prec}f}"
        v[f"{k}_lo"], v[f"{k}_hi"] = (f"{x:.{prec}f}" for x in err_lo_hi(p[k]))
    mem = R["stage2"]["membership"]
    v.update(plx_mean=f"{mem['plx_wmean']:.4f}", d_plx=f"{mem['d_plx_raw_kpc']:.3f}",
             d_plx_err=f"{mem['d_plx_raw_kpc'] * np.hypot(mem['plx_wmean_err'], GAIA_PLX_SYS) / mem['plx_wmean']:.3f}",
             d_plx_quasar=f"{mem['d_plx_zpcorr_kpc']:.3f}", n_plx=str(mem.get("n_plx", "")),
             n_by_pmemb=f"{mem['n_by_pmemb']:,}", n_by_kinematics=f"{mem['n_by_kinematics']:,}",
             pmra0=f"{mem['pmra0']:+.3f}", pmdec0=f"{mem['pmdec0']:+.3f}")
    s3 = R["stage3"]
    v.update(n_single=f"{s3['pops']['n_single']:,}", n_binary=str(s3['pops']['n_binary']), n_bss=str(s3['pops']['n_bss']),
             n_rc=str(s3['pops']['n_rc_region']), sgb_n=str(s3['fit']['feature_counts']['SGB']),
             rc_n=str(s3['fit']['feature_counts']['RC']), teff_dwarfs=f"{s3['teff']['median_dwarfs']:+.0f}",
             teff_giants=f"{s3['teff']['median_giants']:+.0f}", n_teff=str(s3['teff']['n']),
             redd_max=f"{s3['reddening']['max_zone_offset_mag']:.3f}", redd_cal_max=f"{s3['reddening']['cal_max_offset']:.3f}",
             redd_sig_raw=f"{s3['reddening']['sigma_raw']:.3f}", redd_sig_corr=f"{s3['reddening']['sigma_corrected']:.3f}")
    b1 = R["stage1"]["bands"]["r"]
    v.update(n_matched=f"{R['stage1']['n_matched']:,}", n_fallback=f"{b1['n_fallback']:,}",
             ap_offset=f"{b1['aperture_offset']:+.3f}", sky_r=f"{b1['sky_median']:.0f}", n_psf_stars=str(b1['n_psf_stars']),
             airmass_r=f"{b1['airmass']:.2f}", excess_r18=f"{[r for r in cnt['rows'] if 'r' in r['sample'] and '18' in r['sample']][0]['excess']:.0f}")
    F = R.get("comparison", {}).get("fits", {})
    for label, suffix in (("gaia_circle", "gcircle"), ("gaia_free", "free")):
        if label in F:
            q = F[label]["params"]
            for k, prec in (("age_gyr", 2), ("mu0", 3), ("ebv", 3), ("mMV", 3), ("d_kpc", 3), ("mto", 2)):
                v[f"{k}_{suffix}"] = f"{q[k]['value']:.{prec}f}"
                v[f"{k}_{suffix}_total"] = f"{q[k]['total']:.{prec}f}"
                v[f"{k}_{suffix}_lo"], v[f"{k}_{suffix}_hi"] = (f"{x:.{prec}f}" for x in err_lo_hi(q[k]))
            v[f"n_{suffix}"] = f"{F[label]['n_stars']:,}"
    d = R.get("comparison", {}).get("decontamination")
    if d:
        v["n_stat_members"] = f"{d['n_stat_members']:,}"
        v["frac_stat_that_are_gaia"] = f"{d['frac_stat_that_are_gaia']:.0%}"
        v["frac_gaia_recovered"] = f"{d['frac_gaia_recovered']:.0%}"
    return v


def read_report_text(path=REPORT_TEXT_FILE):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(REPORT_TEMPLATE)
        print(f"Created {path} (write your own explanations there, then run --report-only)")
    with open(path) as f:
        txt = re.sub(r"<!--.*?-->", "", f.read(), flags=re.S)
    sections, key = {}, None
    for line in txt.splitlines():
        m = re.match(r"^##\s+(\w+)\s*$", line)
        if m:
            key = m.group(1)
            sections[key] = []
        elif key:
            sections[key].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def build_pdf(R):
    """ReportLab PDF in the order required by the course: original goals, data analysis, results and
    interpretation, references."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table as RLTable, TableStyle,
                                    Image as RLImage, Preformatted, KeepTogether, PageBreak)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fdir = "/usr/share/fonts/truetype/dejavu"
    if os.path.exists(os.path.join(fdir, "DejaVuSans.ttf")):
        for n, fn in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                      ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
            pdfmetrics.registerFont(TTFont(n, os.path.join(fdir, fn)))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
        base, bold, ital, mono = "DV", "DV-B", "DV-I", "DVM"
    else:
        base, bold, ital, mono = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"

    ss = getSampleStyleSheet()
    S = dict(
        title=ParagraphStyle("t", parent=ss["Normal"], fontName=bold, fontSize=16, leading=20, alignment=1, spaceAfter=6),
        auth=ParagraphStyle("a", parent=ss["Normal"], fontName=base, fontSize=10, alignment=1, spaceAfter=12),
        h1=ParagraphStyle("h1", parent=ss["Normal"], fontName=bold, fontSize=13, leading=17, spaceBefore=12, spaceAfter=6,
                          textColor=colors.HexColor("#0B2545"), keepWithNext=True),
        h2=ParagraphStyle("h2", parent=ss["Normal"], fontName=bold, fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4,
                          textColor=colors.HexColor("#13315C"), keepWithNext=True),
        body=ParagraphStyle("b", parent=ss["Normal"], fontName=base, fontSize=9.3, leading=13, spaceAfter=6),
        h3=ParagraphStyle("h3", parent=ss["Normal"], fontName=bold, fontSize=9.6, leading=13, spaceBefore=6, spaceAfter=3),
        todo=ParagraphStyle("td", parent=ss["Normal"], fontName=ital, fontSize=8.8, leading=12, textColor=colors.HexColor("#666666"),
                            backColor=colors.HexColor("#F1F1F1"), borderPadding=5, spaceBefore=4, spaceAfter=10),
        cap=ParagraphStyle("c", parent=ss["Normal"], fontName=base, fontSize=8.2, leading=11, textColor=colors.HexColor("#333333"),
                           spaceAfter=10),
        cell=ParagraphStyle("ce", parent=ss["Normal"], fontName=base, fontSize=7.8, leading=10),
        head=ParagraphStyle("he", parent=ss["Normal"], fontName=bold, fontSize=7.8, leading=10, textColor=colors.white),
        code=ParagraphStyle("co", parent=ss["Normal"], fontName=mono, fontSize=6.4, leading=7.8, backColor=colors.HexColor("#F7F7F7")),
        ref=ParagraphStyle("r", parent=ss["Normal"], fontName=base, fontSize=8.3, leading=11, leftIndent=12, firstLineIndent=-12),
    )
    W = A4[0] - 2 * 50
    st = []
    pv = placeholder_values(R)
    # figure/table numbers in document order (the artificial-star and comparison items are optional);
    # also exposed as placeholders {{fig_<key>}} / {{tab_<key>}} for report_text.md
    has_ast, has_cmp = os.path.exists(AST_FILE_JSON), bool(R.get("comparison", {}).get("fits"))
    FIG = {k: i + 1 for i, k in enumerate(["phase1", "phase2"] + ["ast"] * has_ast + ["fit"] + ["compare"] * has_cmp)}
    TAB = {k: i + 1 for i, k in enumerate(["plan"] + ["results", "counts"] + ["compare"] * has_cmp + ["bench"])}
    pv.update({f"fig_{k}": str(v) for k, v in FIG.items()})
    pv.update({f"tab_{k}": str(v) for k, v in TAB.items()})
    texts = read_report_text()
    missing = []

    def esc(s):
        return html.escape(str(s), quote=False)

    def inline(s):
        s = esc(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", s)
        s = re.sub(r"`(.+?)`", rf'<font face="{mono}">\1</font>', s)
        s = re.sub(r"_\{(.+?)\}", r"<sub>\1</sub>", s)
        s = re.sub(r"\^\{(.+?)\}", r"<super>\1</super>", s)
        return s

    def image(path, caption=None, width=W):
        if not os.path.exists(path):
            st.append(Paragraph(f"[missing image: {esc(path)}]", S["todo"]))
            return
        iw, ih = ImageReader(path).getSize()
        h = width * ih / iw
        if h > 640:
            width, h = width * 640 / h, 640
        items = [RLImage(path, width=width, height=h)]
        if caption:
            items.append(Paragraph(caption, S["cap"]))
        st.append(KeepTogether(items))

    def student(key, title=None):
        txt = re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: pv.get(m.group(1), "{{" + m.group(1) + "?}}"), texts.get(key, ""))
        if title:
            st.append(Paragraph(title, S["h2"]))
        if not txt or txt.lstrip().upper().startswith("TODO"):
            missing.append(key)
            prompt = txt.lstrip()[4:].lstrip(": ").strip() if txt else "section missing from report_text.md"
            st.append(Paragraph(f"[To be written by the student in report_text.md, section '{key}'] " + inline(prompt), S["todo"]))
            return
        for block in re.split(r"\n\s*\n", txt):
            lines = [l for l in block.splitlines() if l.strip()]
            if not lines:
                continue
            m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", lines[0].strip())
            imgs = [re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", l.strip()) for l in lines]
            if len(lines) > 1 and all(imgs):           # consecutive image lines: side by side
                gap, n = 10, len(imgs)
                cw = (W - gap * (n - 1)) / n
                cells = []
                for mm in imgs:
                    if not os.path.exists(mm.group(2)):
                        cells.append([Paragraph(f"[missing image: {esc(mm.group(2))}]", S["todo"])])
                        continue
                    iw, ih = ImageReader(mm.group(2)).getSize()
                    cell = [RLImage(mm.group(2), width=cw, height=cw * ih / iw)]
                    if mm.group(1):
                        cell.append(Paragraph(inline(mm.group(1)), S["cap"]))
                    cells.append(cell)
                row = RLTable([cells], colWidths=[cw + (gap if i < n - 1 else 0) for i in range(n)])
                row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                         ("RIGHTPADDING", (0, 0), (-2, -1), gap), ("RIGHTPADDING", (-1, 0), (-1, -1), 0)]))
                st.append(row)
            elif m and len(lines) == 1:
                image(m.group(2), inline(m.group(1)) if m.group(1) else None, W * 0.5)
            elif lines[0].startswith("### "):
                st.append(Paragraph(inline(lines[0][4:]), S["h3"]))
                rest = lines[1:]
                if rest and all(re.match(r"^\s*[-*] ", l) for l in rest):
                    for l in rest:
                        st.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", l)), S["body"], bulletText="•"))
                elif rest:
                    st.append(Paragraph(inline(" ".join(rest)), S["body"]))
            elif all(re.match(r"^\s*[-*] ", l) for l in lines):
                for l in lines:
                    st.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", l)), S["body"], bulletText="•"))
            else:
                st.append(Paragraph(inline(" ".join(lines)), S["body"]))

    def table(rows, widths, header=True):
        data = [[Paragraph(str(c), S["head"] if (header and i == 0) else S["cell"]) for c in r] for i, r in enumerate(rows)]
        t = RLTable(data, colWidths=widths, repeatRows=1 if header else 0)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2545") if header else colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        st.append(t)
        st.append(Spacer(1, 8))

    s1, s2, s3, g, circ = R["stage1"], R["stage2"], R["stage3"], R["geometry"], R["circle"]
    br, bgn = s1["bands"]["r"], s1["bands"]["g"]
    cal, mem, cnt = s2["calibration"], s2["membership"], s2["spatial_counts"]
    p, teff, redd, pops = s3["fit"]["params"], s3["teff"], s3["reddening"], s3["pops"]

    st.append(Paragraph("A Calibrated SDSS Colour-Magnitude Diagram of NGC 6819 (The Foxhead) from the "
                        "0.7 m GROWTH-India Telescope", S["title"]))
    st.append(Paragraph(f"Veeresh S K (4th year B.Tech, Electrical Engineering) &middot; Observational astrophysics project "
                        f"&middot; pipeline run {esc(R['run']['timestamp'])} ({esc(R['run']['mode'])})", S["auth"]))

    # ---------------- 1. Original goals ----------------
    st.append(Paragraph("1. Original goals", S["h1"]))
    student("original_goals")
    table([["Target quantity", "Measurement method (GIT data)", "Verification benchmark (literature)"],
           ["Distance modulus (μ<sub>0</sub>)", "Vertical offset of the fitted isochrone",
            "(m-M)<sub>V</sub> = 12.38 &plusmn; 0.04 (eclipsing binaries); 11.85 to 12.20 (asteroseismology)"],
           ["Distance (d)", "d = 10<super>(mu0/5+1)</super> pc", "2.3 to 2.8 kpc (method dependent)"],
           ["Age", "Shape of the turnoff &amp; subgiant branch", "2.2 to 2.5 Gyr (binaries &amp; alternative fits)"],
           ["Reddening E(B-V)", "Horizontal offset of the fitted isochrone", "0.10 to 0.16 (independent BV photometry)"],
           ["Turnoff mass", "Read directly from best-fit isochrone", "1.25 &plusmn; 0.06 M<sub>sun</sub> (dynamical orbit)"],
           ["Effective temps", "From dereddened g'-r' via isochrone", "Gaia DR3 spectra (star-by-star comparison)"]],
          [95, 160, W - 255])
    st.append(Paragraph(f"Table {TAB['plan']}: Measurement plan and benchmarks (proposal Section 5, verbatim). Fixed inputs "
                        f"(Section 4): [Fe/H] = {FEH:+.2f} (Z = {Z_INI}), PARSEC v1.2S, R<sub>V</sub> = {R_V}; the proposal's "
                        f"A<sub>r'</sub> = 2.27 E(B-V) was replaced by {A_R_PER_EBV} E(B-V) (Section 3.1).", S["cap"]))

    # ---------------- 2. Data analysis ----------------
    st.append(Paragraph("2. Data analysis: from images to magnitudes and positions", S["h1"]))
    st.append(Paragraph("2.1 Data", S["h2"]))
    table([["Frame", "Band", "t<sub>exp</sub> [s]", "UTC start", "Airmass", "Gain [e-/ADU]", "Saturation [ADU]"],
           *[[FRAMES[b], s1["bands"][b]["filter"] + "'", f"{s1['bands'][b]['exptime']:.0f}",
              s1["bands"][b]["date_obs"].replace("T", " "),
              f"{s1['bands'][b]['airmass']:.3f}", f"{s1['bands'][b]['gain']:.2f}", f"{s1['bands'][b]['saturate']:.0f}"]
             for b in ("r", "g")]], [138, 38, 42, 108, 52, 58, W - 436])
    st.append(Paragraph(
        f"GIT-processed frames (bias, flat, TPV WCS); {g['nx']} x {g['ny']} px at {g['pix_scale']:.3f}\"/px = "
        f"{g['fov_x_arcmin']:.1f}' x {g['fov_y_arcmin']:.1f}'; pointing centre {esc(g['centre_hms'])}.", S["body"]))
    st.append(Paragraph("2.2 Method", S["h2"]))
    student("data_analysis_overview")
    st.append(Paragraph("2.3 Measured quantities", S["h2"]))
    rows = [["Quantity", "r'", "g'"]]
    def both(fmt, key):
        return [fmt.format(s1["bands"][b][key]) if s1["bands"][b].get(key) is not None else "n/a" for b in ("r", "g")]
    rows += [["Sky median / rms [ADU]", *[f"{s1['bands'][b]['sky_median']:.1f} / {s1['bands'][b]['sky_rms']:.2f}" for b in "rg"]],
             ["Seeing FWHM (2-D Gaussian, isolated stars)", *[f"{s1['bands'][b]['fwhm_pix']:.2f} px = {s1['bands'][b]['fwhm_arcsec']:.2f}\" "
                                                             f"(N={s1['bands'][b]['n_fwhm_stars']})" for b in "rg"]],
             ["ePSF stars (2x oversampled)", *both("{}", "n_psf_stars")],
             ["Sources detected (> 4 sigma)", *both("{:,}", "n_detected")],
             ["Science aperture / sky annulus [px]", *[f"{s1['bands'][b]['r_aper_pix']:.2f} / {s1['bands'][b]['r_sky_in_pix']:.1f}-"
                                                      f"{s1['bands'][b]['r_sky_out_pix']:.1f}" for b in "rg"]],
             ["m<sub>PSF</sub> - m<sub>ap</sub> offset [mag]", *[f"{s1['bands'][b]['aperture_offset']:+.4f} &plusmn; "
                                                                 f"{s1['bands'][b]['aperture_offset_scatter']:.4f} (N={s1['bands'][b]['n_offset_stars']})" for b in "rg"]],
             ["Stars using aperture fallback", *both("{:,}", "n_fallback")],
             ["Saturated (raw peak &ge; SATURATE)", *both("{:,}", "n_saturated")],
             ["Zero point ZP [mag]", *[f"{cal[b]['zp']:.4f} &plusmn; {cal[b]['zp_err']:.4f}" for b in "rg"]],
             ["Colour term C", *[f"{cal[b]['colour_term']:+.4f} &plusmn; {cal[b]['colour_term_err']:.4f}" for b in "rg"]],
             ["Calibration robust RMS per star [mag]", *[f"{cal[b]['rms']:.3f}" for b in "rg"]],
             ["ZP systematic used in the fit [mag]", *[f"{cal[b]['zp_sys']:.3f}" for b in "rg"]]]
    table(rows, [190, (W - 190) / 2, (W - 190) / 2])
    st.append(Paragraph(
        f"{s1['n_matched']:,} stars matched in r' and g' ({s1['n_snr10']:,} with S/N &gt; 10); calibration with {cal['n_ref']} PS1 stars. "
        f"Membership: {mem['n_members']:,} Gaia DR3 members = Cantat-Gaudin et al. (2018) P<sub>memb</sub> &ge; {PMEMB_MIN} "
        f"({mem['n_by_pmemb']:,}) or RUWE &lt; {RUWE_MAX} (Fabricius et al. 2021) + proper motion within {PM_NSIGMA * PM_SIGMA:.2f} mas/yr "
        f"of ({mem['pmra0']:+.2f}, {mem['pmdec0']:+.2f}) + {PLX_RANGE[0]}-{PLX_RANGE[1]} mas parallax ({mem['n_by_kinematics']:,}).",
        S["body"]))
    image("phase1_diagnostics.png", f"<b>Figure {FIG['phase1']}.</b> Stage 1 diagnostics: (a) Background2D sky model, (b) r' ePSF with its "
          "radial profile, (c) per-star uncertainties against the expected star + sky noise, (d) the PSF - aperture offset used "
          "for stars brighter than 30,000 ADU or with a bad PSF fit.", W * 0.8)
    st.append(Paragraph(
        "<b>How panels (c) and (d) are obtained.</b> (c) Each star's reported magnitude error is plotted against its instrumental "
        "magnitude; the dashed curves are the error expected from photon noise alone, "
        "σ<sub>m</sub> = 1.0857 √(F/g + N<sub>pix</sub> σ<sub>sky</sub><super>2</super>) / F for a 1.5 FWHM aperture "
        "(F = counts, g = gain, σ<sub>sky</sub> = measured sky noise), so points above the curve show the extra error from the "
        "PSF fit and the flat-field term. (d) For well-fitted, isolated stars (no neighbour within 18 px, PSF error &lt; 0.04 mag, "
        "peak 2,000-30,000 ADU) the PSF and 1.5 FWHM aperture magnitudes are differenced; the median of that difference "
        f"({s1['bands']['r']['aperture_offset']:+.3f} mag in r', {s1['bands']['g']['aperture_offset']:+.3f} in g') is added to the "
        "aperture magnitudes of the bright stars so they are on the PSF scale.", S["body"]))
    image("phase2_calibration_and_membership.png", f"<b>Figure {FIG['phase2']}.</b> (a, b) Zero-point and colour-term fits to Pan-STARRS stars "
          "(points are data, line is the Siegel fit), (c) r' residuals across the detector, (d) proper-motion diagram, "
          "(e) parallaxes, (f) all Gaia-matched stars vs. members.", W * 0.8)
    st.append(Paragraph(
        "<b>How the panels are obtained.</b> (a, b) For each reference star, the PS1 magnitude converted to SDSS minus the GIT "
        "instrumental magnitude, against instrumental colour; the line is the fitted zero point and colour term. (c) Each reference "
        "star's residual from that fit, plotted at its position on the chip. (d-f) Gaia DR3 proper motions, parallaxes and the "
        "calibrated CMD of all matched stars, with the members in green.", S["body"]))
    st.append(Paragraph("2.4 Zero points and uncertainties", S["h2"]))
    student("zero_points_and_uncertainties")
    s2 = 5
    if os.path.exists(AST_FILE_JSON):
        with open(AST_FILE_JSON) as f:
            ast = json.load(f)
        st.append(Paragraph(f"2.{s2} Artificial-star test: completeness and photometric bias", S["h2"]))
        s2 += 1
        Z = list(ast["zones"])
        def zrow(z, lo):
            return next(r for r in ast["zones"][z] if abs(r["r_lo"] - lo) < 1e-6)
        st.append(Paragraph(
            "Real stars cannot tell how many stars were missed or whether the measured magnitudes are biased, because their true "
            "brightness is unknown. The standard check in crowded-field photometry is therefore to add artificial stars of known "
            f"brightness to the images and measure them like real ones (DAOPHOT's ADDSTAR does the same). {ast['n_injected']:,} copies "
            f"of each frame's ePSF with Poisson noise, at random positions and r' = {AST_MAG_RANGE[0]:.0f}-{AST_MAG_RANGE[1]:.0f}, "
            f"were added in {ast['n_pass']} passes (at least {AST_MIN_SEP_PIX:.0f} px apart, so the crowding is not changed), the "
            f"unchanged photometry was rerun, and a star counted as recovered if found in both bands within {AST_MATCH_PIX} px and "
            f"{AST_MATCH_MAG} mag of its input. Results (Figure {FIG['ast']}): at r' = 17-18 the completeness is "
            + ", ".join(f"{zrow(z, 17)['completeness']:.0%} ({esc(z.replace('ring (field)', 'field ring'))})" for z in Z)
            + "; the PSF magnitudes are unbiased to 0.02 mag down to r' = 18.5, while aperture magnitudes are too faint in the core; "
            "and the real scatter is about twice the error the pipeline reports (3-4 times in the core). "
            "<b>Conclusion:</b> the PSF r' magnitudes are accurate, and the g'-r' colours are good to 0.01 mag brighter than "
            "r' = 17, so the turnoff, subgiant and red-clump positions that set the age and distance can be trusted; PSF photometry "
            "was the right choice for this field. Fainter than r' = 17 the colours come out 0.02-0.05 mag too red (most in the core), "
            "which can slightly affect the faint end of the main-sequence feature. Crowding mainly costs completeness: faint core "
            "stars are lost, which is why the star-count excess is smaller than the number of Gaia members in the circle. The "
            "per-star error bars are lower limits; they do not enter the isochrone fit, whose errors come from the bootstrap and "
            "the zero-point shifts.", S["body"]))
        image(AST_FIG, f"<b>Figure {FIG['ast']}.</b> Artificial-star test: (a) completeness, (b) bias and scatter of the final r' magnitudes, "
              "(c) PSF-only vs. aperture-only magnitudes, (d) measured scatter vs. the error the pipeline reports.", W * 0.8)
    st.append(Paragraph(f"2.{s2} Manual checks with SAOImageDS9", S["h2"]))
    student("ds9_apt_checks")
    shots = sorted(f for f in (os.listdir(SCREENSHOT_DIR) if os.path.isdir(SCREENSHOT_DIR) else [])
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    used = texts.get("ds9_apt_checks", "") + " ".join(texts.values())
    for f in shots:
        if f not in used:
            image(os.path.join(SCREENSHOT_DIR, f), f"Screenshot: {esc(f)}", W * 0.85)
    st.append(Paragraph(f"2.{s2 + 1} Use of AI tools", S["h2"]))
    student("ai_declaration")

    # ---------------- 3. Results and interpretation ----------------
    st.append(Paragraph("3. Results and interpretation", S["h1"]))
    st.append(Paragraph("3.1 How the numbers become physical parameters", S["h2"]))
    student("fit_method")
    st.append(Paragraph("3.2 Results", S["h2"]))
    rows = [["Parameter", "Value", "Statistical", "Calibration sys.", "Total"]]
    for k, lab, prec in (("age_gyr", "Age [Gyr]", 2), ("logAge", "log(age/yr)", 3), ("mu0", "μ<sub>0</sub> [mag]", 3),
                         ("ebv", "E(B-V) [mag]", 3), ("mMV", "(m-M)<sub>V</sub> = μ<sub>0</sub> + 3.1 E(B-V) [mag]", 3),
                         ("d_kpc", "Distance d [kpc]", 3), ("mto", "Turnoff mass M<sub>TO</sub> [M<sub>sun</sub>]", 3)):
        q = p[k]
        lo, hi = err_lo_hi(q)
        st_lo, st_hi = q.get("stat_lo", q["stat"]), q.get("stat_hi", q["stat"])
        rows.append([lab, f"{q['value']:.{prec}f}", f"+{st_hi:.{prec}f} / -{st_lo:.{prec}f}", f"&plusmn;{q['sys']:.{prec}f}",
                     f"<b>+{hi:.{prec}f} / -{lo:.{prec}f}</b>"])
    rows.append(["Teff(GIT) - Teff(Gaia GSP-Phot), median [K]", f"{teff['median']:+.0f}", f"{teff['median_err']:.0f}",
                 "-", f"robust σ {teff['robust_sigma']:.0f} (N={teff['n']})"])
    rows.append(["Gaia parallax distance of the members (weighted 1/parallax) [kpc]",
                 f"{mem['d_plx_raw_kpc']:.3f} (no zero-point correction)",
                 f"&plusmn;{mem['d_plx_raw_kpc'] * mem['plx_wmean_err'] / mem['plx_wmean']:.3f}",
                 f"&plusmn;{mem['d_plx_raw_kpc'] * GAIA_PLX_SYS / mem['plx_wmean']:.3f} (0.010 mas floor)",
                 f"parallax {mem['plx_wmean']:.4f} mas"])
    table(rows, [190, 75, 70, 85, W - 420])
    corr = p.get("mu0_ebv_corr")
    st.append(Paragraph(
        f"Table {TAB['results']}: Unforced fit. Best grid point logAge = {s3['fit']['best']['logAge']:.2f}, loss = {s3['fit']['best']['loss']:.4f}. "
        + (f"Bootstrap correlation between μ<sub>0</sub> and E(B-V): {corr:+.2f}. " if corr is not None else "")
        + f"M<sub>TO</sub> is the initial mass at the bluest point of the upper main sequence of the best isochrone "
          f"(M<sub>r</sub> = {s3['fit']['turnoff_Mr']:.2f}). Teff: nearest isochrone point in the dereddened CMD, members with "
          f"r' &lt; {teff['mag_limit']} (dwarfs {teff['median_dwarfs']:+.0f} K, N={teff['n_dwarfs']}; giants {teff['median_giants']:+.0f} K, "
          f"N={teff['n_giants']})."
        + (f" <b>Warning:</b> best fit on the search boundary for {', '.join(s3['fit']['at_edge'])}." if s3["fit"]["at_edge"] else ""),
        S["cap"]))
    image(LIT_FIG, f"<b>Figure {FIG['fit']}.</b> This work's fit compared with published solutions. (a) Calibrated CMD of the "
          "Gaia members with this work's best isochrone (red) and PARSEC isochrones drawn for the age, (m-M)<sub>V</sub> and "
          "E(B-V) of Brewer et al. (2016) and Kalirai et al. (2001). (b) Distance-reddening plane at the best age: loss surface, "
          "bootstrap fits, this work's best fit and the literature values.", W * 0.9)
    st.append(Paragraph(
        "<b>How the panels are obtained.</b> (a) The literature isochrones use the same PARSEC grid ([Fe/H] = +0.09) and the "
        "same extinction coefficients as this work, at the published age (nearest grid age), with μ<sub>0</sub> = "
        "(m-M)<sub>V</sub> - 3.1 E(B-V); the published studies used other models and metallicities, so the comparison shows "
        "where their parameters put the cluster in my photometry, not their own fits. (b) The loss is evaluated on the full "
        "(E(B-V), μ<sub>0</sub>) grid at the best age (light = better fit); the diagonal valley is the distance-reddening "
        "degeneracy, and the cyan points are the 100 bootstrap fits.", S["body"]))
    st.append(Paragraph("3.3 Cluster over-density in the DS9 circle", S["h2"]))
    rows = [["Sample", "Inside circle", "Outside", f"Ring {cnt['ring_arcmin'][0]:.1f}-{cnt['ring_arcmin'][1]:.1f}'",
             "Field density [arcmin<super>-2</super>]", "Background-subtracted excess"]]
    for r in cnt["rows"]:
        rows.append([r["sample"], f"{r['n_inside']:,}", f"{r['n_outside']:,}", f"{r['n_ring']:,}",
                     f"{r['bkg_density_per_arcmin2']:.2f}", f"{r['excess']:.0f} &plusmn; {r['excess_err']:.0f}"])
    rows.append(["Gaia DR3 astrometric members", f"{cnt['members_inside']:,}", f"{cnt['members_outside']:,}",
                 f"{cnt['members_in_ring']:,}", "-", "-"])
    if cnt["hybrid"]:
        rows.append(["Gaia-matched, ring members removed", "", "", "", f"{cnt['hybrid']['field_density']:.2f}",
                     f"{cnt['hybrid']['excess']:.0f} &plusmn; {cnt['hybrid']['excess_err']:.0f}"])
    table(rows, [130, 55, 50, 60, 85, W - 380])
    st.append(Paragraph(
        f"Table {TAB['counts']}: Star counts for the circle r = {cnt['radius_arcmin']:.2f}' (on-chip area {cnt['area_inside']:.1f} arcmin<super>2</super>) "
        f"and the background ring (on-chip area {cnt['area_ring']:.1f} of {cnt['area_ring_full']:.1f} arcmin<super>2</super>). "
        "Excess = N<sub>in</sub> - (N<sub>ring</sub>/A<sub>ring</sub>) A<sub>in</sub>, Poisson errors.", S["cap"]))
    sec = 4
    cmp_ = R.get("comparison", {})
    if cmp_.get("fits"):
        dc, F = cmp_["decontamination"], cmp_["fits"]
        keys = ("main", "gaia_circle", "gaia_free")
        st.append(Paragraph(f"3.{sec} Membership with and without Gaia", S["h2"]))
        sec += 1
        rows = [["Quantity"] + [esc(cmp_["labels"][k]) for k in keys] + ["Benchmark"]]
        rows.append(["Stars fitted (MS / MSTO / SGB / RC anchors)"] + [
            f"{F[k]['n_stars']:,} ({F[k]['feature_counts']['MS']} / {F[k]['feature_counts']['MSTO']} / "
            f"{F[k]['feature_counts']['SGB']} / {F[k]['feature_counts']['RC']})" for k in keys] + ["-"])
        B = BENCH
        for key, lab, prec, bench in (
                ("age_gyr", "Age [Gyr]", 2, f"{B['age_gyr'][0]} to {B['age_gyr'][1]}"),
                ("mMV", "(m-M)<sub>V</sub> [mag]", 3, f"{B['mMV'][0]:.2f} &plusmn; {B['mMV'][1]:.2f}"),
                ("mu0", "μ<sub>0</sub> [mag]", 3, f"{B['mu0_astero'][0]:.2f} to {B['mu0_astero'][1]:.2f}"),
                ("ebv", "E(B-V) [mag]", 3, f"{B['ebv'][0]:.2f} to {B['ebv'][1]:.2f}"),
                ("d_kpc", "Distance d [kpc]", 3, f"{B['dist_kpc'][0]} to {B['dist_kpc'][1]}"),
                ("mto", "M<sub>TO</sub> [M<sub>sun</sub>]", 2, f"{B['mto'][0]} &plusmn; {B['mto'][1]}")):
            rows.append([lab] + [pm_html(F[k]['params'][key], prec) for k in keys] + [bench])

        def crit(k):
            v = {b["quantity"]: b["verdict"] for b in F[k]["benchmarks"]}
            return " / ".join("met" if v[q].startswith("criterion MET") else "not met" for q in ("(m-M)_V", "Age"))
        rows.append(["Proposal criteria: (m-M)<sub>V</sub> / age"] + [crit(k) for k in keys] + ["&le; 0.15 mag / &le; 0.3 Gyr"])
        table(rows, [120, 92, 92, 92, W - 396])
        st.append(Paragraph(
            f"Table {TAB['compare']}: The same calibration and the same unforced PARSEC fit (bootstrap and zero-point systematics included) "
            "applied to three membership choices. Columns 2 and 3 differ only in area; columns 3 and 4 cover the same "
            "area and differ only in how field stars are removed.", S["cap"]))
        image("NGC6819_Gaia_vs_GaiaFree.png", f"<b>Figure {FIG['compare']}.</b> (a) all stars in the circle, (b) the ring field scaled to the "
              "circle's area, (c) Gaia-free survivors coloured by P<sub>stat</sub> with their own fit, (d) Gaia members inside "
              "the circle, (e) Gaia members on the whole frame (main result), (f) fitted parameters of the three samples "
              "against the literature benchmarks (green).", W * 0.8)
        st.append(Paragraph(
            "<b>How the panels are obtained.</b> (b) is one random draw of ring stars scaled to the circle's area; (c) shows the "
            "stars that survive 100 realisations of the statistical subtraction, coloured by survival probability; every sample "
            "was fitted with the identical isochrone method, and (f) compares their best fits and errors.", S["body"]))
        student("membership_comparison")
    st.append(Paragraph(f"3.{sec} Comparison with the literature (proposal Section 5)", S["h2"]))
    rows = [["Quantity", "This work", "Benchmark", "Source", "Assessment"]]
    for b in R["benchmarks"]:
        prec = 2 if b["quantity"] in ("Age", "M_TO") else 3
        rows.append([b["quantity"], f"{b['value']:.{prec}f}<super>+{b.get('err_hi', b['err']):.{prec}f}</super><sub>-{b.get('err_lo', b['err']):.{prec}f}</sub>", esc(b["bench"]), esc(b["source"]),
                     esc(b["verdict"])])
    rows.append(["Teff", f"{teff['median']:+.0f} &plusmn; {teff['median_err']:.0f} K", "Gaia DR3 GSP-Phot", "Andrae et al. (2023)",
                 f"robust σ {teff['robust_sigma']:.0f} K"])
    table(rows, [55, 75, 90, 130, W - 350])
    st.append(Paragraph(f"Table {TAB['bench']}: Assessment computed by the pipeline against the proposal's benchmarks and success criteria "
                        "(distance modulus within 0.15 mag, age within 0.3 Gyr).", S["cap"]))
    student("literature_comparison", f"3.{sec + 1} Discussion")
    student("what_worked_failed", f"3.{sec + 2} What worked and what failed")
    student("improvements", f"3.{sec + 3} What could be improved")
    student("conclusion", f"3.{sec + 4} Conclusion")

    st.append(Paragraph("References", S["h1"]))
    for ref in REFERENCES:
        st.append(Paragraph(esc(ref), S["ref"]))
    st.append(Spacer(1, 10))
    st.append(Paragraph(f'<b>My GitHub link:</b> <a href="{GITHUB_URL}" color="blue">{GITHUB_URL}</a> '
                        "(code, inputs and results of this report)", S["body"]))

    def footer(canv, doc):
        canv.saveState()
        canv.setFont(base, 7.5)
        canv.setFillColor(colors.HexColor("#555555"))
        canv.drawRightString(A4[0] - 50, 28, f"page {doc.page}")
        canv.restoreState()

    SimpleDocTemplate(PDF_FILE, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=45, bottomMargin=45,
                      title="NGC 6819 GIT report", author="Veeresh S K").build(st, onFirstPage=footer, onLaterPages=footer)
    with open("report_placeholders.txt", "w") as f:
        f.write("Placeholders usable in report_text.md as {{name}} (current values):\n")
        for k, v in pv.items():
            f.write(f"  {{{{{k}}}}} = {v}\n")
    print(f"Built {PDF_FILE}")
    if missing:
        print(f"  {len(missing)} section(s) in {REPORT_TEXT_FILE} still TODO: {', '.join(missing)}")
    return missing


# ==================================================================================================
# MAIN
# ==================================================================================================

# ==================================================================================================
# ARTIFICIAL-STAR TEST (run separately:  python ngc6819_master_pipeline.py --ast)
# ==================================================================================================

AST_FILE_STARS = "ast_stars.csv"
AST_FILE_JSON = "ast_results.json"
AST_FIG = "artificial_star_test.png"
AST_BOX = (1100, 3700, 1100, 3700)      # x0, x1, y0, y1: the circle plus a 60-90 deg sector of the ring (r' frame px)
AST_ZONES = (("core 0-3'", 0.0, 3.0, 60), ("circle 3-9.5'", 3.0, None, 90), ("ring (field)", None, None, 100))
AST_MAG_RANGE = (14.0, 20.0)
AST_MIN_SEP_PIX = 40.0                  # artificial stars never closer than this to each other (keeps crowding unchanged)
AST_EDGE_PIX = 70.0                     # and never closer than this to the cut-out edge
AST_MATCH_PIX = 1.5                     # recovered = detected within 1.5 px ...
AST_MATCH_MAG = 0.75                    # ... and within 0.75 mag of the input r'
AST_MAG_BINS = (14, 16, 17, 18, 18.5, 19, 19.5, 20)


def _inject(img, epsf, xs, ys, fluxes, gain, rng, half=15):
    """Add ePSF-shaped stars with Poisson noise (in electrons) to img (in place)."""
    ny, nx = img.shape
    for x, y, f in zip(xs, ys, fluxes):
        xi, yi = int(round(x)), int(round(y))
        x0, x1, y0, y1 = max(xi - half, 0), min(xi + half + 1, nx), max(yi - half, 0), min(yi + half + 1, ny)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = epsf.copy(); m.flux, m.x_0, m.y_0 = f, x, y
        stamp = m(xx, yy)
        stamp = stamp + rng.normal(0.0, np.sqrt(np.clip(stamp, 0, None) / gain))
        img[y0:y1, x0:x1] += stamp.astype(img.dtype)


def _photometry_box(band, img, epsf, info, wcs, box):
    """Exactly the Stage 1 measurement (sky map, detection, aperture + ePSF photometry) on a cut-out."""
    from photutils.background import Background2D, MedianBackground
    from types import SimpleNamespace
    import contextlib, io
    x0, x1, y0, y1 = box
    bkg = Background2D(img, (BKG_BOX, BKG_BOX), filter_size=(3, 3), bkg_estimator=MedianBackground())
    sub = img[y0:y1, x0:x1]
    b = SimpleNamespace(background=bkg.background[y0:y1, x0:x1], background_rms=bkg.background_rms[y0:y1, x0:x1])
    with contextlib.redirect_stdout(io.StringIO()):
        df, st = aperture_and_psf_photometry(band, sub, b, sub - b.background, epsf, info,
                                             wcs.slice((slice(y0, y1), slice(x0, x1))))
    df["x"] += x0
    df["y"] += y0
    return df, st


def _ast_positions(rng, circle, wcs, n_by_zone, ring):
    """Random positions (r' frame px) in each zone, uniform in area, kept apart and away from the box edge."""
    x0, x1, y0, y1 = AST_BOX
    R = circle["radius_arcmin"]
    xs, ys, zs = [], [], []
    for name, lo, hi, n in n_by_zone:
        lo = ring[0] if lo is None else lo
        hi = (R if name.startswith("circle") else ring[1]) if hi is None else hi
        tries = 0
        got = 0
        while got < n and tries < 200 * n:
            tries += 1
            x, y = rng.uniform(x0 + AST_EDGE_PIX, x1 - AST_EDGE_PIX), rng.uniform(y0 + AST_EDGE_PIX, y1 - AST_EDGE_PIX)
            r = circle["centre"].separation(wcs.pixel_to_world(x, y)).arcmin
            if not (lo <= r < hi):
                continue
            if xs and np.min(np.hypot(np.array(xs) - x, np.array(ys) - y)) < AST_MIN_SEP_PIX:
                continue
            xs.append(x); ys.append(y); zs.append(name)
            got += 1
    return np.array(xs), np.array(ys), np.array(zs)


def run_artificial_star_test(circle, n_pass=12, seed=2026):
    """
    Artificial-star test: add stars of known r', g' (made from the frame's own ePSF, with Poisson noise) to both
    frames, a few hundred per pass so the crowding is not changed, rerun the Stage 1 measurement on a cut-out holding
    the DS9 circle and part of the background ring, calibrate with the pipeline's zero points, and compare output
    with input. Gives completeness (recovered in both bands) and photometric bias / scatter versus radius and
    magnitude, separately for the PSF and the aperture magnitudes.
    """
    banner(f"ARTIFICIAL-STAR TEST ({n_pass} passes)")
    with open(RESULTS_JSON) as f:
        R = json.load(f)
    cal = R["stage2"]["calibration"]
    zr, zg, cr, cg = cal["r"]["zp"], cal["g"]["zp"], cal["r"]["colour_term"], cal["g"]["colour_term"]
    ring = tuple(R["stage2"]["spatial_counts"]["ring_arcmin"])
    with open(STAGE1_JSON) as f:
        off_r = json.load(f)["bands"]["r"]["aperture_offset"]    # full-frame aperture -> PSF offset
    mem = pd.read_csv("cluster_members_cleaned.csv")
    rb = np.arange(11.0, 21.01, 0.5)
    gr_ridge = [np.nanmedian(mem.loc[mem["r_sdss"].between(a, a + 0.5), "g_r_sdss"]) for a in rb]
    ok = np.isfinite(gr_ridge)
    frames = {}
    for band in ("r", "g"):
        data, hdr, wcs = load_frame(band)
        _, _, epsf, info = image_models(band, data, hdr)
        frames[band] = (data, wcs, epsf, info)
    wr = frames["r"][1]
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for p in range(n_pass):
        xs, ys, zs = _ast_positions(rng, circle, wr, AST_ZONES, ring)
        r_in = rng.uniform(*AST_MAG_RANGE, len(xs))
        gr_in = np.interp(r_in, rb[ok], np.array(gr_ridge)[ok])
        g_in = r_in + gr_in
        c_inst = (gr_in - (zg - zr)) / (1.0 + cg - cr)            # invert the colour-term calibration
        r_inst = r_in - zr - cr * c_inst
        g_inst = r_inst + c_inst
        sky = wr.pixel_to_world(xs, ys)
        out = {}
        for band, m_inst in (("r", r_inst), ("g", g_inst)):
            data, wcs, epsf, info = frames[band]
            bx, by = (np.asarray(v) for v in wcs.world_to_pixel(sky))
            img = data.copy()
            _inject(img, epsf, bx, by, 10 ** (-0.4 * m_inst) * info["exptime"], info["gain"], rng)
            out[band], _ = _photometry_box(band, img, epsf, info, wcs, AST_BOX)
            del img
        m = crossmatch_bands(out["r"], out["g"])
        c = m["color_inst_g_r"]
        m["r_cal"] = m["mag_inst_r"] + zr + cr * c
        m["g_cal"] = m["mag_inst_g"] + zg + cg * c
        m["r_psf_cal"] = m["mag_psf_r"] + zr + cr * c
        m["r_ap_cal"] = m["mag_aper_1.5_r"] + off_r + zr + cr * c
        tree = cKDTree(np.c_[m["x_r"], m["y_r"]])
        d, j = tree.query(np.c_[xs, ys])
        rad = circle["centre"].separation(sky).arcmin
        for i in range(len(xs)):
            hit = d[i] < AST_MATCH_PIX and abs(m["r_cal"].iloc[j[i]] - r_in[i]) < AST_MATCH_MAG
            mm = m.iloc[j[i]]
            rows.append(dict(pass_=p, x=xs[i], y=ys[i], r_arcmin=rad[i], zone=zs[i], r_in=r_in[i], g_in=g_in[i],
                             gr_in=gr_in[i], recovered=bool(hit), sep_pix=d[i],
                             r_out=mm["r_cal"] if hit else np.nan, g_out=mm["g_cal"] if hit else np.nan,
                             r_psf=mm["r_psf_cal"] if hit else np.nan, r_ap=mm["r_ap_cal"] if hit else np.nan,
                             r_err=mm["mag_err_inst_r"] if hit else np.nan, fallback=bool(mm["use_aperture_fallback_r"]) if hit else False,
                             nn_pix=mm["nn_dist_pix_r"] if hit else np.nan))
        done = pd.DataFrame(rows)
        print(f"  pass {p + 1}/{n_pass}: {len(xs)} stars injected, {done['recovered'].iloc[-len(xs):].mean():.0%} recovered "
              f"({time.time() - t0:.0f} s)")
    ast = pd.DataFrame(rows).rename(columns={"pass_": "pass"})
    return ast


def summarise_ast(ast):
    """Completeness and output - input statistics per zone and r' bin (r_out = the pipeline's final magnitude,
    r_psf / r_ap = PSF-only and aperture-only versions, all calibrated the same way)."""
    res = {"n_injected": int(len(ast)), "n_pass": int(ast["pass"].nunique()), "zones": {}}
    for zname, _, _, _ in AST_ZONES:
        z = ast[ast["zone"] == zname]
        rows = []
        for lo, hi in zip(AST_MAG_BINS[:-1], AST_MAG_BINS[1:]):
            s = z[z["r_in"].between(lo, hi, inclusive="left")]
            rec = s[s["recovered"]]
            def med(v):
                return float(np.median(v)) if len(v) >= 5 else float("nan")
            def rsd(v):
                return float(robust_std(v)) if len(v) >= 5 else float("nan")
            dr = rec["r_out"] - rec["r_in"]
            rows.append(dict(r_lo=lo, r_hi=hi, n=int(len(s)), completeness=float(s["recovered"].mean()) if len(s) else float("nan"),
                             dr_median=med(dr), dr_scatter=rsd(dr), dr_err_reported=med(rec["r_err"]),
                             dr_psf_median=med((rec["r_psf"] - rec["r_in"]).dropna()),
                             dr_ap_median=med((rec["r_ap"] - rec["r_in"]).dropna()),
                             dgr_median=med((rec["g_out"] - rec["r_out"]) - rec["gr_in"]),
                             frac_brighter_0p2=float(np.mean(dr < -0.2)) if len(dr) else float("nan")))
        res["zones"][zname] = rows
    return res


def fig_ast(res):
    cols = {"core 0-3'": "crimson", "circle 3-9.5'": "darkorange", "ring (field)": "tab:blue"}
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    for z, rows in res["zones"].items():
        x = [(r["r_lo"] + r["r_hi"]) / 2 for r in rows]
        col = cols[z]
        axs[0, 0].plot(x, [r["completeness"] for r in rows], "o-", color=col, label=z)
        axs[0, 1].errorbar(x, [r["dr_median"] for r in rows], yerr=[r["dr_scatter"] for r in rows], fmt="o-",
                           color=col, capsize=3, label=z)
        axs[1, 0].plot(x, [r["dr_psf_median"] for r in rows], "o-", color=col, label=f"{z}: PSF")
        axs[1, 0].plot(x, [r["dr_ap_median"] for r in rows], "s--", color=col, mfc="none", label=f"{z}: aperture")
        axs[1, 1].plot(x, [r["dr_scatter"] for r in rows], "o-", color=col, label=f"{z}: measured scatter")
        axs[1, 1].plot(x, [r["dr_err_reported"] for r in rows], "s--", color=col, mfc="none", label=f"{z}: pipeline's own error")
    a = axs[0, 0]; a.set_ylim(0, 1.05); a.axhline(0.5, color="grey", ls=":"); a.set_ylabel("fraction recovered in r' and g'")
    a.set_title("(a) Completeness", fontweight="bold")
    a = axs[0, 1]; a.axhline(0, color="k", lw=0.8); a.set_ylabel("r'(out) - r'(in) [mag]  (median, bars = robust scatter)")
    a.set_title("(b) Bias of the final magnitudes", fontweight="bold")
    a = axs[1, 0]; a.axhline(0, color="k", lw=0.8); a.set_ylabel("median r'(out) - r'(in) [mag]")
    a.set_title("(c) PSF (solid) vs. aperture (dashed) magnitudes", fontweight="bold")
    a = axs[1, 1]; a.set_yscale("log"); a.set_ylabel("[mag]")
    a.set_title("(d) Real scatter vs. the error the pipeline reports", fontweight="bold")
    for a in axs.ravel():
        a.set_xlabel("input r' [mag]"); a.grid(alpha=0.3); a.legend(fontsize=7.5)
    fig.suptitle(f"Artificial-star test: {res['n_injected']:,} stars in {res['n_pass']} passes (ePSF-shaped, Poisson noise)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(AST_FIG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {AST_FIG}")


def print_ast(res):
    for z, rows in res["zones"].items():
        print(f"  {z}")
        for r in rows:
            print(f"    r' {r['r_lo']:4.1f}-{r['r_hi']:4.1f}: N {r['n']:4d}  complete {r['completeness']:5.1%}  "
                  f"dr {r['dr_median']:+.3f} (scatter {r['dr_scatter']:.3f}, reported {r['dr_err_reported']:.3f})  "
                  f"PSF {r['dr_psf_median']:+.3f}  aperture {r['dr_ap_median']:+.3f}  d(g-r) {r['dgr_median']:+.3f}")



def main():
    ap = argparse.ArgumentParser(description="NGC 6819 GIT end-to-end pipeline (FITS -> calibrated CMD -> PARSEC fit -> PDF)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--use-cache", action="store_true", help="reuse Stage 1 photometry (photometry_matched.csv); default")
    mode.add_argument("--from-scratch", action="store_true", help="redo Stage 1 from the FITS pixels")
    mode.add_argument("--report-only", action="store_true", help="rebuild the PDF from pipeline_results.json + report_text.md")
    ap.add_argument("--circle", default=CIRCLE_FILE, help="DS9 region file with your cluster circle")
    ap.add_argument("--spatial-filter", choices=("none", "circle"), default="none",
                    help="'circle': use only Gaia members inside the DS9 circle for the fit (default: all members)")
    ap.add_argument("--diff-reddening", choices=("auto", "on", "off"), default="auto",
                    help="remove local differential reddening before the fit: auto = only if the zone test finds it")
    ap.add_argument("--n-boot", type=int, default=100, help="bootstrap resamples for the isochrone fit (default 100)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel processes for the bootstrap")
    ap.add_argument("--refresh-catalogs", action="store_true", help="re-query PS1/Gaia/Cantat-Gaudin from VizieR")
    ap.add_argument("--ast", action="store_true",
                    help="run the artificial-star test (uses pipeline_results.json from a normal run), then rebuild the PDF")
    ap.add_argument("--ast-passes", type=int, default=12, help="passes of the artificial-star test (default 12)")
    ap.add_argument("--compare", choices=("on", "off"), default="on",
                    help="also fit Gaia members inside the circle and the Gaia-free statistical sample (default on)")
    args = ap.parse_args()
    os.chdir(HERE)

    if args.report_only:
        with open(RESULTS_JSON) as f:
            R = json.load(f)
        R["benchmarks"] = benchmark_table(R["stage3"]["fit"], R["stage2"]["membership"])   # refresh sources/verdicts
        with open(RESULTS_JSON, "w") as f:
            json.dump(R, f, indent=2)
        build_pdf(R)
        return

    if args.ast:
        _, hdr_r, wcs_r = load_frame("r", with_data=False)
        _, hdr_g, wcs_g = load_frame("g", with_data=False)
        geom = read_field_geometry(hdr_r, wcs_r)
        circle = load_manual_circle(args.circle, {"r": (wcs_r, geom["nx"], geom["ny"]),
                                                  "g": (wcs_g, int(hdr_g["NAXIS1"]), int(hdr_g["NAXIS2"]))})
        ast = run_artificial_star_test(circle, args.ast_passes)
        ast.to_csv(AST_FILE_STARS, index=False)
        res = summarise_ast(ast)
        with open(AST_FILE_JSON, "w") as f:
            json.dump(to_jsonable(res), f, indent=2)
        print_ast(res)
        fig_ast(res)
        with open(RESULTS_JSON) as f:
            build_pdf(json.load(f))
        return

    t_start = time.time()
    mode_name = "from-scratch" if args.from_scratch else "use-cache"
    banner(f"NGC 6819 MASTER PIPELINE ({mode_name})")
    _, hdr_r, wcs_r = load_frame("r", with_data=False)
    geom = read_field_geometry(hdr_r, wcs_r)
    print(f"Pointing centre {geom['centre_hms']} ({geom['centre_source']}); FOV {geom['fov_x_arcmin']:.2f}' x "
          f"{geom['fov_y_arcmin']:.2f}' at {geom['pix_scale']:.4f}\"/px; radii inscribed {geom['r_inscribed_arcmin']:.2f}', "
          f"circumscribed {geom['r_circumscribed_arcmin']:.2f}', corner search {geom['r_search_arcmin']:.2f}'")
    _, hdr_g, wcs_g = load_frame("g", with_data=False)
    circle = load_manual_circle(args.circle, {
        "r": (wcs_r, geom["nx"], geom["ny"]),
        "g": (wcs_g, int(hdr_g["NAXIS1"]), int(hdr_g["NAXIS2"]))})

    matched, diag = run_stage1(args.from_scratch)
    apt = export_apt_sample(matched, diag)
    cal, members, calib, refs, mstats, counts, cov = run_stage2(matched, geom, circle, hdr_r, wcs_r,
                                                                 args.refresh_catalogs, args.spatial_filter)
    stat_circ, ring, decon = statistical_decontamination(matched, cal, circle, counts)
    s3 = run_stage3(members, calib, refs, circle["centre"], args.n_boot, args.jobs, args.diff_reddening)
    comp = (run_comparison(cal, stat_circ, calib, refs, circle["centre"], args.n_boot, args.jobs, s3)
            if args.compare == "on" else None)

    banner("STAGE 4: FIGURES, EXPORTS, RESULTS FILE, PDF")
    fig_phase1(matched, diag)
    fig_phase2(cal, members, calib, refs, mstats)
    fig_phase3(cal, members, s3)
    fig_fit_vs_literature(members, s3["fit"], s3["_fitter"], s3["_boot"])
    fig_final_cmd(cal, members, s3, circle, counts, geom)
    apt_stats = fig_apt(apt, diag)
    export_ds9_regions(cal, members, circle, counts)
    if comp:
        fig_membership_comparison(stat_circ, ring, decon, s3, comp, members)

    R = dict(
        run=dict(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"), mode=mode_name, spatial_filter=args.spatial_filter,
                 n_boot=args.n_boot, diff_reddening=args.diff_reddening, stage1_source=diag.get("source"), runtime_s=None),
        geometry={k: v for k, v in geom.items() if k != "centre"},
        circle={k: v for k, v in circle.items() if k != "centre"},
        catalog_coverage=cov,
        stage1={k: v for k, v in diag.items() if k != "_arrays"},
        stage2=dict(calibration=calib, membership=mstats, spatial_counts=counts),
        stage3={k: v for k, v in s3.items() if not k.startswith("_")},
        apt=apt_stats,
    )
    R["benchmarks"] = benchmark_table(s3["fit"], mstats)
    R["comparison"] = comparison_summary(s3, comp, decon) if comp else dict(decontamination=decon)
    R["run"]["runtime_s"] = round(time.time() - t_start)
    R = to_jsonable(R)
    with open(RESULTS_JSON, "w") as f:
        json.dump(R, f, indent=2)
    print(f"Saved {RESULTS_JSON}")
    build_pdf(R)
    banner(f"DONE in {time.time() - t_start:.0f} s")
    for b in R["benchmarks"]:
        print(f"  {b['quantity']:<8s} {b['value']:.3f} +{b['err_hi']:.3f}/-{b['err_lo']:.3f}  vs {b['bench']:<22s} -> {b['verdict']}")


if __name__ == "__main__":
    main()
