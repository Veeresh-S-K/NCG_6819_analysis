# NGC 6819 with the GROWTH-India Telescope

A calibrated SDSS colour-magnitude diagram of the open cluster NGC 6819 (the Foxhead) from two 180 s frames
(r', g') of the 0.7 m GROWTH-India Telescope, with an unforced PARSEC isochrone fit for age, distance and reddening.
Observational astrophysics course project, Veeresh S K.

The final report is [`NGC6819_Final_Report.pdf`](NGC6819_Final_Report.pdf); the original proposal is
[`observation_report.pdf`](observation_report.pdf).

## Result (main fit, 1,544 Gaia DR3 members)

| Quantity | Value |
|---|---|
| Age | 2.69 +0.13 / -0.06 Gyr |
| Distance modulus mu0 | 12.124 +0.082 / -0.144 (d = 2.66 kpc) |
| E(B-V) | 0.115 +0.031 / -0.029 |
| (m-M)_V | 12.481 +0.024 / -0.105 |
| Turnoff mass | 1.53 M_sun |

Errors are bootstrap (16th/84th percentile) and zero-point systematics combined. All numbers are in
`pipeline_results.json`.

## What the pipeline does

`ngc6819_master_pipeline.py` runs everything from FITS pixels to the PDF:

1. **Photometry:** Background2D sky, DAOStarFinder detection, ePSF (Anderson & King 2000) from stars spread over the
   chip, PSF photometry with a 1.5 FWHM aperture fallback for the brightest stars, r'/g' matching on the sky.
2. **Calibration and membership:** Pan-STARRS DR1 converted to SDSS (Tonry et al. 2012), zero point and colour term
   with a robust (Siegel) fit and bootstrap errors; Gaia DR3 membership (Cantat-Gaudin et al. 2018 probabilities, or
   proper motion + parallax + RUWE); star counts in the manual DS9 circle; a Gaia-free statistical field subtraction for
   comparison; an artificial-star test for completeness and bias.
3. **Isochrone fit:** PARSEC v1.2S ([Fe/H] = +0.09), distance and reddening free, four CMD features each matched to
   its own evolutionary phase, grid + Nelder-Mead, bootstrap and zero-point systematics.
4. **Report:** figures, `pipeline_results.json`, and `NGC6819_Final_Report.pdf`. Numbers come from the JSON; the
   written sections come from `report_text.md` (placeholders like `{{age_gyr}}` are filled automatically).

## Files

| Path | Content |
|---|---|
| `ngc6819_master_pipeline.py` | the whole analysis and report generator |
| `extinction_coefficients.py`, `filter_curves/` | derivation of A_r' and E(g'-r') per E(B-V) (CCM89 / O'Donnell 94 over the SDSS filter curves) |
| `make_setup_figure.py` | the set-up figure in `screenshots/processing1.png` |
| `manual_cluster_circle.reg` | the cluster circle drawn in SAOImageDS9 (ICRS) |
| `report_text.md` | the written sections of the report |
| `parsec_grid_ngc6819.dat` | PARSEC v1.2S isochrone grid (CMD web tool, SDSS ugriz) |
| `reference_catalogs/` | cached Pan-STARRS DR1, Gaia DR3 and Cantat-Gaudin 2018 catalogues (`SOURCES.txt` records provenance) |
| `photometry_*.csv`, `stage1_diagnostics.*` | Stage 1 photometry, reused by `--use-cache` |
| `cluster_members_cleaned.csv`, `statistical_membership.csv`, `isochrone_bootstrap_samples*.csv`, `ast_*` | results |
| `*.png`, `ds9_cluster_members.reg` | figures and a DS9 overlay of the members |

The GIT FITS frames (`20261012170-258-RA.wcs.proc.fits`, r'; `20261012170-727-RA.wcs.proc.fits`, g') are not in the
repository because of their size; they are needed only for `--from-scratch` and `--ast` runs.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python ngc6819_master_pipeline.py --report-only    # rebuild the PDF from the saved results (seconds)
.venv/bin/python ngc6819_master_pipeline.py --use-cache      # redo Stages 2-4 from the cached photometry (~15 min)
.venv/bin/python ngc6819_master_pipeline.py --from-scratch   # redo everything from the FITS frames (needs the frames)
.venv/bin/python ngc6819_master_pipeline.py --ast            # artificial-star test (needs the frames)
.venv/bin/python extinction_coefficients.py                  # print the extinction coefficients
```

## AI use

The code was written with Claude Code (Anthropic) and the references were checked on NASA ADS; see Section 2.7 of the
report for the full declaration.
