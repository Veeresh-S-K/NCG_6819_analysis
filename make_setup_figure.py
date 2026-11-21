"""Setup overview for the NGC 6819 report: circle (from .reg), background ring, catalogue cone, frame, defects."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from astropy.visualization import ZScaleInterval
import astropy.units as u
import ngc6819_master_pipeline as P

data, hdr, wcs = P.load_frame("r")
_, hg, wg = P.load_frame("g", with_data=False)
geom = P.read_field_geometry(hdr, wcs)
circ = P.load_manual_circle(P.CIRCLE_FILE, {"r": (wcs, geom["nx"], geom["ny"]), "g": (wg, hg["NAXIS1"], hg["NAXIS2"])})
B = 4; ny, nx = data.shape
s = data[: ny // B * B, : nx // B * B].reshape(ny // B, B, nx // B, B).mean(axis=(1, 3))
lo, hi = ZScaleInterval().get_limits(s)
scale = geom["pix_scale"]
px = lambda arcmin: arcmin * 60 / scale / B
cx, cy = circ["x0"] / B, circ["y0"] / B
tx, ty = (v / B for v in wcs.world_to_pixel(geom["centre"]))
R = circ["radius_arcmin"]; r1, r2 = P.BKG_ANNULUS_FACTORS[0] * R, P.BKG_ANNULUS_FACTORS[1] * R

fig = plt.figure(figsize=(15, 11), facecolor="white")
ax = fig.add_axes([0.02, 0.04, 0.66, 0.9])
ax.imshow(s, origin="lower", cmap="gray_r", vmin=lo, vmax=hi, extent=(-0.5 / B, nx / B, -0.5 / B, ny / B))
ax.add_patch(Rectangle((0, 0), nx / B, ny / B, fill=False, ec="darkorange", lw=2.5))
ax.add_patch(Circle((tx, ty), px(geom["r_search_arcmin"]), fill=False, ec="goldenrod", lw=2, ls=(0, (7, 4))))
ax.add_patch(Circle((cx, cy), px(r1), fill=False, ec="tab:blue", lw=1.8, ls="--"))
ax.add_patch(Circle((cx, cy), px(r2), fill=False, ec="tab:blue", lw=1.8, ls="--"))
ax.add_patch(Circle((cx, cy), circ["r_pix"] / B, fill=False, ec="deepskyblue", lw=3))
ax.plot(tx, ty, "+", color="goldenrod", ms=16, mew=2.5)
# compass from the WCS
c0 = geom["centre"]
for lab, pa in (("N", 0), ("E", 90)):
    tip = c0.directional_offset_by(pa * u.deg, 3.5 * u.arcmin)
    x1, y1 = (v / B for v in wcs.world_to_pixel(tip))
    ox, oy = 1010, 1080
    dx, dy = x1 - tx, y1 - ty
    ax.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text(ox + 1.25 * dx, oy + 1.25 * dy, lab, fontsize=13, fontweight="bold", ha="center", va="center")
ax.plot([-230, -230 + px(10)], [1180, 1180], "k-", lw=3); ax.text(-230 + px(5), 1200, "10'", ha="center", fontsize=12)
st = [pe.withStroke(linewidth=3, foreground="white")]
defects = [  # (x, y) in binned pixels, label
    ((176.5, 90), "Dead columns x = 704-709 (lower part of chip):\nlose a few stars, add no fake ones"),
    ((25, 25), "Vignetted corners: flat field cannot fix them\n(noisy, extreme pixels); outside circle and ring"),
    ((1000, 25), None),
    ((60, 890), "Ghost / reflection arc (stray light);\nalmost entirely outside the ring"),
    ((600, 680), "Saturated stars with bleed trails / spikes:\n111 flagged (peak >= 58,000 ADU); no extra\ndetections found along the trails"),
]
n = 0
for (x, y), text in defects:
    if text is not None:
        n += 1
    ax.text(x, y, str(n), color="red", fontsize=14, fontweight="bold", ha="center", va="center",
            path_effects=st, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="red", lw=1.5))
ax.set_xlim(-260, nx / B + 260); ax.set_ylim(-260, ny / B + 260)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("NGC 6819, GIT r' 180 s frame (binned 4x4, inverted grey scale)", fontsize=13, fontweight="bold")

leg = fig.add_axes([0.69, 0.04, 0.30, 0.9]); leg.axis("off")
handles = [
    Line2D([], [], color="deepskyblue", lw=3, label=f"YOUR circle (manual_cluster_circle.reg)\n"
           f"centre {circ['centre_hms']}, r = {R:.1f}'\n-> cluster + field stars"),
    Line2D([], [], color="tab:blue", lw=1.8, ls="--", label=f"Background ring {r1:.2f}'-{r2:.2f}'\n"
           f"(script setting: 1.5R-2.5R, not drawn by you)\n-> field-star density per arcmin$^2$"),
    Line2D([], [], color="darkorange", lw=2.5, label=f"Detector frame {geom['fov_x_arcmin']:.1f}' x {geom['fov_y_arcmin']:.1f}'\n"
           f"({nx} x {ny} px, {scale:.3f}\"/px)"),
    Line2D([], [], color="goldenrod", lw=2, ls=(0, (7, 4)), label=f"Catalogue download cone r = {geom['r_search_arcmin']:.2f}'\n"
           "(PS1, Gaia; from the WCS: reaches\nevery frame corner)"),
    Line2D([], [], color="goldenrod", marker="+", ls="", ms=14, mew=2.5, label="Pointing centre (TARRA/TARDEC)"),
]
leg.legend(handles=handles, loc="upper left", fontsize=10.5, frameon=True, labelspacing=1.4, title="What is on the sky",
           title_fontproperties=dict(weight="bold", size=12))
y = 0.44
leg.text(0.0, y + 0.04, "Image defects (red numbers)", fontsize=12, fontweight="bold", transform=leg.transAxes)
for i, (_, text) in enumerate([d for d in defects if d[1]], 1):
    leg.text(0.0, y, f"{i}", color="red", fontsize=12, fontweight="bold", transform=leg.transAxes, va="top")
    leg.text(0.06, y, text, fontsize=10, transform=leg.transAxes, va="top")
    y -= 0.045 + 0.02 * text.count("\n")
leg.text(0.0, 0.0, "Bias and flat field were applied by the GIT pipeline\n(header keywords MBIAS, MFLAT) before the data\n"
         "were delivered; the script does not mask defects but\nrejects them through the r'+g' cross-match, the\n"
         "saturation flag and the PS1/Gaia matches.", fontsize=9.5, transform=leg.transAxes, va="bottom", style="italic")
fig.savefig(sys.argv[1], dpi=150)
print("saved", sys.argv[1])
