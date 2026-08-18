"""
Graphical abstract for

  Reorganisation of internal tide energetics by Hurricane Catarina
  over the critical slope of the South Brazil Bight
  Dialectaquiz, Mazzini and Dottori

Two cross-margin schematics, unforced and wind-forced, drawn on identical
geometry, so every difference between the panels is a result of the paper.
The depth axis is stretched over the upper 400 m, where the storm mixing
acts, and compressed below, so the shelf, the slope and the basin all stay
legible in one frame. The four headline numbers sit in the footer strip.

Figure proportion 13 x 5.6 in at 400 dpi, 5200 x 2240 px, above the Elsevier
graphical abstract minimum of 1328 x 531 px.

Outputs graphical_abstract.png and graphical_abstract.pdf.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ----------------------------------------------------------------- palette
C_SURF = "#f0f6fb"      # near-surface water
C_DEEP = "#8fb0cb"      # deep water
C_BED = "#c5baa8"       # sea bed
C_BEDE = "#8f8574"      # sea bed edge
C_ISO = "#5d7f9e"       # density surfaces
C_BT = "#1f3a5f"        # barotropic tide
C_BEAM = "#0e7c7b"      # coherent baroclinic beams
C_FRAG = "#3f9694"      # scattered baroclinic flux
C_STORM = "#b02418"     # hurricane, wind stress
C_NIW = "#6a3d9a"       # near-inertial waves
C_DISS = "#c1651a"      # local retention, downscale transfer
C_CRIT = "#8a6300"      # criticality
C_UPW = "#1b6ca8"       # tidal upwelling
C_MIX = "#e4eaee"       # mixed, unstratified water
C_TXT = "#141414"
C_MUTE = "#5a6169"

plt.rcParams.update({
	"font.family": "DejaVu Sans",
	"font.size": 7.4,
	"text.color": C_TXT,
	"axes.linewidth": 0.7,
	"pdf.fonttype": 42,
	"ps.fonttype": 42,
})

# ---------------------------------------------------------------- geometry
XMAX = 100.0            # km across the margin
AIR = 430.0             # plot units above the sea surface
WATER = 1000.0          # plot units, surface to the deepest bed
ZBREAK, FBREAK = 400.0, 0.55    # stretch the top 400 m over 55% of the panel
HMAX = 3000.0
XSB, HSHELF, HDEEP = 22.0, 150.0, 2800.0
RAY = -20.0             # metres of descent per kilometre, M2 characteristic


def T(z):
	"""Real depth in metres to plot units, piecewise linear."""
	d = -np.asarray(z, dtype=float)
	shallow = -(d / ZBREAK) * FBREAK
	deep = -(FBREAK + (1.0 - FBREAK) * (d - ZBREAK) / (HMAX - ZBREAK))
	return np.where(d <= ZBREAK, shallow, deep) * WATER


def bed(x):
	return -(HSHELF + (HDEEP - HSHELF) * 0.5 * (1.0 + np.tanh((x - 31.0) / 5.0)))


# ------------------------------------------------------------- primitives
def water_gradient(ax):
	grad = np.linspace(0, 1, 512).reshape(-1, 1)
	cmap = LinearSegmentedColormap.from_list("sea", [C_DEEP, C_SURF])
	ax.imshow(grad, extent=[0, XMAX, -WATER, 0], aspect="auto", cmap=cmap,
			  origin="lower", zorder=0, interpolation="bilinear")


def draw_bed(ax):
	x = np.linspace(0, XMAX, 800)
	ax.fill_between(x, T(bed(x)), -WATER, color=C_BED, zorder=3, lw=0)
	ax.plot(x, T(bed(x)), color=C_BEDE, lw=1.2, zorder=3.1)


def isopycnals(ax, levels, x0=0.0, alpha=0.75, lw=0.6):
	x = np.linspace(x0, XMAX, 500)
	for z0 in levels:
		z = z0 - 0.18 * z0 * np.exp(-((x - 30.0) / 15.0) ** 2)
		z = np.where(z > bed(x) + 30, z, np.nan)
		ax.plot(x, T(z), color=C_ISO, lw=lw, alpha=alpha, zorder=2)


def turbulence(ax, x0, x1, ytop, ybot, n=30, seed=0, color="#8b949b", alpha=0.6):
	rng = np.random.default_rng(seed)
	t = np.linspace(0, 2 * np.pi, 60)
	for _ in range(n):
		cx, cy = rng.uniform(x0, x1), rng.uniform(ybot, ytop)
		w, h = rng.uniform(1.3, 2.4), rng.uniform(9, 17)
		ax.plot(cx + w * np.sin(t), cy + h * np.sin(t) * np.cos(t),
				color=color, lw=0.5, alpha=alpha, zorder=2.4)


def arrow(ax, p0, p1, color, lw=1.5, ls="-", mut=7, alpha=1.0, z=5):
	ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
								 lw=lw, ls=ls, color=color, alpha=alpha,
								 shrinkA=0, shrinkB=0, zorder=z))


def leader(ax, p0, p1, color):
	ax.annotate("", xy=p1, xytext=p0, zorder=7,
				arrowprops=dict(arrowstyle="-", lw=0.55, color=color, alpha=0.9))


def ray(ax, x0, z0, length, color, lw=2.1, mut=10):
	xs = np.linspace(x0, x0 + length, 120)
	ys = T(z0 + RAY * (xs - x0))
	ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=5,
			solid_capstyle="round")
	arrow(ax, (xs[-7], ys[-7]), (xs[-1], ys[-1]), color, lw=lw, mut=mut)


def wave_train(ax, x0, z0, z1, dx=7.0, amp=2.2, ncyc=5, color=C_NIW, lw=1.0):
	t = np.linspace(0, 1, 300)
	xs = x0 + dx * t + amp * np.sin(2 * np.pi * ncyc * t) * np.sin(np.pi * t)
	ys = T(z0 + (z1 - z0) * t)
	ax.plot(xs, ys, color=color, lw=lw, zorder=4.5, solid_capstyle="round")
	arrow(ax, (xs[-8], ys[-8]), (xs[-1], ys[-1]), color, lw=lw, mut=7, z=4.5)


def tag(ax, x, y, text, color, ha="center", va="center", fs=6.6,
		weight="bold", alpha=0.92, pad=0.30):
	ax.text(x, y, text, color=color, ha=ha, va=va, fontsize=fs,
			fontweight=weight, zorder=8, linespacing=1.35,
			bbox=dict(boxstyle=f"round,pad={pad}", fc="white", ec=color,
					  lw=0.6, alpha=alpha))


def hurricane(ax, cx, cy, r=3.4, yscale=11.0, color=C_STORM):
	th = np.linspace(0, 1.75 * np.pi, 200)
	rr = r * 0.34 * np.exp(0.34 * th)
	for s in (1, -1):
		ax.plot(cx + s * rr * np.cos(th), cy + s * rr * np.sin(th) * yscale,
				color=color, lw=1.3, zorder=7, solid_capstyle="round")
	ax.plot([cx], [cy], marker="o", ms=2.4, color="white", mec=color,
			mew=0.9, zorder=7.1)


def panel_frame(ax, title, subtitle, tcolor):
	ax.set_xlim(0, XMAX)
	ax.set_ylim(-WATER, AIR)
	ax.set_xticks([])
	ax.set_yticks([])
	for s in ax.spines.values():
		s.set_color("#c8ced3")
	ax.axhline(0, color="#2c3e50", lw=1.0, zorder=4)
	ax.text(1.8, AIR - 22, title, fontsize=10.2, fontweight="bold",
			color=tcolor, va="top", zorder=9)
	ax.text(1.8, AIR - 150, subtitle, fontsize=7.1, color=C_MUTE, va="top",
			zorder=9, style="italic")


def footer(ax, note, note_color=C_MUTE):
	for d, lab in [(200, "200 m"), (1000, "1000 m"), (2800, "2800 m")]:
		y = T(-d)
		ax.plot([97.2, 99.3], [y, y], color=C_MUTE, lw=0.6, zorder=6)
		ax.text(96.7, y, lab, fontsize=5.9, color=C_MUTE, ha="right",
				va="center", zorder=6)
	ax.text(10.5, -978, "SHELF,  $h < 250$ m", fontsize=6.2, color="#7a7060",
			ha="center", va="center", zorder=6, fontweight="bold")
	ax.text(70.0, -978, "DEEP OCEAN,  250 to 3500 m", fontsize=6.2,
			color="#7a7060", ha="center", va="center", zorder=6,
			fontweight="bold")
	ax.annotate("", xy=(30, -978), xytext=(40, -978),
				arrowprops=dict(arrowstyle="<->", lw=0.7, color="#7a7060"))
	ax.text(35, -958, "10 km", fontsize=5.9, color="#7a7060", ha="center",
			va="bottom", zorder=6)
	ax.text(58, -890, note, color=note_color, fontsize=6.8, ha="center",
			va="center", zorder=8, linespacing=1.3, fontweight="bold",
			bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#c8ced3",
					  lw=0.5, alpha=0.90))


# ================================================================== figure
fig = plt.figure(figsize=(13.0, 5), dpi=400)
fig.patch.set_facecolor("white")
axL = fig.add_axes([0.028, 0.180, 0.462, 0.655])
axR = fig.add_axes([0.510, 0.180, 0.462, 0.655])

# ------------------------------------------------------------- left panel
water_gradient(axL)
isopycnals(axL, [-40, -80, -130, -190, -260, -340, -450, -650, -950,
				 -1400, -2000, -2500])
draw_bed(axL)
panel_frame(axL, "Unforced", "tidal forcing only", C_BT)

xt = np.linspace(34, 66, 300)
axL.plot(xt, 165 + 40 * np.sin(2 * np.pi * (xt - 34) / 10.5), color=C_BT,
		 lw=1.3, zorder=6)
arrow(axL, (67, 165), (73, 165), C_BT, lw=1.3, mut=8, z=6)
arrow(axL, (33, 165), (27, 165), C_BT, lw=1.3, mut=8, z=6)
axL.text(50, 268, "barotropic $M_2$ tide, spring", color=C_BT, fontsize=7.1,
		 ha="center", fontweight="bold", zorder=8)

arrow(axL, (18.5, T(-140)), (18.5, T(-22)), C_UPW, lw=1.7, mut=9, z=6)
leader(axL, (12.0, 62), (18.3, 22), C_UPW)
tag(axL, 12.0, 122, "phase-locked tidal upwelling\nat the shelf break,  "
					"$\\mathcal{S} < 0$", C_UPW, fs=6.5)

axL.plot([XSB], [T(bed(XSB))], marker="*", ms=10, color="#e3b23c",
		 mec=C_CRIT, mew=0.6, zorder=6)
leader(axL, (11.5, -370), (XSB - 0.9, T(bed(XSB)) - 12), C_CRIT)
tag(axL, 11.0, -430, "critical to supercritical\nslope,  $\\alpha \\approx 1$",
	C_CRIT)

for zoff, length in [(-25, 68), (-130, 62), (-260, 56)]:
	ray(axL, XSB + 1.2, bed(XSB) + zoff, length, C_BEAM)
axL.text(68, -270, "coherent mode 1 to 4 beams\nradiate to the deep ocean",
		 color=C_BEAM, fontsize=7.2, fontweight="bold", ha="center", zorder=8,
		 linespacing=1.35)

footer(axL, "stratification intact, the water column carries the modes")

# ------------------------------------------------------------ right panel
water_gradient(axR)
isopycnals(axR, [-260, -340, -450, -650, -950, -1400, -2000, -2500], x0=14)
xx = np.linspace(0, XMAX, 600)
axR.fill_between(xx, np.maximum(T(bed(xx)), T(-150)), 0.0, color=C_MIX,
				 alpha=0.95, zorder=2.2, lw=0)
turbulence(axR, 1.5, 98.5, -8, T(-140), n=44, seed=3)
turbulence(axR, 1.5, 20.0, -8, T(-145), n=14, seed=7, alpha=0.75)
draw_bed(axR)
panel_frame(axR, "Wind-forced", "Hurricane Catarina, 24 to 30 March 2004",
			C_STORM)

hurricane(axR, 80.0, 300.0)
arrow(axR, (75.0, 300), (46.0, 300), C_STORM, lw=1.3, ls=(0, (4, 2)), mut=9, z=7)
axR.text(60, 372, "westward track, landfall 28 March", color=C_STORM,
		 fontsize=7.0, ha="center", fontweight="bold", zorder=8)
for x0 in np.arange(34, 82, 8.0):
	arrow(axR, (x0, 200), (x0 - 4.2, 28), C_STORM, lw=1.1, mut=6, alpha=0.9, z=6)
axR.text(92.0, 120, "wind\nstress", color=C_STORM, fontsize=7.0,
		 fontweight="bold", ha="center", zorder=8, linespacing=1.3)

axR.plot([18.5, 18.5], [T(-140), T(-22)], color=C_UPW, lw=1.5, ls=(0, (2, 2)),
		 alpha=0.55, zorder=5)
axR.plot([17.1, 19.9], [T(-112), T(-48)], color="#8f2d2d", lw=1.7, zorder=6)
axR.plot([17.1, 19.9], [T(-48), T(-112)], color="#8f2d2d", lw=1.7, zorder=6)
leader(axR, (12.0, 62), (18.3, 22), "#8f2d2d")
tag(axR, 12.0, 122, "tidal upwelling weakened,\nreversed at the storm peak",
	"#8f2d2d", fs=6.5)

axR.text(56, T(-76), "upper 100 to 150 m mixed,  shelf column mixed through",
		 color=C_MUTE, fontsize=6.5, ha="center", va="center", zorder=8,
		 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#b9c2c8",
				   lw=0.5, alpha=0.93))

axR.plot([XSB], [T(bed(XSB))], marker="*", ms=10, color="#e3b23c",
		 mec=C_CRIT, mew=0.6, zorder=6)
leader(axR, (11.5, -370), (XSB - 0.9, T(bed(XSB)) - 12), C_CRIT)
tag(axR, 11.0, -430, "shelf towards subcritical,\n$\\Delta\\alpha < 0$", C_CRIT)

rng = np.random.default_rng(4)
for zoff, length in [(-25, 34), (-130, 28), (-260, 24)]:
	xc, zc, used = XSB + 1.2, bed(XSB) + zoff, 0.0
	while used < length:
		L = rng.uniform(3.0, 5.5)
		xs = np.linspace(xc, xc + L, 30)
		zs = zc + RAY * (xs - xc) * rng.uniform(0.6, 1.5)
		axR.plot(xs, T(zs), color=C_FRAG, lw=1.6, zorder=5,
				 solid_capstyle="round")
		gap = rng.uniform(1.8, 3.4)
		zc, xc = zs[-1] + rng.uniform(-60, 60), xc + L + gap
		used += L + gap
leader(axR, (56, T(-1450)), (50, T(-870)), C_FRAG)
tag(axR, 57, T(-1560), "scattered, incoherent flux", C_FRAG, fs=6.9)

for cx, cy, r in [(31.0, T(-330), 0.85), (34.0, T(-620), 1.0),
				  (38.5, T(-1150), 1.05)]:
	th = np.linspace(0.3, 1.95 * np.pi, 140)
	xs, ys = cx + 2.6 * r * np.cos(th), cy + 46 * r * np.sin(th)
	axR.plot(xs, ys, color=C_DISS, lw=1.4, zorder=5.5)
	arrow(axR, (xs[-4], ys[-4]), (xs[-1], ys[-1]), C_DISS, lw=1.4, mut=7, z=5.5)
tag(axR, 27.0, T(-1900), "retained and dissipated\nnear the generation sites",
	C_DISS, fs=6.9)

for x0 in (66.0, 82.0):
	wave_train(axR, x0, -170, -1250)
tag(axR, 86.0, T(-1750), "near-inertial\nwaves", C_NIW, fs=6.7)

footer(axR, "eroded stratification, the modes are no longer supported")

# --------------------------------------------------------------- headline
#fig.text(0.5, 0.975, "Hurricane Catarina turns an exporting internal tide "
					 #"into a locally dissipating one over the South Brazil Bight",
		 #ha="center", va="top", fontsize=11.6, fontweight="bold", color=C_TXT)
#fig.text(0.5, 0.922, "Twin 1-km ROMS simulations, identical except for the "
					 #"atmospheric forcing, spanning the passage during spring tide",
		 #ha="center", va="top", fontsize=7.9, color=C_MUTE, style="italic")
#
## ------------------------------------------------------------------ chips
chips = [
	("$M_2$ conversion, domain total", "295 $\\rightarrow$ 220 MW", C_BT),
	("Deep-ocean export, after the passage", "63% $\\rightarrow$ 15%", C_FRAG),
	("Retained near generation, after the passage", "37% $\\rightarrow$ 85%", C_DISS),
	("Tidal-to-supertidal transfer, after the wind relaxed",
	 "6.4$\\times$ the unforced level", C_NIW),
]
x0, w, gap = 0.028, 0.2255, 0.0075
for i, (lab, val, col) in enumerate(chips):
	xc = x0 + i * (w + gap)
	fig.patches.append(FancyBboxPatch(
		(xc, 0.030), w, 0.098, boxstyle="round,pad=0.004,rounding_size=0.012",
		transform=fig.transFigure, fc="#f7f9fa", ec=col, lw=1.0, zorder=1))
	fig.text(xc + w / 2, 0.104, lab, ha="center", va="center", fontsize=6.7,
			 color=C_MUTE)
	fig.text(xc + w / 2, 0.059, val, ha="center", va="center", fontsize=9.3,
			 color=col, fontweight="bold")

fig.savefig("graphical_abstract.pdf", facecolor="white",dpi = 400)
print("saved")