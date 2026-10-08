"""Reproduce Fig. S31 from accompanying computational replay source tables.

All eligible matched quartets are retained. The Agreement score is not a
probability and is therefore excluded from the probability-threshold panels.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SKILL_SCRIPTS = Path(__file__).resolve().parent / "qa_tools"
sys.path.insert(0, str(SKILL_SCRIPTS))
from panel_alignment import require_matplotlib_panel_alignment

OUT = Path(__file__).resolve().parent
DATA = OUT / "source_data"
BASE = OUT / "Fig_S31"
COLORS = {"Global": "#727A83", "Event": "#2366A5", "Combined": "#AC7B42"}
MODELS = ["Global", "Event", "Combined"]
THRESHOLDS = [0.05, 0.10, 0.20]
MARKERS = {0.05: "o", 0.10: "s", 0.20: "^"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "legend.frameon": False,
    "savefig.facecolor": "white",
})

summary = pd.read_csv(DATA / "quartet_concordance_summary.csv")
quartets = pd.read_csv(DATA / "quartet_pair_details.csv")
decisions = pd.read_csv(DATA / "fixed_threshold_decisions.csv")
assert len(quartets) == 4096
assert quartets.groupby(["target", "residual", "snr_db"]).size().eq(4).all()
mixed = quartets.loc[quartets["mixed"].eq(1)].pivot(
    index=["target", "residual", "snr_db", "source"],
    columns="model", values="quartet_concordance"
)
assert len(mixed) == 116 and mixed[MODELS].notna().all().all()
assert np.isfinite(mixed[MODELS].to_numpy()).all()
diff = (mixed["Event"] - mixed["Global"]).round(12)
distribution = diff.value_counts().sort_index()
assert distribution.sum() == 116
for model in MODELS:
    s = summary.loc[summary.scope.eq("pooled") & summary.model.eq(model)].iloc[0]
    assert np.isclose(s.concordance_quartet_equal_primary, mixed[model].mean())
for _, row in decisions.iterrows():
    assert np.isclose(row.coverage, row.retained / row.episodes)
    assert np.isclose(row.retained_failure_fraction, row.retained_failures / row.retained)
    assert row.retained_failures + row.all_nonfailures - row.discarded_nonfailures == row.retained

# Dimensions in inches equal 183 × 145 mm at the final export size.
fig, axs = plt.subplots(2, 2, figsize=(7.2047244094, 5.7086614173))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.115, top=0.925, wspace=0.38, hspace=0.62)

def decorate(ax, letter, title):
    ax.set_label(letter)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-29, 17),
                textcoords="offset points", ha="left", va="bottom", weight="bold", fontsize=9)
    ax.set_title(title, loc="left", pad=12)
    ax.tick_params(direction="out")

# a: Equal quartet weighting isolates the within-quartet timing comparison.
ax = axs[0, 0]
decorate(ax, "a", "Discrimination within matched quartets")
strata = ["G1", "G2", "G3", "G4", "all"]
for model in MODELS:
    rows = summary.loc[summary.scope.isin(["source", "pooled"]) & summary.model.eq(model)].set_index("stratum")
    values = rows.loc[strata, "concordance_quartet_equal_primary"].to_numpy()
    ax.plot(range(5), values, color=COLORS[model], lw=1.15 if model == "Event" else 0.8,
            marker="o", markersize=4 if model == "Event" else 3.4, label=model)
ax.axhline(0.5, ls="--", color="#B7BDC3", lw=0.7, zorder=0)
ax.set_xlim(-0.28, 4.28)
ax.set_ylim(0.32, 1)
ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_xticks(range(5), ["G1", "G2", "G3", "G4", "Pooled"])
ax.set_xlabel("Held-out recording source")
ax.set_ylabel("Mean quartet concordance")
ax.legend(loc="upper left", ncol=3, handlelength=1.5, columnspacing=0.9, borderaxespad=0.15)

# b: Complete paired distribution; exact rational differences are displayed.
ax = axs[0, 1]
decorate(ax, "b", "Paired changes include reversals")
xs = distribution.index.to_numpy(float)
counts = distribution.to_numpy(int)
barcolors = [COLORS["Event"] if x > 0 else ("#A2A8AE" if x < 0 else "#545E68") for x in xs]
ax.bar(xs, counts, width=0.06, color=barcolors, edgecolor="none")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(0, 28)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([0, 10, 20])
ax.set_xlabel("Event − Global quartet concordance")
ax.set_ylabel("Mixed quartets (count)")
ax.text(0.03, 0.94, f"All {len(mixed)} mixed quartets", transform=ax.transAxes, va="top", fontsize=7)

# c: Exact descriptive research thresholds, no optimization on held-out data.
ax = axs[1, 0]
decorate(ax, "c", "Pooled costs at fixed probabilities")
for model in MODELS:
    rows = decisions.loc[decisions.scope.eq("pooled") & decisions.model.eq(model)].sort_values("threshold")
    assert rows.threshold.tolist() == THRESHOLDS
    ax.plot(100 * rows.coverage, 100 * rows.retained_failure_fraction, color=COLORS[model],
            lw=1.1 if model == "Event" else 0.8)
    for _, row in rows.iterrows():
        ax.scatter(100 * row.coverage, 100 * row.retained_failure_fraction,
                   color=COLORS[model], marker=MARKERS[row.threshold], s=21, zorder=4)
ax.set_xlim(59.5, 73)
ax.set_ylim(0, 6.1)
ax.set_xticks([60, 65, 70])
ax.set_yticks([0, 2, 4, 6])
ax.set_xlabel("Retained / all episodes (%)")
ax.set_ylabel("Failures / retained episodes (%)")
threshold_handles = [Line2D([0], [0], color="#343C44", marker=MARKERS[t],
                            lw=0, markersize=4, label=f"{t:.2f}") for t in THRESHOLDS]
ax.legend(handles=threshold_handles, title="Maximum probability", title_fontsize=7,
          loc="upper left", ncol=3, handletextpad=0.4, columnspacing=0.65, borderaxespad=0.15)

# d: The same Event probability threshold exhibits source heterogeneity.
ax = axs[1, 1]
decorate(ax, "d", "Source costs for Event at p ≤ 0.10")
rows = decisions.loc[decisions.scope.eq("source") & decisions.model.eq("Event") & decisions.threshold.eq(0.1)]
assert len(rows) == 4 and rows.episodes.eq(1024).all()
offsets = {"G1": (7, 4), "G2": (7, -2), "G3": (-6, 10), "G4": (7, 1)}
for _, row in rows.iterrows():
    x, y = 100 * row.coverage, 100 * row.retained_failure_fraction
    ax.scatter(x, y, color=COLORS["Event"], marker="s", s=23, zorder=4)
    dx, dy = offsets[row.stratum]
    ax.annotate(f"{row.stratum}  {row.retained_failures}/{row.retained}",
                xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha="right" if dx < 0 else "left", va="bottom", fontsize=7)
ax.set_xlim(59.5, 82)
ax.set_ylim(-0.3, 4.8)
ax.set_xticks([60, 65, 70, 75, 80])
ax.set_yticks([0, 2, 4])
ax.set_xlabel("Retained / all source episodes (%)")
ax.set_ylabel("Failures / retained source episodes (%)")

fig.canvas.draw()
require_matplotlib_panel_alignment(
    fig, json_out=str(BASE) + ".alignment.json", overlay_svg=str(BASE) + ".alignment.svg",
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
)
fig.savefig(str(BASE) + ".pdf")
fig.savefig(str(BASE) + ".svg")
fig.savefig(str(BASE) + ".png", dpi=600)
fig.savefig(str(BASE) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(str(BASE) + "_preview.png", dpi=300)
plt.close(fig)

provenance = {
    "backend": "Python/matplotlib",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob("*.csv"))},
    "eligible_matched_quartets": int(len(mixed)),
    "difference_distribution_counts": {str(k): int(v) for k, v in distribution.items()},
    "mean_paired_difference": float(diff.mean()),
    "excluded_from_discrimination": "Nonmixed quartets have no failure/nonfailure pair; no episode rows removed from source tables.",
    "uncertainty": "No IID confidence interval or hypothesis test; reused replay components and one coded recording series.",
}
(OUT / "Fig_S31_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"output": str(BASE), "mean_paired_difference": float(diff.mean()),
                  "positive": int(diff.gt(0).sum()), "zero": int(diff.eq(0).sum()),
                  "negative": int(diff.lt(0).sum())}))
