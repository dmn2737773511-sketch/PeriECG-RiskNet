"""Fig. S32: annotated continuous ECG baseline and matched replay outcomes.

All target windows are included in model training. The clean-success analysis
is a conditional subset of the original out-of-fold predictions, not a refit.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SKILL_SCRIPTS = Path(__file__).resolve().parent / "qa_tools"
sys.path.insert(0, str(SKILL_SCRIPTS))
from panel_alignment import require_matplotlib_panel_alignment

OUT = Path(__file__).resolve().parent
DATA = OUT / "public_source_data"
BASE = OUT / "Fig_S32"
COLORS = {"Global": "#727A83", "Event": "#2366A5", "Combined": "#AC7B42", "Agreement": "#75558B"}
MODELS = ["Global", "Event", "Combined", "Agreement"]
SUBSETS = ["all_targets", "clean_primary_success"]
LABELS = {"all_targets": "All targets", "clean_primary_success": "Clean-primary success"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.65, "xtick.major.width": 0.65, "ytick.major.width": 0.65,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "legend.frameon": False, "savefig.facecolor": "white",
})

clean = pd.read_csv(DATA / "clean_targets.csv")
quartets = pd.read_csv(DATA / "matched_quartets.csv")
summary = pd.read_csv(DATA / "quartet_summary.csv")
decisions = pd.read_csv(DATA / "fixed_threshold_retention.csv")
pred = pd.read_csv(DATA / "heldout_predictions.csv")
assert len(clean) == 192 and clean.target.nunique() == 192
assert clean.patient_group.nunique() == 47
assert len(pred) == 18432 and len(quartets) == 4608
assert clean.baseline_primary_success.eq(clean.clean_f1.ge(0.95).astype(int)).all()
assert clean.baseline_primary_success.sum() == 170
assert pred.groupby(["target", "residual", "snr_db"]).size().eq(4).all()
assert pred.groupby("target").size().eq(96).all()
assert pred.shift_samples.isin([0, 69, 196, 367]).all()
assert quartets.mixed_failure.eq(quartets.n_failure.between(1, 3).astype(int)).all()
assert quartets.patient_group.nunique() == 47
for _, row in decisions.iterrows():
    assert np.isclose(row.coverage, row.n_retained / row.n_total)
    assert np.isclose(row.failure_rate_retained, row.n_retained_failure / row.n_retained)
    mask = np.ones(len(pred), bool) if row.subset == "all_targets" else pred.baseline_primary_success.eq(1).to_numpy()
    mask &= pred["risk_" + row.method].le(row.risk_threshold).to_numpy()
    assert int(mask.sum()) == row.n_retained
    assert int(pred.loc[mask, "failure"].sum()) == row.n_retained_failure

eq_rows = []
for subset in SUBSETS:
    q = quartets if subset == "all_targets" else quartets.loc[quartets.baseline_primary_success.eq(1)]
    mixed = q.loc[q.mixed_failure.eq(1)]
    assert len(mixed) == (477 if subset == "all_targets" else 394)
    for model in MODELS:
        vals = mixed["pair_concordance_" + model].to_numpy(float)
        assert np.isfinite(vals).all() and ((vals >= 0) & (vals <= 1)).all()
        eq_rows.append({"subset": subset, "model": model, "n_quartets": len(q),
                        "n_mixed_quartets": len(mixed),
                        "n_eligible_pairs": int(mixed.n_success_failure_pairs.sum()),
                        "equal_quartet_concordance": float(vals.mean())})
equal = pd.DataFrame(eq_rows)
equal.to_csv(OUT / "Fig_S32_equal_quartet_concordance.csv", index=False)

# Final publication dimensions: 183 × 145 mm.
fig, axs = plt.subplots(2, 2, figsize=(7.2047244094, 5.7086614173))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.115, top=0.925, wspace=0.38, hspace=0.62)

def decorate(ax, letter, title):
    ax.set_label(letter)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-29, 17),
                textcoords="offset points", ha="left", va="bottom", weight="bold", fontsize=9)
    ax.set_title(title, loc="left", pad=12)
    ax.tick_params(direction="out")

subset_handles = [Line2D([0], [0], marker="o", color="#525B63", lw=0,
                         markerfacecolor="#525B63" if sub == "all_targets" else "white",
                         markersize=4, label=LABELS[sub]) for sub in SUBSETS]

# a: Complete baseline distribution, including all zero or low F1 windows.
ax = axs[0, 0]
decorate(ax, "a", "Clean detector performance has failures")
for col, label, color, ls in [("clean_f1", "Primary", COLORS["Event"], "-"),
                              ("clean_f1_amplitude", "Amplitude", COLORS["Global"], "--")]:
    x = np.sort(clean[col].to_numpy(float))
    y = 100 * np.arange(1, len(x) + 1) / len(x)
    ax.step(np.r_[-0.02, x, 1.02], np.r_[0, y, 100], where="post",
            color=color, ls=ls, lw=1.15, label=label)
ax.axvline(0.95, color="#A2A8AE", ls=":", lw=0.75)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 103)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("Clean-window beat F1")
ax.set_ylabel("Windows at or below F1 (%)")
ax.legend(loc="upper left", borderaxespad=0.15)
ax.text(0.03, 0.56, "Primary < 0.95: 22/192\nAmplitude < 0.95: 17/192",
        transform=ax.transAxes, va="top", linespacing=1.6, fontsize=7)
ax.text(0.91, 94, "0.95", ha="right", va="center", fontsize=7)

# b: Label crossing after controlled rotation, keeping the denominators explicit.
ax = axs[0, 1]
decorate(ax, "b", "Timing changes failure status")
snrs = [20.0, 10.0, 0.0, -5.0]
for j, sub in enumerate(SUBSETS):
    rows = summary.loc[summary.subset.eq(sub) & summary.snr_db.ne("all")].copy()
    rows["snr_db"] = rows.snr_db.astype(float)
    rows = rows.set_index("snr_db").loc[snrs]
    positions = np.arange(4) + (-0.18 if j == 0 else 0.18)
    heights = rows.mixed_fraction.to_numpy() * 100
    ax.bar(positions, heights, width=0.30,
           color="#818B94" if j == 0 else "white", edgecolor="#535F69", lw=0.65)
    for x, y, n in zip(positions, heights, rows.n_mixed_quartets):
        ax.text(x, y + 0.65, str(int(n)), ha="center", va="bottom", fontsize=7)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0, 33)
ax.set_xticks(range(4), ["20", "10", "0", "−5"])
ax.set_yticks([0, 10, 20, 30])
ax.set_xlabel("Target / noise ratio (dB)")
ax.set_ylabel("Mixed / all matched quartets (%)")
bar_handles = [Patch(facecolor="#818B94" if sub == "all_targets" else "white",
                     edgecolor="#535F69", lw=0.65, label=LABELS[sub]) for sub in SUBSETS]
ax.legend(handles=bar_handles, loc="upper left", borderaxespad=0.15, handletextpad=0.5)

# c: The stricter matched question retains the untrained ranking comparator.
ax = axs[1, 0]
decorate(ax, "c", "Matched ranking retains a strong control")
for i, model in enumerate(MODELS):
    rows = equal.loc[equal.model.eq(model)].set_index("subset").loc[SUBSETS]
    y = rows.equal_quartet_concordance.to_numpy()
    xs = [i - 0.12, i + 0.12]
    ax.plot(xs, y, color=COLORS[model], lw=0.85)
    for j, sub in enumerate(SUBSETS):
        ax.scatter(xs[j], y[j], s=26, facecolor=COLORS[model] if j == 0 else "white",
                   edgecolor=COLORS[model], lw=1.0, zorder=4)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0.48, 0.85)
ax.set_yticks([0.5, 0.6, 0.7, 0.8])
ax.set_xticks(range(4), ["Global", "Event", "Combined", "Agreement"])
ax.set_xlabel("Reliability measure")
ax.set_ylabel("Mean quartet concordance")
ax.legend(handles=subset_handles, loc="upper left", borderaxespad=0.15, handletextpad=0.5)

# d: Same research threshold; probabilities only, without any test-set refit.
ax = axs[1, 1]
decorate(ax, "d", "Retention costs for p ≤ 0.10")
for model in MODELS[:3]:
    rows = decisions.loc[decisions.method.eq(model)].set_index("subset").loc[SUBSETS]
    x = 100 * rows.coverage.to_numpy()
    y = 100 * rows.failure_rate_retained.to_numpy()
    ax.plot(x, y, color=COLORS[model], lw=0.85)
    for j, sub in enumerate(SUBSETS):
        ax.scatter(x[j], y[j], s=26, facecolor=COLORS[model] if j == 0 else "white",
                   edgecolor=COLORS[model], lw=1.0, zorder=4)
    # Direct labels use different horizontal anchors for nearby method points.
    offsets = {"Global": (0, 6, "center"), "Event": (-8, 6, "right"),
               "Combined": (8, 6, "left")}
    dx, dy, ha = offsets[model]
    ax.annotate(model, xy=(x[0], y[0]), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom", color=COLORS[model], fontsize=7)
ax.set_xlim(25, 48)
ax.set_ylim(0, 8.3)
ax.set_xticks([25, 30, 35, 40, 45])
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_xlabel("Retained / subset episodes (%)")
ax.set_ylabel("Failures / retained episodes (%)")
ax.legend(handles=subset_handles, loc="upper right", borderaxespad=0.15, handletextpad=0.5)

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
    "new_computation": "Equal-quartet means of existing within-quartet concordance values; no model fitting or threshold optimization.",
    "equal_quartet_concordance": eq_rows,
    "clean_primary_failure_windows": int(clean.clean_f1.lt(0.95).sum()),
    "clean_amplitude_failure_windows": int(clean.clean_f1_amplitude.lt(0.95).sum()),
    "training": "All 192 targets; 47 leave-one-patient-group-out folds; same six noise windows shared across folds.",
    "conditional_subset": "170 clean-primary-success targets select existing OOF results only.",
    "uncertainty": "No IID interval or p value; reused replay components.",
}
(OUT / "Fig_S32_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"output": str(BASE), "clean_windows": len(clean), "mixed_quartets": [477, 394]}))
