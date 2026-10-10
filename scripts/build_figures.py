# -*- coding: utf-8 -*-
"""One-off script: builds publication-quality figures (300 DPI) from the
pipeline's actual output data, using the validated categorical palette."""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- validated categorical palette (light mode), fixed slot order ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

OUT = "D:/Desktop/endo-fibro-lit-mining/figures"
import os
os.makedirs(OUT, exist_ok=True)


def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ============================================================
# Figure 1: corpus scale + false-positive audit
# ============================================================
corpora = ["Joint\n(n=97 fetched)", "Endometriosis-only\n(n=35,777 fetched)", "Fibromyalgia-only\n(n=15,486 fetched)"]
kept = [91, 34732, 15217]
excluded = [6, 1045, 269]

fig, ax = plt.subplots(figsize=(7, 4.3))
y = range(len(corpora))
ax.barh(y, kept, color=BLUE, label="Retained (literal disease-term match)", height=0.5)
ax.barh(y, excluded, left=kept, color=CRITICAL, label="Excluded (PubMed term-mapping false positive)", height=0.5)
ax.set_xscale("log")
ax.set_xlim(50, 200000)
ax.set_yticks(list(y))
ax.set_yticklabels(corpora)
ax.invert_yaxis()
ax.set_xlabel("Number of records (log scale)")
ax.set_title("Corpus scale and false-positive audit", fontsize=11, fontweight="bold", loc="left")
for i, (k, e) in enumerate(zip(kept, excluded)):
    pct = e / (k + e) * 100
    ax.text(k + e, i, f"  final n={k:,}  ({pct:.1f}% excluded)", va="center", fontsize=8.5, color=INK_SECONDARY)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=GRIDLINE, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, fontsize=8.5)
fig.tight_layout()
savefig(fig, "fig1_corpus_audit")

# ============================================================
# Figure 2: extraction breakdown (comorbidity focus + mechanism)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))

# 2a: comorbidity focus
focus_labels = ["Primary focus", "Secondary mention", "Incidental co-mention"]
focus_vals = [16, 31, 44]
colors_a = [BLUE, AQUA, INK_MUTED]
axa = axes[0]
bars = axa.bar(focus_labels, focus_vals, color=colors_a, width=0.6)
for b, v in zip(bars, focus_vals):
    axa.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v}\n({v/91*100:.1f}%)", ha="center", fontsize=8.5, color=INK_SECONDARY)
axa.set_ylabel("Papers (of 91)")
axa.set_title("(a) How central is the comorbidity\nto each paper?", fontsize=10, fontweight="bold", loc="left")
axa.spines[["top", "right"]].set_visible(False)
axa.set_ylim(0, 52)
axa.tick_params(axis="x", labelsize=8.5)

# 2b: mechanism
mech_labels = ["Not\nspecified", "Central\nsensitization", "Autoimmune", "Other", "Nociplastic\npain", "Hormonal", "Neuropathic/\nsmall-fiber", "Psychiatric"]
mech_vals = [48, 26, 4, 4, 3, 3, 2, 1]
axb = axes[1]
bars2 = axb.bar(mech_labels, mech_vals, color=BLUE, width=0.6)
for b, v in zip(bars2, mech_vals):
    axb.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=8.5, color=INK_SECONDARY)
axb.set_ylabel("Papers (of 91)")
axb.set_title("(b) Named mechanism linking the\ntwo conditions", fontsize=10, fontweight="bold", loc="left")
axb.spines[["top", "right"]].set_visible(False)
axb.tick_params(axis="x", labelsize=7.5, rotation=0)
axb.set_ylim(0, 54)

for ax in axes:
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

fig.tight_layout()
savefig(fig, "fig2_extraction_breakdown")

# ============================================================
# Figure 3: topic model (BERTopic) sizes
# ============================================================
d = json.load(open("D:/Desktop/endo-fibro-lit-mining/data/topics/joint_bertopic_topics.json", encoding="utf-8"))
topics = sorted(d["topics"], key=lambda t: t["size"])
labels = [", ".join(t["top_words"][:4]) for t in topics]
sizes = [t["size"] for t in topics]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.barh(range(len(topics)), sizes, color=BLUE, height=0.6)
ax.set_yticks(range(len(topics)))
ax.set_yticklabels(labels, fontsize=8.5)
for i, v in enumerate(sizes):
    ax.text(v + 0.15, i, str(v), va="center", fontsize=8.5, color=INK_SECONDARY)
ax.set_xlabel("Papers assigned to topic (of 91)")
ax.set_title("Topic structure of the joint corpus (BERTopic)", fontsize=11, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=GRIDLINE, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
savefig(fig, "fig3_topic_model")

# ============================================================
# Figure 4: validation reliability (Cohen's kappa per field)
# ============================================================
fields = ["Sample size\n(exact match)", "Mechanism", "Study design", "Comorbidity\nfocus", "Stance", "FM measure\ntype"]
kappas = [None, 0.894, 0.746, 0.690, 0.478, 0.318]  # sample_size reported as % match instead
pct = [100.0, 93.3, 80.0, 80.0, 73.3, 80.0]

fig, ax = plt.subplots(figsize=(7.5, 4))
x = range(len(fields))
bar_colors = [AQUA if k is None else (BLUE if k >= 0.6 else (YELLOW if k >= 0.4 else CRITICAL)) for k in kappas]
plot_vals = [1.0 if k is None else k for k in kappas]
bars = ax.bar(x, plot_vals, color=bar_colors, width=0.6)
for i, (k, p) in enumerate(zip(kappas, pct)):
    label = f"{p:.0f}% exact match" if k is None else f"kappa={k:.2f}\n({p:.0f}% agree)"
    ax.text(i, plot_vals[i] + 0.03, label, ha="center", fontsize=8, color=INK_SECONDARY)
ax.set_xticks(list(x))
ax.set_xticklabels(fields, fontsize=8.5)
ax.set_ylabel("Cohen's kappa (chance-corrected agreement)")
ax.set_ylim(0, 1.15)
ax.axhline(0.81, color=BASELINE, linewidth=0.8, linestyle="--")
ax.axhline(0.61, color=BASELINE, linewidth=0.8, linestyle="--")
ax.axhline(0.41, color=BASELINE, linewidth=0.8, linestyle="--")
ax.text(5.55, 0.81, "almost perfect", fontsize=7, color=INK_MUTED, va="bottom")
ax.text(5.55, 0.61, "substantial", fontsize=7, color=INK_MUTED, va="bottom")
ax.text(5.55, 0.41, "moderate", fontsize=7, color=INK_MUTED, va="bottom")
ax.set_title("Extraction reliability vs. independent human annotation (n=15)", fontsize=11, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
savefig(fig, "fig4_validation")

# ============================================================
# Figure 5 (flagship): bipartite network of confirmed cross-literature connections
# ============================================================
conn = json.load(open("D:/Desktop/endo-fibro-lit-mining/data/discovery/connections.json", encoding="utf-8"))
confirmed = conn["confirmed_connections"]

# group by a coarse mechanism bucket for edge coloring
def bucket(mech):
    m = mech.lower()
    if "nlrp3" in m or "inflammasome" in m or "pyroptosis" in m:
        return "NLRP3 inflammasome", ORANGE
    if "gut" in m or "microbiome" in m or "microbiota" in m:
        return "Gut-brain / microbiome", AQUA
    if "igf" in m or "growth hormone" in m:
        return "GH/IGF-1 signaling", VIOLET
    if "myofascial" in m or "trigger point" in m:
        return "Myofascial trigger points", MAGENTA
    if "migraine" in m or "hyperalgesia" in m:
        return "Migraine / hyperalgesia", YELLOW
    return "Central sensitization / nociplastic pain", BLUE

fm_nodes, endo_nodes = [], []
fm_index, endo_index = {}, {}
edges = []
for c in confirmed:
    fp, ep = c["fibro_pmid"], c["endo_pmid"]
    if fp not in fm_index:
        fm_index[fp] = len(fm_nodes)
        fm_nodes.append(fp)
    if ep not in endo_index:
        endo_index[ep] = len(endo_nodes)
        endo_nodes.append(ep)
    label, color = bucket(c["shared_mechanism"])
    edges.append((fm_index[fp], endo_index[ep], color, label, c["confidence"]))

n_fm, n_endo = len(fm_nodes), len(endo_nodes)
fig, ax = plt.subplots(figsize=(8, 7.5))

fm_y = [i for i in range(n_fm)]
fm_y = [1 - (i / max(n_fm - 1, 1)) for i in range(n_fm)]
endo_y = [1 - (i / max(n_endo - 1, 1)) for i in range(n_endo)]

x_fm, x_endo = 0.0, 1.0

seen_labels = {}
for fi, ei, color, label, conf in edges:
    y1, y2 = fm_y[fi], endo_y[ei]
    ax.plot([x_fm, x_endo], [y1, y2], color=color, linewidth=0.9 + 2.2 * conf, alpha=0.55, zorder=1,
             solid_capstyle="round")
    seen_labels[label] = color

ax.scatter([x_fm] * n_fm, fm_y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
ax.scatter([x_endo] * n_endo, endo_y, s=70, color=VIOLET, edgecolor=SURFACE, linewidth=1.2, zorder=3)

ax.text(x_fm, 1.06, f"Fibromyalgia literature\n(n={n_fm} papers)", ha="center", fontsize=10, fontweight="bold", color=INK_PRIMARY)
ax.text(x_endo, 1.06, f"Endometriosis literature\n(n={n_endo} papers)", ha="center", fontsize=10, fontweight="bold", color=INK_PRIMARY)

ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.08, 1.14)
ax.axis("off")
ax.set_title(
    f"22 verified latent connections between two literatures that never cite each other\n"
    f"(edge thickness = model confidence; color = shared mechanism)",
    fontsize=11, fontweight="bold", pad=10
)

legend_handles = [Line2D([0], [0], color=c, lw=3, label=l) for l, c in seen_labels.items()]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8)

fig.tight_layout()
savefig(fig, "fig5_discovery_network")

print("\nAll figures saved to", OUT)
