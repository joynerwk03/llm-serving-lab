"""README chart: where one batch-1 decode step goes (FINDINGS.md sections 3-4).

Horizontal stacked bars, busy vs idle per step, light and dark versions.
Palette: dataviz reference slots 1-2 (validated both modes, adjacent pairs).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))  # writes next to this script

# (label, GPU busy ms, GPU idle ms) from the torch-profiler analysis (LOG.md)
ROWS = [
    ("vLLM, WSL2 default", 7.56, 1.29),
    ("vLLM, pinned memory on", 7.61, 0.16),
    ("SGLang", 7.15, 0.14),
]
FLOOR = 5.19  # 4.86 GB read per step / 936 GB/s

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e6e5e1",
                  busy="#2a78d6", idle="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", grid="#383835",
                 busy="#3987e5", idle="#d95926"),
}

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def draw(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    h = 0.52
    gap = 0.04          # ~2px surface gap between the two segments at this scale
    xmax = 10.0
    for i, (label, busy, idle) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        # busy segment: square at the baseline
        ax.add_patch(Rectangle((0, y - h / 2), busy - gap / 2, h, facecolor=t["busy"],
                               edgecolor="none", zorder=3))
        # idle segment: rounded outer end, squared inner end
        x0 = busy + gap / 2
        w = max(idle - gap / 2, 0.02)
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                                    mutation_aspect=1 / 9, facecolor=t["idle"], edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x0, y - h / 2), min(w, 0.06), h, facecolor=t["idle"],
                               edgecolor="none", zorder=3))
        total = busy + idle
        ax.text(total + 0.12, y, f"{total:.2f} ms", va="center", ha="left", fontsize=10.5,
                color=t["text"], fontweight="bold", zorder=4)
        ax.text(-0.15, y, label, va="center", ha="right", fontsize=10.5, color=t["text"])
    # the story's number, labelled directly on the first bar
    b0, i0 = ROWS[0][1], ROWS[0][2]
    ax.annotate("idle 1.29 ms", xy=(b0 + i0 / 2, len(ROWS) - 1 + h / 2), xytext=(b0 + i0 / 2, len(ROWS) - 1 + 0.55),
                ha="center", va="bottom", fontsize=9.5, color=t["text2"], zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", facecolor=t["surface"], edgecolor="none"),
                arrowprops=dict(arrowstyle="-", color=t["text2"], lw=0.8))
    # physical floor
    ax.axvline(FLOOR, color=t["text2"], lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.text(FLOOR - 0.08, -0.62, "memory-bandwidth floor, 5.2 ms", ha="right", va="center",
            fontsize=9, color=t["text2"])
    # axes: recessive
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.85, len(ROWS) - 0.2)
    ax.set_yticks([])
    ax.set_xticks(range(0, 11, 2))
    ax.tick_params(axis="x", colors=t["text2"], labelsize=9, length=0, pad=4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
    ax.set_xlabel("milliseconds per output token, one request", color=t["text2"], fontsize=9.5, labelpad=6)
    # legend (2 series) + title
    handles = [Rectangle((0, 0), 1, 1, facecolor=t["busy"]), Rectangle((0, 0), 1, 1, facecolor=t["idle"])]
    leg = fig.legend(handles, ["GPU busy", "GPU idle"], loc="upper right", bbox_to_anchor=(0.985, 0.985),
                     ncol=2, frameon=False, fontsize=9.5, handlelength=1.1, handleheight=0.9, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    fig.text(0.012, 0.95, "Where one decoding step goes", fontsize=13, fontweight="bold",
             color=t["text"], ha="left", va="top")
    fig.text(0.012, 0.865, "Nearly the same GPU work in both engines; the difference is mostly time the GPU sits idle.",
             fontsize=9.5, color=t["text2"], ha="left", va="top")
    fig.subplots_adjust(left=0.27, right=0.9, top=0.74, bottom=0.2)
    path = os.path.join(OUT, f"decode-step-{mode}.png")
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", path)


for m in ("light", "dark"):
    draw(m)
