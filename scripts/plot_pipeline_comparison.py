"""Plot all B×G pipeline configs in one comparison image.

Shows 3 micro-batches flowing through each config with GPU 0 (gen) on top
and GPU 1 (train) on bottom. Config labels and times only, no analysis text.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Measured data from all experiments on 2× RTX PRO 4500 Blackwell
CONFIGS = [
    {
        "label": "B×G=4  (B=1, G=4, 512 tok)",
        "t_gen": 1691,     # ms per batch (gen + reward + ref)
        "t_train": 120,    # ms per batch (fwd + bwd)
        "t_opt": 38,       # ms optimizer step
        "t_sync": 91,      # ms weight sync
        "t_ref": 190,      # included in gen phase on GPU 0
        "t_reward": 10,
    },
    {
        "label": "B×G=16 (B=4, G=4, 512 tok)",
        "t_gen": 2002,
        "t_train": 539,
        "t_opt": 120,
        "t_sync": 95,
        "t_ref": 190,
        "t_reward": 15,
    },
    {
        "label": "B×G=32 (B=8, G=4, 512 tok)",
        "t_gen": 2411,
        "t_train": 1800,
        "t_opt": 300,
        "t_sync": 99,
        "t_ref": 190,
        "t_reward": 20,
    },
    {
        "label": "B×G=48 (B=12, G=4, 384 tok)",
        "t_gen": 2187,
        "t_train": 2200,
        "t_opt": 400,
        "t_sync": 100,
        "t_ref": 190,
        "t_reward": 20,
    },
    {
        "label": "B×G=56 (B=14, G=4, 256 tok)",
        "t_gen": 1466,
        "t_train": 1900,
        "t_opt": 550,
        "t_sync": 97,
        "t_ref": 190,
        "t_reward": 20,
    },
]

N_BATCHES = 3  # show 3 batches flowing through

# Colors
C_GEN = "#4285F4"       # blue - generation (vLLM)
C_REF = "#EA4335"        # red - ref forward
C_REWARD = "#FBBC04"     # yellow - reward
C_FWD = "#34A853"        # green - policy forward
C_BWD = "#A1522A"        # brown - backward
C_OPT = "#9C27B0"        # purple - optimizer
C_SYNC = "#78909C"       # gray-blue - weight sync
C_IDLE = "#E0E0E0"       # light gray - idle/waiting

BAR_HEIGHT = 0.35


def draw_config(ax_gen, ax_train, cfg):
    """Draw pipeline timeline for one config on two axes."""
    t_gen_total = cfg["t_gen"]  # total gen GPU time per batch
    t_gen_core = t_gen_total - cfg["t_ref"] - cfg["t_reward"]
    t_train_total = cfg["t_train"]
    t_opt = cfg["t_opt"]
    t_sync = cfg["t_sync"]

    # Split train into fwd (~25%) and bwd (~75%)
    t_fwd = int(t_train_total * 0.25)
    t_bwd = t_train_total - t_fwd

    # --- GPU 0 (gen): batches are back-to-back ---
    gen_starts = []
    gen_ends = []
    t = 0
    for i in range(N_BATCHES):
        start = t
        gen_starts.append(start)
        # gen core
        ax_gen.barh(0, t_gen_core, left=t, height=BAR_HEIGHT, color=C_GEN, edgecolor="white", linewidth=0.5)
        t += t_gen_core
        # reward
        ax_gen.barh(0, cfg["t_reward"], left=t, height=BAR_HEIGHT, color=C_REWARD, edgecolor="white", linewidth=0.5)
        t += cfg["t_reward"]
        # ref forward
        ax_gen.barh(0, cfg["t_ref"], left=t, height=BAR_HEIGHT, color=C_REF, edgecolor="white", linewidth=0.5)
        t += cfg["t_ref"]
        gen_ends.append(t)

    # Sync after last batch
    ax_gen.barh(0, t_sync, left=t, height=BAR_HEIGHT, color=C_SYNC, edgecolor="white", linewidth=0.5)
    total_time = t + t_sync

    # --- GPU 1 (train): starts when each batch arrives from gen ---
    train_cursor = 0
    for i in range(N_BATCHES):
        batch_ready = gen_ends[i]
        # Idle time waiting for batch
        if batch_ready > train_cursor:
            idle = batch_ready - train_cursor
            ax_train.barh(0, idle, left=train_cursor, height=BAR_HEIGHT, color=C_IDLE, edgecolor="white", linewidth=0.5)
            train_cursor = batch_ready

        # Forward
        ax_train.barh(0, t_fwd, left=train_cursor, height=BAR_HEIGHT, color=C_FWD, edgecolor="white", linewidth=0.5)
        train_cursor += t_fwd
        # Backward
        ax_train.barh(0, t_bwd, left=train_cursor, height=BAR_HEIGHT, color=C_BWD, edgecolor="white", linewidth=0.5)
        train_cursor += t_bwd

    # Optimizer step
    ax_train.barh(0, t_opt, left=train_cursor, height=BAR_HEIGHT, color=C_OPT, edgecolor="white", linewidth=0.5)
    train_cursor += t_opt

    total_time = max(total_time, train_cursor)

    # Dotted lines: batch handoff from gen to train
    for i in range(N_BATCHES):
        for ax in [ax_gen, ax_train]:
            ax.axvline(gen_ends[i], color="#999999", linestyle=":", linewidth=0.7, alpha=0.5)

    # Format axes
    for ax in [ax_gen, ax_train]:
        ax.set_xlim(0, total_time * 1.02)
        ax.set_yticks([0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    ax_gen.set_yticklabels(["GPU 0\n(gen)"], fontsize=7)
    ax_train.set_yticklabels(["GPU 1\n(train)"], fontsize=7)
    ax_gen.set_xticklabels([])
    ax_train.set_xlabel("Time (ms)", fontsize=7)
    ax_train.tick_params(axis="x", labelsize=6)

    # Title with config and times
    ratio = t_gen_total / max(t_train_total, 1)
    ax_gen.set_title(
        f"{cfg['label']}   —   T_gen={t_gen_total}ms  T_train={t_train_total}ms  ratio={ratio:.1f}×",
        fontsize=8, fontweight="bold", pad=4,
    )


def main():
    n = len(CONFIGS)
    fig, axes = plt.subplots(n * 2, 1, figsize=(14, n * 1.6 + 0.8),
                              gridspec_kw={"hspace": 0.15})

    for i, cfg in enumerate(CONFIGS):
        ax_gen = axes[i * 2]
        ax_train = axes[i * 2 + 1]
        draw_config(ax_gen, ax_train, cfg)

        # Add separator line between configs
        if i < n - 1:
            ax_train.axhline(-0.8, color="#CCCCCC", linewidth=0.5)

    # Legend at bottom
    legend_items = [
        mpatches.Patch(color=C_GEN, label="Generation (vLLM)"),
        mpatches.Patch(color=C_REF, label="Ref forward"),
        mpatches.Patch(color=C_REWARD, label="Reward"),
        mpatches.Patch(color=C_FWD, label="Policy forward"),
        mpatches.Patch(color=C_BWD, label="Backward"),
        mpatches.Patch(color=C_OPT, label="Optimizer step"),
        mpatches.Patch(color=C_SYNC, label="Weight sync"),
        mpatches.Patch(color=C_IDLE, label="Idle (waiting)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=8, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Disaggregated Pipeline: 2× RTX PRO 4500 Blackwell — All B×G Configs",
                 fontsize=11, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = "results/pipeline_all_configs.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
