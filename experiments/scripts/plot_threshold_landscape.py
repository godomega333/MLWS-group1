from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("experiments") / "results"
FIGURES_DIR = Path("experiments") / "figures"
SWEEP_PATH = RESULTS_DIR / "threshold_sweep_oof.csv"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SWEEP_PATH).sort_values("threshold").reset_index(drop=True)
    df["rank"] = df["oof_acc"].rank(ascending=False, method="min").astype(int)
    df["rank_smooth"] = df["rank"].rolling(window=5, center=True, min_periods=1).mean()

    lb_points = pd.DataFrame({
        "threshold": [0.4635, 0.4725, 0.5000],
        "lb": [0.81762, 0.81973, 0.81879],
        "label": ["control 0.4635", "anchor 0.4725", "pre-Stage6 0.5"],
    })

    fig, ax_rank = plt.subplots(figsize=(13, 7.5))
    ax_rank.plot(
        df["threshold"],
        df["rank"],
        color="lightsteelblue",
        alpha=0.55,
        linewidth=0.9,
        label="OOF rank (raw, 301 thresholds)",
    )
    ax_rank.plot(
        df["threshold"],
        df["rank_smooth"],
        color="steelblue",
        linewidth=2.4,
        label="OOF rank (rolling mean, window=5)",
    )
    ax_rank.set_xlabel("Threshold", fontsize=12)
    ax_rank.set_ylabel("OOF rank (1 = best, 301 = worst)", color="steelblue", fontsize=12)
    ax_rank.invert_yaxis()
    ax_rank.set_xlim(0.40, 0.55)
    ax_rank.tick_params(axis="y", labelcolor="steelblue")
    ax_rank.axvspan(0.4575, 0.4740, alpha=0.13, color="seagreen", label="OOF-high plateau")
    ax_rank.axvline(0.4745, color="firebrick", linestyle="--", alpha=0.7, linewidth=1.6, label="OOF cliff")
    ax_rank.axvline(0.4725, color="black", linestyle="-", alpha=0.9, linewidth=2.0, label="Chosen 0.4725")
    ax_rank.grid(True, alpha=0.25)

    ax_lb = ax_rank.twinx()
    ax_lb.plot(lb_points["threshold"], lb_points["lb"], marker="o", markersize=14, color="darkred", linewidth=2.4, label="Public LB")
    for _, row in lb_points.iterrows():
        ax_lb.annotate(
            f"{row['lb']:.5f}\n({row['label']})",
            (row["threshold"], row["lb"]),
            textcoords="offset points",
            xytext=(10, 8),
            color="darkred",
            fontsize=10,
            fontweight="bold",
        )
    ax_lb.set_ylabel("Public LB", color="darkred", fontsize=12)
    ax_lb.tick_params(axis="y", labelcolor="darkred")
    ax_lb.set_ylim(0.8170, 0.8205)

    lines_rank, labels_rank = ax_rank.get_legend_handles_labels()
    lines_lb, labels_lb = ax_lb.get_legend_handles_labels()
    ax_rank.legend(lines_rank + lines_lb, labels_rank + labels_lb, loc="lower left", fontsize=9.5, framealpha=0.92)

    plt.title(
        "Threshold landscape: OOF rank versus Public LB\n"
        "CatBoost locked: iter=1358, lr=0.0666, depth=5, l2=8.036, bag_t=0.424, border=211, seed=78",
        fontsize=11,
    )
    plt.tight_layout()
    output_path = FIGURES_DIR / "threshold_landscape.png"
    plt.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
