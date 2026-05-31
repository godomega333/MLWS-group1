from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("experiments") / "results"
SWEEP_PATH = RESULTS_DIR / "threshold_sweep_oof.csv"


def main() -> None:
    df = pd.read_csv(SWEEP_PATH)
    print("columns:", list(df.columns))
    print("total thresholds swept:", len(df))

    threshold_col = next(c for c in df.columns if "thresh" in c.lower())
    acc_col = next(c for c in df.columns if "acc" in c.lower())
    print(f"using threshold_col='{threshold_col}', acc_col='{acc_col}'\n")

    ranked = df.sort_values(acc_col, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    n_rows = len(ranked)

    target = ranked[(ranked[threshold_col] - 0.4725).abs() < 1e-6]
    if len(target) == 0:
        ranked["distance_to_target"] = (ranked[threshold_col] - 0.4725).abs()
        target = ranked.nsmallest(1, "distance_to_target")

    target_rank = int(target["rank"].iloc[0])
    target_acc = float(target[acc_col].iloc[0])
    best_acc = float(ranked[acc_col].iloc[0])

    print("=== Rank of threshold 0.4725 in the OOF sweep ===")
    print(target[["rank", threshold_col, acc_col]].to_string(index=False))
    print(f"rank {target_rank} / {n_rows} = top {target_rank / n_rows * 100:.2f}%")
    print(
        f"OOF acc gap vs argmax = {target_acc - best_acc:+.5f} "
        f"(about {round((target_acc - best_acc) * 8693)} rows on 8693-row OOF)"
    )

    print("\n=== Top 10 by OOF accuracy ===")
    print(ranked.head(10)[["rank", threshold_col, acc_col]].to_string(index=False))

    print("\n=== Threshold 0.4725 +/- 0.02 neighborhood ===")
    neighborhood = ranked[
        (ranked[threshold_col] >= 0.4525)
        & (ranked[threshold_col] <= 0.4925)
    ].sort_values(threshold_col)
    print(neighborhood[[threshold_col, acc_col, "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
