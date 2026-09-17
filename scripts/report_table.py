"""Build the report's main table from both result folders, so no number is copied by hand.

One row per model variant (architecture x training). Within-level statistics are the median over
the 13 corruption levels for each seed, then the mean over seeds; the seed range is kept too.

    python scripts/report_table.py
"""

import numpy as np
import pandas as pd

from fpr.data import REPO_ROOT

FOLDERS = ("results/models", "results/models_skip")
VARIANTS = [  # (group, architecture, training)
    ("dae_lam0", "bottleneck", "none"),
    ("dae_lam0.1w5", "bottleneck", "lambda 0.1"),
    ("dae_lam1w5", "bottleneck", "lambda 1"),
    ("dae_lam1w5_inner", "bottleneck", "lambda 1, inner only"),
    ("dae_lam1w5_outer", "bottleneck", "lambda 1, outer only"),
    ("dae_lam1", "bottleneck", "lambda 1, no warm-up"),
    ("unet_lam0", "skip", "none"),
    ("unet_lam1w5", "skip", "lambda 1"),
    ("unet_lam1", "skip", "lambda 1, no warm-up"),
]


def group_of(model):
    return model.rsplit("_seed", 1)[0]


def load(name):
    frames = [pd.read_csv(REPO_ROOT / folder / name) for folder in FOLDERS]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["model"].str.contains("_seed")]
    return frame.assign(group=frame["model"].map(group_of)) if "group" not in frame else \
        frame.assign(model_group=frame["model"].map(group_of))


def per_seed_median(metrics, signal, statistic, families=None):
    rows = metrics[(metrics["scope"] == "condition") & (metrics["target"] == "e")
                   & (metrics["signal"] == signal)]
    if families:
        rows = rows[rows["group"].str.startswith(families)]
    return rows.groupby("model")[statistic].median()


def flagged_share(per_image, family):
    """Among the worst-quartile errors of one family, the share whose g is in its top quartile.

    Both quartiles are taken over all 13 levels together, so a random score gives 25%.
    """
    shares = {}
    for model, frame in per_image.groupby("model", sort=False):
        if np.ptp(frame["g"]) <= 1e-9:
            continue
        worst = frame["e"] >= frame["e"].quantile(0.75)
        flagged = frame["g"] >= frame["g"].quantile(0.75)
        shares[model] = flagged[worst & (frame["family"] == family)].mean()
    return pd.Series(shares)


def main():
    metrics = load("metrics.csv")
    summary = load("summary.csv")
    columns = ["model", "family", "e", "g"]
    per_image = pd.concat([pd.read_csv(REPO_ROOT / folder / "per_image.csv.gz", usecols=columns)
                           for folder in FOLDERS], ignore_index=True)
    per_image = per_image[per_image["model"].str.contains("_seed")]

    statistics = {
        "rho_g": per_seed_median(metrics, "g", "spearman"),
        "rho_g_noise": per_seed_median(metrics, "g", "spearman", families="noise"),
        "partial_g": per_seed_median(metrics, "g", "partial"),
        "rho_d": per_seed_median(metrics, "d", "spearman"),
        "rho_rA": per_seed_median(metrics, "r_A", "spearman"),
        "rho_d_noise": per_seed_median(metrics, "d", "spearman", families="noise"),
    }
    pooled = metrics[(metrics["scope"] == "all") & (metrics["target"] == "e")]
    statistics["pooled_g"] = pooled[pooled["signal"] == "g"].set_index("model")["spearman"]
    statistics["pooled_g_level_median"] = pooled[pooled["signal"] == "g@level"].set_index("model")["spearman"]
    statistics["auroc_g"] = pooled[pooled["signal"] == "g"].set_index("model")["auroc"]
    statistics["flagged_pixel_g"] = flagged_share(per_image, "pixel_mask")
    statistics["flagged_box_g"] = flagged_share(per_image, "box_mask")
    per_model = pd.DataFrame(statistics)
    per_model["error"] = summary.groupby("model")["e"].mean()
    per_model["residual"] = summary.groupby("model")["g"].mean()
    per_model["group"] = per_model.index.map(group_of)

    rows = []
    for group, architecture, training in VARIANTS:
        seeds = per_model[per_model["group"] == group]
        if seeds.empty:
            continue
        row = {"architecture": architecture, "training": training, "seeds": len(seeds)}
        for column in ("error", "residual", "rho_g", "rho_g_noise", "partial_g", "rho_d", "rho_rA",
                       "rho_d_noise", "pooled_g", "pooled_g_level_median", "auroc_g",
                       "flagged_pixel_g", "flagged_box_g"):
            row[column] = seeds[column].mean()
        row["rho_g_min"], row["rho_g_max"] = seeds["rho_g"].min(), seeds["rho_g"].max()
        rows.append(row)
    table = pd.DataFrame(rows)
    path = REPO_ROOT / "results" / "report_table.csv"
    table.to_csv(path, index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (REPO_ROOT / "report" / "table_main.tex").write_text(latex_rows(table))
    print(f"\nWrote {path} and report/table_main.tex")


def latex_rows(table):
    """The complete tabular for the report (inputting rows inside a tabular breaks \\multicolumn).

    Collapsed models get dashes: their signals are degenerate.
    """
    def number(value, digits=2):
        return "--" if pd.isna(value) else f"{value:.{digits}f}".replace("-", "$-$")

    def percent(value):
        if pd.isna(value):
            return "--"
        # A share that rounds to 0% but is not zero, as the text says "under 1%".
        return "$<$1\\%" if 0 < value < 0.005 else f"{100 * value:.0f}\\%"

    lines = [
        r"\begin{tabular}{@{}lccccccccccc@{}}",
        r"\toprule",
        r"& & & \multicolumn{5}{c}{signal $g$} & $d$ & $r_A$ & \multicolumn{2}{c}{$g$ flagged} \\",
        r"\cmidrule(lr){4-8} \cmidrule(l){11-12}",
        r"training & $e$ & $g$ & within & noise & partial & pooled & AUROC & within & within & pixel & box \\",
        r"\midrule",
    ]
    for architecture in ("bottleneck", "skip"):
        rows = table[table["architecture"] == architecture]
        lines.append(rf"\multicolumn{{12}}{{@{{}}l}}{{\emph{{{architecture} model}}}} \\")
        for _, row in rows.iterrows():
            collapsed = "no warm-up" in row["training"]
            training = row["training"].replace("none", "lambda 0").replace("lambda", r"$\lid =$")
            cells = [training, number(row["error"], 3), number(row["residual"], 3)]
            if collapsed:
                cells += ["--"] * 9
            else:
                cells += [number(row[c]) for c in ("rho_g", "rho_g_noise", "partial_g", "pooled_g",
                                                    "auroc_g", "rho_d", "rho_rA")]
                cells += [percent(row["flagged_pixel_g"]), percent(row["flagged_box_g"])]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
