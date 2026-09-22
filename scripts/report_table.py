"""Build the report's tables from the result folders, so no number is copied by hand.

table_main.tex: one row per model variant (architecture x training). Within-level statistics are
the median over the 13 corruption levels for each seed, then the mean over seeds; the seed range
is kept too.

table_coverage.tex: conformal coverage and selective risk, from results/conformal and
results/selective. Skipped when those folders are missing.

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
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby("model")[statistic].median()


def flagged_share(per_image, family, signal="g"):
    """Among the worst-quartile errors of one family, the share whose signal is in its top quartile.

    Both quartiles are taken over all 13 levels together, so a random score gives 25%.
    """
    shares = {}
    for model, frame in per_image.groupby("model", sort=False):
        if signal not in frame or frame[signal].isna().all() or np.ptp(frame[signal].dropna()) <= 1e-9:
            continue
        worst = frame["e"] >= frame["e"].quantile(0.75)
        flagged = frame[signal] >= frame[signal].quantile(0.75)
        shares[model] = flagged[worst & (frame["family"] == family)].mean()
    return pd.Series(shares)


def per_image_columns(path, wanted=("model", "family", "e", "g", "dis")):
    """Only the columns the table needs, and only those the dump has (dis may be missing)."""
    available = set(pd.read_csv(path, nrows=0).columns)
    return [c for c in wanted if c in available]


def main():
    metrics = load("metrics.csv")
    summary = load("summary.csv")
    paths = [REPO_ROOT / folder / "per_image.csv.gz" for folder in FOLDERS]
    per_image = pd.concat([pd.read_csv(path, usecols=per_image_columns(path)) for path in paths],
                          ignore_index=True)
    per_image = per_image[per_image["model"].str.contains("_seed")]

    statistics = {
        "rho_g": per_seed_median(metrics, "g", "spearman"),
        "rho_g_noise": per_seed_median(metrics, "g", "spearman", families="noise"),
        "partial_g": per_seed_median(metrics, "g", "partial"),
        "rho_d": per_seed_median(metrics, "d", "spearman"),
        "rho_rA": per_seed_median(metrics, "r_A", "spearman"),
        "rho_d_noise": per_seed_median(metrics, "d", "spearman", families="noise"),
        "rho_dis": per_seed_median(metrics, "dis", "spearman"),
        "rho_g2": per_seed_median(metrics, "g2", "spearman"),
    }
    pooled = metrics[(metrics["scope"] == "all") & (metrics["target"] == "e")]
    statistics["pooled_g"] = pooled[pooled["signal"] == "g"].set_index("model")["spearman"]
    statistics["pooled_g_level_median"] = \
        pooled[pooled["signal"] == "g@level"].set_index("model")["spearman"]
    statistics["auroc_g"] = pooled[pooled["signal"] == "g"].set_index("model")["auroc"]
    statistics["flagged_pixel_g"] = flagged_share(per_image, "pixel_mask", "g")
    statistics["flagged_box_g"] = flagged_share(per_image, "box_mask", "g")
    if "dis" in per_image.columns:
        statistics["flagged_pixel_dis"] = flagged_share(per_image, "pixel_mask", "dis")
        statistics["flagged_box_dis"] = flagged_share(per_image, "box_mask", "dis")
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
                       "rho_d_noise", "pooled_g", "pooled_g_level_median", "auroc_g", "rho_dis",
                       "rho_g2", "flagged_pixel_g", "flagged_box_g", "flagged_pixel_dis",
                       "flagged_box_dis"):
            row[column] = seeds[column].mean() if column in seeds else float("nan")
        row["rho_g_min"], row["rho_g_max"] = seeds["rho_g"].min(), seeds["rho_g"].max()
        rows.append(row)
    table = pd.DataFrame(rows)
    path = REPO_ROOT / "results" / "report_table.csv"
    table.to_csv(path, index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (REPO_ROOT / "report" / "table_main.tex").write_text(latex_rows(table))
    print(f"\nWrote {path} and report/table_main.tex")
    coverage_table()


GROUP_LABELS = {"dae_lam0": r"bottleneck, $\lid{=}0$",
                "dae_lam1w5": r"bottleneck, $\lid{=}1$",
                "unet_lam0": r"skip, $\lid{=}0$",
                "unet_lam1w5": r"skip, $\lid{=}1$"}


def coverage_table(groups=("dae_lam0", "dae_lam1w5", "unet_lam0", "unet_lam1w5")):
    """Conformal coverage (pooled, per family, under shift) and selective risk."""
    conformal_dir = REPO_ROOT / "results" / "conformal"
    selective_dir = REPO_ROOT / "results" / "selective"
    needed = [conformal_dir / "coverage.csv", conformal_dir / "coverage_family.csv",
              conformal_dir / "shift.csv", selective_dir / "selective.csv"]
    if not all(p.exists() for p in needed):
        print("skipping table_coverage.tex: run scripts/conformal.py and scripts/selective.py")
        return

    pooled = pd.read_csv(conformal_dir / "coverage.csv")
    family = pd.read_csv(conformal_dir / "coverage_family.csv")
    shift = pd.read_csv(conformal_dir / "shift.csv")
    selective = pd.read_csv(selective_dir / "selective.csv")

    rows = []
    for group in groups:
        pooled_g = pooled[(pooled["model_group"] == group)]
        if pooled_g.empty:
            continue
        row = {"group": group}
        normalized = pooled_g[(pooled_g["signal"] == "g") & (pooled_g["setting"] == "normalized")]
        row["cov_pooled"] = normalized["coverage"].mean()
        row["width_g"] = normalized["width"].mean()
        row["width_marginal"] = pooled_g[pooled_g["setting"] == "marginal"]["width"].mean()
        row["width_severity"] = pooled_g[pooled_g["setting"] == "severity"]["width"].mean()
        fam = family[family["model_group"] == group]
        for signal in ("g", "dis"):
            for name, key in (("pixel_mask", "pixel"), ("box_mask", "box")):
                cell = fam[(fam["signal"] == signal) & (fam["group"] == name)]["coverage"]
                row[f"cov_{key}_{signal}"] = cell.mean() if len(cell) else float("nan")
        for signal in ("g", "dis"):
            cell = shift[(shift["model_group"] == group) & (shift["signal"] == signal)]["coverage"]
            row[f"shift_{signal}"] = cell.mean() if len(cell) else float("nan")
        sel = selective[selective["group"] == group]
        for signal in ("g", "dis", "g2"):
            cell = sel[sel["signal"] == signal]["aurc_norm"]
            row[f"aurc_{signal}"] = cell.mean() if len(cell) else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(REPO_ROOT / "results" / "report_coverage.csv", index=False, float_format="%.4f")

    def num(value, digits=2):
        return "--" if pd.isna(value) else f"{value:.{digits}f}"

    lines = [
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{width} & \multicolumn{3}{c}{coverage, $g$}"
        r" & \multicolumn{2}{c}{nAURC} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(l){7-8}",
        r"model & $g$ & level & pixel & box & shift & $g$ & $\mathrm{dis}$ \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [GROUP_LABELS.get(row["group"], row["group"]),
                 num(row["width_g"], 3), num(row["width_severity"], 3),
                 num(row["cov_pixel_g"]), num(row["cov_box_g"]),
                 num(row["shift_g"]), num(row["aurc_g"]), num(row["aurc_dis"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (REPO_ROOT / "report" / "table_coverage.tex").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 220)
    print("\nCoverage and selective risk")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Wrote report/table_coverage.tex")


def latex_rows(table):
    """The complete tabular for the report (inputting rows inside a tabular breaks \\multicolumn).

    Columns: training, e, g, within/noise/partial/pooled/AUROC for g, within for d, r_A and dis,
    flagged pixel/box for g. The runs without a warm-up collapse to a constant map, so every
    ranking is degenerate; they stay in results/report_table.csv but not in the report's table,
    where the text gives their error and residual.
    """
    def number(value, digits=2):
        return "--" if pd.isna(value) else f"{value:.{digits}f}".replace("-", "$-$")

    def percent(value):
        if pd.isna(value):
            return "--"
        # A share that rounds to 0% but is not zero, as the text says "under 1%".
        return "$<$1\\%" if 0 < value < 0.005 else f"{100 * value:.0f}\\%"

    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccccccccc@{}}",
        r"\toprule",
        r"& & & \multicolumn{5}{c}{signal $g$} & $d$ & $r_A$ & $\mathrm{dis}$"
        r" & \multicolumn{2}{c}{$g$ flagged} \\",
        r"\cmidrule(lr){4-8} \cmidrule(l){12-13}",
        r"training & $e$ & $g$ & within & noise & partial & pooled & AUROC & within & within"
        r" & within & pixel & box \\",
        r"\midrule",
    ]
    for architecture in ("bottleneck", "skip"):
        rows = table[(table["architecture"] == architecture)
                     & ~table["training"].str.contains("no warm-up")]
        for _, row in rows.iterrows():
            training = row["training"].replace("none", "lambda 0").replace("lambda", r"$\lid{=}$")
            cells = [f"{architecture}, {training}", number(row["error"], 3),
                     number(row["residual"], 3)]
            cells += [number(row[c]) for c in ("rho_g", "rho_g_noise", "partial_g", "pooled_g",
                                               "auroc_g", "rho_d", "rho_rA", "rho_dis")]
            cells += [percent(row["flagged_pixel_g"]), percent(row["flagged_box_g"])]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
