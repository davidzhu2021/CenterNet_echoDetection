from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 7
import matplotlib.pyplot as plt
import pandas as pd

from cfar_baseline.generate_srr_sweep_dataset import srr_level_token
from false_alarm_eval_postprocess import postprocess_experiment
from hard_eval_from_detections import evaluate_experiment
from run_hard_mf_false_alarm_compare import (
    DEFAULT_MODELS,
    MF_MODULE_PATH,
    REPO_ROOT,
    choose_best_threshold,
    ensure_centernet_raw,
    ensure_dir,
    evaluate_thresholds_for_raw_dir,
    extract_metrics,
    load_mf_module,
    parse_float_list,
    parse_int_list,
    run_command,
    threshold_grid_from_arg,
)


DEFAULT_DATA_DIR = REPO_ROOT / "data" / "sonar_srr_sweep_v1"
DEFAULT_SAVE_ROOT = REPO_ROOT / "eval_srr_sweep_v1"


METHOD_LABELS = {
    "sonar_cfar_mf": "MF + CA-CFAR",
    "sonar_swin_fpn_endpoint2": "EATFD",
    "sonar_swin_tiny_endpoint2": "Swin-Tiny endpoint2",
    "sonar_swin_fpn_mixed": "Swin-FPN mixed",
}


# Legacy values used by the committed Fig. 5 before the SRR summary CSV was regenerated.
LEGACY_FIG5_LEVELS = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
LEGACY_FIG5_VALUES = {
    "sonar_swin_fpn_endpoint2": {
        "negative_FA_rate": [0.048, 0.038, 0.012, 0.009, 0.001, 0.0],
        "positive_recall": [0.60, 0.85, 0.94, 0.99, 0.98, 1.00],
        "det_f1": [0.65, 0.87, 0.96, 0.97, 0.99, 1.00],
        "AP_overall": [0.68, 0.92, 0.96, 0.99, 0.99, 1.00],
    },
    "sonar_swin_tiny_endpoint2": {
        "negative_FA_rate": [0.068, 0.042, 0.028, 0.013, 0.006, 0.0],
        "positive_recall": [0.50, 0.76, 0.91, 0.94, 0.99, 1.00],
        "det_f1": [0.55, 0.81, 0.92, 0.95, 0.98, 1.00],
        "AP_overall": [0.55, 0.85, 0.91, 0.98, 0.98, 1.00],
    },
    "sonar_swin_fpn_mixed": {
        "negative_FA_rate": [0.075, 0.065, 0.032, 0.022, 0.007, 0.0],
        "positive_recall": [0.40, 0.67, 0.81, 0.89, 0.96, 1.00],
        "det_f1": [0.45, 0.69, 0.86, 0.90, 0.97, 1.00],
        "AP_overall": [0.49, 0.70, 0.89, 0.94, 0.97, 1.00],
    },
    "sonar_cfar_mf": {
        "negative_FA_rate": [0.092, 0.082, 0.068, 0.048, 0.012, 0.0],
        "positive_recall": [0.20, 0.43, 0.66, 0.81, 0.89, 1.00],
        "det_f1": [0.25, 0.49, 0.71, 0.80, 0.93, 1.00],
        "AP_overall": [0.29, 0.55, 0.70, 0.82, 0.92, 1.00],
    },
}


def load_legacy_fig5_summary() -> pd.DataFrame:
    """Return the data encoded by the committed legacy Fig. 5 vector figure."""
    rows = []
    for exp_id, values in LEGACY_FIG5_VALUES.items():
        for idx, srr_level in enumerate(LEGACY_FIG5_LEVELS):
            rows.append({
                "exp_id": exp_id,
                "srr_level": srr_level,
                **{metric: series[idx] for metric, series in values.items()},
            })
    return pd.DataFrame(rows)


def load_srr_levels(data_dir: Path) -> list[float]:
    df = pd.read_csv(data_dir / "labels.csv")
    return sorted(df["srr_level"].astype(float).unique().tolist())


def split_file_for_srr(data_dir: Path, split: str, srr_level: float) -> Path:
    return data_dir / "splits" / f"{split}_srr_{srr_level_token(srr_level)}.txt"


def build_labels_paths(data_dir: Path, split: str) -> tuple[Path, Path]:
    return data_dir / "labels.csv", data_dir / "splits" / f"{split}.txt"


def run_mf_val_selection(mf_module, args, save_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    static_records = mf_module.load_split_static(args.data_dir, "val")
    rows = []
    for num_train in parse_int_list(args.mf_train_grid):
        for num_guard in parse_int_list(args.mf_guard_grid):
            for pfa in parse_float_list(args.mf_pfa_grid):
                records = mf_module.evaluate_static_records(static_records, num_train=num_train, num_guard=num_guard, pfa=pfa)
                ap = mf_module.compute_ap(records)
                for thresh in threshold_grid_from_arg(args.threshold_grid):
                    metrics = mf_module.evaluate_records(records, threshold=float(thresh), center_tol=args.center_tol)
                    details = metrics["details"]
                    negatives = details[details["gt_exists"] == False]  # noqa: E712
                    positives = details[details["gt_exists"] == True]  # noqa: E712
                    negative_fp_count = int(negatives["pred_exists"].fillna(0).astype(int).sum())
                    positive_tp = int(positives["matched"].fillna(0).astype(int).sum())
                    rows.append({
                        "exp_id": "sonar_cfar_mf",
                        "arch": "matched_filter_ca_cfar",
                        "hm_mode": "cfar",
                        "threshold": float(thresh),
                        "num_train": int(num_train),
                        "num_guard": int(num_guard),
                        "pfa": float(pfa),
                        "img_accuracy": metrics["img_accuracy"],
                        "img_precision": metrics["img_precision"],
                        "img_recall": metrics["img_recall"],
                        "img_f1": metrics["img_f1"],
                        "det_precision": metrics["det_precision"],
                        "det_recall": metrics["det_recall"],
                        "det_f1": metrics["det_f1"],
                        "mean_loc_error": metrics["mean_loc_error"],
                        "AP_overall": ap,
                        "negative_FA_rate": negative_fp_count / len(negatives) if len(negatives) else float("nan"),
                        "negative_FP_count": negative_fp_count,
                        "positive_recall": positive_tp / len(positives) if len(positives) else float("nan"),
                        "num_negative": int(len(negatives)),
                        "num_positive": int(len(positives)),
                    })
    df = pd.DataFrame(rows)
    ensure_dir(save_root / "val_selection")
    df.to_csv(save_root / "val_selection" / "sonar_cfar_mf_val_grid.csv", index=False)
    best = choose_best_threshold(df)
    (save_root / "val_selection" / "sonar_cfar_mf_best_config.json").write_text(
        json.dumps(best.to_dict(), indent=2),
        encoding="utf-8",
    )
    return df, best


def run_mf_raw_for_split(model: dict, best: pd.Series, split: str, save_dir: Path, args) -> None:
    if save_dir.exists() and args.force_eval:
        shutil.rmtree(save_dir)
    ensure_dir(save_dir)
    cmd = [
        str(args.python_exe),
        "tools/cfar_baseline/mf_cfar_1d.py",
        "eval",
        "--data_dir", str(args.data_dir),
        "--split", split,
        "--save_dir", str(save_dir),
        "--num_train", str(int(best["num_train"])),
        "--num_guard", str(int(best["num_guard"])),
        "--pfa", str(float(best["pfa"])),
        "--score_thresh", str(float(best["threshold"])),
        "--center_tol", str(args.center_tol),
        "--exp_id", model["exp_id"],
        "--arch_name", model["arch"],
        "--hm_mode", model["hm_mode"],
    ]
    run_command(cmd, REPO_ROOT)


def postprocess_split(exp_dir: Path, split_file: Path, model: dict, threshold: float, protocol: str, args) -> pd.Series:
    postprocess_experiment(
        exp_dir=exp_dir,
        labels_csv=args.data_dir / "labels.csv",
        split_file=split_file,
        exp_id=model["exp_id"],
        arch=model["arch"],
        hm_mode=model["hm_mode"],
        protocol=protocol,
        level="test",
        center_tol=args.center_tol,
        score_thresh=float(threshold),
    )
    return pd.read_csv(exp_dir / "hard_false_alarm_summary.csv").iloc[0]


def run_deep_val_selection(model: dict, save_root: Path, args) -> pd.Series:
    val_dir = save_root / "val_raw" / model["exp_id"]
    ensure_centernet_raw(model, "val", val_dir, args)
    val_df = evaluate_thresholds_for_raw_dir(val_dir, "val", model, args)
    ensure_dir(save_root / "val_selection")
    val_df.to_csv(save_root / "val_selection" / f"{model['exp_id']}_val_grid.csv", index=False)
    return choose_best_threshold(val_df)


def selected_rows_to_csv(rows: list[dict], save_root: Path) -> None:
    ensure_dir(save_root / "val_selection")
    pd.DataFrame(rows).to_csv(save_root / "val_selection" / "selected_operating_points.csv", index=False)


def run_srr_test_splits(models: list[dict], selected: dict[str, pd.Series], args) -> pd.DataFrame:
    rows = []
    for srr in load_srr_levels(args.data_dir):
        split = f"test_srr_{srr_level_token(srr)}"
        split_file = split_file_for_srr(args.data_dir, "test", srr)
        protocol = f"srr_sweep_{srr_level_token(srr)}"
        for model in models:
            exp_id = model["exp_id"]
            exp_dir = args.save_root / "test" / f"srr_{srr_level_token(srr)}" / exp_id
            best = selected[exp_id]
            if model["kind"] == "cfar":
                run_mf_raw_for_split(model, best, split, exp_dir, args)
            else:
                ensure_centernet_raw(model, split, exp_dir, args)
            row = postprocess_split(exp_dir, split_file, model, float(best["threshold"]), protocol, args)
            row = row.to_dict()
            row["srr_level"] = float(srr)
            row["method_label"] = METHOD_LABELS.get(exp_id, exp_id)
            rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(["srr_level", "exp_id"], kind="mergesort")
    out.to_csv(args.save_root / "srr_sweep_summary.csv", index=False)
    wide = out.pivot_table(
        index="srr_level",
        columns="exp_id",
        values=["negative_FA_rate", "positive_recall", "det_f1", "AP_overall", "mean_loc_error"],
        aggfunc="first",
    )
    wide.to_csv(args.save_root / "srr_sweep_summary_wide.csv")
    return out


def plot_srr_summary(summary: pd.DataFrame, save_root: Path, dpi: int = 220) -> None:
    fig_dir = save_root / "figures"
    ensure_dir(fig_dir)
    legend_kwargs = {
        "loc": "lower right",
        "fontsize": 7,
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "0.4",
        "framealpha": 0.88,
        "handlelength": 1.4,
        "borderpad": 0.25,
    }
    metrics = [
        ("negative_FA_rate", "False Positive Rate", "lower is better"),
        ("positive_recall", "Positive Recall", "higher is better"),
        ("det_f1", "Detection F1", "higher is better"),
        ("AP_overall", "AP Overall", "higher is better"),
    ]
    colors = {
        "sonar_cfar_mf": "#555555",
        "sonar_swin_fpn_endpoint2": "#0B5CAD",
        "sonar_swin_tiny_endpoint2": "#D17C00",
        "sonar_swin_fpn_mixed": "#7A3E9D",
    }
    markers = {
        "sonar_cfar_mf": "o",
        "sonar_swin_fpn_endpoint2": "s",
        "sonar_swin_tiny_endpoint2": "^",
        "sonar_swin_fpn_mixed": "D",
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), dpi=dpi, constrained_layout=True)
    for panel_idx, (ax, (metric, ylabel, hint)) in enumerate(zip(axes.ravel(), metrics)):
        for exp_id, group in summary.groupby("exp_id", sort=False):
            group = group.sort_values("srr_level")
            ax.plot(
                group["srr_level"],
                group[metric],
                marker=markers.get(exp_id, "o"),
                linewidth=2.0,
                markersize=5.0,
                color=colors.get(exp_id),
                label=METHOD_LABELS.get(exp_id, exp_id),
            )
        ax.set_xlabel("SRR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        panel_legend_kwargs = dict(legend_kwargs)
        if panel_idx == 0:
            panel_legend_kwargs["loc"] = "upper right"
        ax.legend(**panel_legend_kwargs)
    fig.savefig(fig_dir / "srr_sweep_4panel.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(fig_dir / "srr_sweep_4panel.svg", bbox_inches="tight")
    fig.savefig(fig_dir / "srr_sweep_4panel.pdf", bbox_inches="tight")
    plt.close(fig)

    for metric, ylabel, hint in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.5), dpi=dpi, constrained_layout=True)
        for exp_id, group in summary.groupby("exp_id", sort=False):
            group = group.sort_values("srr_level")
            ax.plot(
                group["srr_level"],
                group[metric],
                marker=markers.get(exp_id, "o"),
                linewidth=2.0,
                markersize=5.0,
                color=colors.get(exp_id),
                label=METHOD_LABELS.get(exp_id, exp_id),
            )
        ax.set_xlabel("SRR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.legend(**legend_kwargs)
        fig.savefig(fig_dir / f"{metric}_vs_srr.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(fig_dir / f"{metric}_vs_srr.svg", bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SRR sweep comparison with MF+CA-CFAR and CenterNet models.")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--input_res", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--center_tol", type=float, default=40.0)
    parser.add_argument("--raw_score_thresh", type=float, default=0.001)
    parser.add_argument("--threshold_grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--endpoint_search_radius", type=int, default=16)
    parser.add_argument("--mf_train_grid", default="40,80,120")
    parser.add_argument("--mf_guard_grid", default="10,20,30")
    parser.add_argument("--mf_pfa_grid", default="0.01,0.04,0.1")
    parser.add_argument("--protocol_name", default="srr_sweep")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument(
        "--legacy_fig5",
        action="store_true",
        help="Regenerate the committed Fig. 5 curves without rerunning evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Missing data_dir: {args.data_dir}")
    ensure_dir(args.save_root)
    if args.legacy_fig5:
        plot_srr_summary(load_legacy_fig5_summary(), args.save_root)
        print(f"Saved legacy Fig. 5 plots under {args.save_root / 'figures'}")
        return
    mf_module = load_mf_module(MF_MODULE_PATH)

    selected: dict[str, pd.Series] = {}
    selected_rows: list[dict] = []
    for model in DEFAULT_MODELS:
        if model["kind"] == "cfar":
            _, best = run_mf_val_selection(mf_module, args, args.save_root)
        else:
            best = run_deep_val_selection(model, args.save_root, args)
        row = best.to_dict()
        row["exp_id"] = model["exp_id"]
        row["arch"] = model["arch"]
        row["hm_mode"] = model["hm_mode"]
        selected[model["exp_id"]] = best
        selected_rows.append(row)

    selected_rows_to_csv(selected_rows, args.save_root)
    summary = run_srr_test_splits(DEFAULT_MODELS, selected, args)
    if not args.skip_plots:
        plot_srr_summary(summary, args.save_root)
    print("Selected operating points:")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print("\nSRR sweep summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
