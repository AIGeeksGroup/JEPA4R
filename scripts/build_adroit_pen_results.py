"""Render the audited Adroit Pen comparison from its shareable results JSON.

Run from any working directory:
    python scripts/build_adroit_pen_results.py

Requires Matplotlib. Defaults are resolved relative to this repository.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_MARKERS = ("o", "s", "^")
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")
METRICS = (
    ("best_of_20", "Best-of-20", "#0072B2"),
    ("final", "Final (epoch 99)", "#D55E00"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_results(data):
    """Recompute all summaries and paired differences before plotting."""
    protocol = data["protocol"]
    require(data["schema_version"] == 1, "Unsupported results schema")
    require(protocol["metric_unit"] == "percent", "Expected success percentages")
    require(protocol["difference_unit"] == "percentage_points", "Expected paired differences in pp")
    require(protocol["policy_seeds"] == [100, 200, 300], "Unexpected policy seeds")
    require(
        (protocol["num_demos"], protocol["epochs"], protocol["eval_frequency"],
         protocol["eval_num_traj"]) == (100, 100, 5, 25),
        "Expected the matched 100-demo, 100-epoch, 25-trajectory protocol",
    )
    require(protocol["eval_epochs"] == list(range(5, 100, 5)) + [99], "Unexpected eval grid")
    require(set(data["modalities"]) == {"rgb", "pcd"}, "Expected RGB and PCD")
    for modality, block in data["modalities"].items():
        for group in ("pretrained", "random", "random_minus_pretrained"):
            rows = block[group]["seeds"]
            require([r["seed"] for r in rows] == protocol["policy_seeds"], "Seed alignment mismatch")
            for metric, _, _ in METRICS:
                values = [r[metric] for r in rows]
                require(all(math.isfinite(v) for v in values), "Non-finite metric")
                if group != "random_minus_pretrained":
                    require(all(0 <= v <= 100 for v in values), "Success outside [0, 100]")
                expected = {"mean": statistics.mean(values),
                            "population_std": statistics.pstdev(values)}
                for name, value in expected.items():
                    require(
                        math.isclose(block[group]["summary"][metric][name], value,
                                     rel_tol=1e-12, abs_tol=1e-12),
                        f"{modality}/{group}/{metric}: incorrect {name}",
                    )
        for pretrained, random, delta in zip(
            block["pretrained"]["seeds"], block["random"]["seeds"],
            block["random_minus_pretrained"]["seeds"],
        ):
            for metric, _, _ in METRICS:
                require(
                    math.isclose(delta[metric], random[metric] - pretrained[metric], abs_tol=1e-12),
                    f"{modality}: incorrect paired difference",
                )
        for group in ("random", "pretrained"):
            for row in block[group]["seeds"]:
                if group == "pretrained" and "eval_success" not in row:
                    continue
                curve = row["eval_success"]
                require(len(curve) == len(protocol["eval_epochs"]), "Incomplete eval curve")
                require(all(math.isfinite(v) and 0 <= v <= 100 for v in curve), "Invalid eval curve")
                require(max(curve) == row["best_of_20"], "Curve/Best-of-20 mismatch")
                require(curve[-1] == row["final"], "Curve/final mismatch")
                if group == "random":
                    audit = row["audit"]
                    require(
                        audit["status"] == "PASS" and audit["finite"] is True
                        and (audit["loss_records"], audit["eval_records"],
                             audit["last_eval_epoch"], audit["exit_code"]) == (100, 20, 99, 0),
                        "Random run gates failed",
                    )


def style_axis(ax):
    ax.set_ylim(0, 103)
    ax.set_yticks(range(0, 101, 20))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#A5ADB5")
    ax.tick_params(colors="#26313B", length=3)
    ax.set_ylabel("Evaluator success (%)")


def comparison_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.24, top=0.78, wspace=0.18)
    fig.suptitle("Adroit Pen: pretrained vs random frozen representations", y=0.98,
                 fontsize=19, fontweight="bold")
    fig.text(0.5, 0.921, "100 demos | 3 policy seeds | 25 trajectories per evaluation",
             ha="center", fontsize=12)
    handles = [Patch(facecolor=color, alpha=0.72, label=label) for _, label, color in METRICS]
    handles += [
        Line2D([], [], linestyle="", marker=marker, color="#20252B", label=f"Seed {seed}")
        for seed, marker in zip(data["protocol"]["policy_seeds"], SEED_MARKERS)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.883),
               ncol=5, frameon=False, fontsize=11)
    for ax, modality, title in zip(axes, ("rgb", "pcd"),
                                   ("RGB (official protocol)", "PCD (3D extension)")):
        block = data["modalities"][modality]
        style_axis(ax)
        ax.set_title(title, fontsize=15, pad=12)
        for group_index, group in enumerate(("pretrained", "random")):
            label_heights = [
                max(max(r[metric] for r in block[group]["seeds"]),
                    block[group]["summary"][metric]["mean"]
                    + block[group]["summary"][metric]["population_std"]) + 4
                for metric, _, _ in METRICS
            ]
            if abs(label_heights[0] - label_heights[1]) < 7:
                label_heights[1] = max(label_heights) + 8
            for metric_index, (metric, _, color) in enumerate(METRICS):
                x = group_index + (-0.18 if metric_index == 0 else 0.18)
                values = [r[metric] for r in block[group]["seeds"]]
                summary = block[group]["summary"][metric]
                mean, std = summary["mean"], summary["population_std"]
                ax.bar(x, mean, width=0.31, color=color, alpha=0.72,
                       edgecolor="white", linewidth=1, zorder=2,
                       hatch=None if metric_index == 0 else "//")
                ax.errorbar(x, mean, yerr=std, color="#20252B", capsize=5,
                            linewidth=1.5, zorder=3)
                for value, offset, marker in zip(values, (-0.055, 0, 0.055), SEED_MARKERS):
                    ax.scatter(x + offset, value, s=46, marker=marker, facecolor="white",
                               edgecolor="#20252B", linewidth=1.2, zorder=4)
                label_y = label_heights[metric_index]
                ax.text(x, label_y, f"{mean:.1f} +/- {std:.1f}", ha="center",
                        va="bottom", fontsize=10.5)
        ax.set_xticks([0, 1], ["Pretrained", "Random"])
        ax.set_xlim(-0.53, 1.53)
        ax.tick_params(axis="x", labelsize=13, pad=9)
        delta = block["random_minus_pretrained"]["summary"]
        ax.text(0.5, -0.18,
                f"Random - pretrained: best {delta['best_of_20']['mean']:+.2f} pp"
                f" | final {delta['final']['mean']:+.2f} pp",
                transform=ax.transAxes, ha="center", fontsize=11)
    fig.text(0.065, 0.082, "Bars: mean +/- population SD. Points: individual policy seeds.",
             fontsize=10.5)
    fig.text(0.065, 0.047,
             "Pretrained: validated snapshot. Random: audited local copy. PCD is not an official RGB score.",
             fontsize=10.5)
    return fig


def curves_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.20, top=0.78, wspace=0.18)
    fig.suptitle("Adroit Pen: random frozen evaluation curves", y=0.98,
                 fontsize=19, fontweight="bold")
    fig.text(0.5, 0.921,
             "100 demos | 25 trajectories per evaluation | Shared representation initialization seed 0",
             ha="center", fontsize=12)
    handles = [
        Line2D([], [], color=color, marker=marker, linewidth=2, label=f"Seed {seed}")
        for seed, color, marker in zip(data["protocol"]["policy_seeds"], SEED_COLORS, SEED_MARKERS)
    ]
    has_pretrained = any("eval_success" in row for block in data["modalities"].values()
                         for row in block["pretrained"]["seeds"])
    if has_pretrained:
        handles += [Line2D([], [], color="#20252B", linestyle=style, label=label)
                    for style, label in (("-", "Random"), ("--", "Pretrained snapshot"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.883),
               ncol=len(handles), frameon=False, fontsize=11)
    for ax, modality, title in zip(axes, ("rgb", "pcd"),
                                   ("RGB random", "PCD random (3D extension)")):
        style_axis(ax)
        ax.set_title(title, fontsize=15, pad=12)
        block = data["modalities"][modality]
        for group, linestyle, alpha in (("random", "-", 1.0), ("pretrained", "--", 0.6)):
            for row, color, marker in zip(block[group]["seeds"], SEED_COLORS, SEED_MARKERS):
                if "eval_success" not in row:
                    continue
                ax.plot(data["protocol"]["eval_epochs"], row["eval_success"], color=color,
                        marker=marker, linestyle=linestyle, alpha=alpha, markersize=4.5,
                        linewidth=1.8, zorder=3)
        ax.set_xlim(2, 102)
        ax.set_xticks([5, 25, 45, 65, 85, 99])
        ax.set_xlabel("Epoch (zero-based)", labelpad=9)
    fig.text(0.065, 0.082,
             "Each point is the logged evaluator success, not a cumulative best or a smoothed estimate.",
             fontsize=10.5)
    note = ("Dashed curves use per-epoch evidence from the pretrained snapshot." if has_pretrained
            else "Pretrained per-epoch metrics are unavailable in the supplied reference snapshot.")
    fig.text(0.065, 0.047, note, fontsize=10.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=REPO_ROOT / "docs/results/adroit_pen_100demo_random_control.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs/figures")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    require(args.dpi > 0, "DPI must be positive")
    raw = args.results.read_bytes()
    data = json.loads(raw)
    validate_results(data)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "text.color": "#20252B",
        "axes.labelcolor": "#26313B", "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Software": f"Matplotlib {matplotlib.__version__}",
                "Results SHA256": hashlib.sha256(raw).hexdigest()}
    for name, render in (
        ("adroit_pen_random_control.png", comparison_figure),
        ("adroit_pen_evaluation_curves.png", curves_figure),
    ):
        fig = render(data)
        path = args.output_dir / name
        fig.savefig(path, dpi=args.dpi, metadata=metadata)
        plt.close(fig)
        print(json.dumps({"file": str(path.resolve()),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "results_sha256": metadata["Results SHA256"]}))
    print(json.dumps({
        modality: {group: block[group]["summary"] for group in
                   ("pretrained", "random", "random_minus_pretrained")}
        for modality, block in data["modalities"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
