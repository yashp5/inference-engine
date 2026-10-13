# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations
from collections.abc import Callable
from typing import cast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

N_SLOTS = 8
RESULTS = "bench/results"
TIMELINE_CONCURRENCY = 8

# workload name -> (static label, continuous label). Each pair is one
# controlled comparison: same seed and mix, only the admission policy differs.
PAIRS = {
    "mixed": ("static", "continuous"),
    "uniform": ("static-uniform", "continuous-uniform"),
}

# one colour per policy, shared by every figure, so a policy reads the same
# everywhere. continuous takes categorical slot 1, static slot 2.
COLORS = {"continuous": "#2a78d6", "static": "#eb6834"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False,
    "font.size": 9.5, "lines.linewidth": 2,
})


def policy(label: str) -> str:
    return "static" if label.startswith("static") else "continuous"


req = pd.read_csv(f"{RESULTS}/requests.csv")
st = pd.read_csv(f"{RESULTS}/stats.csv")
runs = pd.read_csv(f"{RESULTS}/runs.csv")
req = cast(pd.DataFrame, req.loc[req["label"] != "warmup"])
runs = cast(pd.DataFrame, runs.loc[runs["label"] != "warmup"])
# active_slots is -1 while the server has no engine session (before the first
# step, or across a worker restart). Left in, it drags slot_util down and makes
# "waiting" read -2.
st = cast(pd.DataFrame, st.loc[(st["label"] != "warmup") & (st["active_slots"] >= 0)])

ok = cast(pd.DataFrame, req.loc[req["status"] == 200]).copy()
assert (ok["tokens"] == ok["max_tokens"]).all(), "ignore_eos not on: lengths aren't controlled"

# by the actual length rather than short/long: the uniform workload is a
# single class, and naming it "long" would be misleading
ok["cls"] = ok["max_tokens"].map(lambda t: f"{t} tok")


def q(p: float) -> Callable[[pd.Series], float]:
    def quantile(s: pd.Series) -> float:
        return float(s.quantile(p))

    return quantile


summary = cast(pd.DataFrame, ok.groupby(["label", "concurrency", "cls"]).agg(
    n=("idx", "size"),
    ttft_p50=("ttft_ms", q(.5)), ttft_p99=("ttft_ms", q(.99)),
    e2e_p50=("latency_ms", q(.5)), e2e_p99=("latency_ms", q(.99)),
))
print(summary.round(0))

slot_util = cast(pd.Series, st.groupby(["label", "concurrency"])["active_slots"].mean()) / N_SLOTS
print(slot_util.rename("slot_util").round(2))


def plot_timeline(name: str, labels: tuple[str, str]) -> None:
    """Active slots and waiting requests over one run: the sawtooth figure."""
    fig, raw_axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6), constrained_layout=True)
    ax = cast(tuple[Axes, Axes], raw_axes)
    data = st.loc[(st["concurrency"] == TIMELINE_CONCURRENCY) & st["label"].isin(labels)]
    for label in labels:
        g = data.loc[data["label"] == label]
        if g.empty:
            continue
        c = COLORS[policy(label)]
        t = g["elapsed_ms"] / 1000
        ax[0].step(t, g["active_slots"], where="post", color=c, label=label)
        ax[1].step(t, g["queue_depth"] + g["engine_waiting"], where="post", color=c, label=label)
    ax[0].set_ylim(0, N_SLOTS + 0.5)
    ax[0].set_title(f"Active slots ({name} workload, concurrency {TIMELINE_CONCURRENCY})")
    ax[0].set_ylabel("active slots")
    ax[1].set_title("Requests waiting (Go queue + engine waiting)")
    ax[1].set_ylabel("waiting requests")
    ax[1].set_xlabel("seconds into run")
    # outside the plot: both series spend most of the run at the top edge
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(f"{RESULTS}/timeline_{name}.png", dpi=150)
    plt.close(fig)


def plot_sweep(name: str, labels: tuple[str, str]) -> None:
    """Headline numbers against concurrency. Three panels instead of a dual
    axis: latency and throughput don't share a scale."""
    fig, raw_axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ax = cast(tuple[Axes, Axes, Axes], raw_axes)
    for label in labels:
        g = runs.loc[runs["label"] == label].sort_values("concurrency")
        if g.empty:
            continue
        c = COLORS[policy(label)]
        x = g["concurrency"]
        ax[0].plot(x, g["ttft_p50_ms"] / 1000, color=c, marker="o", markersize=6, label=f"{label} p50")
        ax[0].plot(x, g["ttft_p99_ms"] / 1000, color=c, marker="o", markersize=6, linestyle="--", label=f"{label} p99")
        ax[1].plot(x, g["p50_ms"] / 1000, color=c, marker="o", markersize=6, label=f"{label} p50")
        ax[1].plot(x, g["p99_ms"] / 1000, color=c, marker="o", markersize=6, linestyle="--", label=f"{label} p99")
        ax[2].plot(x, g["output_tok_s"], color=c, marker="o", markersize=6, label=label)
    for a, title, ylabel in [
        (ax[0], "Time to first token", "seconds"),
        (ax[1], "End-to-end latency", "seconds"),
        (ax[2], "Output throughput", "tokens / s"),
    ]:
        a.set_title(f"{title} ({name})")
        a.set_ylabel(ylabel)
        a.set_xlabel("concurrency")
        a.set_xscale("log", base=2)
        a.set_xticks(sorted(runs["concurrency"].unique()))
        a.get_xaxis().set_major_formatter(ScalarFormatter())
        a.set_ylim(bottom=0)
        a.legend(fontsize=8)
    fig.savefig(f"{RESULTS}/sweep_{name}.png", dpi=150)
    plt.close(fig)


def plot_ttft_cdf(name: str, labels: tuple[str, str]) -> None:
    """TTFT distribution per request length. Solid and dashed separate the
    length classes; colour stays on the policy."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    data = ok.loc[(ok["concurrency"] == TIMELINE_CONCURRENCY) & ok["label"].isin(labels)]
    styles = ["-", "--", ":"]
    for label in labels:
        for i, (cls, g) in enumerate(sorted(data.loc[data["label"] == label].groupby("cls"),
                                            key=lambda kv: int(str(kv[0]).split()[0]))):
            x = g["ttft_ms"].sort_values().to_numpy() / 1000
            y = [(k + 1) / len(x) for k in range(len(x))]
            ax.step(x, y, where="post", color=COLORS[policy(label)],
                    linestyle=styles[i % len(styles)], label=f"{label}, {cls}")
    ax.set_title(f"TTFT distribution ({name} workload, concurrency {TIMELINE_CONCURRENCY})")
    ax.set_xlabel("time to first token (s)")
    ax.set_ylabel("share of requests")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right")
    fig.savefig(f"{RESULTS}/ttft_cdf_{name}.png", dpi=150)
    plt.close(fig)


for name, labels in PAIRS.items():
    missing = [l for l in labels if l not in set(runs["label"])]
    if missing:
        print(f"skipping {name}: no runs for {missing}")
        continue
    plot_timeline(name, labels)
    plot_sweep(name, labels)
    plot_ttft_cdf(name, labels)
    print(f"wrote {RESULTS}/{{timeline,sweep,ttft_cdf}}_{name}.png")
