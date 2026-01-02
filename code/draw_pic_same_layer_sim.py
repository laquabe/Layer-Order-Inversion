import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 配置区 =====
ROOT = Path("/data/xkliu/Knowledge_Edit/analyze/outputs/gpt-j-6B_classified_split_results/analyze_same_layer_sim")
CLASSES = {
    "class_1_results": "Both Correct",
    "class_3_results": "Single Correct, Multi Wrong"
}
MODULE = "mlp_fc_in"    # 自选模块: mlp.fc_in / mlp.fc_out / attn_out_proj
TOKENS = ["last_subject", "last"]
JUMP_COMBINATIONS = ["1-of-2", "2-of-2", "1-of-3", "2-of-3", "3-of-3", "1-of-4", "3-of-4", "2-of-4",  "4-of-4"]  # 跳数组合
OUT_DIR = ROOT / f"aggregated_bar_{MODULE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 收集统计数据 =====
records = []
for cdir, cname in CLASSES.items():
    for jump in JUMP_COMBINATIONS:
        for m_t in TOKENS:
            for s_t in TOKENS:
                pair = f"{m_t}__{s_t}"
                fpath = ROOT / cdir / MODULE / "combined" / f"single_{jump}__{pair}_L28.npy"
                # print(fpath)
                if not fpath.exists():
                    print(fpath)
                    continue
                A = np.load(fpath)
                records.append({
                    "class": cname,
                    "jump": jump,
                    "pair": pair,
                    "module": MODULE,
                    "mean": A.mean(),
                    "std": A.std(),
                    "min": A.min(),
                    "max": A.max(),
                })

df = pd.DataFrame(records)
csv_path = OUT_DIR / f"aggregated_summary_{MODULE}.csv"
df.to_csv(csv_path, index=False)
print(f"[SAVED] Summary CSV: {csv_path}")

# ===== 绘制柱状图（均值 + 方差） =====
# ===== 绘制 mean ± std 箱图风格图 =====
pairs = sorted(df["pair"].unique())
x_jump = np.arange(len(JUMP_COMBINATIONS))
box_w = 0.28  # 控制箱体宽度

plt.style.use("seaborn-v0_8-whitegrid")

# 配色
COLORS = {
    "Both Correct": "#4C72B0",        
    "Single Correct, Multi Wrong": "#DD8452"
}

HATCH = {
    "Both Correct": "///",
    "Single Correct, Multi Wrong": ".."
}

def draw_box(ax, x_center, mean, std, color, hatch):
    top = mean + std
    bottom = mean - std

    # 箱体
    ax.fill_between(
        [x_center - box_w/2, x_center + box_w/2],
        [top, top],
        [bottom, bottom],
        color=color,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.4,
        hatch=hatch
    )

    # 中线（均值）
    ax.plot(
        [x_center - box_w/2, x_center + box_w/2],
        [mean, mean],
        color="black",
        linewidth=1.8
    )


for pair in pairs:

    plt.figure(figsize=(12, 5), dpi=150)
    ax = plt.gca()

    pair_df = df[df["pair"] == pair]

    # y 轴动态范围
    ymin = pair_df["mean"].min() - pair_df["std"].max() * 1.6
    ymax = pair_df["mean"].max() + pair_df["std"].max() * 1.6

    for i, cname in enumerate(CLASSES.values()):
        subset = df[(df["pair"] == pair) & (df["class"] == cname)]

        # 按 jump 顺序取 mean/std
        means = [
            subset[subset["jump"] == j]["mean"].values[0]
            if j in subset["jump"].values else np.nan
            for j in JUMP_COMBINATIONS
        ]
        stds = [
            subset[subset["jump"] == j]["std"].values[0]
            if j in subset["jump"].values else 0
            for j in JUMP_COMBINATIONS
        ]

        for k, j in enumerate(JUMP_COMBINATIONS):
            mean = means[k]
            std = stds[k]

            if np.isnan(mean):
                continue

            x_pos = x_jump[k] + (i - 0.5) * box_w * 1.2  # 左右错开

            draw_box(
                ax,
                x_pos,
                mean,
                std,
                COLORS[cname],
                HATCH[cname]
            )

    # x 轴设置
    ax.set_xticks(x_jump)
    ax.set_xticklabels(JUMP_COMBINATIONS, rotation=30, fontsize=10)
    ax.set_xlabel("Jump Combination (n-of-m)", fontsize=12)

    # y 轴
    ax.set_ylabel("Similarity (mean ± std)", fontsize=12)
    ax.set_ylim(ymin, ymax)

    # 图例构造（自定义标签）
    handles = []
    for cname in CLASSES.values():
        handles.append(
            plt.Rectangle((0,0), 1, 1,
                          facecolor=COLORS[cname], alpha=0.4,
                          edgecolor="black", hatch=HATCH[cname])
        )
    ax.legend(handles, CLASSES.values(), fontsize=11, framealpha=0.8)

    # 标题
    plt.title(f"{pair} · {MODULE}", fontsize=14, weight="bold")

    plt.tight_layout()

    save_path = OUT_DIR / f"aggregated_{pair}_{MODULE}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVED] {save_path}")


