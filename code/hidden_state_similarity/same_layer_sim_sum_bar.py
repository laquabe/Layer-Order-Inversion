import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ======== 配置区域（按你的磁盘实际情况修改）========
SRC_ROOT = Path("outputs/gpt-j-6B_classified_split_results")
CLASS_DIRS = ["class_1_results", "class_3_results"]          # 你要分析的类目录（按磁盘实际）
MODULE_NAME = "mlp_fc_out"                                 # 或 "mlp.fc_in" / "mlp.fc_out"

JUMP_COMBINATIONS =  ['1-of-2', '1-of-3', '1-of-4', '2-of-2', '2-of-3', '2-of-4', '3-of-3', '3-of-4', '4-of-4']  # 示例跳数组合（实际数据中会有）
TOKENS = ["last_subject", "last"]

L_EXPECT = 28                                                 # 你的矩阵层数（Lm=Ls=28），如不同可设为 None 并自动检查
LAYER_IDX = 25                                               # 选定具体层(0-based)。None 表示不选层；绘图时可选择取均值/指定层

# 只画哪些 token 对（可选）。None 表示全部；也可列举少量关心的pair降低图的复杂度
TOKEN_PAIRS = None  # 例如: [("last","last"), ("last_subject","last")]

# ======== 读取与组织： results[class_dir][(m_t,s_t)][jump_comb] = diag(np.ndarray [L]) ========
results = {}

pairs = TOKEN_PAIRS if TOKEN_PAIRS is not None else [(mt, st) for mt in TOKENS for st in TOKENS]

for class_dir in CLASS_DIRS:
    results[class_dir] = {}
    for (m_t, s_t) in pairs:
        results[class_dir][(m_t, s_t)] = {}
        for jump in JUMP_COMBINATIONS:
            pattern = SRC_ROOT / class_dir / MODULE_NAME / "combined" / f"single_{jump}__{m_t}__{s_t}_Lm28_Ls28.npy"
            if not pattern.exists():
                # 若不存在可跳过或给出提示
                # print(f"[WARN] Missing: {pattern}")
                continue
            A = np.load(pattern)  # [Lm, Ls]
            # 校验方阵与层数（可选）
            if L_EXPECT is not None:
                if A.shape[0] != L_EXPECT or A.shape[1] != L_EXPECT:
                    print(f"[WARN] Shape mismatch: {pattern} -> {A.shape}, expect ({L_EXPECT},{L_EXPECT})")
            diag = np.diag(A)     # [L]
            results[class_dir][(m_t, s_t)][jump] = diag

# ======== 简单检查：统计每类下读取到的 (pair, jump) 数量 ========
for class_dir in CLASS_DIRS:
    cnt = sum(len(v) for v in results[class_dir].values())
    print(f"[INFO] {class_dir}: loaded {cnt} (pair, jump) diagonals")

# ======== 绘图：每个类一张图，横轴=跳数组合，颜色=token对；Y取某一层或层均值 ========
#   - X: 跳数组合（n-of-m）
#   - color: token对 (m_t__s_t)
#   - Y: 如果 LAYER_IDX is not None -> diag[LAYER_IDX]；否则用对角线均值（这一步不是“再平均样本”，只是把同层序列压成一个展示标量）
def pick_scalar(diag: np.ndarray, layer_idx=None):
    if layer_idx is None:
        # 仅用于图形展示，把对角线的 28 维汇总成一个数（不是再平均样本）
        return float(diag.mean())
    else:
        if layer_idx < 0 or layer_idx >= len(diag):
            raise ValueError(f"Invalid layer_idx={layer_idx}, diag length={len(diag)}")
        return float(diag[layer_idx])

for class_dir in CLASS_DIRS:
    plt.figure(figsize=(12, 6))
    
    # 横轴：token pairs 的索引
    x = np.arange(len(pairs))  # token pair 数量
    bar_w = 0.8 / len(JUMP_COMBINATIONS)  # 每个柱子的宽度

    # 遍历所有 token 对
    for i, jump in enumerate(JUMP_COMBINATIONS):
        ys = []
        for m_t, s_t in pairs:
            diag = results[class_dir].get((m_t, s_t), {}).get(jump, None)
            val = pick_scalar(diag, LAYER_IDX) if diag is not None else np.nan
            ys.append(val)
        # 绘制柱形图，x轴加偏移量，y轴是均值
        bars = plt.bar(x + i * bar_w, ys, width=bar_w, label=f"{jump}")
        plt.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    # 设置标签和标题
    plt.xticks(x + (len(JUMP_COMBINATIONS) - 1) * bar_w / 2, [f"{m_t}__{s_t}" for m_t, s_t in pairs])  # token pair 标签
    title = f"{class_dir} · {MODULE_NAME} · " + (f"Layer {LAYER_IDX}" if LAYER_IDX is not None else "Diag-Mean")
    plt.title(title)
    plt.ylabel("Similarity (same-layer)")
    plt.xlabel("Token Pairs")
    plt.legend(title="Hops", ncols=3, fontsize=9)
    plt.tight_layout()

    # 保存图表
    out_png = f"./outputs/gpt-j-6B_classified_split_results/same_layer_pic/{MODULE_NAME}/hops/{class_dir}__{MODULE_NAME}__{'layer'+str(LAYER_IDX) if LAYER_IDX is not None else 'diagmean'}.png"
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print(f"[SAVED] {out_png}")
    plt.close()