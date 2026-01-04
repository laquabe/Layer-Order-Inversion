import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -----------------------
# 配置你的 class 目录根路径
# -----------------------
root = Path("outputs/Llama-3-8B_classified_split_results/analyze_same_layer_sim")

# 自动发现 class_x_results 目录
class_dirs = sorted([d for d in root.glob("class_*_results") if d.is_dir()])

print("发现以下 class 目录：")
for d in class_dirs:
    print(" -", d)

# -----------------------
# 扫描所有 class 下的 combined/*.npy
# -----------------------
all_files = defaultdict(dict)  # key: filename → {class_name: path}
module = "mlp_fc_out"  # 这里指定要对比的模块名称

for class_dir in class_dirs:
    class_name = class_dir.name  # 例如 class_1_results
    combined_dir = class_dir / module / "combined"
    
    if not combined_dir.exists():
        continue
    
    for npy_file in combined_dir.glob("*.npy"):
        all_files[npy_file.name][class_name] = npy_file

print("\n共发现同名 npy 组：", len(all_files))

# 输出对比图目录
out_dir = root / "comparison_plots" / module
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# 绘制每个同名 npy 的对比折线图
# -----------------------
for filename, mapping in all_files.items():

    plt.figure(figsize=(8, 5))

    for class_name, file_path in mapping.items():
        vec = np.load(file_path)
        plt.plot(vec, marker="o", label=class_name)

    plt.title(f"Comparison across classes: {filename}")
    plt.xlabel("Layer")
    plt.ylabel("Similarity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = out_dir / f"compare_{filename.replace('.npy','.png')}"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("✔ Saved:", save_path)

print("\n所有对比图已生成，目录：", out_dir)
