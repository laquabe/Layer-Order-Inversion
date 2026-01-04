import jsonlines
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

def plot_statistics(df, fig_path):
    """根据汇总的统计数据绘制柱状图和折线图在同一图表中"""
    
    # 设置 Seaborn 样式
    sns.set(style="whitegrid")
    
    # 创建图形和两个 y 轴
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 创建一个新的列 "hop_count_hop_idx" 用于 x 轴
    df["hop_count_hop_idx"] = df["hop_count"].astype(str) + "-" + df["hop_idx"].astype(str)

    # --- 1. 绘制柱状图：实体出现百分比 ---
    bar_plot = sns.barplot(x="hop_count_hop_idx", y="percentage", hue="hop_count", data=df, ax=ax1, ci=None)
    ax1.set_title("Entity Appearance Percentage and Average First Layer")
    ax1.set_xlabel("hop_count - hop_idx")
    ax1.set_ylabel("Appearance Percentage (%)")
    ax1.legend(title="hop_count", loc="upper left")

    # 在柱形上方添加数值
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.1f') + '%',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=10,
                          xytext=(0, 8), textcoords='offset points')

    # --- 2. 使用双 y 轴，创建第二个 y 轴 ---
    ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴

    # --- 3. 绘制折线图：最早层数平均值 ---
    line_plot = sns.lineplot(x="hop_count_hop_idx", y="avg_first_layer", hue="hop_count", data=df, ax=ax2, markers="o", linewidth=2, palette="muted")
    ax2.set_ylabel("Average First Layer")

    ax1.grid(False)
    ax2.grid(False)
    
    # 设置整体布局
    plt.tight_layout()
    plt.xticks(rotation=45)  # 旋转 x 轴标签，避免重叠
    # 保存图表
    plt.savefig(fig_path)
    # plt.show()


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    with jsonlines.open(file_path, 'r') as reader:
        data = [line for line in reader]
    return data


def analyze_entity_stats(jsonl_data):
    """根据 hop_count 和 hop_idx 汇总生成结果统计信息"""
    
    # 用于统计每个 hop_count - hop_idx 类别的生成结果出现次数，最早层数
    stats = defaultdict(lambda: {"count": 0, "first_layers": []})

    total_count = defaultdict(int)  # 记录每个 hop_count - hop_idx 类别的总问题数
    
    for entry in jsonl_data:
        hop_count = entry["hop_count"]
        hop_idx = entry["hop_idx"]
        first_layer = entry["first_layer"]

        # 更新 total_count，统计不同 hop_count - hop_idx 类别的问题数
        total_count[(hop_count, hop_idx)] += 1

        # 如果该条目有有效的生成结果（appear_ratio > 0）
        if entry["appear_ratio"] > 0:
            # 更新该类别下的统计信息
            stats[(hop_count, hop_idx)]["count"] += 1
            stats[(hop_count, hop_idx)]["first_layers"].append(first_layer)

    # 汇总统计结果
    result = []
    for (hop_count, hop_idx), data in sorted(total_count.items()):
        count = stats[(hop_count, hop_idx)]["count"]
        first_layers = stats[(hop_count, hop_idx)]["first_layers"]
        percentage = (count / total_count[(hop_count, hop_idx)]) * 100
        avg_first_layer = sum(first_layers) / len(first_layers) if first_layers else None  # 计算平均值

        result.append({
            "hop_count": hop_count,
            "hop_idx": hop_idx,
            "count": count,
            "percentage": percentage,
            "avg_first_layer": avg_first_layer  # 保存该类别的平均 first_layer
        })

    # 转为 DataFrame
    df = pd.DataFrame(result)
    return df

def save_summary(df, output_path):
    """保存汇总结果到 CSV 文件"""
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Summary saved to {output_path}")

def main():
    file_name = "class_1_both_correct_globalFilter90"  # 替换成你的文件名
    path_name = "results/Llama-3-8B_classified_split_results_sample/last/class_1_both_correct"
    input_jsonl = f"{path_name}/{file_name}.jsonl"  # 替换成你的 JSONL 文件路径
    output_summary = f"{path_name}/{file_name}_summray.csv" # 输出 CSV 文件路径
    fig_path = f"{path_name}/{file_name}_stats.png"  # 输出图表路径
    # 加载数据
    jsonl_data = load_jsonl(input_jsonl)

    # 汇总统计信息
    df_summary = analyze_entity_stats(jsonl_data)

    # 保存结果
    save_summary(df_summary, output_summary)

    plot_statistics(df_summary, fig_path)

if __name__ == "__main__":
    main()
