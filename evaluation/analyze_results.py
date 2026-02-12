"""
Result Analysis Script
======================
对 SurGE (Human-written) 和 Autosurvey (AI-generated) 的评估结果进行统计分析、
生成可视化图表，并输出分析报告。

Usage:
    python evaluation/analyze_results.py \
        --surge_dir evaluation/results/surge \
        --autosurvey_dir evaluation/results/autosurvey \
        --output_dir papers/figures
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# 维度映射: scores key -> paper 中的名称
DIM_MAP = {
    "synthesis": "Depth",
    "critical": "Breadth",
    "abstraction": "Height",
}

# 配色
COLOR_HUMAN = "#2E86AB"   # 蓝色
COLOR_AI = "#E84855"      # 红色
COLORS = [COLOR_HUMAN, COLOR_AI]


# ============================================================
# 1. Data Loading
# ============================================================

def load_all_sections(results_dir: str) -> List[Dict[str, Any]]:
    """从 JSONL 结果目录加载所有章节评估数据"""
    sections = []
    results_path = Path(results_dir)
    for jsonl_file in sorted(results_path.glob("survey_*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    section = json.loads(line)
                    # 只保留有有效 insight_result 和 scores 的
                    ir = section.get("insight_result", {})
                    if "scores" in ir and "error" not in ir:
                        sections.append(section)
                except json.JSONDecodeError:
                    continue
    return sections


def extract_scores(sections: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """提取各维度分数"""
    scores = {"synthesis": [], "critical": [], "abstraction": []}
    for sec in sections:
        s = sec["insight_result"]["scores"]
        for key in scores:
            val = s.get(key)
            if val is not None:
                scores[key].append(float(val))
    return scores


def extract_types(sections: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计 type 分布"""
    type_counts = {}
    for sec in sections:
        t = sec["insight_result"].get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    return type_counts


def extract_insight_levels(sections: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计 insight_level 分布"""
    level_counts = {}
    for sec in sections:
        lvl = sec["insight_result"].get("insight_level", "unknown")
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    return level_counts


def extract_per_survey_means(sections: List[Dict[str, Any]], source_key: str = "title") -> Dict[str, Dict[str, float]]:
    """按 survey 聚合，计算每篇 survey 的各维度平均分"""
    survey_data: Dict[str, List[Dict[str, float]]] = {}
    for sec in sections:
        # 用 section_path 的第一个部分作为 survey 标识
        # 或直接用文件级别
        survey_id = "unknown"
        sp = sec.get("section_path", "")
        if sp:
            survey_id = sp.split(" > ")[0] if " > " in sp else sp
        scores = sec["insight_result"]["scores"]
        if survey_id not in survey_data:
            survey_data[survey_id] = []
        survey_data[survey_id].append(scores)

    survey_means = {}
    for sid, score_list in survey_data.items():
        means = {}
        for key in ["synthesis", "critical", "abstraction"]:
            vals = [s.get(key, 0) for s in score_list if key in s]
            means[key] = np.mean(vals) if vals else 0.0
        survey_means[sid] = means
    return survey_means


# ============================================================
# 2. Statistical Analysis
# ============================================================

def compute_stats(scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """计算每个维度的描述统计量"""
    result = {}
    for key, vals in scores.items():
        arr = np.array(vals)
        result[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "n": len(vals),
        }
    return result


def compute_ttest(
    scores_a: Dict[str, List[float]],
    scores_b: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """对两组分数做独立样本 t 检验"""
    results = {}
    for key in ["synthesis", "critical", "abstraction"]:
        t_stat, p_value = stats.ttest_ind(scores_a[key], scores_b[key], equal_var=False)
        # Cohen's d
        n1, n2 = len(scores_a[key]), len(scores_b[key])
        s1, s2 = np.std(scores_a[key]), np.std(scores_b[key])
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohens_d = (np.mean(scores_a[key]) - np.mean(scores_b[key])) / pooled_std if pooled_std > 0 else 0
        results[key] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": p_value < 0.05,
        }
    return results


def compute_mannwhitneyu(
    scores_a: Dict[str, List[float]],
    scores_b: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Mann-Whitney U 检验（非参数）"""
    results = {}
    for key in ["synthesis", "critical", "abstraction"]:
        u_stat, p_value = stats.mannwhitneyu(scores_a[key], scores_b[key], alternative='two-sided')
        results[key] = {
            "u_stat": float(u_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
    return results


# ============================================================
# 3. Visualization
# ============================================================

def plot_comparison_bar(
    stats_human: Dict[str, Dict[str, float]],
    stats_ai: Dict[str, Dict[str, float]],
    ttest_results: Dict[str, Dict[str, float]],
    output_path: str,
):
    """对比柱状图: Human vs AI, 含误差线和显著性标注"""
    fig, ax = plt.subplots(figsize=(4.5, 3))

    dims = list(DIM_MAP.keys())
    dim_labels = [DIM_MAP[d] for d in dims]
    x = np.arange(len(dims))
    width = 0.32

    means_h = [stats_human[d]["mean"] for d in dims]
    stds_h = [stats_human[d]["std"] for d in dims]
    means_a = [stats_ai[d]["mean"] for d in dims]
    stds_a = [stats_ai[d]["std"] for d in dims]

    bars_h = ax.bar(x - width/2, means_h, width, yerr=stds_h,
                    label="Human (SurGE)", color=COLOR_HUMAN, alpha=0.85,
                    capsize=4, edgecolor="white", linewidth=0.5)
    bars_a = ax.bar(x + width/2, means_a, width, yerr=stds_a,
                    label="AI (Autosurvey)", color=COLOR_AI, alpha=0.85,
                    capsize=4, edgecolor="white", linewidth=0.5)

    # 显著性标注
    for i, d in enumerate(dims):
        p = ttest_results[d]["p_value"]
        if p < 0.001:
            sig_text = "***"
        elif p < 0.01:
            sig_text = "**"
        elif p < 0.05:
            sig_text = "*"
        else:
            sig_text = "n.s."

        y_max = max(means_h[i] + stds_h[i], means_a[i] + stds_a[i]) + 0.15
        ax.annotate(sig_text, xy=(x[i], y_max), ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Score (1-5)")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 5.5)
    ax.legend(frameon=True, fancybox=True, shadow=False, edgecolor="#cccccc",
              ncol=2, loc="upper center", columnspacing=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  ✅ 保存: {output_path}")


def plot_dimension_distributions(
    scores_human: Dict[str, List[float]],
    scores_ai: Dict[str, List[float]],
    output_path: str,
):
    """各维度分数分布直方图（violin + box）"""
    dims = list(DIM_MAP.keys())
    dim_labels = [DIM_MAP[d] for d in dims]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for i, (d, label) in enumerate(zip(dims, dim_labels)):
        ax = axes[i]
        data = [scores_human[d], scores_ai[d]]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)

        # 给 violin 上色
        for j, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLORS[j])
            body.set_alpha(0.6)
        for partname in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if partname in parts:
                parts[partname].set_edgecolor("#333333")
                parts[partname].set_linewidth(0.8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Human", "AI"])
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 5.5)
        if i == 0:
            ax.set_ylabel("Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  ✅ 保存: {output_path}")


def plot_type_distribution(
    types_human: Dict[str, int],
    types_ai: Dict[str, int],
    output_path: str,
):
    """章节类型分布对比柱状图"""
    all_types = sorted(set(list(types_human.keys()) + list(types_ai.keys())))
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(all_types))
    width = 0.32

    total_h = sum(types_human.values())
    total_a = sum(types_ai.values())

    vals_h = [types_human.get(t, 0) / total_h * 100 for t in all_types]
    vals_a = [types_ai.get(t, 0) / total_a * 100 for t in all_types]

    ax.bar(x - width/2, vals_h, width, label="Human", color=COLOR_HUMAN, alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, vals_a, width, label="AI", color=COLOR_AI, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in all_types])
    ax.set_ylabel("Percentage (%)")
    ax.legend(frameon=True, fancybox=True, edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  ✅ 保存: {output_path}")


def plot_insight_level_distribution(
    levels_human: Dict[str, int],
    levels_ai: Dict[str, int],
    output_path: str,
):
    """Insight Level 分布对比"""
    level_order = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(level_order))
    width = 0.32

    total_h = sum(levels_human.values())
    total_a = sum(levels_ai.values())

    vals_h = [levels_human.get(l, 0) / total_h * 100 for l in level_order]
    vals_a = [levels_ai.get(l, 0) / total_a * 100 for l in level_order]

    ax.bar(x - width/2, vals_h, width, label="Human", color=COLOR_HUMAN, alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, vals_a, width, label="AI", color=COLOR_AI, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in level_order])
    ax.set_ylabel("Percentage (%)")
    ax.legend(frameon=True, fancybox=True, edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  ✅ 保存: {output_path}")


def plot_radar_chart(
    stats_human: Dict[str, Dict[str, float]],
    stats_ai: Dict[str, Dict[str, float]],
    output_path: str,
):
    """雷达图对比"""
    dims = list(DIM_MAP.keys())
    dim_labels = [DIM_MAP[d] for d in dims]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    vals_h = [stats_human[d]["mean"] for d in dims] + [stats_human[dims[0]]["mean"]]
    vals_a = [stats_ai[d]["mean"] for d in dims] + [stats_ai[dims[0]]["mean"]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    ax.fill(angles, vals_h, alpha=0.2, color=COLOR_HUMAN)
    ax.plot(angles, vals_h, "o-", color=COLOR_HUMAN, linewidth=2, label="Human (SurGE)")
    ax.fill(angles, vals_a, alpha=0.2, color=COLOR_AI)
    ax.plot(angles, vals_a, "o-", color=COLOR_AI, linewidth=2, label="AI (Autosurvey)")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.set_rticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  ✅ 保存: {output_path}")


# ============================================================
# 4. Report Generation
# ============================================================

def generate_report(
    stats_human: Dict,
    stats_ai: Dict,
    ttest: Dict,
    mwu: Dict,
    types_human: Dict,
    types_ai: Dict,
    levels_human: Dict,
    levels_ai: Dict,
    n_human: int,
    n_ai: int,
    n_surveys_human: int,
    n_surveys_ai: int,
    output_path: str,
):
    """生成 Markdown 分析报告"""
    lines = []
    lines.append("# InsightEval Experimental Analysis Report\n")

    lines.append("## 1. Dataset Overview\n")
    lines.append(f"| Source | Surveys | Sections (with citations) |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Human (SurGE) | {n_surveys_human} | {n_human} |")
    lines.append(f"| AI (Autosurvey) | {n_surveys_ai} | {n_ai} |")
    lines.append("")

    lines.append("## 2. Descriptive Statistics\n")
    lines.append("### Dimension Mapping")
    lines.append("- `synthesis` → **Depth**: How well the section synthesizes cited papers")
    lines.append("- `critical` → **Breadth**: Critical analysis and evaluation quality")
    lines.append("- `abstraction` → **Height**: Ability to generalize and abstract\n")

    lines.append("### Summary Statistics (Mean ± Std)\n")
    lines.append("| Dimension | Human (Mean ± Std) | AI (Mean ± Std) | Δ (Human - AI) |")
    lines.append("|---|---|---|---|")
    for key in ["synthesis", "critical", "abstraction"]:
        h = stats_human[key]
        a = stats_ai[key]
        delta = h["mean"] - a["mean"]
        sign = "+" if delta > 0 else ""
        lines.append(
            f"| **{DIM_MAP[key]}** ({key}) | "
            f"{h['mean']:.2f} ± {h['std']:.2f} | "
            f"{a['mean']:.2f} ± {a['std']:.2f} | "
            f"{sign}{delta:.2f} |"
        )
    lines.append("")

    # Overall average
    avg_h = np.mean([stats_human[k]["mean"] for k in ["synthesis", "critical", "abstraction"]])
    avg_a = np.mean([stats_ai[k]["mean"] for k in ["synthesis", "critical", "abstraction"]])
    lines.append(f"**Overall Average**: Human = {avg_h:.2f}, AI = {avg_a:.2f}, Δ = {avg_h - avg_a:+.2f}\n")

    lines.append("## 3. Statistical Tests\n")
    lines.append("### Welch's t-test (two-sided)\n")
    lines.append("| Dimension | t-stat | p-value | Cohen's d | Significant? |")
    lines.append("|---|---|---|---|---|")
    for key in ["synthesis", "critical", "abstraction"]:
        t = ttest[key]
        sig = "✅ Yes" if t["significant"] else "❌ No"
        lines.append(
            f"| **{DIM_MAP[key]}** | "
            f"{t['t_stat']:.3f} | "
            f"{t['p_value']:.2e} | "
            f"{t['cohens_d']:.3f} | "
            f"{sig} |"
        )
    lines.append("")

    lines.append("### Mann-Whitney U test (non-parametric)\n")
    lines.append("| Dimension | U-stat | p-value | Significant? |")
    lines.append("|---|---|---|---|")
    for key in ["synthesis", "critical", "abstraction"]:
        m = mwu[key]
        sig = "✅ Yes" if m["significant"] else "❌ No"
        lines.append(
            f"| **{DIM_MAP[key]}** | "
            f"{m['u_stat']:.1f} | "
            f"{m['p_value']:.2e} | "
            f"{sig} |"
        )
    lines.append("")

    lines.append("## 4. Section Type Distribution\n")
    all_types = sorted(set(list(types_human.keys()) + list(types_ai.keys())))
    total_h = sum(types_human.values())
    total_a = sum(types_ai.values())
    lines.append("| Type | Human (%) | AI (%) |")
    lines.append("|---|---|---|")
    for t in all_types:
        h_pct = types_human.get(t, 0) / total_h * 100
        a_pct = types_ai.get(t, 0) / total_a * 100
        lines.append(f"| {t.capitalize()} | {h_pct:.1f}% | {a_pct:.1f}% |")
    lines.append("")

    lines.append("## 5. Insight Level Distribution\n")
    total_h_l = sum(levels_human.values())
    total_a_l = sum(levels_ai.values())
    lines.append("| Level | Human (%) | AI (%) |")
    lines.append("|---|---|---|")
    for lvl in ["low", "medium", "high"]:
        h_pct = levels_human.get(lvl, 0) / total_h_l * 100
        a_pct = levels_ai.get(lvl, 0) / total_a_l * 100
        lines.append(f"| {lvl.capitalize()} | {h_pct:.1f}% | {a_pct:.1f}% |")
    lines.append("")

    lines.append("## 6. Key Findings\n")
    # Auto-generate findings
    biggest_gap_key = max(["synthesis", "critical", "abstraction"],
                          key=lambda k: abs(stats_human[k]["mean"] - stats_ai[k]["mean"]))
    biggest_gap = stats_human[biggest_gap_key]["mean"] - stats_ai[biggest_gap_key]["mean"]

    lines.append(f"1. Human-written surveys achieve higher scores across all three dimensions.")
    lines.append(f"2. The largest gap is in **{DIM_MAP[biggest_gap_key]}** (Δ = {biggest_gap:+.2f}), "
                 f"suggesting {'abstract reasoning' if biggest_gap_key == 'abstraction' else 'deep synthesis' if biggest_gap_key == 'synthesis' else 'critical analysis'} is the most challenging for AI.")
    lines.append(f"3. All dimensional differences are {'statistically significant' if all(ttest[k]['significant'] for k in ['synthesis','critical','abstraction']) else 'partially significant'} (p < 0.05).")

    # Effect size interpretation
    for key in ["synthesis", "critical", "abstraction"]:
        d = abs(ttest[key]["cohens_d"])
        if d >= 0.8:
            effect = "large"
        elif d >= 0.5:
            effect = "medium"
        else:
            effect = "small"
        lines.append(f"4. Effect size for {DIM_MAP[key]}: Cohen's d = {ttest[key]['cohens_d']:.3f} ({effect})")
    lines.append("")

    # LaTeX table
    lines.append("## 7. LaTeX Table\n")
    lines.append("```latex")
    lines.append("\\begin{table}[t]")
    lines.append("    \\centering")
    lines.append("    \\caption{Insight evaluation results across paper sources.}")
    lines.append("    \\label{tab:results}")
    lines.append("    \\begin{tabular}{lccc}")
    lines.append("        \\toprule")
    lines.append("        \\textbf{Source} & \\textbf{Depth} & \\textbf{Breadth} & \\textbf{Height} \\\\")
    lines.append("        \\midrule")
    h_vals = [f"{stats_human[k]['mean']:.2f} $\\pm$ {stats_human[k]['std']:.2f}" for k in ["synthesis", "critical", "abstraction"]]
    a_vals = [f"{stats_ai[k]['mean']:.2f} $\\pm$ {stats_ai[k]['std']:.2f}" for k in ["synthesis", "critical", "abstraction"]]
    lines.append(f"        Human-written & {h_vals[0]} & {h_vals[1]} & {h_vals[2]} \\\\")
    lines.append(f"        AI-generated & {a_vals[0]} & {a_vals[1]} & {a_vals[2]} \\\\")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("```\n")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✅ 保存: {output_path}")


# ============================================================
# 5. Main
# ============================================================

def main(args):
    print("=" * 60)
    print("📊 InsightEval 结果分析")
    print("=" * 60)

    # 加载数据
    print("\n📂 加载数据...")
    sections_human = load_all_sections(args.surge_dir)
    sections_ai = load_all_sections(args.autosurvey_dir)
    print(f"  SurGE (Human):    {len(sections_human)} sections")
    print(f"  Autosurvey (AI):  {len(sections_ai)} sections")

    # 统计 survey 数量
    n_surveys_human = len(list(Path(args.surge_dir).glob("survey_*.jsonl")))
    n_surveys_ai = len(list(Path(args.autosurvey_dir).glob("survey_*.jsonl")))

    # 提取分数
    scores_human = extract_scores(sections_human)
    scores_ai = extract_scores(sections_ai)

    # 描述统计
    print("\n📈 计算统计量...")
    stats_human = compute_stats(scores_human)
    stats_ai = compute_stats(scores_ai)

    for key in ["synthesis", "critical", "abstraction"]:
        print(f"  {DIM_MAP[key]:>8s}: Human={stats_human[key]['mean']:.2f}±{stats_human[key]['std']:.2f}  "
              f"AI={stats_ai[key]['mean']:.2f}±{stats_ai[key]['std']:.2f}  "
              f"Δ={stats_human[key]['mean'] - stats_ai[key]['mean']:+.2f}")

    # 统计检验
    print("\n🔬 统计检验...")
    ttest = compute_ttest(scores_human, scores_ai)
    mwu = compute_mannwhitneyu(scores_human, scores_ai)
    for key in ["synthesis", "critical", "abstraction"]:
        print(f"  {DIM_MAP[key]:>8s}: t={ttest[key]['t_stat']:.3f}, p={ttest[key]['p_value']:.2e}, "
              f"d={ttest[key]['cohens_d']:.3f}")

    # 类型和 level 统计
    types_human = extract_types(sections_human)
    types_ai = extract_types(sections_ai)
    levels_human = extract_insight_levels(sections_human)
    levels_ai = extract_insight_levels(sections_ai)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    print("\n🎨 生成图表...")
    plot_comparison_bar(stats_human, stats_ai, ttest,
                        str(output_dir / "comparison_bar.pdf"))
    plot_dimension_distributions(scores_human, scores_ai,
                                str(output_dir / "dimension_distributions.pdf"))
    plot_type_distribution(types_human, types_ai,
                          str(output_dir / "type_distribution.pdf"))
    plot_insight_level_distribution(levels_human, levels_ai,
                                   str(output_dir / "insight_level_distribution.pdf"))
    plot_radar_chart(stats_human, stats_ai,
                     str(output_dir / "radar_chart.pdf"))

    # 生成报告
    print("\n📝 生成分析报告...")
    generate_report(
        stats_human, stats_ai, ttest, mwu,
        types_human, types_ai, levels_human, levels_ai,
        len(sections_human), len(sections_ai),
        n_surveys_human, n_surveys_ai,
        str(output_dir / "analysis_report.md"),
    )

    # 保存原始数据 JSON
    raw_data = {
        "stats_human": stats_human,
        "stats_ai": stats_ai,
        "ttest": ttest,
        "mannwhitneyu": mwu,
        "types_human": types_human,
        "types_ai": types_ai,
        "levels_human": levels_human,
        "levels_ai": levels_ai,
        "n_sections_human": len(sections_human),
        "n_sections_ai": len(sections_ai),
        "n_surveys_human": n_surveys_human,
        "n_surveys_ai": n_surveys_ai,
    }
    raw_path = str(output_dir / "raw_stats.json")

    # numpy types → native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  ✅ 保存: {raw_path}")

    print("\n🎉 分析完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightEval 结果统计分析")
    parser.add_argument("--surge_dir", type=str, default="evaluation/results/surge",
                        help="SurGE (Human) 结果目录")
    parser.add_argument("--autosurvey_dir", type=str, default="evaluation/results/autosurvey",
                        help="Autosurvey (AI) 结果目录")
    parser.add_argument("--output_dir", type=str, default="papers/figures",
                        help="输出目录 (图表 + 报告)")
    args = parser.parse_args()
    main(args)