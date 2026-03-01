import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import gseapy as gp
import matplotlib.pyplot as plt
import sys
import os

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("功能模块 GO/KEGG 富集分析")
print("="*60)

# ==========================================
# 1. 快速重建网络与模块 (复用之前的逻辑)
# ==========================================
filename = 'string_interactions_short.tsv'
print(f"1. 正在读取网络数据: {filename} ...")

try:
    df = pd.read_csv(filename, sep='\t', header=None)
    last_col = df.shape[1] - 1
    df.rename(columns={0: 'node1', 1: 'node2', last_col: 'weight'}, inplace=True)
    df = df[['node1', 'node2', 'weight']]
    # 清洗
    df = df[df['node1'].astype(str) != 'node1']
    df = df[df['node1'].astype(str) != '#node1']
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df.dropna(subset=['weight'], inplace=True)
    # 归一化
    if df['weight'].max() > 1.0:
        df['weight'] = df['weight'] / 1000.0
except Exception as e:
    print(f"读取失败: {e}")
    sys.exit(1)

# 构建网络
G = nx.from_pandas_edgelist(df, 'node1', 'node2', edge_attr='weight')
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()

print(f"-> 网络构建完成: {G_lcc.number_of_nodes()} 节点")

# 划分模块
print("2. 正在执行 Louvain 模块划分...")
partition = community_louvain.best_partition(G_lcc, weight='weight')

# 整理模块数据
from collections import defaultdict
mod_groups = defaultdict(list)
for node, mod_id in partition.items():
    mod_groups[mod_id].append(node)

# 按大小排序，取前3个
top_modules = sorted(mod_groups.items(), key=lambda x: len(x[1]), reverse=True)[:3]

print(f"-> 识别到 {len(mod_groups)} 个模块，将对前 3 个核心模块进行富集分析。")

# ==========================================
# 2. 定义富集分析函数
# ==========================================
def run_enrichment(gene_list, module_name, description):
    print(f"\n>>> 正在分析 {module_name} ({description})...")
    print(f"    包含基因数: {len(gene_list)} (如: {gene_list[:3]}...)")
    
    # 选用的基因集库
    # GO_Biological_Process_2021: GO生物学过程
    # KEGG_2021_Human: KEGG通路
    gene_sets = ['GO_Biological_Process_2021', 'KEGG_2021_Human']
    
    try:
        # 调用 gseapy 访问 Enrichr API
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=gene_sets,
                         organism='Human', 
                         outdir=None, # 不自动输出，手动画图
                         cutoff=0.05) # P值阈值
        
        # 结果存入 DataFrame
        res = enr.results
        
        if res.empty:
            print(f"警告: {module_name} 未发现显著富集结果。")
            return
        
        # 筛选 P-value < 0.01 的显著项
        res_sig = res[res['P-value'] < 0.05].copy()
        res_sig = res_sig.sort_values('P-value') # 按显著性排序
        
        # 保存表格
        csv_name = f"enrichment_{module_name}.csv"
        res_sig.to_csv(csv_name, index=False)
        print(f"结果已保存至: {csv_name}")
        
        # === 可视化 (绘制气泡图) ===
        # 为了图表美观，只取最显著的前 10 个 GO 和前 10 个 KEGG
        top_terms = pd.concat([
            res_sig[res_sig['Gene_set'].str.contains('GO')].head(10),
            res_sig[res_sig['Gene_set'].str.contains('KEGG')].head(10)
        ])
        
        if top_terms.empty:
            return

        # 使用 gseapy 自带的绘图功能 (Dotplot)
        # 这里的 top_term 参数控制显示的条目数
        ax = gp.dotplot(top_terms,
                        title=f"{module_name} GO/KEGG Enrichment",
                        cmap='viridis_r', # 颜色方案
                        size=10, # 字体大小
                        figsize=(8, len(top_terms)/2 + 2)) # 动态调整高度
        
        # 保存图片
        img_name = f"enrichment_plot_{module_name}.png"
        plt.savefig(img_name, bbox_inches='tight', dpi=300)
        print(f"富集气泡图已保存至: {img_name}")
        plt.close() # 关闭画布，防止内存溢出

    except Exception as e:
        print(f"分析出错 (可能是网络问题): {e}")

# ==========================================
# 3. 批量执行分析
# ==========================================
# 模块名称映射
module_names = ["Module_1 (Growth)", "Module_2 (Cell_Cycle)", "Module_3 (DNA_Repair)"]

for i, (mod_id, genes) in enumerate(top_modules):
    name = f"Module_{i+1}"
    # 简单的描述
    desc = f"Size: {len(genes)}"
    run_enrichment(genes, name, desc)

print("\n"+"="*60)
print("所有富集分析已完成！请查看生成的 .csv 和 .png 文件。")
print("="*60)