import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
import numpy as np

# ==========================================
# 1. 加载数据 & 构建网络 (保持上下文完整)
# ==========================================
filename = 'ppi_data.tsv'
try:
    df = pd.read_csv(filename, sep='\t')
except FileNotFoundError:
    print("【错误】找不到 ppi_data.tsv 文件，请确认路径。")
    exit()

# 清洗列名与分数归一化
df.columns = [c.strip('#').strip() for c in df.columns]
rename_map = {
    'string_identifier_1': 'node1', 'preferredName_A': 'node1', 
    'string_identifier_2': 'node2', 'preferredName_B': 'node2',
    'score': 'combined_score'
}
df.rename(columns=rename_map, inplace=True)
if df['combined_score'].max() > 1.0:
    df['combined_score'] = df['combined_score'] / 1000.0
df_clean = df[df['combined_score'] >= 0.7].copy()

# 构建带权重的网络 (权重有助于更准确地划分模块)
G = nx.from_pandas_edgelist(df_clean, 'node1', 'node2', edge_attr='combined_score')

# ==========================================
# 2. 核心算法：功能模块挖掘
# ==========================================
print("正在执行模块挖掘算法 (Greedy Modularity Maximization)...")

# 使用 NetworkX 的贪婪模块度算法
communities = nx_comm.greedy_modularity_communities(G, weight='combined_score')

# 按模块大小排序（大的在前）
communities = sorted(communities, key=len, reverse=True)

# 计算模块度 (Q值) - 这是题目要求的"质量评估"指标
q_score = nx_comm.modularity(G, communities, weight='combined_score')

# ==========================================
# 3. 结果输出与自动注释
# ==========================================
print("=" * 60)
print(f"【模块挖掘结果与质量评估】")
print(f"检测到的功能模块数量: {len(communities)}")
print(f"网络模块度 (Modularity Q): {q_score:.4f}")
if q_score > 0.3:
    print("-> 评价: 极佳 (Q > 0.3)，说明网络存在非常显著的功能复合物结构。")
else:
    print("-> 评价: 一般 (网络过于致密，模块边界可能不清晰)。")
print("-" * 60)

# 准备绘图颜色
node_color_map = {}
# 使用 tab10 调色板，保证颜色区分度
colors = plt.cm.tab10(np.linspace(0, 1, 10))

print("【各功能模块详细分析】")
for i, comm in enumerate(communities):
    # 将基因按度数(Degree)从高到低排序，排在前面的就是该模块的"核心/Leader"
    gene_list = sorted(list(comm), key=lambda x: G.degree(x), reverse=True)
    top_genes = gene_list[:8] # 取前8个展示
    
    # 颜色分配
    color = colors[i % len(colors)]
    for node in gene_list:
        node_color_map[node] = color
    
    print(f"\n>>> 模块 {i+1} (包含 {len(gene_list)} 个基因)")
    print(f"   核心成员: {', '.join(top_genes)} ...")
    
    # --- 自动生物学功能推测逻辑 (辅助写报告) ---
    func_guess = "辅助/未知功能模块"
    # 规则1: DNA修复相关
    if any(g in gene_list for g in ['BRCA1', 'ATM', 'RAD50', 'H2AX', 'TP53']):
        func_guess = "DNA 损伤修复 / 基因组稳定性 (DNA Repair)"
    # 规则2: 细胞周期相关
    elif any(g in gene_list for g in ['CCND1', 'CDK4', 'CDK2', 'RB1', 'E2F1']):
        func_guess = "细胞周期调控 (Cell Cycle Regulation)"
    # 规则3: 生长信号相关
    elif any(g in gene_list for g in ['EGFR', 'ERBB2', 'PIK3CA', 'PTEN', 'AKT1']):
        func_guess = "生长因子信号通路 (PI3K-Akt / Growth Signaling)"
    elif any(g in gene_list for g in ['MLH1', 'MSH2']):
        func_guess = "错配修复 (Mismatch Repair)"
        
    print(f"   [推测生物学功能]: {func_guess}")

print("\n" + "=" * 60)

# ==========================================
# 4. 可视化模块结构
# ==========================================
plt.figure(figsize=(12, 10))
# 使用 spring layout，k参数大一点可以让不同团块分得更开
pos = nx.spring_layout(G, seed=42, k=0.25) 

node_colors = [node_color_map.get(n, '#999999') for n in G.nodes()]

nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, alpha=0.9, edgecolors='white')
nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', font_weight='bold')

plt.title(f"Functional Modules in PPI Network (Q={q_score:.2f})", fontsize=16)
plt.axis('off')

save_path = 'ppi_modules_final.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"模块可视化图已保存为 '{save_path}'")
plt.show()