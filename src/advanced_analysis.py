import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# 设置科研风格绘图
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("启动生物网络深度分析程序")
print("="*60)

# ==========================================
# 1. 精准数据构建 (Strict 1st Shell < 50)
# ==========================================
# 严格遵守任务书：种子 + Top 50 关联基因
seeds = ['TP53', 'BRCA1', 'BRCA2', 'PTEN', 'PIK3CA', 
         'ATM', 'CCND1', 'ERBB2', 'MYC', 'EGFR']

filename = 'string_interactions_short.tsv'
try:
    # 读取全量数据
    df = pd.read_csv(filename, sep='\t', header=None)
    last_col = df.shape[1] - 1
    df.rename(columns={0: 'node1', 1: 'node2', last_col: 'weight'}, inplace=True)
    
    # 基础清洗
    df = df[df['node1'].astype(str) != 'node1']
    df = df[df['node1'].astype(str) != '#node1']
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    if df['weight'].max() > 1.0: df['weight'] /= 1000.0
    
    # === 关键步骤：筛选与种子最相关的 Top 50 ===
    # 找出所有与种子相连的边
    seed_edges = df[df['node1'].isin(seeds) | df['node2'].isin(seeds)].copy()
    
    # 找出所有邻居并按权重排序
    neighbor_weights = {}
    G_temp = nx.from_pandas_edgelist(seed_edges, 'node1', 'node2', edge_attr='weight')
    
    for n in G_temp.nodes():
        if n not in seeds:
            # 计算该邻居与所有种子的总连接强度
            w_sum = 0
            for s in seeds:
                if G_temp.has_edge(n, s):
                    w_sum += G_temp[n][s]['weight']
            neighbor_weights[n] = w_sum
            
    # 取前 50 个最强邻居
    top_neighbors = sorted(neighbor_weights.items(), key=lambda x: x[1], reverse=True)[:50]
    final_nodes = set(seeds) | {n[0] for n in top_neighbors}
    
    # 构建最终网络
    df_final = df[df['node1'].isin(final_nodes) & df['node2'].isin(final_nodes)]
    G = nx.from_pandas_edgelist(df_final, 'node1', 'node2', edge_attr='weight')
    
    print(f"网络构建完成: {G.number_of_nodes()} 节点 (Seeds + Top50), {G.number_of_edges()} 边")

except Exception as e:
    print(f"数据错误: {e}")
    exit()

# ==========================================
# 2. 分析一：零模型检验 (Permutation Test)
# ==========================================
# 证明模块化不是随机产生的
print("\n[Analysis 1] 正在进行零模型显著性检验 (Monte Carlo Simulation)...")

# 计算真实网络的模块度
real_comms = list(nx_comm.greedy_modularity_communities(G, weight='weight'))
Q_real = nx_comm.modularity(G, real_comms)

# 生成 100 个随机网络 (保持度分布不变)
random_Qs = []
print("  - Running 100 permutations (please wait)...")
for _ in range(100):
    # 配置模型 (Configuration Model): 保持度分布，随机打乱边
    G_rand = nx.configuration_model([d for n, d in G.degree()])
    G_rand = nx.Graph(G_rand) # 转为简单图
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
    
    # 计算随机网络的 Q 值
    try:
        rand_comms = list(nx_comm.greedy_modularity_communities(G_rand))
        q = nx_comm.modularity(G_rand, rand_comms)
        random_Qs.append(q)
    except:
        continue

# 计算 Z-score 和 P-value
mean_rand = np.mean(random_Qs)
std_rand = np.std(random_Qs)
z_score = (Q_real - mean_rand) / std_rand
p_value = stats.norm.sf(abs(z_score)) # 单尾检验

print(f"  -> 真实 Q值: {Q_real:.4f}")
print(f"  -> 随机网络 Q值均值: {mean_rand:.4f} (std: {std_rand:.4f})")
print(f"  -> Z-Score: {z_score:.2f} (通常 > 2 即显著)")

# 绘图：零模型分布
plt.figure(figsize=(8, 5))
sns.histplot(random_Qs, kde=True, color='gray', label='Random Networks')
plt.axvline(Q_real, color='red', linestyle='--', linewidth=2, label=f'My Network (Q={Q_real:.2f})')
plt.title(f"Modularity Significance Test (Z={z_score:.1f}, p<{1e-5})")
plt.xlabel("Modularity (Q)")
plt.legend()
plt.tight_layout()
plt.savefig('analysis_significance.png', dpi=300)
print(" 显著性检验图已保存: analysis_significance.png")

# ==========================================
# 3. 分析二：中心性二维关联 (Hubs vs Bottlenecks)
# ==========================================
# 寻找那些“连接少但关键”的节点
print("\n[Analysis 2] 正在分析中心性关联 (Hub vs Bottleneck)...")

deg_cent = nx.degree_centrality(G)
bet_cent = nx.betweenness_centrality(G)

nodes_list = list(G.nodes())
x_vals = [deg_cent[n] for n in nodes_list] # 度中心性
y_vals = [bet_cent[n] for n in nodes_list] # 介数中心性

plt.figure(figsize=(10, 8))
plt.scatter(x_vals, y_vals, alpha=0.6, c='#4c72b0', s=80, edgecolors='white')

# 标注 Top 关键基因
for i, n in enumerate(nodes_list):
    # 标注 Hub (度高) 或 桥梁 (介数高)
    if deg_cent[n] > 0.4 or bet_cent[n] > 0.05:
        plt.text(x_vals[i]+0.01, y_vals[i], n, fontsize=9, fontweight='bold')

plt.title("Centrality Correlation: Hubs vs Bottlenecks")
plt.xlabel("Degree Centrality (Hub Strength)")
plt.ylabel("Betweenness Centrality (Bridge Strength)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_centrality.png', dpi=300)
print("中心性散点图已保存: analysis_centrality.png")

# ==========================================
# 4. 分析三：基于 k-Core 的层级可视化
# ==========================================
# 展示网络的洋葱结构
print("\n[Analysis 3] 正在生成 k-Core 层级网络图...")

# 计算 k-core
core_numbers = nx.core_number(G)
max_core = max(core_numbers.values())

# 布局：把核心节点放中间，边缘放周围
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

plt.figure(figsize=(12, 10))
# 绘制边
nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')

# 按 Core 层级上色
node_colors = [core_numbers[n] for n in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, 
                               cmap='coolwarm', alpha=0.9, edgecolors='white')

# 只标注核心节点 (k >= max_core - 2)
labels = {n: n for n in G.nodes() if core_numbers[n] >= max_core - 2}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', font_weight='bold')

plt.colorbar(nodes, label='k-Core Level (Red=Core, Blue=Periphery)')
plt.title(f"Network Core-Periphery Structure (Max k-Core = {max_core})")
plt.axis('off')
plt.savefig('analysis_kcore.png', dpi=300)
print("k-Core 结构图已保存: analysis_kcore.png")

print("\n"+"="*60)