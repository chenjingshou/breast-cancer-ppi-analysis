import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 加载数据 & 构建网络
# ==========================================
filename = 'ppi_data.tsv'
try:
    df = pd.read_csv(filename, sep='\t')
except FileNotFoundError:
    print("【错误】找不到 ppi_data.tsv 文件，请检查路径。")
    exit()

# 1.1 清洗列名 (自动去除表头的 # 号，处理 STRING 格式)
df.columns = [c.strip('#').strip() for c in df.columns]

# 1.2 确保列名统一
# STRING 导出通常包含 node1, node2, combined_score
# 有时可能是 string_identifier_1 等，这里做个保险映射
rename_map = {
    'string_identifier_1': 'node1', 'preferredName_A': 'node1', 
    'string_identifier_2': 'node2', 'preferredName_B': 'node2',
    'score': 'combined_score'
}
df.rename(columns=rename_map, inplace=True)

# 1.3 确保分数在 0-1 之间
if df['combined_score'].max() > 1.0:
    df['combined_score'] = df['combined_score'] / 1000.0

# 1.4 筛选置信度 (双重保险)
df_clean = df[df['combined_score'] >= 0.7].copy()

# 构建无向图
G = nx.from_pandas_edgelist(df_clean, 'node1', 'node2', edge_attr='combined_score')

# ==========================================
# 2. 计算关键拓扑指标
# ==========================================
# (1) 基础指标
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degrees = [d for n, d in G.degree()]
# 平均度 = 2 * 边数 / 节点数
avg_degree = sum(degrees) / num_nodes

# (2) 平均聚类系数 (Average Clustering Coefficient)
# 反映节点的"抱团"程度
avg_clustering = nx.average_clustering(G)

# (3) 路径长度与直径 (必须在连通图中计算)
if nx.is_connected(G):
    G_main = G
    status = "完全连通"
else:
    # 如果有孤立节点，取最大的连通子图 (Giant Component)
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    status = f"非完全连通 (基于最大子图 {G_main.number_of_nodes()} 节点计算)"

avg_path_len = nx.average_shortest_path_length(G_main)
diameter = nx.diameter(G_main)

# ==========================================
# 3. 输出结果
# ==========================================
print("=" * 50)
print("【PPI 网络全局拓扑特性分析结果】")
print("=" * 50)
print(f"1. 节点总数 (N):            {num_nodes}")
print(f"2. 边总数 (E):              {num_edges}")
print(f"3. 网络连通性:              {status}")
print(f"4. 平均度 (Avg Degree):     {avg_degree:.2f}")
print(f"5. 平均聚类系数 (Avg CC):   {avg_clustering:.4f}")
print(f"6. 平均路径长度 (L):        {avg_path_len:.4f}")
print(f"7. 网络直径 (Diameter):     {diameter}")
print("-" * 50)

# 找出 Top 5 枢纽蛋白 (Hubs)
print("【关键枢纽蛋白 (Top 5 Hubs)】")
sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
for i, (node, deg) in enumerate(sorted_nodes[:5]):
    print(f"  {i+1}. {node} (Degree: {deg})")
print("=" * 50)

# ==========================================
# 4. 绘图：度分布直方图
# ==========================================
plt.figure(figsize=(8, 6))
# 绘制直方图
plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), 
         color='#4c72b0', edgecolor='white', alpha=0.8)
plt.title("Degree Distribution of PPI Network", fontsize=14)
plt.xlabel("Degree (k)", fontsize=12)
plt.ylabel("Frequency P(k)", fontsize=12)
# 画一条平均线
plt.axvline(avg_degree, color='red', linestyle='dashed', linewidth=1.5, label=f'Avg: {avg_degree:.1f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 保存图片
plt.savefig('degree_distribution.png', dpi=300)
print("图表已保存为 'degree_distribution.png'")
plt.show()