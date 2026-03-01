import pandas as pd
import networkx as nx
import numpy as np

# ==========================================
# 拓扑特性精确计算脚本 (Topology Calculator)
# ==========================================
print("正在读取数据并计算精确拓扑指标...")

# 1. 暴力读取 (与之前的成功代码一致)
df = pd.read_csv('string_interactions_short.tsv', sep='\t', header=None)
last_col_idx = df.shape[1] - 1
df.rename(columns={0: 'node1', 1: 'node2', last_col_idx: 'weight'}, inplace=True)
df = df[['node1', 'node2', 'weight']]
# 清洗
df = df[df['node1'].astype(str) != 'node1']
df = df[df['node1'].astype(str) != '#node1']
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df.dropna(subset=['weight'], inplace=True)

# 2. 构建网络
G = nx.from_pandas_edgelist(df, 'node1', 'node2')
# 提取最大连通子图 (LCC) - 计算路径长度必须用这个
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()

print("-" * 30)
print(f"【真实实验数据结果】")
print(f"节点数 (Nodes): {G.number_of_nodes()}")
print(f"边数 (Edges):   {G.number_of_edges()}")
print(f"平均度 (Avg Degree): {2 * G.number_of_edges() / G.number_of_nodes():.4f}")

# 3. 开始计算耗时指标 (可能需要几十秒)
print("正在计算聚类系数 (这可能需要一点时间)...")
avg_clustering = nx.average_clustering(G)
print(f"平均聚类系数 (Avg Clustering): {avg_clustering:.4f}")

print("正在计算路径长度 (这也需要一点时间)...")
avg_path = nx.average_shortest_path_length(G_lcc)
diameter = nx.diameter(G_lcc)
print(f"平均最短路径 (Avg Path Length): {avg_path:.4f}")
print(f"网络直径 (Diameter): {diameter}")
print("-" * 30)