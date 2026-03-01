import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
import sys

# 设置风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)

# ==========================================
# 1. 加载全量数据 (智能适配格式)
# ==========================================
filename = 'string_interactions_short.tsv'
print(f"1. 读取全量数据: {filename} ...")

try:
    df = pd.read_csv(filename, sep='\t', header=None)
    last_col = df.shape[1] - 1
    df.rename(columns={0: 'node1', 1: 'node2', last_col: 'weight'}, inplace=True)
    
    # 清洗：去除表头
    df = df[df['node1'].astype(str) != 'node1']
    df = df[df['node1'].astype(str) != '#node1']
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df.dropna(subset=['weight'], inplace=True)

    # === 自动判断数据范围 ===
    max_val = df['weight'].max()
    print(f"   -> 检测到权重最大值: {max_val}")
    
    if max_val > 1.0:
        print("   -> 数据格式为 0-1000，执行归一化...")
        df['weight'] = df['weight'] / 1000.0
        # 过滤阈值设为 400 (对应 0.4)
        df = df[df['weight'] > 0.4] 
    else:
        print("   -> 数据格式为 0-1，保持原样...")
        # 过滤阈值设为 0.4
        df = df[df['weight'] > 0.4]
    
    if len(df) == 0:
        print("过滤后数据为空！请检查阈值设置。")
        sys.exit(1)

    print(f"   -> 保留高置信度边数: {len(df)}")

    # 构建网络
    G = nx.from_pandas_edgelist(df, 'node1', 'node2', edge_attr='weight')
    
    # 提取最大连通子图
    if len(G) == 0:
         print("错误：图为空。")
         sys.exit(1)
         
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()
    
    print(f"   -> 网络加载完成: {G_lcc.number_of_nodes()} 节点 | {G_lcc.number_of_edges()} 边")

except Exception as e:
    print(f"数据处理失败: {e}")
    sys.exit(1)

# ==========================================
# 2. 构建邻接矩阵
# ==========================================
print("2. 构建邻接矩阵...")
nodes = list(G_lcc.nodes())
adj_matrix = nx.to_numpy_array(G_lcc, nodelist=nodes)
np.fill_diagonal(adj_matrix, 0) # 去自环

# ==========================================
# 3. 执行谱双聚类
# ==========================================
n_clusters = 5
print(f"3. 执行谱双聚类 (N_clusters={n_clusters})...")

try:
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(adj_matrix)
    
    # 获取重排后的矩阵
    fit_data = adj_matrix[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    
    print("   -> 双聚类完成。")

except Exception as e:
    print(f"聚类失败: {e}")
    sys.exit(1)

# ==========================================
# 4. 可视化：混沌 vs 秩序
# ==========================================
print("4. 生成矩阵重排对比图...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 原始矩阵
ax1.spy(adj_matrix, markersize=0.1, color='#4c72b0')
ax1.set_title(f"原始邻接矩阵 (Original)\nNodes: {len(nodes)}", fontsize=14)
ax1.set_xlabel("Gene Index")
ax1.set_ylabel("Gene Index")

# 重排矩阵
ax2.spy(fit_data, markersize=0.1, color='#c44e52')
ax2.set_title(f"双聚类重排矩阵 (Spectral Co-Clustering)\nBlock Structure Revealed", fontsize=14)
ax2.set_xlabel("Reordered Index")
ax2.set_yticks([])

plt.tight_layout()
plt.savefig('biclustering_matrix.png', dpi=300)
print("双聚类矩阵图已保存: biclustering_matrix.png")
print("-" * 30)