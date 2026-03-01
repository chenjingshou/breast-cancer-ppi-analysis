import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置与数据加载
# ==========================================
FILENAME = 'ppi_data.tsv'  # 下载的文件名

def load_data(filename):
    """读取STRING导出的TSV文件，如果不存在则生成模拟数据"""
    if os.path.exists(filename):
        print(f"正在读取本地文件: {filename}")
        df = pd.read_csv(filename, sep='\t')
    else:
        print("【提示】未找到本地文件，正在生成'模拟数据'以供演示...")
        # 模拟一份类似 STRING 的数据结构
        data = {
            'node1': ['TP53', 'TP53', 'BRCA1', 'BRCA1', 'PTEN', 'AKT1', 'MYC', 'EGFR'],
            'node2': ['BRCA1', 'MDM2', 'BRCA2', 'TP53', 'PIK3CA', 'PIK3CA', 'CCND1', 'ERBB2'],
            'combined_score': [0.99, 0.95, 0.98, 0.99, 0.96, 0.95, 0.92, 0.97]
        }
        df = pd.DataFrame(data)
    return df

# ==========================================
# 2. 数据清洗与网络构建
# ==========================================
df = load_data(FILENAME)

# 打印列名，STRING导出的列名可能不同，通常是 'node1', 'node2', 'combined_score' 
# 或者 'preferredName_A', 'preferredName_B', 'score'
# 这里做一个简单的列名映射处理
rename_map = {
    '#node1': 'node1', 'string_identifier_1': 'node1', 'preferredName_A': 'node1',
    'node2': 'node2', 'string_identifier_2': 'node2', 'preferredName_B': 'node2',
    'score': 'combined_score'
}
df.rename(columns=rename_map, inplace=True)

# 题目要求：仅保留置信度 > 0.7
# 注意：STRING下载文件中的 score 可能是 0-1 或 0-1000
if df['combined_score'].max() > 1.0:
    df['combined_score'] = df['combined_score'] / 1000.0

df_clean = df[df['combined_score'] >= 0.7].copy()

# 构建无向图
G = nx.from_pandas_edgelist(df_clean, 'node1', 'node2', edge_attr='combined_score')

# ==========================================
# 3.不仅是构建，先看一眼基本信息
# ==========================================
print("-" * 30)
print(f"网络构建完成！")
print(f"节点数量 (Nodes): {G.number_of_nodes()}")
print(f"边数量 (Edges):   {G.number_of_edges()}")
print("-" * 30)

# 简单画个图预览一下
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
plt.title("Protein-Protein Interaction Network Preview")
plt.show()