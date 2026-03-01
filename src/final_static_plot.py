import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain 
import numpy as np
import sys

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("="*50)

# ==========================================
# 1. (按位置读取)
# ==========================================
filename = 'string_interactions_short.tsv'
print(f"正在读取文件: {filename} ...")

try:
    # sep='\t': 强制用 Tab 分隔
    df = pd.read_csv(filename, sep='\t', header=None)
except Exception as e:
    print(f"读取失败: {e}")
    sys.exit(1)

print(f"-> 原始数据行数: {len(df)}")
if len(df) < 2:
    print("数据行数太少，请检查文件！")
    sys.exit(1)

# ==========================================
# 2. 强制指定列 (Mapping by Index)
# ==========================================
# STRING 格式：第0列=Node1, 第1列=Node2, 最后一列=Combined_Score
# 只取这三列
last_col_idx = df.shape[1] - 1
print(f"-> 锁定列索引: Node1=0, Node2=1, Score={last_col_idx}")

# 重命名
df.rename(columns={0: 'node1', 1: 'node2', last_col_idx: 'weight'}, inplace=True)

# 只保留这三列，防止其他列干扰
df = df[['node1', 'node2', 'weight']]

# ==========================================
# 3. 数据清洗 (去除混入的表头)
# ==========================================
# 如果读入了原来的表头行（包含 'node1' 或 '#node1' 字样），删掉它
df = df[df['node1'].astype(str) != 'node1']
df = df[df['node1'].astype(str) != '#node1']

# 确保权重是数字
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df.dropna(subset=['weight'], inplace=True)

# 归一化分数 (0-1000 -> 0-1)
if df['weight'].max() > 1.0:
    df['weight'] = df['weight'] / 1000.0

print(f"-> 清洗后有效互作数量: {len(df)}")

# ==========================================
# 4. 构建网络与社区发现
# ==========================================
print("正在构建网络图...")
G = nx.from_pandas_edgelist(df, 'node1', 'node2', edge_attr='weight')

# 提取最大连通子图 (LCC)
if nx.number_connected_components(G) > 0:
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()
else:
    G_lcc = G.copy()

print(f"网络构建成功! 当前规模: {G_lcc.number_of_nodes()} 节点 | {G_lcc.number_of_edges()} 边")

print("正在执行 Louvain 社区划分...")
partition = community_louvain.best_partition(G_lcc, weight='weight')
q_score = community_louvain.modularity(partition, G_lcc)
print(f"模块度 (Q-value): {q_score:.4f}")

# ==========================================
# 5. 绘图 (Visualization)
# ==========================================
print("正在计算布局 (Spring Layout)...")
plt.figure(figsize=(16, 14)) 

# 布局
pos = nx.spring_layout(G_lcc, k=0.2, seed=42, iterations=50)

# 颜色
try:
    cmap = plt.get_cmap('tab20')
except:
    cmap = plt.cm.tab20

node_colors = [cmap(partition[node] % 20) for node in G_lcc.nodes()]
degrees = dict(G_lcc.degree())
# 节点大小：根据度数
node_sizes = [degrees[n] * 3 + 30 for n in G_lcc.nodes()]

# 画边
nx.draw_networkx_edges(G_lcc, pos, alpha=0.05, edge_color='#999999')
# 画点
nx.draw_networkx_nodes(G_lcc, pos, node_size=node_sizes, node_color=node_colors, 
                       alpha=0.9, linewidths=0.5, edgecolors='white')

# 标签：Top 20 Hubs
top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
labels = {n[0]: n[0] for n in top_nodes}

text_items = nx.draw_networkx_labels(G_lcc, pos, labels, font_size=12, font_weight='bold', font_color='#222222')
import matplotlib.patheffects as path_effects
for t in text_items.values():
    t.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

plt.title(f"Protein Interaction Functional Modules (Q={q_score:.2f})", fontsize=22, pad=20)
plt.axis('off') 

output_file = "final_module_map_v3.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"最终大图已保存为: {output_file}")

# ==========================================
# 6. 生成报告数据 (前3大模块)
# ==========================================
from collections import defaultdict
mod_groups = defaultdict(list)
for node, mod_id in partition.items():
    mod_groups[mod_id].append(node)

print("\n" + "="*50)
top_modules = sorted(mod_groups.items(), key=lambda x: len(x[1]), reverse=True)[:3]

for i, (mod_id, nodes) in enumerate(top_modules):
    local_degrees = sorted([(n, degrees[n]) for n in nodes], key=lambda x: x[1], reverse=True)
    top_hubs = [n[0] for n in local_degrees[:6]]
    print(f"模块 #{i+1} | 基因数: {len(nodes)}")
    print(f"核心 Hubs: {', '.join(top_hubs)}")
    print("-" * 30)