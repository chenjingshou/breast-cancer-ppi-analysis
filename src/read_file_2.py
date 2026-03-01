import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置绘图风格，符合学术发表标准
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 加载真实数据 (Real Data Loading)
# ==========================================
filename = 'string_interactions_short.tsv' # 使用短版本
print(f"正在读取数据文件: {filename} ...")

#表头包含 #node1, node2, combined_score
df = pd.read_csv(filename, sep='\t')

# 清洗列名：去除 # 号
df.columns = [c.strip('#').strip() for c in df.columns]

# 确保列名正确映射
rename_map = {
    'node1': 'node1', 
    'node2': 'node2',
    'combined_score': 'weight' # 将分数重命名为权重
}
# 数据显示 0.915, 0.724，说明已经是归一化后的数据 [cite: 3]
if df['combined_score'].max() > 1.0:
    df['combined_score'] = df['combined_score'] / 1000.0

df['weight'] = df['combined_score']

# 构建网络
print("正在构建 NetworkX 图对象...")
G = nx.from_pandas_edgelist(df, 'node1', 'node2', edge_attr='weight')

# 提取最大连通子图 (LCC) - 生物网络分析通常只关注最大的那个团
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()

print("="*40)
print(f"【网络基本概况】")
print(f"原始节点数: {G.number_of_nodes()}")
print(f"原始边数:   {G.number_of_edges()}")
print(f"最大连通子图节点数 (LCC): {G_lcc.number_of_nodes()}")
print(f"LCC 边数: {G_lcc.number_of_edges()}")
print("="*40)

# ==========================================
# 2. 幂律分布验证 (Power-law Validation)
# ==========================================
# 这是证明网络具有"无尺度"特性的核心证据
degrees = [d for n, d in G_lcc.degree()]

plt.figure(figsize=(12, 6))

# 子图1: 普通直方图
plt.subplot(1, 2, 1)
sns.histplot(degrees, bins=50, kde=True, color='skyblue')
plt.title("度分布 (Degree Distribution)", fontsize=14)
plt.xlabel("Degree (k)")
plt.ylabel("Count")

# 子图2: 双对数坐标图 (Log-Log Plot)
# 如果这是一条直线，就证明它是 Scale-free 的
plt.subplot(1, 2, 2)
degree_freq = nx.degree_histogram(G_lcc)
degrees_x = range(len(degree_freq))
plt.loglog(degrees_x, degree_freq, 'go-', markersize=4, alpha=0.6) 
plt.title("双对数度分布 (Log-Log Plot)", fontsize=14)
plt.xlabel("Degree (k) - Log scale")
plt.ylabel("Frequency - Log scale")
plt.text(0.1, 0.1, "直线特征 = 无尺度特性", transform=plt.gca().transAxes, color='red')

plt.tight_layout()
plt.savefig("topology_analysis.png", dpi=300)
print("拓扑分析图已保存为 topology_analysis.png")
plt.show()

import random

# ==========================================
# 3. 网络鲁棒性压力测试 (Attack Simulation)
# ==========================================
def simulate_attack(graph, mode='random'):
    """
    模拟攻击过程：逐步移除节点，记录最大连通子图的相对大小
    """
    g_temp = graph.copy()
    initial_size = g_temp.number_of_nodes()
    history = []
    
    # 确定移除顺序
    nodes = list(g_temp.nodes())
    if mode == 'random':
        random.shuffle(nodes)
    elif mode == 'targeted':
        # 按度中心性排序，优先攻击 Hub (如 EGFR, TP53)
        nodes = sorted(g_temp.degree, key=lambda x: x[1], reverse=True)
        nodes = [n[0] for n in nodes]
        
    # 执行移除循环 (模拟删除前 40% 的节点)
    remove_count = int(initial_size * 0.4) 
    
    # 记录初始状态
    history.append(1.0) # 100%
    
    for i in range(remove_count):
        node_to_remove = nodes[i]
        g_temp.remove_node(node_to_remove)
        
        # 计算当前的 LCC 比例
        if g_temp.number_of_nodes() > 0:
            if nx.number_connected_components(g_temp) > 0:
                core = max(nx.connected_components(g_temp), key=len)
                ratio = len(core) / initial_size
                history.append(ratio)
            else:
                history.append(0)
        else:
            history.append(0)
            
    return history

print("正在进行随机攻击模拟 (Random Failure)...")
res_random = simulate_attack(G_lcc, mode='random')

print("正在进行靶向攻击模拟 (Targeted Attack)...")
res_targeted = simulate_attack(G_lcc, mode='targeted')

# 绘图
plt.figure(figsize=(10, 6))
x_axis = np.linspace(0, 40, len(res_random)) # 移除比例 0-40%

plt.plot(x_axis, res_random, label='随机故障 (Random Failure)', color='green', linewidth=2.5, alpha=0.8)
plt.plot(x_axis, res_targeted, label='靶向攻击 (Targeted Attack)', color='red', linewidth=2.5, linestyle='--')

plt.title("网络鲁棒性分析：Hub 节点的重要性", fontsize=15)
plt.xlabel("移除节点的比例 (%)", fontsize=12)
plt.ylabel("最大连通子图的相对大小 (LCC Size)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='-.', alpha=0.5)

# 标注关键差异
plt.annotate('网络迅速崩溃', xy=(5, 0.4), xytext=(10, 0.6),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig("network_robustness.png", dpi=300)
print("鲁棒性分析曲线已保存为 network_robustness.png")
plt.show()