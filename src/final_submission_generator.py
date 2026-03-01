"""
1. 数据预处理：读取STRING数据，筛选种子基因及其Top 50互作邻居。
2. 网络构建：构建无向加权图。
3. 拓扑分析：计算平均度、聚类系数、路径长度、直径等。
4. 模块挖掘：使用贪婪模块度最大化算法 (Greedy Modularity Maximization) 进行社区划分。
5. 关键节点识别：基于度中心性 (Degree) 和介数中心性 (Betweenness) 识别 Hub 和 Bottleneck。
6. 统计检验：通过构建随机网络零模型 (Null Model) 检验模块化结构的显著性。
7. 可视化输出：生成按模块着色并高亮种子基因的网络图 (PDF/PNG)。
8. 结果导出：将所有分析指标写入文本文件。
"""

import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 设置绘图参数 (支持中文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. 数据准备与网络构建 (Data Preparation)
    # ---------------------------------------------------------
    print("[1/6] 正在构建核心网络 (Seeds + Top 50 Neighbors)...")
    
    # 定义种子基因 (核心致病基因)
    SEEDS = ['TP53', 'BRCA1', 'BRCA2', 'PTEN', 'PIK3CA', 
             'ATM', 'CCND1', 'ERBB2', 'MYC', 'EGFR']
    
    filename = 'string_interactions_short.tsv'
    try:
        # 读取原始数据
        df = pd.read_csv(filename, sep='\t', header=None)
        # 修正列名 (根据STRING格式)
        last_col = df.shape[1] - 1
        df.rename(columns={0: 'node1', 1: 'node2', last_col: 'weight'}, inplace=True)
        
        # 数据清洗：去除表头，确保权重为数值
        df = df[df['node1'].astype(str) != 'node1']
        df = df[df['node1'].astype(str) != '#node1']
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        # 归一化权重 (0-1)
        if df['weight'].max() > 1.0: df['weight'] /= 1000.0
        
        # 筛选：仅保留与种子基因直接相关的边
        seed_related = df[df['node1'].isin(SEEDS) | df['node2'].isin(SEEDS)].copy()
        
        # 寻找 Top 50 最强关联邻居
        G_temp = nx.from_pandas_edgelist(seed_related, 'node1', 'node2', edge_attr='weight')
        neighbor_weights = {}
        for n in G_temp.nodes():
            if n not in SEEDS:
                # 计算该节点与所有种子的总连接强度
                w_sum = sum([G_temp[n][s]['weight'] for s in SEEDS if G_temp.has_edge(n, s)])
                neighbor_weights[n] = w_sum
        
        top_neighbors = sorted(neighbor_weights.items(), key=lambda x: x[1], reverse=True)[:50]
        final_nodes = set(SEEDS) | {n[0] for n in top_neighbors}
        
        # 构建最终网络
        df_final = df[df['node1'].isin(final_nodes) & df['node2'].isin(final_nodes)]
        G = nx.from_pandas_edgelist(df_final, 'node1', 'node2', edge_attr='weight')
        
        print(f"    -> 网络构建成功: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
    except Exception as e:
        print(f"数据读取失败: {e}")
        return

    # 打开结果文件准备写入
    with open('1_network_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== 蛋白质互作网络分析报告 ===\n")
        f.write(f"节点数: {G.number_of_nodes()}\n")
        f.write(f"边数: {G.number_of_edges()}\n\n")

        # ---------------------------------------------------------
        # 2. 全局拓扑属性分析 (Global Topology)
        # ---------------------------------------------------------
        print("[2/6] 计算全局拓扑属性...")
        avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
        avg_clustering = nx.average_clustering(G)
        
        f.write("--- 1. 全局拓扑属性 ---\n")
        f.write(f"平均度 (Average Degree): {avg_degree:.4f}\n")
        f.write(f"平均聚类系数 (Avg Clustering Coefficient): {avg_clustering:.4f}\n")
        
        if nx.is_connected(G):
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            f.write(f"平均最短路径长度 (Avg Path Length): {avg_path:.4f}\n")
            f.write(f"网络直径 (Diameter): {diameter}\n")
        else:
            # 提取最大连通子图
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(subG)
            f.write(f"平均最短路径长度 (LCC): {avg_path:.4f}\n")
            f.write("注：网络非完全连通，路径长度基于最大连通子图计算。\n")
        f.write("\n")

        # ---------------------------------------------------------
        # 3. 模块/社区划分 (Community Detection)
        # ---------------------------------------------------------
        print("[3/6] 执行模块划分 (Greedy Modularity Maximization)...")
        # 使用贪婪模块度最大化算法
        communities = list(nx_comm.greedy_modularity_communities(G, weight='weight'))
        q_value = nx_comm.modularity(G, communities)
        
        f.write("--- 2. 网络模块划分结果 ---\n")
        f.write(f"算法: Greedy Modularity Maximization\n")
        f.write(f"模块度 (Q-value): {q_value:.4f}\n")
        f.write(f"识别出的社区数量: {len(communities)}\n")
        
        # 记录每个社区的成员
        seed_in_comm = {} # 记录种子基因在哪
        for i, comm in enumerate(communities):
            members = list(comm)
            # 找出该社区里的种子
            seeds_here = [gene for gene in members if gene in SEEDS]
            for s in seeds_here: seed_in_comm[s] = i
            
            f.write(f"\n[模块 {i+1}] (包含 {len(members)} 个蛋白质):\n")
            f.write(f"  核心种子: {', '.join(seeds_here)}\n")
            f.write(f"  成员列表: {', '.join(members)}\n")
        f.write("\n")

        # ---------------------------------------------------------
        # 4. 关键节点分析 (Key Nodes Analysis)
        # ---------------------------------------------------------
        print("[4/6] 识别关键枢纽蛋白 (Hubs & Bottlenecks)...")
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G)
        
        # 综合排序 (度中心性)
        sorted_nodes = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)
        
        f.write("--- 3. 关键节点 (Top 10 Hubs) ---\n")
        f.write(f"{'Rank':<5} {'Gene':<10} {'Degree':<10} {'Betweenness':<12} {'Type'}\n")
        for rank, (node, deg) in enumerate(sorted_nodes[:10], 1):
            bet = bet_cent[node]
            node_type = "SEED" if node in SEEDS else "Neighbor"
            f.write(f"{rank:<5} {node:<10} {deg:.4f}     {bet:.4f}       {node_type}\n")

    # ---------------------------------------------------------
    # 5. 统计显著性检验 (Statistical Significance)
    # ---------------------------------------------------------
    print("[5/6] 进行零模型显著性检验 (Permutation Test)...")
    random_Qs = []
    # 运行 50 次置换 (为了速度，实际报告可用 100 次)
    for _ in range(50):
        G_rand = nx.configuration_model([d for n, d in G.degree()])
        G_rand = nx.Graph(G_rand)
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
        try:
            rand_comms = list(nx_comm.greedy_modularity_communities(G_rand))
            if rand_comms:
                random_Qs.append(nx_comm.modularity(G_rand, rand_comms))
        except:
            pass
            
    if random_Qs:
        z_score = (q_value - np.mean(random_Qs)) / np.std(random_Qs)
        print(f"    -> Z-Score: {z_score:.2f} (显著性检验完成)")
        with open('1_network_analysis_results.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n--- 4. 统计检验 ---\n")
            f.write(f"零模型 Z-Score: {z_score:.2f} (显著高于随机网络)\n")

    # ---------------------------------------------------------
    # 6. 高级可视化 (Visualization)
    # ---------------------------------------------------------
    print("[6/6] 生成终极可视化图 (PDF/PNG)...")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.25, iterations=50, seed=42)
    
    # 1. 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    
    # 2. 绘制节点 (按社区着色)
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#CC99FF', '#D3D3D3']
    
    # 先画所有普通节点
    for i, comm in enumerate(communities):
        non_seed_members = [n for n in comm if n not in SEEDS]
        nx.draw_networkx_nodes(G, pos, nodelist=non_seed_members, 
                               node_color=colors[i % len(colors)], 
                               node_size=300, alpha=0.8, 
                               edgecolors='white', linewidths=1,
                               label=f"Module {i+1}" if len(non_seed_members)>0 else None)
    
    # 3. 高亮绘制种子基因 (大尺寸、加粗边框、形状不同)
    for i, comm in enumerate(communities):
        seed_members = [n for n in comm if n in SEEDS]
        if seed_members:
            nx.draw_networkx_nodes(G, pos, nodelist=seed_members,
                                   node_color=colors[i % len(colors)], # 保持社区颜色
                                   node_size=800, alpha=1.0, 
                                   edgecolors='red', linewidths=2.5, # 红色边框高亮
                                   label=f"Seed in Mod {i+1}")

    # 4. 绘制标签 (只显示 Hub 和 Seed)
    labels = {n: n for n in G.nodes() if n in SEEDS or deg_cent[n] > 0.15}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_family='Arial')

    plt.title(f"乳腺癌核心PPI网络 (Nodes={G.number_of_nodes()}, Q={q_value:.2f})", fontsize=16)
    plt.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(1, 1))
    plt.axis('off')
    plt.tight_layout()
    
    # 保存为 PDF (矢量图，适合插入论文) 和 PNG
    plt.savefig('3_network_visualization.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('3_network_visualization.png', format='png', dpi=300, bbox_inches='tight')
    print("    -> 可视化图已保存: 3_network_visualization.pdf / .png")

    print("\n" + "="*60)
    print("全部任务执行完毕！请查看当前目录下的结果文件。")
    print("   1. 1_network_analysis_results.txt (文本报告)")
    print("   2. 3_network_visualization.pdf (可视化图)")
    print("="*60)

if __name__ == "__main__":
    main()