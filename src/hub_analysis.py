import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import sys

# ==========================================
# 0. 预置：癌症关键基因功能字典 (模拟查询功能)
# ==========================================
# 内置了常见 Hub 基因的生物学描述
GENE_DB = {
    # 模块 1 (生长/代谢) 常见基因
    'EGFR': '表皮生长因子受体，接收外部生长信号，驱动细胞增殖 (Ras-MAPK通路)。',
    'PIK3CA': 'PI3K激酶的催化亚基，激活Akt通路，促进细胞存活和代谢。',
    'SRC': '非受体酪氨酸激酶，参与细胞粘附、生长和迁移，与肿瘤转移密切相关。',
    'ERBB2': '即 HER2，在乳腺癌中常过表达，驱动侵袭性生长。',
    'PIK3R1': 'PI3K的调节亚基，对PI3K的稳定性和活性至关重要。',
    'GRB2': '接头蛋白，连接EGFR和Ras，是信号传导的关键桥梁。',
    'AKT1': '丝氨酸/苏氨酸激酶，抑制凋亡，调控细胞生存。',
    
    # 模块 2 (细胞周期/综合) 常见基因
    'TP53': '基因组卫士，监控DNA损伤。若无法修复则启动凋亡，是突变最频繁的抑癌基因。',
    'MYC': '转录因子，调控大量基因表达，驱动细胞周期的G1/S转换。',
    'CCND1': '细胞周期蛋白D1，与CDK4/6结合，推动细胞进入分裂期。',
    'STAT3': '转录因子，介导炎症与癌症的关联，促进持续增殖。',
    'CTNNB1': 'β-catenin，Wnt通路核心成员，调控细胞粘附和基因转录。',
    'HSP90AA1': '热休克蛋白，帮助许多致癌蛋白（如HER2, Akt）维持正确折叠。',
    
    # 模块 3 (DNA修复) 常见基因
    'BRCA1': '乳腺癌易感基因1，负责DNA双链断裂的同源重组修复。',
    'BRCA2': '乳腺癌易感基因2，招募RAD51进行DNA修复。',
    'ATM': 'DNA损伤感应激酶，检测到双链断裂后激活P53和BRCA1。',
    'RAD51': 'DNA重组酶，直接执行DNA链的交换和修复。',
    'PALB2': '连接BRCA1和BRCA2的桥梁蛋白，对复合物稳定性至关重要。',
    'BARD1': '与BRCA1形成异二聚体，维持其E3泛素连接酶活性。'
}

def get_func(gene_name):
    return GENE_DB.get(gene_name, "需手动查询 (Potential Driver Gene)")

# ==========================================
# 1. 构建网络 (Robust Loading)
# ==========================================
print("="*60)
print("枢纽蛋白 (Hub) 分析")
print("="*60)

filename = 'string_interactions_short.tsv'
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
except Exception as e:
    print(f"读取失败: {e}")
    sys.exit(1)

G = nx.from_pandas_edgelist(df, 'node1', 'node2')
# 提取 LCC
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
print(f"-> 网络加载完成: {G_lcc.number_of_nodes()} 节点 | {G_lcc.number_of_edges()} 边")

# ==========================================
# 2. 全局枢纽分析 (Global Hubs)
# ==========================================
print("\n" + "-"*30)
print("【全网 Top 3 枢纽蛋白】(Global Top 3)")
print("-"*30)

# 按度数排序
global_degrees = sorted(G_lcc.degree, key=lambda x: x[1], reverse=True)
top_global = global_degrees[:3]

for rank, (gene, degree) in enumerate(top_global, 1):
    desc = get_func(gene)
    print(f"No.{rank}: {gene} (Degree: {degree})")
    print(f"    功能: {desc}")

# ==========================================
# 3. 模块内枢纽分析 (Module Hubs)
# ==========================================
print("\n" + "-"*30)
print("【各模块 Top 3 枢纽蛋白】(Local Module Hubs)")
print("-"*30)

# 划分模块
partition = community_louvain.best_partition(G_lcc, weight='weight') 

# 整理模块
from collections import defaultdict
mod_groups = defaultdict(list)
for node, mod_id in partition.items():
    mod_groups[mod_id].append(node)

# 取最大的3个模块
top_modules = sorted(mod_groups.items(), key=lambda x: len(x[1]), reverse=True)[:3]

for i, (mod_id, nodes) in enumerate(top_modules):
    print(f"\n>>> 模块 {i+1} (包含 {len(nodes)} 个基因)")
    
    # 在该模块内部找度数最高的 (这里计算的是全网度数，更能反映影响力)
    local_hubs = sorted([(n, G_lcc.degree(n)) for n in nodes], key=lambda x: x[1], reverse=True)[:3]
    
    for rank, (gene, degree) in enumerate(local_hubs, 1):
        desc = get_func(gene)
        print(f"    {rank}. {gene} (Degree: {degree})")
        print(f"       功能: {desc}")

print("\n" + "="*60)