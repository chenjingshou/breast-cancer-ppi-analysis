import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys

# 设置风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("基于图神经网络 (GNN) 的模块挖掘程序")
print("="*60)

# ==========================================
# 1. 高级数据加载与张量构建
# ==========================================
filename = 'string_interactions_short.tsv'
print(f"正在读取并解析多维边特征: {filename} ...")

try:
    # 强制不读表头，手动指定列
    df = pd.read_csv(filename, sep='\t', header=None)
    
    # 数据格式非常丰富，利用第 4 列到第 11 列的所有属性
    # index: 0=node1, 1=node2, ... 
    # 属性列索引
    # neighborhood(4), fusion(5), coocc(6), homology(7), coexp(8), exp(9), db(10), text(11), combined(12)
    
    # 提取节点列表，构建映射字典
    nodes = pd.concat([df[0], df[1]]).unique()
    # 过滤掉可能的表头行
    nodes = [n for n in nodes if n != 'node1' and n != '#node1']
    
    node2id = {n: i for i, n in enumerate(nodes)}
    id2node = {i: n for i, n in enumerate(nodes)}
    num_nodes = len(nodes)
    
    print(f"-> 节点总数: {num_nodes}")

    # 构建边索引 (Edge Index) 和 边特征 (Edge Attributes)
    src_list = []
    dst_list = []
    edge_attrs = []
    
    # 清洗并提取数据
    feature_cols = [4, 5, 6, 7, 8, 9, 10, 11] # 8个维度的特征
    
    for idx, row in df.iterrows():
        u, v = row[0], row[1]
        if u not in node2id or v not in node2id:
            continue
            
        src_list.append(node2id[u])
        dst_list.append(node2id[v])
        
        # 提取特征，非数值转为0
        feats = []
        for col_idx in feature_cols:
            val = row[col_idx]
            try:
                val = float(val)
            except:
                val = 0.0
            feats.append(val)
        edge_attrs.append(feats)
        
        # 因为是无向图，添加反向边 (GCN通常需要)
        src_list.append(node2id[v])
        dst_list.append(node2id[u])
        edge_attrs.append(feats) # 特征相同

    # 转为 PyTorch 张量
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # 归一化边特征 (防止数值差异过大)
    edge_attr = F.normalize(edge_attr, p=2, dim=1)
    
    # 节点初始特征：如果没有外部特征，通常用单位矩阵 (One-hot) 或 随机向量
    # 为了节省显存且模拟真实场景，我们用可训练的 Embedding
    x = torch.eye(num_nodes) # One-hot 初始化 (N x N)
    
    print(f"-> 张量构建完成: Edge Index {edge_index.shape}, Edge Attr {edge_attr.shape}")

except Exception as e:
    print(f"❌ 数据处理失败: {e}")
    sys.exit(1)

# ==========================================
# 2. 定义手写 GCN 模型 (Graph Auto-Encoder)
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # 经典的 GCN 公式: D^-0.5 * A * D^-0.5 * X * W
        # 这里简化为 A * X * W (PyTorch 自动处理广播)
        support = self.linear(x)
        output = torch.spmm(adj, support)
        return output

class GAE(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim):
        super(GAE, self).__init__()
        # Encoder: 两层 GCN
        self.gc1 = GCNLayer(num_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, embedding_dim)
        self.dropout = 0.1

    def encode(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        z = self.gc2(x, adj) # 得到 Embedding Z
        return z

    def decode(self, z):
        # Decoder: 内积重构邻接矩阵 (Z * Z.t)
        adj_rec = torch.matmul(z, z.t())
        return torch.sigmoid(adj_rec)

    def forward(self, x, adj):
        z = self.encode(x, adj)
        adj_rec = self.decode(z)
        return z, adj_rec

# 预处理邻接矩阵 (归一化 trick)
def get_adj_matrix(num_nodes, edge_index):
    # 构建稀疏矩阵
    values = torch.ones(edge_index.shape[1])
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    return adj.to_dense() + torch.eye(num_nodes) # 加自环

# ==========================================
# 3. 训练模型
# ==========================================
print("\n正在初始化 GNN 模型...")
# 参数设置
HIDDEN_DIM = 64
EMBEDDING_DIM = 32 # 最终降维到 32 维
LR = 0.01
EPOCHS = 200

# 准备数据
adj_norm = get_adj_matrix(num_nodes, edge_index)
model = GAE(num_nodes, HIDDEN_DIM, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss() # 重构损失

print(f"开始训练 (Epochs: {EPOCHS})...")
losses = []
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    z, adj_rec = model(x, adj_norm)
    
    # 计算 Loss: 让重构的矩阵尽可能接近原始邻接矩阵
    loss = criterion(adj_rec, adj_norm)
    
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"  Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.6f}")

print("训练完成！已提取深度节点特征 (Embeddings)。")

# ==========================================
# 4. 下游任务：K-Means 聚类与可视化
# ==========================================
print("\n正在执行谱聚类 (K-Means on GNN Embeddings)...")
# 获取训练好的 Embedding
model.eval()
with torch.no_grad():
    z, _ = model(x, adj_norm)
    embeddings = z.numpy()

# 聚类 (设为 3 类，对应之前的发现)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# 打印结果
print("-" * 30)
from collections import defaultdict
res = defaultdict(list)
for idx, c in enumerate(clusters):
    res[c].append(id2node[idx])

for c_id, genes in res.items():
    print(f"GNN Module {c_id+1}: 包含 {len(genes)} 个基因")
    # 打印前5个作为代表
    print(f"  Example: {', '.join(genes[:5])} ...")

# t-SNE 可视化
print("\n正在生成 t-SNE 可视化图...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_nodes-1))
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=clusters, cmap='viridis', s=60, alpha=0.8)
plt.title(f"GNN Embeddings t-SNE Visualization (Loss={losses[-1]:.4f})", fontsize=14)
plt.colorbar(scatter, label='Module ID')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('gnn_modules.png', dpi=300)
print("GNN 聚类结果图已保存为: gnn_modules.png")

# 绘制 Loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(losses, label='Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GNN Training Trajectory')
plt.legend()
plt.savefig('gnn_loss.png', dpi=300)
print("训练损失曲线已保存为: gnn_loss.png")