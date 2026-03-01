"""
gnn_gae.py
----------
Graph Auto-Encoder (GAE) using PyTorch to learn low-dimensional node embeddings
for the breast cancer PPI network.

Architecture:
  Encoder: two-layer GCN  (GraphConv)
  Decoder: inner-product between node embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Simple symmetric-normalised graph convolution: A_hat * X * W."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        return torch.mm(adj, support)


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder producing node embeddings of dimension `latent_dim`."""

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1 = GraphConv(n_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, adj))
        z = self.conv2(h, adj)
        return z


class GraphAutoEncoder(nn.Module):
    """
    Graph Auto-Encoder.

    Encoder: GCNEncoder
    Decoder: inner-product  (sigmoid(Z * Z^T))
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = GCNEncoder(n_features, hidden_dim, latent_dim)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.mm(z, z.t()))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, adj)
        adj_recon = self.decode(z)
        return adj_recon, z


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Compute the symmetrically normalised adjacency matrix: D^{-1/2} A D^{-1/2}."""
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    deg = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def train_gae(
    adj: torch.Tensor,
    features: torch.Tensor,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    device: str = "cpu",
) -> tuple[GraphAutoEncoder, torch.Tensor]:
    """
    Train a Graph Auto-Encoder and return the model and learned embeddings.

    Parameters
    ----------
    adj      : raw (unnormalised) adjacency matrix  [N x N]
    features : node feature matrix                  [N x F]
    """
    adj = adj.to(device)
    features = features.to(device)
    adj_norm = normalize_adjacency(adj).to(device)

    n_features = features.size(1)
    model = GraphAutoEncoder(n_features, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        adj_recon, _ = model(features, adj_norm)
        loss = F.binary_cross_entropy(adj_recon, adj)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, embeddings = model(features, adj_norm)

    return model, embeddings.cpu()


if __name__ == "__main__":
    import sys
    import pandas as pd
    import networkx as nx

    if len(sys.argv) < 3:
        print("Usage: python gnn_gae.py <edge_csv> <output_embeddings_csv>")
        sys.exit(1)

    edge_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df = pd.read_csv(edge_csv)
    G = nx.from_pandas_edgelist(df, source="protein1", target="protein2")
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    N = len(nodes)
    adj_np = nx.to_numpy_array(G, nodelist=nodes)
    adj_tensor = torch.FloatTensor(adj_np)
    features = torch.eye(N)

    _, embeddings = train_gae(adj_tensor, features)

    emb_df = pd.DataFrame(embeddings.numpy())
    emb_df.insert(0, "protein", nodes)
    emb_df.to_csv(output_csv, index=False)
    print(f"Node embeddings saved to {output_csv}")
