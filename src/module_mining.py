"""
module_mining.py
----------------
Detect functional modules in the PPI network using:
  1. Louvain community detection (python-louvain / community package).
  2. Spectral Biclustering on the adjacency matrix (scikit-learn).
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralBiclustering


def louvain_communities(G: nx.Graph, random_state: int = 42) -> pd.DataFrame:
    """
    Apply the Louvain algorithm to detect communities.
    Returns a DataFrame with columns ['protein', 'community'].
    """
    try:
        import community as community_louvain
    except ImportError as exc:
        raise ImportError(
            "python-louvain is required for Louvain community detection. "
            "Install it with: pip install python-louvain"
        ) from exc

    partition = community_louvain.best_partition(G, random_state=random_state)
    df = pd.DataFrame(list(partition.items()), columns=["protein", "community"])
    return df.sort_values("community").reset_index(drop=True)


def spectral_biclustering(
    G: nx.Graph,
    n_clusters: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply Spectral Biclustering on the adjacency matrix of the graph.
    Returns a DataFrame with columns ['protein', 'bicluster_row'].
    """
    nodes = list(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)

    model = SpectralBiclustering(n_clusters=n_clusters, random_state=random_state)
    model.fit(adj_matrix)

    df = pd.DataFrame({"protein": nodes, "bicluster_row": model.row_labels_})
    return df.sort_values("bicluster_row").reset_index(drop=True)


def run_module_mining(
    edge_csv: str,
    output_louvain: str,
    output_bicluster: str,
    source_col: str = "protein1",
    target_col: str = "protein2",
    n_clusters: int = 10,
) -> None:
    """End-to-end module mining: load graph, run both methods, save CSVs."""
    df = pd.read_csv(edge_csv)
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)

    louvain_df = louvain_communities(G)
    louvain_df.to_csv(output_louvain, index=False)
    print(f"Louvain communities saved to {output_louvain}")

    bicluster_df = spectral_biclustering(G, n_clusters=n_clusters)
    bicluster_df.to_csv(output_bicluster, index=False)
    print(f"Spectral biclusters saved to {output_bicluster}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python module_mining.py <edge_csv> <output_louvain_csv> <output_bicluster_csv>"
        )
        sys.exit(1)

    run_module_mining(sys.argv[1], sys.argv[2], sys.argv[3])
